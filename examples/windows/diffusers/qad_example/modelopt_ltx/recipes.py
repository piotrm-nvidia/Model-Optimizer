from __future__ import annotations

import copy

from modelopt.torch.quantization.config import FP8_DEFAULT_CFG, NVFP4_DEFAULT_CFG

SENSITIVE_LAYER_PATTERNS = [
    # Video-only workflows provide no activation samples for audio and
    # cross-audio branches. Quantizing those weights would create unpaired
    # scales that cannot satisfy LTX fp8-scaled-mm.
    "*audio*",
    "*patchify_proj*",
    "*adaln_single*",
    "*caption_projection*",
    "*proj_out*",
    "*audio_patchify_proj*",
    "*audio_adaln_single*",
    "*audio_caption_projection*",
    "*audio_proj_out*",
    "*av_ca_video_scale_shift_adaln_single*",
    "*av_ca_a2v_gate_adaln_single*",
    "*av_ca_audio_scale_shift_adaln_single*",
    "*av_ca_v2a_gate_adaln_single*",
]


def should_save_checkpoint(step: int, checkpoint_steps: set[int]) -> bool:
    return not checkpoint_steps or step in checkpoint_steps


def should_run_calibration(initial_modelopt_state: object | None) -> bool:
    return initial_modelopt_state is None


def quantization_only_state(state: dict) -> dict:
    filtered = dict(state)
    filtered["modelopt_state_dict"] = [
        entry for entry in state["modelopt_state_dict"] if entry[0] == "quantize"
    ]
    if not filtered["modelopt_state_dict"]:
        raise ValueError("ModelOpt state does not contain quantization mode")
    return filtered


def validate_calibration_counts(attempted: int, successful: int, failed: int) -> None:
    """Reject empty, inconsistent, or majority-failed calibration runs."""
    if attempted < 1:
        raise RuntimeError("PTQ calibration attempted no batches")
    if successful + failed != attempted:
        raise RuntimeError(
            f"PTQ calibration counts are inconsistent: "
            f"attempted={attempted} successful={successful} failed={failed}"
        )
    if successful < 1:
        raise RuntimeError(f"PTQ calibration produced no successful batches out of {attempted}")
    if failed > attempted * 0.5:
        raise RuntimeError(
            f"Too many PTQ calibration failures ({failed}/{attempted}); "
            f"successful batches={successful}"
        )


def validate_calibration_success(successful_batches: int) -> None:
    """Compatibility wrapper for older callers."""
    if successful_batches < 1:
        raise RuntimeError("PTQ calibration produced no successful batches")


def build_quant_config(
    exclude_blocks: list[int] | None = None,
    quant_recipe: str = "nvfp4",
) -> dict:
    """Build QAD recipe while preserving historical NVFP4 defaults."""
    if exclude_blocks is None:
        exclude_blocks = [0, 1, 46, 47]
    exclusions = [
        *[{"quantizer_name": pattern, "enable": False} for pattern in SENSITIVE_LAYER_PATTERNS],
        *[
            {"quantizer_name": f"*transformer_blocks.{index}.*", "enable": False}
            for index in exclude_blocks
        ],
    ]
    if quant_recipe == "fp8":
        config = copy.deepcopy(FP8_DEFAULT_CFG)
        config["quant_cfg"] = [*config["quant_cfg"], *exclusions]
        return config
    if quant_recipe != "nvfp4":
        raise ValueError(f"Unsupported quantization recipe: {quant_recipe}")

    nvfp4_numerics = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        "axis": None,
    }
    return {
        "quant_cfg": [
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": nvfp4_numerics,
                "enable": True,
            },
            {
                "quantizer_name": "*input_quantizer",
                "cfg": nvfp4_numerics,
                "enable": True,
            },
            *exclusions,
        ],
        "algorithm": NVFP4_DEFAULT_CFG["algorithm"],
    }
