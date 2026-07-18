from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

EXAMPLE_DIR = Path(__file__).parents[3] / "examples/windows/diffusers/qad_example"
sys.path.insert(0, str(EXAMPLE_DIR))

from modelopt_ltx.artifacts import create_deploy_bundle, verify_deploy_bundle
from modelopt_ltx.cli import build_parser
from modelopt_ltx.native_fp8 import convert_transformer_state
from modelopt_ltx.recipes import (
    SENSITIVE_LAYER_PATTERNS,
    build_quant_config,
    quantization_only_state,
    should_run_calibration,
    should_save_checkpoint,
    validate_calibration_counts,
    validate_calibration_success,
)
from modelopt_ltx.runtime import validate_preprocessed
from modelopt_ltx.state import (
    quantizer_state_digest,
    register_dynamic_quantizer_buffers,
    summarize_quantizer_state,
    validate_quantizer_state,
)

from modelopt.torch.quantization.config import FP8_DEFAULT_CFG, QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer


def test_nvfp4_remains_default() -> None:
    config = build_quant_config()
    assert config["algorithm"] == "max"
    assert config["quant_cfg"][0]["cfg"]["num_bits"] == (2, 1)
    assert config["quant_cfg"][0]["cfg"]["block_sizes"][-1] == 16


def test_fp8_uses_copy_of_supported_preset_and_exclusions() -> None:
    original = copy.deepcopy(FP8_DEFAULT_CFG)
    config = build_quant_config([0, 47], quant_recipe="fp8")

    assert config is not FP8_DEFAULT_CFG
    assert original == FP8_DEFAULT_CFG
    assert config["algorithm"] == FP8_DEFAULT_CFG["algorithm"]
    names = [
        entry.get("quantizer_name") for entry in config["quant_cfg"] if isinstance(entry, dict)
    ]
    assert SENSITIVE_LAYER_PATTERNS[0] in names
    assert "*transformer_blocks.0.*" in names
    assert "*transformer_blocks.47.*" in names


def test_unknown_recipe_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        build_quant_config(quant_recipe="int4")


def test_exact_checkpoint_schedule() -> None:
    schedule = {1, 3, 10}
    assert [step for step in range(1, 11) if should_save_checkpoint(step, schedule)] == [
        1,
        3,
        10,
    ]
    assert should_save_checkpoint(7, set())


def test_initial_modelopt_state_skips_recalibration() -> None:
    assert should_run_calibration(None)
    assert not should_run_calibration("modelopt_state_step_00000.pth")


def test_deploy_state_excludes_distillation_mode() -> None:
    state = {
        "modelopt_version": "1.0",
        "modelopt_state_dict": [
            ("quantize", {"config": {}, "metadata": {}}),
            ("kd_loss", {"config": {}, "metadata": {}}),
        ],
    }
    filtered = quantization_only_state(state)
    assert [entry[0] for entry in filtered["modelopt_state_dict"]] == ["quantize"]
    assert len(state["modelopt_state_dict"]) == 2


def test_calibration_requires_successful_batch() -> None:
    with pytest.raises(RuntimeError, match="no successful batches"):
        validate_calibration_success(0)
    validate_calibration_success(1)


@pytest.mark.parametrize(
    ("attempted", "successful", "failed"),
    [(1, 1, 0), (8, 8, 0), (8, 4, 4)],
)
def test_calibration_counts_accept_usable_runs(
    attempted: int, successful: int, failed: int
) -> None:
    validate_calibration_counts(attempted, successful, failed)


@pytest.mark.parametrize(
    ("attempted", "successful", "failed", "message"),
    [
        (0, 0, 0, "attempted no batches"),
        (1, 0, 1, "no successful batches"),
        (8, 3, 5, "Too many"),
        (8, 7, 0, "inconsistent"),
    ],
)
def test_calibration_counts_reject_invalid_runs(
    attempted: int, successful: int, failed: int, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_calibration_counts(attempted, successful, failed)


def test_dynamic_amax_is_registered_before_strict_restore() -> None:
    model = nn.Sequential(TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=None)))
    state = {"0": {"_amax": torch.tensor(12.0)}}
    assert not hasattr(model[0], "_amax")
    assert register_dynamic_quantizer_buffers(model, state) == 1
    model[0].load_state_dict(state["0"])
    assert model[0].amax.item() == 12.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dynamic_amax_restore_can_preserve_existing_model_device() -> None:
    model = nn.Sequential(
        TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=None))
    ).cuda()
    target_device = next(model.parameters(), torch.empty(0, device="cuda")).device
    state = {"0": {"_amax": torch.tensor(12.0)}}

    register_dynamic_quantizer_buffers(model, state)
    model[0].load_state_dict(state["0"])
    model.to(device=target_device)

    assert model[0].amax.device == target_device
    assert model[0].amax.item() == 12.0


def test_quantizer_evidence_requires_enabled_weight_input_and_valid_amax() -> None:
    model = nn.Module()
    model.weight_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=None))
    model.input_quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=None))
    model.weight_quantizer.amax = torch.tensor(12.0)
    model.input_quantizer.amax = torch.tensor(6.0)

    summary = summarize_quantizer_state(model)
    validate_quantizer_state(summary)

    assert summary["enabled_weight_count"] == 1
    assert summary["enabled_input_count"] == 1
    assert summary["amax_total"] == 2
    assert summary["amax_finite"] == 2
    assert summary["amax_positive"] == 2


def test_quantizer_state_digest_changes_with_amax() -> None:
    first = {"q": {"_amax": torch.tensor(1.0)}}
    second = {"q": {"_amax": torch.tensor(2.0)}}

    assert quantizer_state_digest(first) != quantizer_state_digest(second)


def test_native_fp8_conversion_emits_e4m3_weights_and_scalar_scales() -> None:
    prefix = "transformer_blocks.2.attn.to_q"
    trained = {
        f"{prefix}.weight": torch.tensor([[-4.0, 2.0], [1.0, 3.0]], dtype=torch.bfloat16),
        f"{prefix}.bias": torch.ones(2, dtype=torch.float32),
        "transformer_blocks.0.attn.to_q.weight": torch.ones(2, 2, dtype=torch.float32),
    }
    quantizers = {
        f"_student_model.velocity_model.{prefix}.weight_quantizer": {"_amax": torch.tensor(896.0)},
        f"_student_model.velocity_model.{prefix}.input_quantizer": {"_amax": torch.tensor(224.0)},
    }

    converted = convert_transformer_state(trained, quantizers)

    assert converted[f"{prefix}.weight"].dtype == torch.float8_e4m3fn
    assert converted[f"{prefix}.weight_scale"].item() == pytest.approx(2.0)
    assert converted[f"{prefix}.input_scale"].item() == pytest.approx(0.5)
    assert converted[f"{prefix}.bias"].dtype == torch.bfloat16
    assert converted["transformer_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16


def test_deploy_bundle_is_independent_and_bitwise_equal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state = checkpoint / "modelopt_state_step_00000.pth"
    state.write_bytes(b"modelopt-state")
    output = tmp_path / "deploy"
    native = tmp_path / "native.safetensors"
    native.write_bytes(b"native-fp8")
    native_contract = tmp_path / "native.contract.json"
    native_contract.write_text("{}")

    create_deploy_bundle(
        checkpoint,
        output,
        base_model=tmp_path / "base.safetensors",
        recipe="fp8",
        native_checkpoint=native,
        native_contract=native_contract,
    )
    manifest = verify_deploy_bundle(output)

    copied = output / manifest.modelopt_state
    assert copied.read_bytes() == state.read_bytes()
    assert copied.stat().st_ino != state.stat().st_ino
    assert copied.stat().st_nlink == 1


def test_preprocess_rejects_empty_output(tmp_path: Path) -> None:
    (tmp_path / "latents").mkdir()
    (tmp_path / "conditions").mkdir()
    with pytest.raises(RuntimeError, match="Incomplete"):
        validate_preprocessed(tmp_path)
    (tmp_path / "latents/sample.pt").write_bytes(b"latent")
    (tmp_path / "conditions/sample.pt").write_bytes(b"condition")
    validate_preprocessed(tmp_path)


@pytest.mark.parametrize(
    ("arguments", "command"),
    [
        (
            [
                "ptq",
                "--quant-recipe",
                "fp8",
                "--config",
                "config.yaml",
                "--calibration-manifest",
                "train.json",
                "--output",
                "ptq",
            ],
            "ptq",
        ),
        (
            [
                "create-fp8-deploy",
                "--checkpoint",
                "checkpoint",
                "--config",
                "config.yaml",
                "--output",
                "deploy",
            ],
            "create-fp8-deploy",
        ),
    ],
)
def test_cli_contract(arguments: list[str], command: str) -> None:
    assert build_parser().parse_args(arguments).command == command
