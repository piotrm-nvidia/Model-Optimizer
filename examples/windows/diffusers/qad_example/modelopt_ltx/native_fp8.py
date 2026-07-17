from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as functional
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .state import write_json

if TYPE_CHECKING:
    from pathlib import Path

CORRECT_PREFIX = "model.diffusion_model."
NON_TRANSFORMER_PREFIXES = (
    "vae.",
    "audio_vae.",
    "vocoder.",
    "text_embedding_projection.",
    "text_encoders.",
    "first_stage_model.",
    "cond_stage_model.",
    "conditioner.",
)
REMOVABLE_MARKERS = (
    "_amax",
    "_zero_point",
    "input_quantizer",
    "weight_quantizer",
    "output_quantizer",
    "_teacher_model",
    "_loss_modules",
)
STRIP_PREFIXES = (
    CORRECT_PREFIX,
    "_student_model.",
    "module.",
    "_orig_mod.",
    "diffusion_model.",
    "transformer.",
    "velocity_model.",
    "model.",
)


def strip_export_prefix(key: str) -> str:
    previous = None
    while key != previous:
        previous = key
        for prefix in STRIP_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
    return key


def scalar_fp8_scale(name: str, state: dict) -> torch.Tensor | None:
    amax = state.get("_amax")
    if amax is None:
        return None
    if not isinstance(amax, torch.Tensor) or amax.numel() != 1:
        raise ValueError(f"{name} has non-scalar amax")
    amax = amax.detach().cpu().float().abs().reshape(())
    if not torch.isfinite(amax) or amax <= 0:
        raise ValueError(f"{name} has invalid amax {amax.item()}")
    return (amax / 448.0).to(torch.float32)


def extract_fp8_linear_scales(
    quantizer_state: dict[str, dict],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    scales = {}
    suffix = ".weight_quantizer"
    for quantizer_name, state in quantizer_state.items():
        if not quantizer_name.endswith(suffix):
            continue
        weight_scale = scalar_fp8_scale(quantizer_name, state)
        if weight_scale is None:
            continue
        module_name = quantizer_name[: -len(suffix)]
        input_name = f"{module_name}.input_quantizer"
        input_state = quantizer_state.get(input_name)
        if input_state is None:
            raise ValueError(f"Missing {input_name}")
        input_scale = scalar_fp8_scale(input_name, input_state)
        if input_scale is None:
            raise ValueError(f"Missing calibrated amax for {input_name}")
        scales[strip_export_prefix(module_name)] = (weight_scale, input_scale)
    if not scales:
        raise ValueError("ModelOpt state contains no calibrated per-tensor FP8 linears")
    return scales


def convert_transformer_state(
    trained_state: dict[str, torch.Tensor],
    quantizer_state: dict[str, dict],
) -> dict[str, torch.Tensor]:
    scales = extract_fp8_linear_scales(quantizer_state)
    normalized = {}
    for key, value in trained_state.items():
        if any(marker in key for marker in REMOVABLE_MARKERS):
            continue
        if key.startswith(NON_TRANSFORMER_PREFIXES):
            continue
        normalized_key = strip_export_prefix(key)
        if normalized_key in normalized:
            raise ValueError(f"Duplicate normalized checkpoint key: {normalized_key}")
        normalized[normalized_key] = value.detach().cpu()

    converted = {}
    exported = set()
    for key, value in normalized.items():
        module_name = key[: -len(".weight")] if key.endswith(".weight") else None
        if module_name in scales:
            if not value.is_floating_point():
                raise TypeError(f"FP8 linear weight must be floating point: {key}")
            weight_scale, input_scale = scales[module_name]
            converted[key] = ((value.float() / weight_scale).clamp(-448.0, 448.0)).to(
                torch.float8_e4m3fn
            )
            converted[f"{module_name}.weight_scale"] = weight_scale.reshape(())
            converted[f"{module_name}.input_scale"] = input_scale.reshape(())
            exported.add(module_name)
        elif value.is_floating_point():
            converted[key] = value.to(torch.bfloat16)
        else:
            converted[key] = value
    missing = sorted(set(scales) - exported)
    if missing:
        raise KeyError(f"Trained checkpoint missing FP8 linear weights: {missing[:10]}")
    return converted


def validate_native_checkpoint(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        fp8_weights = [
            key
            for key in keys
            if key.endswith(".weight") and handle.get_tensor(key).dtype == torch.float8_e4m3fn
        ]
        if not fp8_weights:
            raise RuntimeError(f"Native FP8 checkpoint contains no E4M3 weights: {path}")
        scales = []
        for weight_key in fp8_weights:
            prefix = weight_key.removesuffix(".weight")
            weight_scale_key = f"{prefix}.weight_scale"
            input_scale_key = f"{prefix}.input_scale"
            if weight_scale_key not in keys or input_scale_key not in keys:
                raise RuntimeError(f"Missing FP8 scale companions for {weight_key}")
            for scale_key in (weight_scale_key, input_scale_key):
                scale = handle.get_tensor(scale_key)
                if (
                    scale.dtype != torch.float32
                    or scale.numel() != 1
                    or not torch.isfinite(scale).all()
                    or not (scale > 0).all()
                ):
                    raise RuntimeError(f"Invalid native FP8 scale {scale_key}: {scale}")
                scales.append(scale_key)
    return {
        "schema_version": 1,
        "checkpoint": str(path),
        "fp8_weight_count": len(fp8_weights),
        "scale_count": len(scales),
        "fp8_weight_keys": sorted(fp8_weights),
    }


def create_native_checkpoint(
    *,
    trained_path: Path,
    modelopt_state_path: Path,
    base_path: Path,
    output_path: Path,
) -> dict:
    from modelopt.torch.export.diffusers_utils import (
        build_layerwise_quant_metadata,
        merge_diffusion_checkpoint,
    )

    for path in (trained_path, modelopt_state_path, base_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    trained_state = load_file(trained_path, device="cpu")
    modelopt_state = torch.load(modelopt_state_path, map_location="cpu", weights_only=False)
    quantizer_state = modelopt_state.get("modelopt_state_weights")
    if not isinstance(quantizer_state, dict):
        raise ValueError(f"Missing modelopt_state_weights in {modelopt_state_path}")
    transformer_state = convert_transformer_state(trained_state, quantizer_state)
    quant_config = {"quant_algo": "FP8"}
    merged, metadata = merge_diffusion_checkpoint(
        transformer_state,
        str(base_path),
        "ltx2",
        hf_quant_config=quant_config,
    )
    metadata["_quantization_metadata"] = build_layerwise_quant_metadata(merged, quant_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(merged, str(temporary), metadata=metadata)
    temporary.replace(output_path)
    report = validate_native_checkpoint(output_path)
    write_json(output_path.with_suffix(".contract.json"), report)
    return report


def compare_exported_weights(
    *,
    trained_path: Path,
    native_path: Path,
    max_layers: int = 8,
) -> dict:
    """Compare source weights with dequantized native E4M3 weights."""
    with safe_open(trained_path, framework="pt", device="cpu") as trained:
        trained_keys = {
            strip_export_prefix(key): key
            for key in trained.keys()  # noqa: SIM118
        }
        with safe_open(native_path, framework="pt", device="cpu") as native:
            native_weights = [
                key
                for key in native.keys()  # noqa: SIM118
                if key.endswith(".weight") and native.get_tensor(key).dtype == torch.float8_e4m3fn
            ][:max_layers]
            if not native_weights:
                raise RuntimeError("No native FP8 weights available for parity")
            rows = []
            for native_key in native_weights:
                normalized = strip_export_prefix(native_key)
                source_key = trained_keys.get(normalized)
                if source_key is None:
                    raise KeyError(f"Source weight missing for {native_key}")
                source = trained.get_tensor(source_key).float()
                weight = native.get_tensor(native_key).float()
                scale = native.get_tensor(
                    native_key.removesuffix(".weight") + ".weight_scale"
                ).float()
                restored = weight * scale
                difference = restored - source
                rows.append(
                    {
                        "key": native_key,
                        "mean_abs_error": difference.abs().mean().item(),
                        "max_abs_error": difference.abs().max().item(),
                        "cosine_similarity": functional.cosine_similarity(
                            restored.reshape(1, -1),
                            source.reshape(1, -1),
                        ).item(),
                        "finite": bool(torch.isfinite(restored).all()),
                    }
                )
    if not all(row["finite"] for row in rows):
        raise RuntimeError(f"Non-finite native FP8 weight parity: {rows}")
    return {"comparison": "source weight vs dequantized native FP8", "rows": rows}
