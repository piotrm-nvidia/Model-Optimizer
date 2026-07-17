from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import get_unwrapped_name


def register_dynamic_quantizer_buffers(model, quantizer_state: dict) -> int:
    """Register lazily-created amax buffers before strict state loading."""
    registered = 0
    for name, module in model.named_modules():
        key = get_unwrapped_name(name, model)
        saved = quantizer_state.get(key, {})
        if (
            isinstance(module, TensorQuantizer)
            and "_amax" in saved
            and not hasattr(module, "_amax")
        ):
            module.amax = saved["_amax"]
            registered += 1
    return registered


def summarize_quantizer_state(model) -> dict:
    """Return fail-closed quantizer and amax inventory."""
    enabled = []
    disabled = []
    amax_total = amax_finite = amax_positive = 0
    for name, module in model.named_modules():
        if not isinstance(module, TensorQuantizer):
            continue
        target = enabled if module.is_enabled else disabled
        target.append(name)
        if not module.is_enabled or not hasattr(module, "_amax"):
            continue
        amax = module.amax.detach()
        amax_total += amax.numel()
        amax_finite += int(torch.isfinite(amax).sum().item())
        amax_positive += int((amax.abs() > 0).sum().item())
    return {
        "enabled_count": len(enabled),
        "disabled_count": len(disabled),
        "enabled_weight_count": sum(name.endswith("weight_quantizer") for name in enabled),
        "enabled_input_count": sum(name.endswith("input_quantizer") for name in enabled),
        "enabled_fqns": sorted(enabled),
        "disabled_fqns": sorted(disabled),
        "amax_total": amax_total,
        "amax_finite": amax_finite,
        "amax_positive": amax_positive,
    }


def validate_quantizer_state(summary: dict) -> None:
    """Reject missing, disabled-only, or invalid FP8 quantizer state."""
    if summary["enabled_weight_count"] < 1 or summary["enabled_input_count"] < 1:
        raise RuntimeError(f"FP8 weight/input quantizers are not enabled: {summary}")
    if summary["amax_total"] < 1:
        raise RuntimeError(f"Enabled quantizers contain no saved amax values: {summary}")
    if summary["amax_finite"] != summary["amax_total"]:
        raise RuntimeError(f"Enabled quantizers contain non-finite amax values: {summary}")
    if summary["amax_positive"] < 1:
        raise RuntimeError(f"Enabled quantizers contain no positive amax values: {summary}")


def quantizer_state_digest(quantizer_state: dict) -> str:
    """Hash quantizer tensor values and names without pickling object metadata."""
    digest = hashlib.sha256()
    for name in sorted(quantizer_state):
        digest.update(name.encode())
        for key, value in sorted(quantizer_state[name].items()):
            digest.update(key.encode())
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    """Write deterministic JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
