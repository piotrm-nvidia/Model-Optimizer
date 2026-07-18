from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
from safetensors.torch import save_file

from modelopt.torch.quantization.nn import TensorQuantizer

from .artifacts import atomic_json, verify_deploy_bundle
from .runtime import (
    _load_fake_transformer,
    _load_native_transformer,
    _validation_config,
    load_config,
)

if TYPE_CHECKING:
    from pathlib import Path

REPRESENTATIVE_LAYERS = (
    "transformer_blocks.2.attn1.to_q",
    "transformer_blocks.2.ff.net.0.proj",
    "transformer_blocks.24.attn1.to_q",
    "transformer_blocks.24.ff.net.0.proj",
    "transformer_blocks.45.attn1.to_q",
    "transformer_blocks.45.ff.net.0.proj",
)
MAX_STATS_VALUES = 65536
MAX_SAVED_VALUES = 4096


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_summary(tensor: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    detached = tensor.detach()
    flat = detached.reshape(-1)
    stats_sample = flat[:MAX_STATS_VALUES].float()
    saved = flat[:MAX_SAVED_VALUES].cpu().contiguous()
    finite = torch.isfinite(stats_sample)
    summary = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": detached.numel(),
        "stats_numel": stats_sample.numel(),
        "finite": int(finite.sum().item()),
        "min": stats_sample.min().item() if stats_sample.numel() else None,
        "max": stats_sample.max().item() if stats_sample.numel() else None,
        "mean": stats_sample.mean().item() if stats_sample.numel() else None,
        "std": stats_sample.std().item() if stats_sample.numel() > 1 else 0.0,
        "saved_values": saved.numel(),
        "saved_sha256": hashlib.sha256(_tensor_bytes(saved)).hexdigest(),
    }
    return summary, saved


def _tensor_items(value: Any, prefix: str):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _tensor_items(
                getattr(value, field.name),
                f"{prefix}.{field.name}",
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_items(item, f"{prefix}.{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_items(item, f"{prefix}.{key}")


class TensorCapture:
    def __init__(self, transformer: torch.nn.Module):
        self.transformer = transformer
        self.summaries: dict[str, dict[str, Any]] = {}
        self.slices: dict[str, torch.Tensor] = {}
        self.input_hashes: dict[str, str] = {}
        self.handles = []
        self.input_captured = False

    def _save(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.summaries:
            return
        summary, saved = _tensor_summary(tensor)
        self.summaries[key] = summary
        self.slices[key.replace(".", "__")] = saved

    def _root_pre_hook(self, _module, args, kwargs) -> None:
        if self.input_captured:
            return
        for key, tensor in _tensor_items(args, "transformer.args"):
            self.input_hashes[key] = hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
            self._save(key, tensor)
        for key, tensor in _tensor_items(kwargs, "transformer.kwargs"):
            self.input_hashes[key] = hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
            self._save(key, tensor)
        self.input_captured = True

    def _root_hook(self, _module, _args, _kwargs, output) -> None:
        for key, tensor in _tensor_items(output, "transformer.output"):
            self._save(key, tensor)

    def _layer_hook(self, name: str):
        def hook(_module, inputs, output) -> None:
            for key, tensor in _tensor_items(inputs, f"{name}.input"):
                self._save(key, tensor)
            for key, tensor in _tensor_items(output, f"{name}.output"):
                self._save(key, tensor)

        return hook

    def __enter__(self):
        modules = dict(self.transformer.named_modules())
        missing = [name for name in REPRESENTATIVE_LAYERS if name not in modules]
        if missing:
            raise RuntimeError(f"Representative attribution layers missing: {missing}")
        self.handles.append(
            self.transformer.register_forward_pre_hook(
                self._root_pre_hook,
                with_kwargs=True,
            )
        )
        self.handles.append(
            self.transformer.register_forward_hook(
                self._root_hook,
                with_kwargs=True,
            )
        )
        for name in REPRESENTATIVE_LAYERS:
            self.handles.append(modules[name].register_forward_hook(self._layer_hook(name)))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()


class FakeQuantProof:
    def __init__(self, transformer: torch.nn.Module):
        self.transformer = transformer
        self.handles = []
        self.calls = defaultdict(int)
        self.changed_calls = defaultdict(int)
        self.enabled: list[tuple[str, TensorQuantizer]] = []

    def _hook(self, name: str):
        def hook(_module, inputs, output) -> None:
            self.calls[name] += 1
            if self.calls[name] != 1 or not inputs or not isinstance(output, torch.Tensor):
                return
            source = inputs[0].detach().reshape(-1)[:MAX_STATS_VALUES].float()
            result = output.detach().reshape(-1)[:MAX_STATS_VALUES].float()
            if source.shape == result.shape and not torch.equal(source, result):
                self.changed_calls[name] += 1

        return hook

    def __enter__(self):
        for name, module in self.transformer.named_modules():
            if isinstance(module, TensorQuantizer) and module.is_enabled:
                self.enabled.append((name, module))
                self.handles.append(module.register_forward_hook(self._hook(name)))
        weight_count = sum(name.endswith("weight_quantizer") for name, _ in self.enabled)
        input_count = sum(name.endswith("input_quantizer") for name, _ in self.enabled)
        if weight_count < 1 or input_count < 1:
            raise RuntimeError("Fake FP8 path contains no enabled weight/input quantizers")
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()

    def report(self) -> dict[str, Any]:
        invoked = {name: count for name, count in self.calls.items() if count}
        changed = {name: count for name, count in self.changed_calls.items() if count}
        report = {
            "enabled_count": len(self.enabled),
            "enabled_weight_count": sum(
                name.endswith("weight_quantizer") for name, _ in self.enabled
            ),
            "enabled_input_count": sum(
                name.endswith("input_quantizer") for name, _ in self.enabled
            ),
            "invoked_count": len(invoked),
            "invoked_weight_count": sum(name.endswith("weight_quantizer") for name in invoked),
            "invoked_input_count": sum(name.endswith("input_quantizer") for name in invoked),
            "changed_count": len(changed),
            "calls": invoked,
            "changed": sorted(changed),
        }
        if (
            report["invoked_weight_count"] < 1
            or report["invoked_input_count"] < 1
            or report["changed_count"] < 1
        ):
            raise RuntimeError(f"Fake FP8 quantizer execution proof failed: {report}")
        return report


def _run_validation(
    *,
    runner,
    transformer: torch.nn.Module,
    output: Path,
    backend: str,
) -> TensorCapture:
    from ltx_trainer.progress import TrainingProgress

    output.mkdir(parents=True, exist_ok=True)
    scaled_mm_calls = 0
    original_scaled_mm = torch._scaled_mm

    def counted_scaled_mm(*args, **kwargs):
        nonlocal scaled_mm_calls
        scaled_mm_calls += 1
        return original_scaled_mm(*args, **kwargs)

    torch._scaled_mm = counted_scaled_mm
    try:
        with (
            TensorCapture(transformer) as capture,
            TrainingProgress(enabled=False, total_steps=1) as progress,
        ):
            results = runner.run(
                transformer=transformer,
                step=0,
                output_dir=output,
                device=torch.device("cuda"),
                progress=progress,
            )
    finally:
        torch._scaled_mm = original_scaled_mm
    if len(results) != 1:
        raise RuntimeError(f"Expected one attribution video, got {len(results)}")
    if backend == "native" and scaled_mm_calls < 1:
        raise RuntimeError("Native attribution executed no torch._scaled_mm calls")
    if backend != "native" and scaled_mm_calls:
        raise RuntimeError(f"{backend} attribution unexpectedly used torch._scaled_mm")
    atomic_json(
        output / "execution.json",
        {
            "backend": backend,
            "torch_scaled_mm_calls": scaled_mm_calls,
            "input_hashes": capture.input_hashes,
            "tensors": capture.summaries,
        },
    )
    save_file(capture.slices, output / "tensor_slices.safetensors")
    return capture


def _native_inventory(transformer: torch.nn.Module) -> dict[str, Any]:
    from ltx_core.quantization.fp8_scaled_mm import FP8Linear

    rows = []
    fake_quantizers = 0
    for name, module in transformer.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            fake_quantizers += 1
        if isinstance(module, FP8Linear):
            valid = (
                module.weight.dtype == torch.float8_e4m3fn
                and module.weight_scale.numel() == 1
                and module.input_scale.numel() == 1
                and torch.isfinite(module.weight_scale).all()
                and torch.isfinite(module.input_scale).all()
                and (module.weight_scale > 0).all()
                and (module.input_scale > 0).all()
            )
            rows.append({"name": name, "valid": bool(valid)})
    if not rows or not all(row["valid"] for row in rows) or fake_quantizers:
        raise RuntimeError(f"Native FP8 inventory invalid: fp8={len(rows)} fake={fake_quantizers}")
    return {
        "fp8_linear_count": len(rows),
        "active_fake_quantizer_count": fake_quantizers,
        "layers": rows,
    }


def _compare_slices(root: Path, backends: tuple[str, ...]) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = {
        backend: load_file(root / backend / "tensor_slices.safetensors") for backend in backends
    }
    common = set.intersection(*(set(values) for values in tensors.values()))
    comparisons = {}
    for lhs, rhs in (("bf16", "fake"), ("bf16", "native"), ("fake", "native")):
        rows = {}
        for key in sorted(common):
            left = tensors[lhs][key].float()
            right = tensors[rhs][key].float()
            if left.shape != right.shape or not left.numel():
                continue
            difference = right - left
            rows[key] = {
                "mean_abs_error": difference.abs().mean().item(),
                "max_abs_error": difference.abs().max().item(),
                "relative_l2": (
                    difference.norm() / left.norm().clamp_min(torch.finfo(torch.float32).eps)
                ).item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    left.reshape(1, -1),
                    right.reshape(1, -1),
                ).item(),
                "finite": bool(torch.isfinite(right).all()),
            }
        comparisons[f"{lhs}_vs_{rhs}"] = rows
    return comparisons


def run_tensor_attribution(
    *,
    bundle: Path,
    config_path: Path,
    manifest_path: Path,
    output: Path,
    sample_id: str,
    seed: int,
) -> None:
    from ltx_trainer.validation_runner import ValidationRunner

    payload, _ = load_config(config_path)
    manifest = json.loads(manifest_path.read_text())
    rows = [row for row in manifest["samples"] if row["id"] == sample_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one sample {sample_id}, found {len(rows)}")
    reduced_manifest = output / "input_manifest.json"
    output.mkdir(parents=True, exist_ok=True)
    reduced_manifest.write_text(json.dumps({"samples": rows, "seeds": [seed]}, indent=2) + "\n")
    deploy = verify_deploy_bundle(bundle)
    validation_config = _validation_config(payload, reduced_manifest)
    runner = ValidationRunner(
        config=validation_config,
        model_path=deploy.base_model,
        text_encoder_path=payload["model"]["text_encoder_path"],
        load_text_encoder_in_8bit=False,
    )
    gc.collect()
    torch.cuda.empty_cache()

    fake_transformer, _ = _load_fake_transformer(bundle, "cuda")
    with FakeQuantProof(fake_transformer) as proof:
        fake_capture = _run_validation(
            runner=runner,
            transformer=fake_transformer,
            output=output / "fake",
            backend="fake",
        )
    fake_report = proof.report()
    atomic_json(output / "fake/quantizer_execution.json", fake_report)

    for _, module in proof.enabled:
        module.disable()
    bf16_capture = _run_validation(
        runner=runner,
        transformer=fake_transformer,
        output=output / "bf16",
        backend="bf16",
    )
    if fake_capture.input_hashes != bf16_capture.input_hashes:
        raise RuntimeError("BF16 and fake FP8 transformer inputs differ")
    del fake_transformer
    gc.collect()
    torch.cuda.empty_cache()

    native_transformer, _ = _load_native_transformer(bundle, "cuda")
    native_inventory = _native_inventory(native_transformer)
    atomic_json(output / "native/module_inventory.json", native_inventory)
    native_capture = _run_validation(
        runner=runner,
        transformer=native_transformer,
        output=output / "native",
        backend="native",
    )
    if fake_capture.input_hashes != native_capture.input_hashes:
        raise RuntimeError("Fake and native FP8 transformer inputs differ")

    comparisons = _compare_slices(output, ("bf16", "fake", "native"))
    fake_output_rows = comparisons["bf16_vs_fake"]
    changed_fake_outputs = [
        row for key, row in fake_output_rows.items() if "output" in key and row["max_abs_error"] > 0
    ]
    if not changed_fake_outputs:
        raise RuntimeError("Fake FP8 output is identical to quantizers-disabled BF16")
    atomic_json(
        output / "summary.json",
        {
            "sample_id": sample_id,
            "seed": seed,
            "input_hashes": fake_capture.input_hashes,
            "fake_quantizer_execution": fake_report,
            "native_inventory": native_inventory,
            "comparisons": comparisons,
        },
    )
