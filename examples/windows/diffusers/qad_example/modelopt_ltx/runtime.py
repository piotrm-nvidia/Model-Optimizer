from __future__ import annotations

import gc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType  # noqa: TC003
from typing import Any

from .artifacts import (
    atomic_json,
    create_deploy_bundle,
    resolve_checkpoint,
    sha256_file,
    verify_deploy_bundle,
)
from .native_fp8 import (
    compare_exported_weights,
    create_native_checkpoint,
    validate_native_checkpoint,
)


def load_qad_module() -> ModuleType:
    script = Path(__file__).parents[1] / "sample_example_qad_diffusers.py"
    spec = importlib.util.spec_from_file_location("modelopt_ltx_qad_runner", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load QAD runner: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import yaml

    payload = yaml.safe_load(os.path.expandvars(path.read_text()))
    qad = payload.pop("qad", {})
    return payload, qad


def validate_preprocessed(output: Path) -> None:
    latents = output / "latents"
    conditions = output / "conditions"
    if (
        not latents.is_dir()
        or not conditions.is_dir()
        or not any(latents.iterdir())
        or not any(conditions.iterdir())
    ):
        raise RuntimeError(f"Incomplete preprocessed dataset: {output}")


def preprocess(
    *,
    config_path: Path,
    manifest_path: Path,
    output: Path,
) -> None:
    payload, _ = load_config(config_path)
    rows = json.loads(manifest_path.read_text())["samples"]
    dataset = [
        {
            "video": row["source"],
            "caption": "extend the scene naturally, consistent with the original footage",
        }
        for row in rows
    ]
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / "dataset.json"
    atomic_json(dataset_path, dataset)

    ltx_dir = Path(os.environ["LTX_DIR"])
    process_script = ltx_dir / "packages/ltx-trainer/scripts/process_dataset.py"
    model = payload["model"]
    subprocess.run(
        [
            sys.executable,
            str(process_script),
            str(dataset_path),
            "--resolution-buckets",
            "1280x704x49",
            "--output-dir",
            str(output),
            "--model-path",
            str(model["model_path"]),
            "--text-encoder-path",
            str(model["text_encoder_path"]),
            "--batch-size",
            "1",
            "--skip-audio",
            "--vae-tiling",
        ],
        check=True,
    )
    validate_preprocessed(output)


def run_ptq(
    *,
    config_path: Path,
    output: Path,
    quant_recipe: str,
    calibration_batches: int,
) -> Path:
    module = load_qad_module()
    from ltx_trainer.config import LtxTrainerConfig

    payload, qad = load_config(config_path)
    payload["output_dir"] = str(output)
    config = LtxTrainerConfig(**payload)
    quant_config = module.build_quant_config(
        exclude_blocks=qad.get("exclude_blocks", [0, 1, 46, 47]),
        quant_recipe=quant_recipe,
    )
    trainer = module.LtxvQADTrainer(
        trainer_config=config,
        quant_cfg=quant_config,
        calib_size=calibration_batches,
        kd_loss_weight=float(qad.get("kd_loss_weight", 1.0)),
        setup_distillation=False,
    )
    state_path = output / "checkpoints/modelopt_state_step_00000.pth"
    trainer.save_ptq_state(state_path)
    atomic_json(
        output / "ptq_manifest.json",
        {
            "schema_version": 1,
            "recipe": quant_recipe,
            "modelopt_state": str(state_path),
            "sha256": sha256_file(state_path),
            "calibration_batches": calibration_batches,
        },
    )
    return state_path


def run_qad(
    *,
    config_path: Path,
    output: Path,
    quant_recipe: str,
    initial_checkpoint: Path,
) -> None:
    module = load_qad_module()
    from ltx_trainer.config import LtxTrainerConfig

    payload, qad = load_config(config_path)
    checkpoint_steps = list(qad.get("checkpoint_steps", [1, 3, 10]))
    payload["output_dir"] = str(output)
    payload["checkpoints"]["interval"] = 1
    payload["checkpoints"]["keep_last_n"] = len(checkpoint_steps)
    config = LtxTrainerConfig(**payload)
    state_path, _ = resolve_checkpoint(initial_checkpoint)
    trainer = module.LtxvQADTrainer(
        trainer_config=config,
        quant_cfg=module.build_quant_config(
            exclude_blocks=qad.get("exclude_blocks", [0, 1, 46, 47]),
            quant_recipe=quant_recipe,
        ),
        calib_size=int(qad.get("calib_size", 8)),
        kd_loss_weight=float(qad.get("kd_loss_weight", 1.0)),
        initial_modelopt_state=state_path,
        checkpoint_steps=checkpoint_steps,
    )
    trainer.train()


def create_deploy(
    *,
    checkpoint: Path,
    output: Path,
    config_path: Path,
    quant_recipe: str,
) -> None:
    payload, _ = load_config(config_path)
    base_model = Path(payload["model"]["model_path"])
    state, weights = resolve_checkpoint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        native = Path(temporary) / "native_fp8.safetensors"
        create_native_checkpoint(
            trained_path=weights or base_model,
            modelopt_state_path=state,
            base_path=base_model,
            output_path=native,
        )
        create_deploy_bundle(
            checkpoint,
            output,
            base_model=base_model,
            recipe=quant_recipe,
            native_checkpoint=native,
            native_contract=native.with_suffix(".contract.json"),
        )


def export_parity(*, source: Path, deploy: Path, output: Path) -> None:
    state, weights = resolve_checkpoint(source)
    manifest = verify_deploy_bundle(deploy)
    source_state_hash = sha256_file(state)
    source_weights_hash = sha256_file(weights) if weights else None
    native_checkpoint = deploy / manifest.native_checkpoint
    native_contract = validate_native_checkpoint(native_checkpoint)
    weight_parity = compare_exported_weights(
        trained_path=weights or Path(manifest.base_model),
        native_path=native_checkpoint,
    )
    valid = (
        source_state_hash == manifest.modelopt_state_sha256
        and source_weights_hash == manifest.model_weights_sha256
        and native_contract["fp8_weight_count"] > 0
        and all(row["finite"] for row in weight_parity["rows"])
    )
    payload = {
        "valid": valid,
        "comparison": "bundle-copy integrity plus native FP8 weight export parity",
        "source_modelopt_state_sha256": source_state_hash,
        "deploy_modelopt_state_sha256": manifest.modelopt_state_sha256,
        "source_model_weights_sha256": source_weights_hash,
        "deploy_model_weights_sha256": manifest.model_weights_sha256,
        "native_contract": native_contract,
        "native_weight_parity": weight_parity,
    }
    atomic_json(output, payload)
    if not valid:
        raise RuntimeError("Deploy bundle differs from source checkpoint")


def _load_fake_transformer(bundle: Path, device: str):
    import torch
    from ltx_trainer.model_loader import load_transformer

    module = load_qad_module()
    manifest = verify_deploy_bundle(bundle)
    transformer = load_transformer(
        checkpoint_path=manifest.base_model,
        device="cpu",
        dtype=torch.bfloat16,
    )
    transformer = module.restore_quantized_model(transformer, bundle / manifest.modelopt_state)
    if manifest.model_weights:
        from safetensors.torch import load_file

        state = load_file(bundle / manifest.model_weights, device="cpu")
        incompatible = transformer.load_state_dict(state, strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if "quantizer" not in key and "_teacher_model" not in key
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected trained-weight keys: {unexpected[:20]}")
    return transformer.to(device).eval(), manifest


def _load_native_transformer(bundle: Path, device: str):
    import torch
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )
    from ltx_core.quantization.fp8_scaled_mm import get_fp8_swap_module_ops

    manifest = verify_deploy_bundle(bundle)
    checkpoint = bundle / manifest.native_checkpoint
    validate_native_checkpoint(checkpoint)
    transformer = SingleGPUModelBuilder(
        model_path=str(checkpoint),
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        module_ops=get_fp8_swap_module_ops(str(checkpoint)),
    ).build(device=torch.device(device), dtype=None)
    return transformer.eval(), manifest


def _validation_config(base_config: dict[str, Any], manifest_path: Path):
    from ltx_trainer.config import ValidationConfig

    heldout = json.loads(manifest_path.read_text())
    samples = [
        {
            "prompt": "extend the scene naturally, consistent with the original footage",
            "conditions": [
                {
                    "type": "spatial_crop",
                    "video": row["conditioning"],
                    "spatial_region": [106, 192, 598, 1088],
                }
            ],
            "video_dims": [1280, 704, 49],
            "seed": seed,
        }
        for row in heldout["samples"]
        for seed in heldout["seeds"]
    ]
    defaults = base_config.get("validation", {})
    return ValidationConfig(
        samples=samples,
        negative_prompt=defaults.get(
            "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
        ),
        video_dims=(1280, 704, 49),
        frame_rate=24.0,
        inference_steps=int(defaults.get("inference_steps", 8)),
        guidance_scale=float(defaults.get("guidance_scale", 1.0)),
        stg_scale=float(defaults.get("stg_scale", 0.0)),
        generate_audio=False,
        interval=None,
    )


def evaluate_bundle(
    *,
    bundle: Path,
    config_path: Path,
    manifest_path: Path,
    output: Path,
    backend: str = "native",
) -> None:
    import torch
    from ltx_trainer.progress import TrainingProgress
    from ltx_trainer.validation_runner import ValidationRunner

    payload, _ = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deploy = verify_deploy_bundle(bundle)
    config = _validation_config(payload, manifest_path)
    runner = ValidationRunner(
        config=config,
        model_path=deploy.base_model,
        text_encoder_path=payload["model"]["text_encoder_path"],
        load_text_encoder_in_8bit=False,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if backend == "native":
        transformer, deploy = _load_native_transformer(bundle, device)
    elif backend == "fake":
        transformer, deploy = _load_fake_transformer(bundle, device)
    else:
        raise ValueError(f"Unknown evaluation backend: {backend}")
    output.mkdir(parents=True, exist_ok=True)
    scaled_mm_calls = 0
    original_scaled_mm = torch._scaled_mm

    def counted_scaled_mm(*args, **kwargs):
        nonlocal scaled_mm_calls
        scaled_mm_calls += 1
        return original_scaled_mm(*args, **kwargs)

    torch._scaled_mm = counted_scaled_mm
    try:
        with TrainingProgress(enabled=False, total_steps=1) as progress:
            results = runner.run(
                transformer=transformer,
                step=0,
                output_dir=output,
                device=torch.device(device),
                progress=progress,
            )
    finally:
        torch._scaled_mm = original_scaled_mm
    if backend == "native" and scaled_mm_calls < 1:
        raise RuntimeError("Native FP8 inference executed no torch._scaled_mm calls")
    atomic_json(
        output
        / (
            "native_kernel_evidence.json"
            if backend == "native"
            else "fake_quant_execution_evidence.json"
        ),
        {
            "backend": "ltx fp8-scaled-mm" if backend == "native" else "ModelOpt fake FP8",
            "torch_scaled_mm_calls": scaled_mm_calls,
            "negative_control": backend == "fake",
        },
    )
    rows = json.loads(manifest_path.read_text())
    expected = [(row["id"], seed) for row in rows["samples"] for seed in rows["seeds"]]
    if len(results) != len(expected):
        raise RuntimeError(f"Expected {len(expected)} outputs, got {len(results)}")
    for (_, generated), (clip_id, seed) in zip(results, expected, strict=True):
        destination = output / f"{clip_id}__seed{seed}.mp4"
        generated.replace(destination)


def inference_matrix(
    *,
    matrix_path: Path,
    config_path: Path,
    manifest_path: Path,
    ptq_bundle: Path,
    qad_root: Path,
    output: Path,
    resume: bool,
) -> None:
    matrix = json.loads(matrix_path.read_text())
    for candidate in matrix["candidates"]:
        candidate_id = candidate["id"]
        if candidate_id == "bf16":
            continue
        candidate_output = output / candidate_id
        expected = [
            candidate_output / f"{sample}__seed{seed}.mp4"
            for sample in matrix["samples"]
            for seed in matrix["seeds"]
        ]
        if resume and all(path.is_file() for path in expected):
            continue
        if candidate["kind"] == "ptq":
            bundle = ptq_bundle
        else:
            bundle = qad_root / f"step{candidate['step']}"
        evaluate_bundle(
            bundle=bundle,
            config_path=config_path,
            manifest_path=manifest_path,
            output=candidate_output,
        )
