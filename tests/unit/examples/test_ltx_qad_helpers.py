# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from examples.windows.diffusers.qad_example import sample_example_qad_diffusers as qad
from examples.windows.diffusers.qad_example.sample_example_qad_diffusers import (
    SENSITIVE_LAYER_PATTERNS,
    build_quant_config,
    cast_model_inputs,
    convert_ltx_fp8_transformer_state,
    create_fp8_deploy_checkpoint,
    restore_quantizer_state,
    summarize_amax_state,
    validate_calibration_counts,
)


@dataclass(frozen=True)
class _FakeModality:
    latent: torch.Tensor
    sigma: torch.Tensor
    timesteps: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    enabled: bool = True


class _AmaxModel(torch.nn.Module):
    def __init__(self, values: list[float]):
        super().__init__()
        self.register_buffer("_amax", torch.tensor(values))


def test_cast_model_inputs_normalizes_compute_tensors_but_preserves_positions():
    modality = _FakeModality(
        latent=torch.ones(1, dtype=torch.float32),
        sigma=torch.ones(1, dtype=torch.float32),
        timesteps=torch.ones(1, dtype=torch.float32),
        positions=torch.ones(1, dtype=torch.float32),
        context=torch.ones(1, dtype=torch.float32),
        context_mask=torch.ones(1, dtype=torch.float32),
    )
    inputs = SimpleNamespace(video=modality, audio=None)

    cast_model_inputs(inputs, torch.bfloat16)

    assert inputs.video.latent.dtype == torch.bfloat16
    assert inputs.video.sigma.dtype == torch.bfloat16
    assert inputs.video.timesteps.dtype == torch.bfloat16
    assert inputs.video.context.dtype == torch.bfloat16
    assert inputs.video.context_mask.dtype == torch.bfloat16
    assert inputs.video.positions.dtype == torch.float32


@pytest.mark.parametrize(
    ("attempted", "successful", "failed"),
    [(1, 1, 0), (10, 10, 0), (10, 5, 5)],
)
def test_validate_calibration_counts_accepts_usable_runs(attempted, successful, failed):
    validate_calibration_counts(attempted, successful, failed)


@pytest.mark.parametrize(
    ("attempted", "successful", "failed", "match"),
    [
        (1, 0, 1, "zero successful"),
        (10, 4, 6, "Too many calibration failures"),
    ],
)
def test_validate_calibration_counts_rejects_bad_runs(attempted, successful, failed, match):
    with pytest.raises(RuntimeError, match=match):
        validate_calibration_counts(attempted, successful, failed)


def test_summarize_amax_state_counts_finite_positive_values():
    assert summarize_amax_state(_AmaxModel([0.0, 1.0, -2.0])) == {
        "total": 3,
        "finite": 3,
        "positive": 2,
    }


@pytest.mark.parametrize("recipe", ["nvfp4", "fp8"])
def test_build_quant_config_preserves_sensitive_and_block_exclusions(recipe):
    config = build_quant_config(exclude_blocks=[3, 9], recipe=recipe)
    disabled = {
        entry["quantizer_name"] for entry in config["quant_cfg"] if entry.get("enable") is False
    }

    assert set(SENSITIVE_LAYER_PATTERNS) <= disabled
    assert "*transformer_blocks.3.*" in disabled
    assert "*transformer_blocks.9.*" in disabled


def test_build_fp8_quant_config_does_not_mutate_default():
    original_length = len(qad.mtq.FP8_DEFAULT_CFG["quant_cfg"])

    config = build_quant_config(exclude_blocks=[7], recipe="fp8")

    assert config is not qad.mtq.FP8_DEFAULT_CFG
    assert len(qad.mtq.FP8_DEFAULT_CFG["quant_cfg"]) == original_length
    assert len(config["quant_cfg"]) > original_length


def test_restore_quantizer_state_uses_quantizer_helper(monkeypatch):
    calls = []
    model = torch.nn.Linear(2, 2)
    quantizer_state = {"linear.weight_quantizer": {"_amax": torch.tensor(2.0)}}
    state = {"modelopt_state_weights": quantizer_state, "modelopt_version": "test"}

    monkeypatch.setattr(qad, "register_ltx2_quant_linear", lambda: calls.append("register"))
    monkeypatch.setattr(
        qad.mto,
        "restore_from_modelopt_state",
        lambda restored_model, restored_state: calls.append(
            ("restore", restored_model, restored_state)
        ),
    )
    monkeypatch.setattr(
        qad,
        "set_quantizer_state_dict",
        lambda restored_model, restored_quantizers: calls.append(
            ("set", restored_model, restored_quantizers)
        ),
    )

    restore_quantizer_state(model, state)

    assert calls == [
        "register",
        ("restore", model, {"modelopt_version": "test"}),
        ("set", model, quantizer_state),
    ]
    assert state["modelopt_state_weights"] is quantizer_state


def _fp8_quantizer_state():
    return {
        "_student_model.velocity_model.transformer_blocks.2.attn.to_q.weight_quantizer": {
            "_amax": torch.tensor(896.0)
        },
        "_student_model.velocity_model.transformer_blocks.2.attn.to_q.input_quantizer": {
            "_amax": torch.tensor(224.0)
        },
    }


def test_convert_ltx_fp8_state_emits_expected_dtype_scales_and_bf16_exclusions():
    trained = {
        "_student_model.velocity_model.transformer_blocks.2.attn.to_q.weight": torch.tensor(
            [[-4.0, 2.0], [1.0, 3.0]], dtype=torch.bfloat16
        ),
        "_student_model.velocity_model.transformer_blocks.2.attn.to_q.bias": torch.ones(
            2, dtype=torch.float32
        ),
        "_student_model.velocity_model.transformer_blocks.0.attn.to_q.weight": torch.ones(
            2, 2, dtype=torch.float32
        ),
    }

    converted = convert_ltx_fp8_transformer_state(trained, _fp8_quantizer_state())
    prefix = "transformer_blocks.2.attn.to_q"

    assert converted[f"{prefix}.weight"].dtype == torch.float8_e4m3fn
    assert converted[f"{prefix}.weight_scale"].dtype == torch.float32
    assert converted[f"{prefix}.weight_scale"].shape == ()
    assert converted[f"{prefix}.weight_scale"].item() == pytest.approx(2.0)
    assert converted[f"{prefix}.input_scale"].item() == pytest.approx(0.5)
    assert converted[f"{prefix}.bias"].dtype == torch.bfloat16
    assert converted["transformer_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16


def test_create_fp8_deploy_checkpoint_uses_ltx_key_contract(tmp_path):
    trained_path = tmp_path / "trained.safetensors"
    base_path = tmp_path / "base.safetensors"
    state_path = tmp_path / "modelopt.pth"
    output_path = tmp_path / "deploy.safetensors"
    save_file(
        {
            "transformer_blocks.2.attn.to_q.weight": torch.ones(2, 2, dtype=torch.bfloat16),
            "transformer_blocks.0.attn.to_q.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        },
        trained_path,
    )
    save_file(
        {
            "vae.decoder.weight": torch.ones(1, dtype=torch.bfloat16),
            "model.diffusion_model.transformer_blocks.2.attn.to_q.weight": torch.zeros(
                2, 2, dtype=torch.bfloat16
            ),
        },
        base_path,
    )
    torch.save({"modelopt_state_weights": _fp8_quantizer_state()}, state_path)

    create_fp8_deploy_checkpoint(
        str(trained_path),
        str(state_path),
        str(base_path),
        str(output_path),
    )

    exported = load_file(output_path)
    prefix = "model.diffusion_model.transformer_blocks.2.attn.to_q"
    assert exported[f"{prefix}.weight"].dtype == torch.float8_e4m3fn
    assert exported[f"{prefix}.weight_scale"].shape == ()
    assert exported[f"{prefix}.input_scale"].shape == ()
    assert (
        exported["model.diffusion_model.transformer_blocks.0.attn.to_q.weight"].dtype
        == torch.bfloat16
    )
    assert exported["vae.decoder.weight"].dtype == torch.bfloat16
