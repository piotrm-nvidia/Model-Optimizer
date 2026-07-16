# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from examples.windows.diffusers.qad_example.sample_example_qad_diffusers import (
    cast_model_inputs,
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
def test_validate_calibration_counts_rejects_bad_runs(
    attempted, successful, failed, match
):
    with pytest.raises(RuntimeError, match=match):
        validate_calibration_counts(attempted, successful, failed)


def test_summarize_amax_state_counts_finite_positive_values():
    assert summarize_amax_state(_AmaxModel([0.0, 1.0, -2.0])) == {
        "total": 3,
        "finite": 3,
        "positive": 2,
    }
