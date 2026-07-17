# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils import (
    get_quantizer_state_dict,
    set_quantizer_state_dict,
)

from examples.windows.diffusers.qad_example.sample_example_qad_diffusers import (
    audit_quantizer_coverage,
    audit_quantizer_state_keys,
    calibration_step_count,
    cast_model_inputs,
    compare_tensor_outputs,
    inventory_digest,
    quantizer_inventory,
    reset_runtime_dynamic_input_amax,
    summarize_amax_state,
    tensor_output_report,
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


class _QuantizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enabled_input_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(
                num_bits=(2, 1),
                block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            ),
            amax=torch.tensor(2.0),
        )
        self.enabled_weight_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(num_bits=8), amax=torch.tensor(3.0)
        )
        self.disabled_input_quantizer = TensorQuantizer(
            QuantizerAttributeConfig(num_bits=8, enable=False), amax=torch.tensor(float("nan"))
        )


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


def test_calibration_step_count_cycles_smaller_representative_dataset():
    assert calibration_step_count(requested=32, dataset_size=8) == 32
    assert calibration_step_count(requested=32, dataset_size=16) == 32


def test_calibration_step_count_rejects_empty_or_nonpositive_requests():
    with pytest.raises(RuntimeError, match="dataset is empty"):
        calibration_step_count(requested=32, dataset_size=0)
    with pytest.raises(ValueError, match="must be positive"):
        calibration_step_count(requested=0, dataset_size=8)


def test_summarize_amax_state_counts_finite_positive_values():
    assert summarize_amax_state(_AmaxModel([0.0, 1.0, -2.0])) == {
        "total": 3,
        "finite": 3,
        "positive": 2,
    }


def test_quantizer_inventory_filters_quantizers_and_records_amax():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), _QuantizerModel())

    inventory = quantizer_inventory(model)

    assert [item["fqn"] for item in inventory] == [
        "1.disabled_input_quantizer",
        "1.enabled_input_quantizer",
        "1.enabled_weight_quantizer",
    ]
    enabled = next(item for item in inventory if item["fqn"].endswith("enabled_input_quantizer"))
    assert enabled["class"].endswith(".TensorQuantizer")
    assert enabled["enabled"] is True
    assert enabled["amax"]["present"] is True
    assert enabled["amax"]["finite"] is True
    assert enabled["amax"]["positive"] is True
    assert len(enabled["amax"]["digest"]) == 64
    assert inventory_digest(inventory) == inventory_digest(quantizer_inventory(model))


def test_coverage_audit_classifies_disabled_nonfinite_without_failure():
    inventory = quantizer_inventory(_QuantizerModel())

    report = audit_quantizer_coverage(inventory, inventory)

    assert report["enabled_nonfinite"] == []
    assert report["disabled_nonfinite"] == ["disabled_input_quantizer"]


def test_coverage_audit_allows_uncalibrated_runtime_dynamic_input():
    model = _QuantizerModel()
    model.enabled_input_quantizer.reset_amax()
    inventory = quantizer_inventory(model)

    report = audit_quantizer_coverage(inventory, inventory)

    dynamic = next(
        item for item in inventory if item["fqn"] == "enabled_input_quantizer"
    )
    assert dynamic["enabled"] is True
    assert dynamic["requires_amax"] is False
    assert report["missing_enabled_amax"] == []


def test_coverage_audit_rejects_enabled_set_and_amax_failures():
    expected = quantizer_inventory(_QuantizerModel())
    actual = quantizer_inventory(_QuantizerModel())
    actual_by_name = {item["fqn"]: item for item in actual}
    actual_by_name["enabled_input_quantizer"]["enabled"] = False
    actual_by_name["enabled_input_quantizer"]["disabled"] = True
    actual_by_name["enabled_weight_quantizer"]["amax"]["finite"] = False

    with pytest.raises(RuntimeError, match="Quantizer coverage audit failed"):
        audit_quantizer_coverage(expected, list(actual_by_name.values()))


def test_coverage_audit_rejects_required_enabled_amax_absence():
    expected = quantizer_inventory(_QuantizerModel())
    actual = quantizer_inventory(_QuantizerModel())
    weight = next(item for item in actual if item["fqn"] == "enabled_weight_quantizer")
    weight["amax"]["present"] = False

    with pytest.raises(RuntimeError, match="missing_enabled_amax"):
        audit_quantizer_coverage(expected, actual)


def test_coverage_audit_rejects_changed_calibrated_amax():
    expected = quantizer_inventory(_QuantizerModel())
    actual_model = _QuantizerModel()
    actual_model.enabled_input_quantizer.amax = torch.tensor(9.0)
    actual = quantizer_inventory(actual_model)

    with pytest.raises(RuntimeError, match="mismatched_enabled_amax"):
        audit_quantizer_coverage(expected, actual)


def test_nested_quantizer_state_round_trip_restores_amax_exactly():
    source = _QuantizerModel()
    target = _QuantizerModel()
    target.enabled_input_quantizer.amax = torch.tensor(9.0)
    state = get_quantizer_state_dict(source)

    set_quantizer_state_dict(target, state)

    audit_quantizer_coverage(
        quantizer_inventory(source),
        quantizer_inventory(target),
    )
    assert target.enabled_input_quantizer.amax.item() == 2.0


def test_coverage_audit_rejects_missing_and_unexpected_quantizers():
    expected = quantizer_inventory(_QuantizerModel())
    actual = [dict(item) for item in expected[1:]]
    unexpected = dict(actual[0])
    unexpected["fqn"] = "unexpected.input_quantizer"
    actual.append(unexpected)

    with pytest.raises(RuntimeError, match="missing_keys"):
        audit_quantizer_coverage(expected, actual)


def test_quantizer_state_key_audit_requires_exact_coverage():
    assert audit_quantizer_state_keys(["a._amax", "b._amax"], ["b._amax", "a._amax"]) == {
        "missing_keys": [],
        "unexpected_keys": [],
    }
    with pytest.raises(RuntimeError, match="missing_keys"):
        audit_quantizer_state_keys(["a._amax"], ["b._amax"])


def test_runtime_dynamic_reset_only_enabled_input_quantizer_amax():
    model = _QuantizerModel()

    reset = reset_runtime_dynamic_input_amax(model)

    assert reset == ["enabled_input_quantizer"]
    assert not hasattr(model.enabled_input_quantizer, "_amax")
    assert hasattr(model.enabled_weight_quantizer, "_amax")
    assert hasattr(model.disabled_input_quantizer, "_amax")
    assert model.enabled_input_quantizer.block_sizes == {
        -1: 16,
        "type": "dynamic",
        "scale_bits": (4, 3),
    }


def test_tensor_output_report_and_comparison():
    reference = (torch.tensor([1.0, 2.0]), {"audio": torch.tensor([3.0])})
    close = (torch.tensor([1.0, 2.000001]), {"audio": torch.tensor([3.0])})
    far = (torch.tensor([1.0, 3.0]), {"audio": torch.tensor([3.0])})

    report = tensor_output_report(reference)
    close_comparison = compare_tensor_outputs(reference, close)
    far_comparison = compare_tensor_outputs(reference, far)

    assert set(report) == {"output.0", "output.1.audio"}
    assert report["output.0"]["finite"] is True
    assert len(report["output.0"]["digest"]) == 64
    assert close_comparison["allclose"] is True
    assert close_comparison["tensors"]["output.0"]["exact_digest"] is False
    assert far_comparison["allclose"] is False
