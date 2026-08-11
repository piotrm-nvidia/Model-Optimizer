# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-checks for the calibration coverage audit.

Runs against a module tree carrying LTX-2 names and stand-in quantizers, so the audit's
classification and its fail-closed behaviour can be checked without a checkpoint, a GPU,
or the quantization runtime:

    pytest test_calib_coverage.py
    python test_calib_coverage.py       # same checks, no pytest needed
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from calib_coverage import audit_calibration_coverage, check_calibration_coverage
from torch import nn


class FakeQuantizer(nn.Module):
    """Stands in for a modelopt TensorQuantizer, with only what the audit reads."""

    def __init__(
        self,
        enabled: bool = True,
        amax: float | None = 1.0,
        dynamic: bool = False,
        pre_quant_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self._disabled = not enabled
        self._dynamic = dynamic
        self.is_mx_format = False
        self._amax = None if amax is None else torch.tensor(amax)
        self.pre_quant_scale = pre_quant_scale

    @property
    def is_enabled(self) -> bool:
        return not self._disabled


class FakeQuantLinear(nn.Module):
    """A quantized Linear: the layer plus the quantizers modelopt attaches to it."""

    def __init__(self, input_quantizer: FakeQuantizer, weight_quantizer: FakeQuantizer):
        super().__init__()
        self.input_quantizer = input_quantizer
        self.weight_quantizer = weight_quantizer


def _calibrated(**kwargs) -> FakeQuantLinear:
    return FakeQuantLinear(FakeQuantizer(**kwargs), FakeQuantizer())


class FakeBackbone(nn.Module):
    """Video and audio FFNs plus a connector, under LTX-2's real module names."""

    def __init__(self, audio_input: FakeQuantizer | None = None):
        super().__init__()
        blocks = nn.Module()
        block = nn.Module()

        video_ff = nn.Module()
        video_net = nn.Module()
        # ff.net.0 / ff.net.2 are the up- and down-projections the classifier keys on.
        setattr(video_net, "0", _calibrated())
        setattr(video_net, "2", _calibrated())
        video_ff.net = video_net
        block.ff = video_ff

        audio_ff = nn.Module()
        audio_net = nn.Module()
        setattr(audio_net, "0", _calibrated())
        setattr(
            audio_net,
            "2",
            FakeQuantLinear(audio_input or FakeQuantizer(), FakeQuantizer()),
        )
        audio_ff.net = audio_net
        block.audio_ff = audio_ff

        setattr(blocks, "0", block)
        self.transformer_blocks = blocks


def _audio_down_quantizers(report: dict) -> dict[str, int]:
    return report["by_layer_class"]["audio_ffn_down"]


def test_a_clean_model_reports_no_offenders():
    report = audit_calibration_coverage(FakeBackbone())
    assert report["uncalibrated"] == [], report["uncalibrated"]
    assert report["totals"]["calibrated"] == 8, report["totals"]


def test_layer_classes_separate_the_audio_branch():
    report = audit_calibration_coverage(FakeBackbone())
    classes = set(report["by_layer_class"])
    assert {"ffn_up", "ffn_down", "audio_ffn_up", "audio_ffn_down"} <= classes, classes


def test_an_uncalibrated_input_quantizer_is_reported_with_its_class():
    report = audit_calibration_coverage(FakeBackbone(audio_input=FakeQuantizer(amax=None)))
    assert len(report["uncalibrated"]) == 1, report["uncalibrated"]
    offender = report["uncalibrated"][0]
    assert offender["role"] == "input_quantizer", offender
    assert offender["layer_class"] == "audio_ffn_down", offender
    assert _audio_down_quantizers(report)["uncalibrated"] == 1


def test_a_dynamic_quantizer_has_nothing_to_calibrate():
    report = audit_calibration_coverage(
        FakeBackbone(audio_input=FakeQuantizer(amax=None, dynamic=True))
    )
    assert report["uncalibrated"] == [], report["uncalibrated"]
    assert report["totals"]["dynamic"] == 1, report["totals"]


def test_a_disabled_quantizer_is_not_an_offender():
    report = audit_calibration_coverage(
        FakeBackbone(audio_input=FakeQuantizer(enabled=False, amax=None))
    )
    assert report["uncalibrated"] == [], report["uncalibrated"]
    assert report["totals"]["disabled"] == 1, report["totals"]


def test_smoothquant_migration_is_counted_per_enabled_input():
    scale = torch.ones(4)
    report = audit_calibration_coverage(
        FakeBackbone(audio_input=FakeQuantizer(pre_quant_scale=scale))
    )
    smooth = report["smoothquant"]
    assert smooth["enabled_input_quantizers"] == 4, smooth
    assert smooth["with_pre_quant_scale"] == 1, smooth


def test_the_check_fails_closed_on_an_uncalibrated_quantizer():
    model = FakeBackbone(audio_input=FakeQuantizer(amax=None))
    try:
        check_calibration_coverage(model)
    except RuntimeError as error:
        assert "audio_ff.net.2" in str(error), str(error)
    else:
        raise AssertionError("an uncalibrated quantizer must stop the run")


def test_the_check_can_record_the_gap_instead_of_failing():
    model = FakeBackbone(audio_input=FakeQuantizer(amax=None))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "coverage.json"
        report = check_calibration_coverage(model, out_path=out, fail_on_uncalibrated=False)
        written = json.loads(out.read_text())
    assert written == report, "the written report must match the returned one"
    assert len(written["uncalibrated"]) == 1, written["uncalibrated"]


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
        except Exception as error:  # noqa: BLE001 - a crash is a failure worth printing
            failures += 1
            print(f"ERROR {test.__name__}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
