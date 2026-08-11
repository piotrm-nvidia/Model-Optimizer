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

"""Self-checks for the INT8 arms' declared configs.

The INT8 tier comparison only means something if the arms differ in numerics and not in
which modules they cover, and if the SmoothQuant strength is stated rather than
defaulted. Both are config-level properties, so they are checked without a checkpoint:

    pytest test_int8_configs.py
    python test_int8_configs.py         # same checks, no pytest needed
"""

from __future__ import annotations

from config import (
    INT8_DEFAULT_CONFIG,
    INT8_PER_CHANNEL_PER_TOKEN_CONFIG,
    INT8_SMOOTHQUANT_CONFIG,
)
from quantize_config import Int8Numerics, QuantAlgo, QuantFormat, QuantizationConfig

INT8_ARMS = {
    "static_max": INT8_DEFAULT_CONFIG,
    "smoothquant": INT8_SMOOTHQUANT_CONFIG,
    "per_token_dynamic": INT8_PER_CHANNEL_PER_TOKEN_CONFIG,
}


def test_every_int8_arm_covers_the_same_modules():
    """Coverage must be identical across arms, or a tier comparison compares coverage."""
    coverage = {name: sorted(cfg["quant_cfg"]) for name, cfg in INT8_ARMS.items()}
    baseline = coverage["static_max"]
    for name, keys in coverage.items():
        assert keys == baseline, f"{name} covers {keys}, static_max covers {baseline}"


def test_every_int8_arm_disables_the_same_extras():
    for name, cfg in INT8_ARMS.items():
        assert cfg["quant_cfg"]["default"] == {"enable": False}, name
        assert cfg["quant_cfg"]["*output_quantizer"] == {"enable": False}, name


def test_the_arms_differ_only_in_activation_numerics():
    baseline = INT8_DEFAULT_CONFIG["quant_cfg"]
    for name, cfg in INT8_ARMS.items():
        quant_cfg = cfg["quant_cfg"]
        assert quant_cfg["*weight_quantizer"] == baseline["*weight_quantizer"], name
        differing = [
            key
            for key in quant_cfg
            if quant_cfg[key] != baseline[key] and key != "*input_quantizer"
        ]
        assert differing == [], f"{name} also differs at {differing}"


def test_smoothquant_activations_are_per_tensor():
    """model_calib.smoothquant only converts a quantizer whose axis is None."""
    assert INT8_SMOOTHQUANT_CONFIG["quant_cfg"]["*input_quantizer"]["axis"] is None
    assert INT8_SMOOTHQUANT_CONFIG["algorithm"] == "smoothquant"


def _config(**kwargs) -> QuantizationConfig:
    return QuantizationConfig(format=QuantFormat.INT8, **kwargs)


def test_smoothquant_without_an_alpha_is_rejected():
    try:
        _config(algo=QuantAlgo.SMOOTHQUANT).validate()
    except ValueError as error:
        assert "--alpha" in str(error), str(error)
    else:
        raise AssertionError("an unstated SmoothQuant alpha must stop the run")


def test_a_pinned_alpha_is_accepted():
    _config(algo=QuantAlgo.SMOOTHQUANT, alpha=0.5).validate()


def test_arms_without_smoothquant_need_no_alpha():
    _config(algo=QuantAlgo.MAX).validate()
    _config(algo=QuantAlgo.MAX, int8_numerics=Int8Numerics.PER_TOKEN_DYNAMIC).validate()


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
