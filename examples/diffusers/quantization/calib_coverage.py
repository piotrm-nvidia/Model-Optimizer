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

"""Check that calibration actually reached every quantizer that stays enabled.

A quantizer whose activation range is never observed keeps the amax it was created
with, and the layer it wraps is then quantized against a scale that came from nowhere.
Nothing in the calibration path reports this: the run completes, the checkpoint saves,
and the damage only shows up as quality loss that looks like the format's fault.

Two known ways to reach that state on LTX-2:

* a branch the calibration forward does not exercise - a modality that the pipeline
  builds but never denoises, or a conditioning path that a given clip geometry skips;
* SmoothQuant, which silently skips any layer it saw no activation for.

Both leave the same fingerprint, an enabled quantizer with no collected amax, so one
audit covers them. Run it after ``mtq.quantize`` and before the checkpoint is written.

Quantizers are matched by duck-typing rather than by importing ``TensorQuantizer``, so
the audit and its tests stay importable without pulling in the quantization runtime.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Reported for a quantizer that has nothing to calibrate, so it can never be a defect.
NOTHING_TO_CALIBRATE = ("dynamic", "mx_format", "disabled")

# Quantizer roles, taken from the leaf name modelopt gives each one.
_ROLE_SUFFIX = "_quantizer"


def _is_quantizer(module: Any) -> bool:
    """Whether a module is a modelopt TensorQuantizer.

    Attribute-based so a test can stand in a small fake, and so importing this module
    does not require the quantization runtime.
    """
    return hasattr(module, "is_enabled") and hasattr(module, "_disabled")


def _quantizer_state(quantizer: Any) -> str:
    """One of: disabled, dynamic, mx_format, calibrated, uncalibrated."""
    if not quantizer.is_enabled:
        return "disabled"
    # Dynamic activation numerics compute a scale per forward, so there is no amax to
    # collect and reading the property would assert.
    if getattr(quantizer, "_dynamic", False):
        return "dynamic"
    if getattr(quantizer, "is_mx_format", False):
        return "mx_format"
    amax = getattr(quantizer, "_amax", None)
    return "calibrated" if amax is not None else "uncalibrated"


def _split_role(name: str) -> tuple[str, str]:
    """Split a quantizer's module path into (owning module, role)."""
    parent, _, leaf = name.rpartition(".")
    role = leaf if leaf.endswith(_ROLE_SUFFIX) else "unknown"
    return parent, role


def _layer_classifier():
    """Return a name -> layer class function, or None when the LTX solver is absent.

    Reuses the tier solver's classifier so the audit groups quantizers by exactly the
    classes the protection tiers are solved over - an uncalibrated set that turns out to
    be one whole class is a different problem from a handful of scattered layers.
    """
    try:
        from ltx2_tier_solver import TokenGeometry, classify
    except ImportError:
        return None

    # Shapes and token counts only feed cost arithmetic the audit does not use.
    tokens = TokenGeometry(video=1, audio=1, text=1)
    return lambda name: classify(name, 1, 1, tokens).layer_class


def audit_calibration_coverage(model: Any) -> dict[str, Any]:
    """Inventory every quantizer in ``model`` by whether calibration reached it."""
    classifier = _layer_classifier()

    by_state: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    uncalibrated: list[dict[str, str]] = []
    smoothquant_migrated = 0
    enabled_inputs = 0

    for name, module in model.named_modules():
        if not _is_quantizer(module):
            continue
        parent, role = _split_role(name)
        state = _quantizer_state(module)
        by_state[state] = by_state.get(state, 0) + 1

        layer_class = classifier(parent) if classifier is not None else "unclassified"
        counts = by_class.setdefault(layer_class, {})
        counts[state] = counts.get(state, 0) + 1

        if role == "input_quantizer" and state not in NOTHING_TO_CALIBRATE:
            enabled_inputs += 1
            if getattr(module, "pre_quant_scale", None) is not None:
                smoothquant_migrated += 1

        if state == "uncalibrated":
            uncalibrated.append({"module": parent, "role": role, "layer_class": layer_class})

    return {
        "totals": by_state,
        "by_layer_class": by_class,
        "uncalibrated": uncalibrated,
        "smoothquant": {
            "enabled_input_quantizers": enabled_inputs,
            "with_pre_quant_scale": smoothquant_migrated,
        },
    }


def summarize_coverage(report: dict[str, Any]) -> str:
    """One-line summary for the run log."""
    totals = report["totals"]
    smooth = report["smoothquant"]
    return (
        f"calibration coverage: {totals.get('calibrated', 0)} calibrated, "
        f"{totals.get('uncalibrated', 0)} uncalibrated, "
        f"{totals.get('dynamic', 0)} dynamic, {totals.get('disabled', 0)} disabled; "
        f"{smooth['with_pre_quant_scale']}/{smooth['enabled_input_quantizers']} "
        f"enabled input quantizers carry a SmoothQuant scale"
    )


def format_uncalibrated(report: dict[str, Any], limit: int = 12) -> str:
    """Human-readable list of the offenders, capped so a log line stays readable."""
    offenders = report["uncalibrated"]
    shown = offenders[:limit]
    lines = [f"  {item['module']}.{item['role']} [{item['layer_class']}]" for item in shown]
    if len(offenders) > limit:
        lines.append(f"  ... and {len(offenders) - limit} more")
    return "\n".join(lines)


def check_calibration_coverage(
    model: Any,
    out_path: Path | None = None,
    logger: Any = None,
    fail_on_uncalibrated: bool = True,
) -> dict[str, Any]:
    """Audit ``model``, log the summary, optionally write it, and fail if it is short.

    Raises RuntimeError when an enabled quantizer never collected an amax, because a
    checkpoint saved in that state cannot be told apart later from one that calibrated
    cleanly - the file records the scale, not where it came from.
    """
    report = audit_calibration_coverage(model)

    if logger is not None:
        logger.info(summarize_coverage(report))
        counted = report["by_layer_class"]
        for layer_class in sorted(counted):
            states = counted[layer_class]
            rendered = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
            logger.debug(f"  {layer_class}: {rendered}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        if logger is not None:
            logger.info(f"Wrote the calibration coverage report to {out_path}")

    offenders = report["uncalibrated"]
    if offenders:
        message = (
            f"{len(offenders)} enabled quantizers never collected an amax, so they would "
            f"be scored on their initialization scale:\n{format_uncalibrated(report)}\n"
            "Either the calibration forward does not exercise those modules, or they were "
            "skipped by the calibration algorithm. Protect them, fix the calibration "
            "coverage, or pass --allow-uncalibrated to record the gap and continue."
        )
        if fail_on_uncalibrated:
            raise RuntimeError(message)
        if logger is not None:
            logger.warning(message)

    return report
