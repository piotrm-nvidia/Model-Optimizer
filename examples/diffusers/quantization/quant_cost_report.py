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

"""Per-variant quantization cost report, emitted next to the calibrated state.

A quality number is not interpretable without the cost it bought. This writes, for one
calibrated model, what each layer actually ended up as and what the whole variant costs:

  * weight bytes at the realized per-layer format, including scale-tensor bytes,
    against the BF16 baseline - MEASURED from the model, not predicted from a tier name
  * effective bits over the quantized module set
  * the quantized share of linear GEMM FLOPs, when a token-annotated inventory is
    supplied, and an Amdahl latency projection built on it - PROJECTED, never measured

The FLOP side deliberately joins against an external ``linear_inventory.json`` - the
token-annotated module inventory emitted by the companion tier cost model - rather than
re-deriving per-module token counts here. One definition of the token geometry, shared
by the tier solver and this report, means a tier's predicted cost and its realized cost
are directly comparable instead of being two independent estimates.
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch

from modelopt.torch.export.quant_utils import get_quantization_format
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    SequentialQuantizer,
    TensorQuantizer,
)

BF16_BITS = 16
SCALE_BYTES = 4  # float32 scale, matching what the unified HF exporter writes

# INT8 : BF16 dense arithmetic ratio. A100 624 TOPS / 312 TFLOPS and H100 SXM
# 1979 TOPS / 989 TFLOPS both come to 2.0.
TOPS_RATIO = {"a100": 2.0, "h100": 2.0}


def _quantizer_bits(quantizer: TensorQuantizer | None) -> int | None:
    """Storage bits for one quantizer, or None when it is absent or disabled."""
    if quantizer is None or not quantizer.is_enabled:
        return None
    if isinstance(quantizer, SequentialQuantizer):
        # Only the narrowest stage determines storage width.
        stages = [q.num_bits for q in quantizer if q.is_enabled]
        if not stages:
            return None
        return min(b if isinstance(b, int) else sum(b) + 1 for b in stages)
    num_bits = quantizer.num_bits
    if isinstance(num_bits, tuple):
        # Float formats are given as (exponent, mantissa); add the sign bit.
        return sum(num_bits) + 1
    return num_bits


def _is_quantizable_linear(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "weight")
        and isinstance(getattr(module, "weight", None), torch.Tensor)
        and module.weight.dim() == 2
        and hasattr(module, "weight_quantizer")
    )


def _load_inventory(inventory_path: Path | None) -> dict[str, dict]:
    """Map module name to its token-annotated inventory entry."""
    if inventory_path is None:
        return {}
    payload = json.loads(Path(inventory_path).read_text())
    return {entry["name"]: entry for entry in payload.get("modules", [])}


def _inventory_lookup(inventory: dict[str, dict], name: str) -> dict | None:
    """Match a live module name against the inventory, tolerating wrapper prefixes."""
    if name in inventory:
        return inventory[name]
    parts = name.split(".")
    for start in range(1, len(parts)):
        candidate = ".".join(parts[start:])
        if candidate in inventory:
            return inventory[candidate]
    return None


def build_quant_cost_report(
    backbone: torch.nn.Module,
    inventory_path: Path | None = None,
    gemm_time_share: float = 0.75,
    meta: dict[str, Any] | None = None,
) -> dict:
    """Walk a calibrated model and total up what its quantization actually costs."""
    inventory = _load_inventory(inventory_path)

    layers: list[dict] = []
    unmatched: list[str] = []
    bytes_bf16 = 0
    bytes_realized = 0
    flops_total = 0
    flops_quantized = 0
    quantized_numel = 0
    total_numel = 0

    for name, module in backbone.named_modules():
        if not _is_quantizable_linear(module):
            continue
        weight_quantizer = getattr(module, "weight_quantizer", None)
        input_quantizer = getattr(module, "input_quantizer", None)
        weight_bits = _quantizer_bits(weight_quantizer)
        act_bits = _quantizer_bits(input_quantizer)

        numel = module.weight.numel()
        out_features = module.weight.shape[0]
        total_numel += numel
        bytes_bf16 += numel * BF16_BITS // 8

        if weight_bits is None:
            realized = numel * BF16_BITS // 8
        else:
            realized = numel * weight_bits // 8
            quantized_numel += numel
            # Per-output-channel scales; a per-tensor weight scale would be one value,
            # but every format used here is axis 0.
            axis = getattr(weight_quantizer, "axis", None)
            realized += out_features * SCALE_BYTES if axis is not None else SCALE_BYTES
        bytes_realized += realized

        entry: dict[str, Any] = {
            "name": name,
            "numel": numel,
            "out_features": out_features,
            "in_features": module.weight.shape[1],
            "weight_bits": weight_bits,
            "activation_bits": act_bits,
            "weight_quantizer_enabled": weight_bits is not None,
            "input_quantizer_enabled": act_bits is not None,
            "dynamic_activation": bool(getattr(input_quantizer, "block_sizes", None))
            if input_quantizer is not None
            else False,
            "format": get_quantization_format(module),
            "weight_bytes_realized": realized,
        }

        inventory_entry = _inventory_lookup(inventory, name)
        if inventory_entry is not None:
            flops = inventory_entry["flops_per_step"]
            entry["flops_per_step"] = flops
            entry["tokens"] = inventory_entry["tokens"]
            entry["layer_class"] = inventory_entry.get("layer_class")
            flops_total += flops
            if weight_bits is not None:
                flops_quantized += flops
        elif inventory:
            unmatched.append(name)

        layers.append(entry)

    flop_fraction = flops_quantized / flops_total if flops_total else None
    projected = {}
    if flop_fraction is not None:
        for gpu, ratio in TOPS_RATIO.items():
            f = gemm_time_share * flop_fraction
            projected[f"projected_speedup_{gpu}"] = round(1.0 / ((1.0 - f) + f / ratio), 4)

    return {
        "meta": meta or {},
        "measured": {
            "linear_modules": len(layers),
            "quantized_modules": sum(1 for entry in layers if entry["weight_quantizer_enabled"]),
            "weight_bytes_bf16": bytes_bf16,
            "weight_bytes_realized": bytes_realized,
            "weight_gib_bf16": round(bytes_bf16 / 2**30, 4),
            "weight_gib_realized": round(bytes_realized / 2**30, 4),
            "vram_fraction_of_bf16": round(bytes_realized / bytes_bf16, 4) if bytes_bf16 else None,
            "effective_bits": round(BF16_BITS * bytes_realized / bytes_bf16, 3)
            if bytes_bf16
            else None,
            "quantized_weight_fraction": round(quantized_numel / total_numel, 4)
            if total_numel
            else None,
        },
        "projected": {
            "note": (
                "Amdahl over the quantized GEMM share. No INT8 GEMM backend is registered "
                "in this path, so an INT8 forward dequantizes to BF16; treat this as an "
                "upper bound, never as a measurement."
            ),
            "gemm_time_share_assumed": gemm_time_share,
            "tops_ratio": TOPS_RATIO,
            "flop_fraction_quantized": round(flop_fraction, 4)
            if flop_fraction is not None
            else None,
            **projected,
        },
        "inventory_join": {
            "inventory": str(inventory_path) if inventory_path else None,
            "matched": sum(1 for entry in layers if "flops_per_step" in entry),
            "unmatched": unmatched[:32],
            "unmatched_count": len(unmatched),
        },
        "layers": layers,
    }


def write_quant_cost_report(
    backbone: torch.nn.Module,
    out_path: Path,
    logger: logging.Logger | None = None,
    inventory_path: Path | None = None,
    gemm_time_share: float = 0.75,
    meta: dict[str, Any] | None = None,
) -> dict:
    """Build the cost report and write it as JSON."""
    report = build_quant_cost_report(
        backbone,
        inventory_path=inventory_path,
        gemm_time_share=gemm_time_share,
        meta=meta,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    if logger is not None:
        measured = report["measured"]
        logger.info(
            "Quant cost: %d/%d linears quantized, %.2f GiB vs %.2f GiB BF16 "
            "(%.1f%%, %.2f effective bits)",
            measured["quantized_modules"],
            measured["linear_modules"],
            measured["weight_gib_realized"],
            measured["weight_gib_bf16"],
            100 * (measured["vram_fraction_of_bf16"] or 0),
            measured["effective_bits"] or 0,
        )
        if report["projected"]["flop_fraction_quantized"] is None:
            logger.info(
                "No FLOP inventory supplied; cost report has weight bytes only. "
                "Pass --cost-inventory to add the projected latency columns."
            )
        if report["inventory_join"]["unmatched_count"]:
            logger.warning(
                "%d quantized linears had no inventory match; FLOP totals are partial.",
                report["inventory_join"]["unmatched_count"],
            )
    return report
