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

"""Cost-anchored INT8 protection tiers for the LTX-2 transformer.

Keeping a layer out of INT8 buys quality and gives back savings. A protection set
written by hand states which layers, but not what it costs, so there is no way to tell
whether one tier gives back 5% or 40% of the quantization win. This module solves the
inverse problem: given a cost target, choose the protection set that lands on it.

Two axes are searched when composing a protection set:

  * layer class - to_gate_logits, ff.net.2, to_out.0, ff.net.0.proj, to_q/to_k/to_v,
                  and the audio / cross-modal streams
  * block id    - within a class, blocks are protected from the ends of the stack
                  inward (0, last, 1, last-1, ...), so depth is a real dimension of the
                  ladder rather than a hard-coded tail

Cost is measured on two axes, which do **not** coincide for this model. The audio stream
carries ~13% of the linear weight bytes but only ~1.5% of the GEMM FLOPs, because it
sees ~51 tokens against video's ~2688 at a typical clip geometry. Protecting audio is
therefore nearly free in projected latency and expensive in VRAM.

  ``vram``     weight bytes, including scale-tensor bytes, against BF16
  ``latency``  quantized share of linear GEMM FLOPs, projected through Amdahl
  ``balanced`` geometric mean of the two

In practice the three orderings trace nearly the same path through cost space: at
matched cost on one axis they deliver nearly identical results on the other, diverging
only after the cheap classes are exhausted and then reconverging. The axis choice
matters much less than where the walk stops, so ``vram`` is the default: weight bytes
are realizable, whereas the latency figure is a projection (see ``projected_speedup``).

Run this file directly to plan cost points from a checkpoint before booking a GPU:

    python ltx2_tier_solver.py --safetensors <fused>.safetensors --targets 0.75,0.5,0.25

Inside the quantization flow, prefer ``solve_protection_from_model``: it builds the
inventory from the live module tree, so the tier is solved against exactly the modules
that were wrapped, with exact shapes.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# --- clip geometry ------------------------------------------------------------------
#
# Mirrors ltx_core.types: VideoLatentShape.from_pixel_shape with scale factors
# (time=8, height=32, width=32) and token_count = frames * height * width;
# AudioLatentShape.from_video_pixel_shape at 16000 / 160 / 4 = 25 latents per second;
# and the Gemma tokenizer's fixed 1024-token pad.

VIDEO_SCALE_TIME = 8
VIDEO_SCALE_HEIGHT = 32
VIDEO_SCALE_WIDTH = 32
AUDIO_LATENTS_PER_SECOND = 16000 / 160 / 4
TEXT_TOKENS = 1024

BF16_BYTES = 2
INT8_BYTES = 1
SCALE_BYTES = 4  # float32 per-output-channel weight_scale, as written by the exporter

# INT8 : BF16 dense arithmetic ratio. A100 624 TOPS / 312 TFLOPS and H100 SXM
# 1979 TOPS / 989 TFLOPS both come to 2.0.
TOPS_RATIO = {"a100": 2.0, "h100": 2.0}

METRICS = ("vram", "latency", "balanced")


@dataclass(frozen=True)
class TokenGeometry:
    """Token counts each stream sees for one clip, at one denoise step."""

    video: int
    audio: int
    text: int = TEXT_TOKENS

    @staticmethod
    def for_clip(height: int, width: int, frames: int, fps: float) -> "TokenGeometry":
        latent_frames = (frames - 1) // VIDEO_SCALE_TIME + 1
        return TokenGeometry(
            video=latent_frames * (height // VIDEO_SCALE_HEIGHT) * (width // VIDEO_SCALE_WIDTH),
            audio=round((frames / fps) * AUDIO_LATENTS_PER_SECOND),
            text=TEXT_TOKENS,
        )

    def as_dict(self) -> dict[str, int]:
        return {"video": self.video, "audio": self.audio, "text": self.text}


# --- module classification ---------------------------------------------------------

# Which stream supplies queries and which supplies keys/values, per attention container.
# Taken from the LTX-2 block definition, including its own "Q: Video, K,V: Audio" split
# for the two cross-modal attentions.
ATTENTION_STREAMS = {
    "attn1": ("video", "video"),
    "attn2": ("video", "text"),
    "audio_attn1": ("audio", "audio"),
    "audio_attn2": ("audio", "text"),
    "audio_to_video_attn": ("video", "audio"),
    "video_to_audio_attn": ("audio", "video"),
}

# Modules kept in high precision at every tier: embeddings, the timestep / adaLN
# conditioning path, patchify, and the output projection. Identical to the set
# filter_func_ltx_video already protects, so a target of 1.0 reproduces stock behaviour.
BASE_PROTECT_PATTERN = re.compile(
    r".*(proj_in|time_embed|caption_projection|proj_out|patchify_proj|adaln_single).*"
)

_BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.")

# How fragile each layer class is expected to be under a uniform INT8 grid. This is a
# PRIOR, not a measurement: it encodes that a heads-wide gate tensor decides whether a
# branch contributes at all, that the FFN down-projection is the class most often
# reported as dominant for INT8, and that bulk q/k/v compute is the most tolerant.
# Override it with ``sensitivity_prior=`` to test a different ordering, or replace it
# outright with an empirical ranking from an AutoQuantize search.
SENSITIVITY_PRIOR: dict[str, float] = {
    "attn_gate": 10.0,
    "xmodal_attn_gate": 10.0,
    "audio_attn_gate": 10.0,
    "ffn_down": 9.0,
    "audio_ffn_down": 9.0,
    "xmodal_attn_k": 6.0,
    "xmodal_attn_v": 6.0,
    "xmodal_attn_q": 5.5,
    "xmodal_attn_out": 5.5,
    "attn_out": 5.0,
    "audio_attn_out": 5.0,
    "ffn_up": 4.0,
    "audio_ffn_up": 4.0,
    "attn_k": 3.0,
    "attn_v": 3.0,
    "audio_attn_k": 3.0,
    "audio_attn_v": 3.0,
    "attn_q": 2.0,
    "audio_attn_q": 2.0,
}


@dataclass
class LinearEntry:
    name: str
    out_features: int
    in_features: int
    block: int | None
    layer_class: str
    tokens: int
    always_protected: bool

    @property
    def numel(self) -> int:
        return self.out_features * self.in_features

    @property
    def flops(self) -> int:
        """Forward GEMM FLOPs for one denoise step, one guidance branch.

        Guidance and the denoise-step count scale every module equally, so they cancel
        out of every fraction reported here.
        """
        return 2 * self.tokens * self.numel


def classify(name: str, out_features: int, in_features: int, tokens: TokenGeometry) -> LinearEntry:
    """Assign a layer class and a token count to one Linear, by module name."""
    block_match = _BLOCK_RE.search(name)
    block = int(block_match.group(1)) if block_match else None

    container = None
    for candidate in ATTENTION_STREAMS:
        # Match on a dotted boundary so audio_attn1 does not also match attn1, and take
        # the longest match so audio_to_video_attn wins over any shorter prefix.
        if f".{candidate}." in name or name.startswith(f"{candidate}."):
            if container is None or len(candidate) > len(container):
                container = candidate

    layer_class = "other"
    if container is not None:
        query_stream, context_stream = ATTENTION_STREAMS[container]
        leaf = name.rsplit(f"{container}.", 1)[1]
        if leaf.startswith("to_q"):
            layer_class, stream = "attn_q", query_stream
        elif leaf.startswith("to_k"):
            layer_class, stream = "attn_k", context_stream
        elif leaf.startswith("to_v"):
            layer_class, stream = "attn_v", context_stream
        elif leaf.startswith("to_out"):
            layer_class, stream = "attn_out", query_stream
        elif leaf.startswith("to_gate_logits"):
            layer_class, stream = "attn_gate", query_stream
        else:
            layer_class, stream = "attn_other", query_stream
        if container in ("audio_to_video_attn", "video_to_audio_attn"):
            layer_class = f"xmodal_{layer_class}"
        elif container.startswith("audio_"):
            layer_class = f"audio_{layer_class}"
    elif "ff.net." in name:
        stream = "audio" if "audio_ff" in name else "video"
        layer_class = "ffn_up" if ".net.0" in name else "ffn_down"
        if stream == "audio":
            layer_class = f"audio_{layer_class}"
    else:
        stream = "audio" if "audio_" in name else "video"
        if "caption_projection" in name:
            stream = "text"
        elif "adaln" in name or "time_embed" in name or "emb." in name:
            # Conditioning is computed once per timestep, not once per token.
            stream = "single"

    token_map = {
        "video": tokens.video,
        "audio": tokens.audio,
        "text": tokens.text,
        "single": 1,
    }
    return LinearEntry(
        name=name,
        out_features=out_features,
        in_features=in_features,
        block=block,
        layer_class=layer_class,
        tokens=token_map[stream],
        always_protected=BASE_PROTECT_PATTERN.match(name) is not None,
    )


def build_inventory_from_model(
    backbone: torch.nn.Module, tokens: TokenGeometry
) -> list[LinearEntry]:
    """Inventory the Linear modules of a live backbone.

    Preferred over the checkpoint reader: shapes are exact and the set is exactly the
    modules present in this configuration, so a class that a given build does not
    instantiate cannot skew the solved tier.
    """
    inventory = []
    for name, module in backbone.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            continue
        if not isinstance(module, torch.nn.Linear) and not hasattr(module, "weight_quantizer"):
            continue
        inventory.append(classify(name, weight.shape[0], weight.shape[1], tokens))
    return inventory


def build_inventory_from_safetensors(path: Path, tokens: TokenGeometry) -> list[LinearEntry]:
    """Inventory from a checkpoint's safetensors header, without loading any tensors.

    Pure stdlib and payload-free, so this runs on a login node against a multi-tens-of-GB
    checkpoint. Use it to choose cost points before a GPU is available.
    """
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))

    inventory = []
    for key, meta in header.items():
        if key == "__metadata__" or not key.endswith(".weight"):
            continue
        shape = meta.get("shape", [])
        if len(shape) != 2:
            continue
        name = key[: -len(".weight")]
        for prefix in ("model.diffusion_model.", "diffusion_model.", "model."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        if name.startswith(("vae.", "video_vae.", "audio_vae.", "text_encoder.")):
            continue
        inventory.append(classify(name, shape[0], shape[1], tokens))
    return inventory


# --- costing ------------------------------------------------------------------------


@dataclass
class Cost:
    quantized_modules: int = 0
    protected_modules: int = 0
    weight_bytes_bf16: int = 0
    weight_bytes_tiered: int = 0
    flops_total: int = 0
    flops_quantized: int = 0

    @property
    def byte_fraction_of_bf16(self) -> float:
        return self.weight_bytes_tiered / self.weight_bytes_bf16

    @property
    def effective_bits(self) -> float:
        return 16.0 * self.byte_fraction_of_bf16

    @property
    def flop_fraction_quantized(self) -> float:
        return self.flops_quantized / self.flops_total if self.flops_total else 0.0

    def projected_speedup(self, gemm_time_share: float, tops_ratio: float) -> float:
        """Amdahl over the quantized GEMM share.

        PROJECTED, not measured. Whether it is achievable depends on a real low-precision
        GEMM backend being registered for the format; without one the forward dequantizes
        and the observed latency can be worse than BF16. Treat this as an upper bound.
        """
        f = gemm_time_share * self.flop_fraction_quantized
        return 1.0 / ((1.0 - f) + f / tops_ratio)


def cost_of(inventory: list[LinearEntry], is_protected) -> Cost:
    cost = Cost()
    for entry in inventory:
        cost.weight_bytes_bf16 += entry.numel * BF16_BYTES
        cost.flops_total += entry.flops
        if entry.always_protected or is_protected(entry):
            cost.protected_modules += 1
            cost.weight_bytes_tiered += entry.numel * BF16_BYTES
        else:
            cost.quantized_modules += 1
            cost.weight_bytes_tiered += entry.numel * INT8_BYTES + entry.out_features * SCALE_BYTES
            cost.flops_quantized += entry.flops
    return cost


def savings(cost: Cost, floor: Cost) -> tuple[float, float]:
    """(latency saving retained, VRAM saving retained) against the all-INT8 floor.

    1.0 is the most aggressive point the base protection set permits; 0.0 is everything
    back in high precision. Defined identically on both axes so the two are comparable.
    """
    latency = cost.flops_quantized / floor.flops_quantized if floor.flops_quantized else 0.0
    achievable = floor.weight_bytes_bf16 - floor.weight_bytes_tiered
    vram = (cost.weight_bytes_bf16 - cost.weight_bytes_tiered) / achievable if achievable else 0.0
    return latency, vram


# --- tier construction --------------------------------------------------------------


@dataclass
class ProtectionSet:
    """Protected modules, as whole layer classes plus per-class block subsets."""

    whole_classes: set[str] = field(default_factory=set)
    partial: dict[str, set[int]] = field(default_factory=dict)

    def protects(self, entry: LinearEntry) -> bool:
        if entry.layer_class in self.whole_classes:
            return True
        blocks = self.partial.get(entry.layer_class)
        return blocks is not None and entry.block in blocks

    def copy(self) -> "ProtectionSet":
        return ProtectionSet(set(self.whole_classes), {k: set(v) for k, v in self.partial.items()})

    def module_names(self, inventory: list[LinearEntry]) -> list[str]:
        """Exact module names, so a solved tier can be pinned and replayed verbatim."""
        return sorted(entry.name for entry in inventory if self.protects(entry))


def protection_order(
    inventory: list[LinearEntry],
    metric: str = "vram",
    sensitivity_prior: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Rank layer classes by expected quality benefit per unit of cost given up.

    Dividing one sensitivity prior by a per-axis cost share is what makes the axes
    behave differently without any per-axis hand-editing: a class that is cheap in FLOPs
    and expensive in bytes moves to the front of the latency order and the back of the
    VRAM order on its own.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}.")
    prior = sensitivity_prior or SENSITIVITY_PRIOR

    flop_totals: dict[str, float] = {}
    byte_totals: dict[str, float] = {}
    for entry in inventory:
        if entry.always_protected:
            continue
        flop_totals[entry.layer_class] = flop_totals.get(entry.layer_class, 0.0) + entry.flops
        byte_totals[entry.layer_class] = (
            byte_totals.get(entry.layer_class, 0.0) + entry.numel * BF16_BYTES
        )
    flop_grand = sum(flop_totals.values()) or 1.0
    byte_grand = sum(byte_totals.values()) or 1.0

    ranked = []
    for layer_class in flop_totals:
        flop_share = flop_totals[layer_class] / flop_grand
        byte_share = byte_totals[layer_class] / byte_grand
        if metric == "latency":
            share = flop_share
        elif metric == "vram":
            share = byte_share
        else:
            share = (flop_share * byte_share) ** 0.5
        ranked.append((layer_class, prior.get(layer_class, 1.0) / max(share, 1e-12)))
    ranked.sort(key=lambda item: -item[1])
    return ranked


def block_ramp_order(blocks: list[int]) -> list[int]:
    """Blocks ordered from the ends of the stack inward: first, last, second, ...

    The first and last blocks of a DiT stack carry the largest activation-range
    excursions, so depth-wise protection starts there.
    """
    ordered = sorted(blocks)
    ramp: list[int] = []
    low, high = 0, len(ordered) - 1
    while low <= high:
        ramp.append(ordered[low])
        if high != low:
            ramp.append(ordered[high])
        low += 1
        high -= 1
    return ramp


def _trace(
    inventory: list[LinearEntry],
    metric: str,
    sensitivity_prior: dict[str, float] | None = None,
) -> list[tuple[ProtectionSet, Cost, str]]:
    """Every protection set one ordering passes through, least to most protected.

    Protection only grows along the trace, so savings only fall. Comparing orderings
    means comparing whole traces: a tier that protects less always shows a larger saving
    on both axes, which says nothing about whether its ordering was better.
    """
    blocks = sorted({entry.block for entry in inventory if entry.block is not None})
    ramp = block_ramp_order(blocks)
    current = ProtectionSet()
    trace = [(current.copy(), cost_of(inventory, current.protects), "none")]

    for layer_class, _ in protection_order(inventory, metric, sensitivity_prior):
        has_blocks = any(
            entry.layer_class == layer_class and entry.block is not None for entry in inventory
        )
        if not has_blocks:
            current = current.copy()
            current.whole_classes.add(layer_class)
            trace.append((current.copy(), cost_of(inventory, current.protects), layer_class))
            continue
        for block in ramp:
            current = current.copy()
            current.partial.setdefault(layer_class, set()).add(block)
            trace.append(
                (current.copy(), cost_of(inventory, current.protects), f"{layer_class}@{block}")
            )
    return trace


def solve_protection(
    inventory: list[LinearEntry],
    target: float,
    metric: str = "vram",
    sensitivity_prior: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Choose the protection set that lands closest to a cost target.

    ``target`` is the fraction of the achievable saving on ``metric`` that the tier
    should still retain: 1.0 protects nothing beyond the base set, 0.0 protects
    everything. Returns the protection set, its realized cost on both axes, and the step
    at which the walk stopped.
    """
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"Protection target must be in [0, 1], got {target}.")
    trace = _trace(inventory, metric, sensitivity_prior)
    floor = trace[0][1]

    def measure(cost: Cost) -> float:
        latency, vram = savings(cost, floor)
        if metric == "latency":
            return latency
        if metric == "vram":
            return vram
        return (latency * vram) ** 0.5

    protection, cost, step = min(trace, key=lambda item: abs(measure(item[1]) - target))
    latency, vram = savings(cost, floor)
    return {
        "metric": metric,
        "target": target,
        "achieved": round(measure(cost), 4),
        "last_step": step,
        "protection": protection,
        "cost": cost,
        "latency_saving_retained": round(latency, 4),
        "vram_saving_retained": round(vram, 4),
    }


def describe_solution(solution: dict[str, Any], gemm_time_share: float = 0.75) -> dict[str, Any]:
    """Flatten a solved tier into a JSON-friendly record, labelling what is projected."""
    cost: Cost = solution["cost"]
    return {
        "metric": solution["metric"],
        "target": solution["target"],
        "achieved_saving_retained": solution["achieved"],
        "last_step": solution["last_step"],
        "measured": {
            "quantized_modules": cost.quantized_modules,
            "protected_modules": cost.protected_modules,
            "weight_gib_bf16": round(cost.weight_bytes_bf16 / 2**30, 3),
            "weight_gib_tiered": round(cost.weight_bytes_tiered / 2**30, 3),
            "vram_fraction_of_bf16": round(cost.byte_fraction_of_bf16, 4),
            "effective_bits": round(cost.effective_bits, 3),
            "vram_saving_retained": solution["vram_saving_retained"],
        },
        "projected": {
            "note": (
                "Amdahl over the quantized GEMM share; an upper bound that assumes a real "
                "low-precision GEMM backend exists for this format."
            ),
            "gemm_time_share_assumed": gemm_time_share,
            "flop_fraction_quantized": round(cost.flop_fraction_quantized, 4),
            "latency_saving_retained": solution["latency_saving_retained"],
            **{
                f"speedup_{gpu}": round(cost.projected_speedup(gemm_time_share, ratio), 4)
                for gpu, ratio in TOPS_RATIO.items()
            },
        },
    }


def solve_protection_from_model(
    backbone: torch.nn.Module,
    target: float,
    tokens: TokenGeometry,
    metric: str = "vram",
    gemm_time_share: float = 0.75,
) -> tuple[list[str], dict[str, Any]]:
    """Solve a protection tier against a live backbone.

    Returns the protected module names and a report of the tier's predicted cost. The
    names are exact, so callers can both apply them now and record them for replay.
    """
    inventory = build_inventory_from_model(backbone, tokens)
    if not inventory:
        raise RuntimeError("No 2-D Linear weights found in the backbone; cannot solve a tier.")
    solution = solve_protection(inventory, target, metric)
    report = describe_solution(solution, gemm_time_share)
    report["tokens"] = tokens.as_dict()
    report["linear_modules"] = len(inventory)
    report["protected_classes_whole"] = sorted(solution["protection"].whole_classes)
    report["protected_classes_by_block"] = {
        k: sorted(v) for k, v in sorted(solution["protection"].partial.items())
    }
    return solution["protection"].module_names(inventory), report


def class_breakdown(inventory: list[LinearEntry]) -> list[dict[str, Any]]:
    """Per-class FLOP and byte shares: where the two columns disagree, the axes do."""
    totals: dict[str, dict[str, Any]] = {}
    flops_all = sum(entry.flops for entry in inventory) or 1
    bytes_all = sum(entry.numel * BF16_BYTES for entry in inventory) or 1
    for entry in inventory:
        row = totals.setdefault(
            entry.layer_class,
            {"layer_class": entry.layer_class, "count": 0, "flops": 0, "bytes": 0},
        )
        row["count"] += 1
        row["flops"] += entry.flops
        row["bytes"] += entry.numel * BF16_BYTES
    rows = sorted(totals.values(), key=lambda r: -r["flops"])
    for row in rows:
        row["flop_share"] = round(row["flops"] / flops_all, 4)
        row["byte_share"] = round(row["bytes"] / bytes_all, 4)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--safetensors", type=Path, required=True, help="Checkpoint to inventory (header only)"
    )
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--targets",
        default="0.75,0.5,0.25",
        help="Fractions of the achievable saving to retain, high to low",
    )
    parser.add_argument("--metric", choices=METRICS, default="vram")
    parser.add_argument("--gemm-time-share", type=float, default=0.75)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write tier_targets.json and protect_<target>.json here",
    )
    args = parser.parse_args()

    tokens = TokenGeometry.for_clip(args.height, args.width, args.frames, args.fps)
    inventory = build_inventory_from_safetensors(args.safetensors, tokens)
    if not inventory:
        print("No 2-D Linear weights found in the checkpoint header.", file=sys.stderr)
        return 1

    floor = cost_of(inventory, lambda entry: False)
    print(f"Tokens per clip: video={tokens.video} audio={tokens.audio} text={tokens.text}")
    print(
        f"Linear modules: {len(inventory)}   BF16 linear weights: "
        f"{floor.weight_bytes_bf16 / 2**30:.2f} GiB   "
        f"per-step GEMM FLOPs: {floor.flops_total / 1e12:.2f} TFLOP"
    )
    print()
    print(f"{'layer_class':<22}{'count':>7}{'flop_share':>12}{'byte_share':>12}")
    for row in class_breakdown(inventory):
        print(
            f"{row['layer_class']:<22}{row['count']:>7}"
            f"{row['flop_share']:>12.4f}{row['byte_share']:>12.4f}"
        )
    print()

    targets = [float(t) for t in args.targets.split(",")]
    records = []
    print(f"Tiers on the {args.metric} axis:")
    print(
        f"{'target':>8}{'lat_ret':>9}{'vram_ret':>10}{'quant_mods':>12}"
        f"{'GiB':>9}{'eff_bits':>10}{'proj_x_a100':>13}"
    )
    for target in [1.0, *targets]:
        solution = solve_protection(inventory, target, args.metric)
        record = describe_solution(solution, args.gemm_time_share)
        record["protect"] = solution["protection"].module_names(inventory)
        records.append(record)
        measured, projected = record["measured"], record["projected"]
        print(
            f"{target:>8.2f}{projected['latency_saving_retained']:>9.3f}"
            f"{measured['vram_saving_retained']:>10.3f}{measured['quantized_modules']:>12}"
            f"{measured['weight_gib_tiered']:>9.2f}{measured['effective_bits']:>10.2f}"
            f"{projected['speedup_a100']:>13.3f}"
        )
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            name = f"protect_{args.metric}{int(round(target * 100)):03d}.json"
            (args.out_dir / name).write_text(
                json.dumps({"target": target, "protect": record["protect"]}, indent=2) + "\n"
            )

    if args.out_dir:
        (args.out_dir / "tier_targets.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(args.safetensors),
                    "tokens": tokens.as_dict(),
                    "metric": args.metric,
                    "sensitivity_prior_is_not_a_measurement": SENSITIVITY_PRIOR,
                    "tiers": records,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nWrote tier definitions to {args.out_dir}")
    print("\nLatency columns are PROJECTED; byte and VRAM columns are realizable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
