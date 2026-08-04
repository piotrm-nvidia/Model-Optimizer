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

"""Self-checks for the LTX-2 protection tier solver.

Runs against a synthetic module tree that reproduces LTX-2's naming, so the solver's
classification and monotonicity can be checked without a 22B checkpoint or a GPU:

    pytest test_ltx2_tier_solver.py
    python test_ltx2_tier_solver.py     # same checks, no pytest needed

The synthetic model is small but shaped like the real one - dual video/audio streams,
cross-modal attention, gated attention, and the same leaf names - because every
classification rule keys off those names.
"""

from __future__ import annotations

import torch
from ltx2_tier_solver import (
    BF16_BYTES,
    SENSITIVITY_PRIOR,
    TokenGeometry,
    build_inventory_from_model,
    cost_of,
    protection_order,
    savings,
    solve_protection,
    solve_protection_from_model,
)
from torch import nn

VIDEO_DIM = 64
AUDIO_DIM = 32
TEXT_DIM = 48
HEADS = 4
N_BLOCKS = 6


class FakeAttention(nn.Module):
    """Attention with LTX-2's leaf names, including the gate-logits projection."""

    def __init__(self, query_dim: int, context_dim: int, gated: bool = True):
        super().__init__()
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim, bias=False)])
        if gated:
            self.to_gate_logits = nn.Linear(query_dim, HEADS, bias=False)


class FakeFeedForward(nn.Module):
    """net.0.proj is the up-projection and net.2 the down-projection, as in LTX-2."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        proj = nn.Module()
        proj.proj = nn.Linear(dim, inner, bias=False)
        self.net = nn.ModuleList([proj, nn.Identity(), nn.Linear(inner, dim, bias=False)])


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = FakeAttention(VIDEO_DIM, VIDEO_DIM)
        self.attn2 = FakeAttention(VIDEO_DIM, TEXT_DIM)
        self.ff = FakeFeedForward(VIDEO_DIM)
        self.audio_attn1 = FakeAttention(AUDIO_DIM, AUDIO_DIM)
        self.audio_attn2 = FakeAttention(AUDIO_DIM, TEXT_DIM)
        self.audio_ff = FakeFeedForward(AUDIO_DIM)
        # Cross-modal: audio_to_video_attn has video queries and audio context.
        self.audio_to_video_attn = FakeAttention(VIDEO_DIM, AUDIO_DIM)
        self.video_to_audio_attn = FakeAttention(AUDIO_DIM, VIDEO_DIM)


class FakeLtx2(nn.Module):
    def __init__(self):
        super().__init__()
        self.patchify_proj = nn.Linear(VIDEO_DIM, VIDEO_DIM, bias=False)
        self.audio_patchify_proj = nn.Linear(AUDIO_DIM, AUDIO_DIM, bias=False)
        self.caption_projection = nn.Linear(TEXT_DIM, VIDEO_DIM, bias=False)
        self.adaln_single = nn.Linear(VIDEO_DIM, 6 * VIDEO_DIM, bias=False)
        self.transformer_blocks = nn.ModuleList([FakeBlock() for _ in range(N_BLOCKS)])
        self.proj_out = nn.Linear(VIDEO_DIM, VIDEO_DIM, bias=False)


def _inventory(tokens: TokenGeometry | None = None):
    tokens = tokens or TokenGeometry.for_clip(512, 768, 49, 24.0)
    return build_inventory_from_model(FakeLtx2(), tokens), tokens


def _by_name(inventory):
    return {entry.name: entry for entry in inventory}


def test_token_geometry_matches_latent_shapes():
    tokens = TokenGeometry.for_clip(512, 768, 121, 24.0)
    # (121 - 1) / 8 + 1 = 16 latent frames, 512/32 = 16 rows, 768/32 = 24 columns.
    assert tokens.video == 16 * 16 * 24
    # 121 frames at 24 fps is ~5.04 s, at 25 audio latents per second.
    assert tokens.audio == 126
    assert tokens.text == 1024


def test_classification_follows_module_names():
    inventory, _ = _inventory()
    entries = _by_name(inventory)

    assert entries["transformer_blocks.0.ff.net.0.proj"].layer_class == "ffn_up"
    assert entries["transformer_blocks.0.ff.net.2"].layer_class == "ffn_down"
    # audio_ff must not be misread as the video FFN, which an over-specific pattern does.
    assert entries["transformer_blocks.0.audio_ff.net.2"].layer_class == "audio_ffn_down"
    assert entries["transformer_blocks.0.attn1.to_gate_logits"].layer_class == "attn_gate"
    assert entries["transformer_blocks.0.audio_attn1.to_q"].layer_class == "audio_attn_q"
    # Longest-match wins, so audio_to_video_attn is cross-modal and not the audio stream.
    assert entries["transformer_blocks.0.audio_to_video_attn.to_q"].layer_class == "xmodal_attn_q"
    assert entries["transformer_blocks.0.audio_to_video_attn.to_k"].layer_class == "xmodal_attn_k"
    assert all(entry.layer_class != "other" for entry in inventory if entry.block is not None)


def test_token_counts_follow_the_stream_that_supplies_the_tensor():
    inventory, tokens = _inventory()
    entries = _by_name(inventory)

    assert entries["transformer_blocks.0.attn1.to_q"].tokens == tokens.video
    assert entries["transformer_blocks.0.audio_attn1.to_q"].tokens == tokens.audio
    # attn2 is text cross-attention: queries are video, keys and values are text.
    assert entries["transformer_blocks.0.attn2.to_q"].tokens == tokens.video
    assert entries["transformer_blocks.0.attn2.to_k"].tokens == tokens.text
    # audio_to_video_attn: video queries, audio keys and values.
    assert entries["transformer_blocks.0.audio_to_video_attn.to_q"].tokens == tokens.video
    assert entries["transformer_blocks.0.audio_to_video_attn.to_k"].tokens == tokens.audio
    # adaLN conditioning is computed once per timestep, not per token.
    assert entries["adaln_single"].tokens == 1


def test_base_set_is_protected_at_every_target():
    inventory, tokens = _inventory()
    always = {entry.name for entry in inventory if entry.always_protected}
    assert {"patchify_proj", "audio_patchify_proj", "caption_projection", "adaln_single", "proj_out"} <= always

    for target in (1.0, 0.75, 0.5, 0.25, 0.0):
        names = set(solve_protection(inventory, target)["protection"].module_names(inventory))
        # module_names lists solver-chosen modules; the base set is protected regardless,
        # so what matters is that no target can quantize it.
        protection = solve_protection(inventory, target)["protection"]
        for entry in inventory:
            if entry.always_protected:
                assert cost_of([entry], protection.protects).quantized_modules == 0
        assert names <= {entry.name for entry in inventory}


def test_target_one_reproduces_the_stock_filter_and_zero_protects_everything():
    inventory, _ = _inventory()
    floor = cost_of(inventory, lambda entry: False)

    top = solve_protection(inventory, 1.0)
    assert top["protection"].module_names(inventory) == []
    assert top["cost"].quantized_modules == floor.quantized_modules

    bottom = solve_protection(inventory, 0.0)
    assert bottom["vram_saving_retained"] < 0.05
    assert bottom["latency_saving_retained"] < 0.05


def test_savings_fall_monotonically_as_the_target_falls():
    inventory, _ = _inventory()
    previous_vram = previous_latency = None
    for target in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
        solution = solve_protection(inventory, target, "vram")
        vram = solution["vram_saving_retained"]
        latency = solution["latency_saving_retained"]
        if previous_vram is not None:
            assert vram <= previous_vram + 1e-9, f"VRAM saving rose at target {target}"
            assert latency <= previous_latency + 1e-9, f"latency saving rose at target {target}"
        previous_vram, previous_latency = vram, latency


def test_targets_are_hit_closely():
    inventory, _ = _inventory()
    for metric in ("vram", "latency"):
        for target in (0.75, 0.5, 0.25):
            solution = solve_protection(inventory, target, metric)
            # The ladder is discrete, so a target is snapped to rather than hit exactly;
            # a wide miss means the walk has a step big enough to jump over a cost point.
            assert abs(solution["achieved"] - target) < 0.15, (
                f"{metric} target {target} landed at {solution['achieved']}"
            )


def test_effective_bits_stay_between_int8_and_bf16():
    inventory, _ = _inventory()
    for target in (1.0, 0.5, 0.0):
        cost = solve_protection(inventory, target)["cost"]
        assert 8.0 <= cost.effective_bits <= 16.0
    # Nothing protected still exceeds 8 bits, because per-channel scales cost bytes too.
    assert solve_protection(inventory, 1.0)["cost"].effective_bits > 8.0


def test_the_axes_disagree_because_audio_is_cheap_in_flops_and_dear_in_bytes():
    inventory, _ = _inventory()
    audio = [entry for entry in inventory if entry.layer_class.startswith("audio_")]
    flop_share = sum(e.flops for e in audio) / sum(e.flops for e in inventory)
    byte_share = sum(e.numel for e in audio) / sum(e.numel for e in inventory)
    assert byte_share > flop_share * 3, (
        "audio should be far cheaper in FLOPs than in bytes; if not, the token geometry "
        "or the stream assignment is wrong"
    )

    # That asymmetry is what makes the two orderings differ, with no per-axis hand-editing.
    latency_first = [c for c, _ in protection_order(inventory, "latency")][:6]
    vram_first = [c for c, _ in protection_order(inventory, "vram")][:6]
    assert latency_first != vram_first


def test_ordering_puts_the_most_sensitive_cheap_class_first():
    inventory, _ = _inventory()
    ranked = [c for c, _ in protection_order(inventory, "vram")]
    # Gate logits are the highest prior and a negligible share of cost on either axis, so
    # any sane cost-normalized ordering protects them before bulk projections.
    assert ranked.index("attn_gate") < ranked.index("attn_q")
    assert SENSITIVITY_PRIOR["attn_gate"] > SENSITIVITY_PRIOR["attn_q"]


def test_savings_are_defined_against_the_achievable_floor():
    inventory, _ = _inventory()
    floor = cost_of(inventory, lambda entry: False)
    latency, vram = savings(floor, floor)
    assert abs(latency - 1.0) < 1e-9
    assert abs(vram - 1.0) < 1e-9


def test_solve_from_model_returns_exact_names_and_a_labelled_report():
    tokens = TokenGeometry.for_clip(512, 768, 49, 24.0)
    model = FakeLtx2()
    names, report = solve_protection_from_model(model, 0.5, tokens, metric="vram")

    present = {name for name, _ in model.named_modules()}
    assert names, "a 0.5 target must protect something"
    assert set(names) <= present, "solved names must be real module paths"
    # The report has to separate what is realizable from what is projected, or the latency
    # number gets quoted as if it were measured.
    assert "vram_saving_retained" in report["measured"]
    assert "speedup_a100" in report["projected"]
    assert report["tokens"]["video"] == tokens.video


def test_scale_bytes_are_counted():
    inventory, _ = _inventory()
    quantized_all = cost_of(inventory, lambda entry: False)
    bare_int8 = sum(entry.numel for entry in inventory if not entry.always_protected)
    protected = sum(entry.numel for entry in inventory if entry.always_protected) * BF16_BYTES
    assert quantized_all.weight_bytes_tiered > bare_int8 + protected, (
        "per-output-channel scales must be part of the byte cost"
    )


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
    torch.manual_seed(0)
    raise SystemExit(_run_all())
