# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from modelopt.torch.quantization.tensor_quant import scaled_e4m3_impl

TOKENS = 3080
HIDDEN_SIZE = 4096


def test_fp8_fake_quant_rejects_cpu_amax_for_cuda_input():
    inputs = torch.randn(1, 8, 16, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="amax must be on the input device"):
        scaled_e4m3_impl(inputs, inputs.abs().max().cpu())


@pytest.mark.parametrize(
    ("quantize_input", "quantize_weight"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_fp8_fake_quant_serving_shape_linear(quantize_input, quantize_weight):
    torch.manual_seed(180100)
    inputs = torch.randn(
        1,
        TOKENS,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    bias = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    if quantize_input:
        inputs = scaled_e4m3_impl(inputs, inputs.abs().max())
    if quantize_weight:
        weight = scaled_e4m3_impl(weight, weight.abs().max())

    output = F.linear(inputs.contiguous(), weight.contiguous(), bias)
    torch.cuda.synchronize()

    assert output.shape == (1, TOKENS, HIDDEN_SIZE)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
