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

import torch.nn as nn
from calib.plugin_calib import PercentileCalibrator

FP8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

INT8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": 0},
        "*output_quantizer": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

# W8A8 SmoothQuant, derived from INT8_DEFAULT_CONFIG so the two static INT8 arms differ
# only in what SmoothQuant itself requires.
#
# The core INT8_SMOOTHQUANT_CFG is not that base. It carries a different skip list -
# no "default": disable, so softmax and BMM quantizers that INT8_DEFAULT_CONFIG leaves
# off would be enabled here - plus a set of LLM-shaped exclusions (lm_head, MoE routers,
# mamba conv1d) that mean nothing for a video DiT. Comparing a tier under one base
# against a tier under the other would be comparing coverage, not numerics.
#
# The one genuine difference: SmoothQuant needs a per-tensor activation quantizer.
# model_calib.smoothquant only converts a quantizer whose axis is None (to -1 for the
# per-channel collection, back to None afterwards), so leaving axis 0 here would collect
# the wrong statistic and silently smooth against it.
INT8_SMOOTHQUANT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*output_quantizer": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "smoothquant",
}

# W8A8 with per-output-channel weights and per-token dynamic activations. Dynamic
# activation scales are computed at runtime from each token's own range, so there is
# nothing to calibrate on the activation side and no SmoothQuant migration to tune,
# which makes this a clean comparison arm against static SmoothQuant INT8. Mirrors the
# core FP8_PER_CHANNEL_PER_TOKEN_CFG spelling
# (modelopt/torch/quantization/config.py) with num_bits 8.
#
# axis 0 on the weight quantizer is required, not cosmetic: the real-INT8 exporter's
# to_quantized_weight indexes the scale as [:, None], so a per-tensor weight scale would
# make --hf-ckpt-dir export unusable and the checkpoint-byte deliverable unmeasurable.
INT8_PER_CHANNEL_PER_TOKEN_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {
            "num_bits": 8,
            "type": "dynamic",
            "block_sizes": {-1: None},
        },
        "*output_quantizer": {"enable": False},
        "default": {"enable": False},
    },
    "algorithm": "max",
}

NVFP4_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*output_quantizer": {"enable": False},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

NVFP4_FP8_MHA_CONFIG = {
    "quant_cfg": {
        "**weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "**input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "*bmm2_output_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": {"method": "svdquant", "lowrank": 32},
}


def set_quant_config_attr(quant_config, trt_high_precision_dtype, quant_algo, **kwargs):
    algo_cfg = {"method": quant_algo}

    if quant_algo == "smoothquant" and "alpha" in kwargs:
        algo_cfg["alpha"] = kwargs["alpha"]
    elif quant_algo == "svdquant" and "lowrank" in kwargs:
        algo_cfg["lowrank"] = kwargs["lowrank"]
    quant_config["algorithm"] = algo_cfg

    for p in quant_config["quant_cfg"].values():
        if "num_bits" in p and "trt_high_precision_dtype" not in p:
            p["trt_high_precision_dtype"] = trt_high_precision_dtype


def reset_set_int8_config(quant_config, percentile, n_steps, collect_method, backbone):
    """
    Configure INT8 quantization with different settings for Conv2d and Linear layers.

    Args:
        quant_config: The quantization configuration dictionary
        percentile: Percentile value for calibration
        n_steps: Number of calibration steps
        collect_method: Method for collecting calibration statistics
        backbone: The model backbone to analyze layer types
    """

    # Build a mapping of layer names to their types
    layer_type_map = {}
    for name, module in backbone.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer_type_map[name] = type(module)

    quant_config["quant_cfg"] = {}
    for layer_name, layer_type in layer_type_map.items():
        wq_name = f"*{layer_name}*weight_quantizer*"
        aq_name = f"*{layer_name}*input_quantizer*"
        if layer_type is nn.Linear:
            quant_config["quant_cfg"][wq_name] = {
                "num_bits": 8,
                "axis": 0,
            }
            quant_config["quant_cfg"][aq_name] = {
                "num_bits": 8,
                "axis": -1,
            }
        else:
            quant_config["quant_cfg"][wq_name] = {
                "num_bits": 8,
                "axis": 0,
            }
            quant_config["quant_cfg"][aq_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": n_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
