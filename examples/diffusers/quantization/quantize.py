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

import argparse
import copy
import json
import logging
import sys
import time as time
from pathlib import Path
from typing import Any

import torch
from calib_coverage import check_calibration_coverage
from calibration import Calibrator
from config import (
    FP8_DEFAULT_CONFIG,
    INT8_DEFAULT_CONFIG,
    INT8_PER_CHANNEL_PER_TOKEN_CONFIG,
    INT8_SMOOTHQUANT_CONFIG,
    NVFP4_DEFAULT_CONFIG,
    NVFP4_FP8_MHA_CONFIG,
    reset_set_int8_config,
    set_quant_config_attr,
)
from diffusers import DiffusionPipeline
from ltx2_tier_solver import METRICS, TokenGeometry, solve_protection_from_model
from models_utils import (
    MODEL_DEFAULTS,
    ModelType,
    get_model_filter_func,
    parse_extra_params,
    resolve_clip_geometry,
)
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from pipeline_manager import PipelineManager
from quant_cost_report import write_quant_cost_report
from quantize_config import (
    CalibrationConfig,
    CollectMethod,
    DataType,
    ExportConfig,
    Int8Numerics,
    ModelConfig,
    QuantAlgo,
    QuantFormat,
    QuantizationConfig,
)
from utils import check_conv_and_mha, check_lora

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose logging

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create custom formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(console_handler)

    # Optionally reduce noise from other libraries
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logger


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a search state into something JSON can hold."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Quantizer:
    """Handles model quantization operations."""

    def __init__(
        self, config: QuantizationConfig, model_config: ModelConfig, logger: logging.Logger
    ):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
            model_config: Model configuration
            logger: Logger instance
        """
        self.config = config
        self.model_config = model_config
        self.logger = logger
        # Populated by solve_protection_tier when a cost target is given.
        self.protect_names: list[str] | None = None
        self.protect_report: dict[str, Any] | None = None

    def solve_protection_tier(
        self, backbone: torch.nn.Module, extra_params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Solve the protection tier for a cost target, against this loaded backbone.

        Called before calibration so the tier's predicted VRAM and projected latency are
        on record, and a target that lands somewhere unintended fails before a calibration
        pass is spent on it. Returns the cost report, or None when no target was given.
        """
        if self.config.ltx_protect_target is None:
            return None
        if self.model_config.model_type != ModelType.LTX2:
            raise ValueError(
                f"--ltx-protect-target only applies to {ModelType.LTX2.value}, "
                f"not {self.model_config.model_type.value}."
            )

        height, width, frames, fps = resolve_clip_geometry(
            self.model_config.model_type, extra_params
        )
        tokens = TokenGeometry.for_clip(height, width, frames, fps)
        self.protect_names, self.protect_report = solve_protection_from_model(
            backbone,
            self.config.ltx_protect_target,
            tokens,
            metric=self.config.ltx_protect_metric,
        )
        self.protect_report["clip_geometry"] = {
            "height": height,
            "width": width,
            "num_frames": frames,
            "frame_rate": fps,
        }
        report = self.protect_report
        self.logger.info(
            f"Solved protection tier on the {report['metric']} axis: "
            f"target {report['target']:.2f}, achieved {report['achieved_saving_retained']:.3f}, "
            f"stopped after {report['last_step']}"
        )
        self.logger.info(
            f"Tier keeps {report['measured']['protected_modules']} modules in high precision "
            f"and quantizes {report['measured']['quantized_modules']}; linear weights "
            f"{report['measured']['weight_gib_bf16']:.2f} -> "
            f"{report['measured']['weight_gib_tiered']:.2f} GiB "
            f"({report['measured']['effective_bits']:.2f} effective bits)"
        )
        self.logger.info(
            f"Projected (upper bound) A100 speedup {report['projected']['speedup_a100']:.3f}x "
            f"at a {report['projected']['gemm_time_share_assumed']:.2f} GEMM time share"
        )
        return report

    def get_quant_config(self, n_steps: int, backbone: torch.nn.Module) -> Any:
        """
        Build quantization configuration based on format.

        Args:
            n_steps: Number of denoising steps

        Returns:
            Quantization configuration object
        """
        self.logger.info(f"Building quantization config for {self.config.format.value}")

        if self.config.format == QuantFormat.INT8:
            if self.config.int8_numerics == Int8Numerics.PER_TOKEN_DYNAMIC:
                quant_config = copy.deepcopy(INT8_PER_CHANNEL_PER_TOKEN_CONFIG)
            elif self.config.algo == QuantAlgo.SMOOTHQUANT:
                # Deep-copied because set_quant_config_attr mutates in place, and a second
                # get_quant_config call in one process would otherwise see the first
                # call's edits.
                quant_config = copy.deepcopy(INT8_SMOOTHQUANT_CONFIG)
            else:
                quant_config = copy.deepcopy(INT8_DEFAULT_CONFIG)
            if self.config.collect_method != CollectMethod.DEFAULT:
                reset_set_int8_config(
                    quant_config,
                    self.config.percentile,
                    n_steps,
                    collect_method=self.config.collect_method.value,
                    backbone=backbone,
                )
        elif self.config.format == QuantFormat.FP8:
            quant_config = copy.deepcopy(FP8_DEFAULT_CONFIG)
        elif self.config.format == QuantFormat.FP4:
            if self.model_config.model_type.value.startswith("flux"):
                quant_config = copy.deepcopy(NVFP4_FP8_MHA_CONFIG)
            else:
                quant_config = copy.deepcopy(NVFP4_DEFAULT_CONFIG)
        else:
            raise NotImplementedError(f"Unknown format {self.config.format}")
        if self.config.quantize_mha:
            quant_config["quant_cfg"]["*[qkv]_bmm_quantizer"] = {"num_bits": (4, 3), "axis": None}  # type: ignore[index]
        set_quant_config_attr(
            quant_config,
            self.model_config.trt_high_precision_dtype.value,
            self.config.algo.value,
            alpha=self.config.alpha,
            lowrank=self.config.lowrank,
        )
        self.logger.info(f"Quant config {quant_config}")
        return quant_config

    def quantize_model(
        self,
        backbone: torch.nn.Module,
        quant_config: Any,
        forward_loop: callable,  # type: ignore[valid-type]
    ) -> torch.nn.Module:
        """
        Apply quantization to the model.

        Args:
            backbone: Model backbone to quantize
            quant_config: Quantization configuration
            forward_loop: Forward pass function for calibration
        """
        self.logger.info("Checking for LoRA layers...")
        check_lora(backbone)

        self.logger.info("Starting model quantization...")
        mtq.quantize(backbone, quant_config, forward_loop)

        model_filter_func = get_model_filter_func(
            self.model_config.model_type,
            protect_names=self.protect_names,
            protect_from_json=self.config.protect_from_json,
        )
        if self.config.protect_from_json is not None:
            self.logger.info(f"Protecting modules listed in {self.config.protect_from_json}")
        elif self.protect_names is not None:
            self.logger.info(f"Protecting {len(self.protect_names)} solved modules")
        else:
            self.logger.info(f"Using filter function for {self.model_config.model_type.value}")

        # Disabling after calibration rather than before is safe for SmoothQuant:
        # TensorQuantizer.forward applies pre_quant_scale before the disabled early
        # return, so a protected layer keeps both halves of the migration (activation
        # divided, weight multiplied) and stays mathematically equivalent to BF16.
        self.logger.info("Disabling protected quantizers...")
        mtq.disable_quantizer(backbone, model_filter_func)

        self.logger.info("Quantization completed successfully")
        return backbone

    def auto_quantize_model(
        self,
        backbone: torch.nn.Module,
        quant_config: Any,
        forward_loop: callable,  # type: ignore[valid-type]
        sensitivity_out: Path | None = None,
    ) -> torch.nn.Module:
        """Search per-layer formats instead of applying a declared protection set.

        The score phase is the expensive part, so it is cached through the searcher's
        own checkpoint: re-running at a different ``effective_bits`` restores the search
        state and only re-solves, which is what makes several cost points affordable
        from one scoring pass.

        ``method="gradient"`` needs a backward pass over the whole backbone, which does
        not fit alongside 22B of weights on a single 80 GB device; ``kl_div`` is
        forward-only and is therefore the default here.
        """
        self.logger.info(
            "Starting AutoQuantize search: method=%s effective_bits=%.2f",
            self.config.auto_quantize_method,
            self.config.effective_bits,
        )
        check_lora(backbone)

        # auto_quantize drives its own forward passes over a data loader; the
        # calibration loop already encapsulates one full pipeline call per prompt, so a
        # single-item loader per step keeps the two paths consistent.
        def forward_step(mod, batch):
            return forward_loop(mod)

        model, search_state = mtq.auto_quantize(
            backbone,
            constraints={"effective_bits": self.config.effective_bits},
            quantization_formats=[quant_config],
            data_loader=[None],
            forward_step=forward_step,
            method=self.config.auto_quantize_method,
            checkpoint=str(self.config.auto_quantize_checkpoint)
            if self.config.auto_quantize_checkpoint
            else None,
            verbose=True,
        )

        if sensitivity_out is not None:
            sensitivity_out.parent.mkdir(parents=True, exist_ok=True)
            sensitivity_out.write_text(
                json.dumps(
                    {
                        "method": self.config.auto_quantize_method,
                        "effective_bits_requested": self.config.effective_bits,
                        "search_state": _jsonable(search_state),
                    },
                    indent=2,
                )
                + "\n"
            )
            self.logger.info(f"Wrote AutoQuantize sensitivity ranking to {sensitivity_out}")
        return model


class ExportManager:
    """Handles model export operations."""

    def __init__(
        self,
        config: ExportConfig,
        logger: logging.Logger,
        pipeline_manager: PipelineManager | None = None,
    ):
        """
        Initialize export manager.

        Args:
            config: Export configuration
            logger: Logger instance
            pipeline_manager: Pipeline manager for per-backbone IO
        """
        self.config = config
        self.logger = logger
        self.pipeline_manager = pipeline_manager

    def _has_conv_layers(self, model: torch.nn.Module) -> bool:
        """
        Check if the model contains any convolutional layers.

        Args:
            model: Model to check

        Returns:
            True if model contains Conv layers, False otherwise
        """
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)) and (
                module.input_quantizer.is_enabled or module.weight_quantizer.is_enabled
            ):
                return True
        return False

    def save_checkpoint(self, backbone: torch.nn.Module) -> None:
        """
        Save quantized model checkpoint.

        Args:
            backbone: The quantized backbone module to save (must be the same instance
                that was passed to mtq.quantize, as it carries the _modelopt_state).
        """
        if not self.config.quantized_torch_ckpt_path:
            return

        ckpt_path = self.config.quantized_torch_ckpt_path
        ckpt_path.mkdir(parents=True, exist_ok=True)
        target_path = ckpt_path / "backbone.pt"
        self.logger.info(f"Saving backbone to {target_path}")
        mto.save(backbone, str(target_path))

        self.logger.info("Checkpoint saved successfully")

    def export_onnx(
        self,
        pipe: DiffusionPipeline,
        backbone: torch.nn.Module,
        model_type: ModelType,
        quant_format: QuantFormat,
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            pipe: Diffusion pipeline
            backbone: Model backbone
            model_type: Type of model
            quant_format: Quantization format
        """
        if not self.config.onnx_dir:
            return

        self.logger.info(f"Starting ONNX export to {self.config.onnx_dir}")

        if quant_format == QuantFormat.FP8 and self._has_conv_layers(backbone):
            self.logger.info(
                "Detected quantizing conv layers in backbone. Generating FP8 scales..."
            )
            generate_fp8_scales(backbone)
        self.logger.info("Preparing models for export...")
        pipe.to("cpu")
        torch.cuda.empty_cache()
        backbone.to("cuda")
        # Export to ONNX
        backbone.eval()
        with torch.no_grad():
            self.logger.info("Exporting to ONNX...")
            modelopt_export_sd(
                backbone, str(self.config.onnx_dir), model_type.value, quant_format.value
            )

        self.logger.info("ONNX export completed successfully")

    def restore_checkpoint(self) -> None:
        """
        Restore a previously quantized model.

        """
        if not self.config.restore_from:
            return

        restore_path = self.config.restore_from
        if self.pipeline_manager is None:
            raise RuntimeError("Pipeline manager is required for per-backbone checkpoints.")

        backbone = self.pipeline_manager.get_backbone()
        if restore_path.exists() and restore_path.is_dir():
            source_path = restore_path / "backbone.pt"
            if not source_path.exists():
                raise FileNotFoundError(f"Backbone checkpoint not found: {source_path}")
            self.logger.info(f"Restoring backbone from {source_path}")
            mto.restore(backbone, str(source_path))
        self.logger.info("Backbone checkpoints restored successfully")

    # TODO: should not do the any data type
    def export_hf_ckpt(self, pipe: Any) -> None:
        """
        Export quantized model to HuggingFace checkpoint format.

        Args:
            pipe: Diffusion pipeline containing the quantized model
        """
        if not self.config.hf_ckpt_dir:
            return

        self.logger.info(f"Exporting HuggingFace checkpoint to {self.config.hf_ckpt_dir}")
        export_hf_checkpoint(pipe, export_dir=self.config.hf_ckpt_dir)
        self.logger.info("HuggingFace checkpoint export completed successfully")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Diffusion Model Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic INT8 quantization with SmoothQuant
            %(prog)s --model flux-dev --format int8 --quant-algo smoothquant --collect-method global_min

            # FP8 quantization with ONNX export
            %(prog)s --model sd3-medium --format fp8 --onnx-dir ./onnx_models/

            # FP8 quantization with weight compression (reduces memory footprint)
            %(prog)s --model flux-dev --format fp8 --compress

            # Quantize LTX-Video model with full multi-stage pipeline
            %(prog)s --model ltx-video-dev --format fp8 --batch-size 1 --calib-size 32

            # Faster LTX-Video quantization (skip upsampler)
            %(prog)s --model ltx-video-dev --format fp8 --batch-size 1 --calib-size 32 --ltx-skip-upsampler

            # Restore and export a previously quantized model
            %(prog)s --model flux-schnell --restore-from checkpoint.pt --onnx-dir ./exports/
        """,
    )
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=[m.value for m in ModelType],
        help="Model to load and quantize",
    )
    model_group.add_argument(
        "--backbone",
        nargs="+",
        default=None,
        help=(
            "Model backbone(s) in the DiffusionPipeline to work on. "
            "Provide one name or multiple names separated by space or comma. "
            "If not provided use default based on model type."
        ),
    )
    model_group.add_argument(
        "--model-dtype",
        type=str,
        default="Half",
        choices=[d.value for d in DataType],
        help="Precision for loading the pipeline. If you want different dtypes for separate components, "
        "please specify using --component-dtype",
    )
    model_group.add_argument(
        "--component-dtype",
        action="append",
        default=[],
        help="Precision for loading each component of the model by format of name:dtype. "
        "You can specify multiple components. "
        "Example: --component-dtype vae:Half --component-dtype transformer:BFloat16",
    )
    model_group.add_argument(
        "--override-model-path", type=str, help="Custom path to model (overrides default)"
    )
    model_group.add_argument(
        "--cpu-offloading", action="store_true", help="Enable CPU offloading for limited VRAM"
    )
    model_group.add_argument(
        "--ltx-skip-upsampler",
        action="store_true",
        help="Skip upsampler pipeline for LTX-Video (faster calibration, only quantizes main transformer)",
    )
    model_group.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra model-specific parameters in KEY=VALUE form. Can be provided multiple times. "
            "These override model-specific CLI arguments when present."
        ),
    )
    quant_group = parser.add_argument_group("Quantization Configuration")
    quant_group.add_argument(
        "--format",
        type=str,
        default="int8",
        choices=[f.value for f in QuantFormat],
        help="Quantization format",
    )
    quant_group.add_argument(
        "--quant-algo",
        type=str,
        default="max",
        choices=[a.value for a in QuantAlgo],
        help="Quantization algorithm",
    )
    quant_group.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Percentile for calibration, works for INT8, not including smoothquant",
    )
    quant_group.add_argument(
        "--collect-method",
        type=str,
        default="default",
        choices=[c.value for c in CollectMethod],
        help="Calibration collection method, works for INT8, not including smoothquant",
    )
    quant_group.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "SmoothQuant migration strength, required with --quant-algo smoothquant. "
            "Has no default because it decides the experiment: alpha=1.0 moves the whole "
            "activation range into the weights, 0.5 is the paper's balance point."
        ),
    )
    quant_group.add_argument("--lowrank", type=int, default=32, help="SVDQuant lowrank parameter")
    quant_group.add_argument(
        "--quantize-mha", action="store_true", help="Quantizing MHA into FP8 if its True"
    )
    quant_group.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Compress quantized weights to reduce memory footprint. Supported for INT8 as "
            "well as FP8/FP4, but only FP8/FP4 have real-quantized GEMM backends, so an "
            "INT8 forward dequantizes to BF16 and warns."
        ),
    )
    quant_group.add_argument(
        "--int8-numerics",
        type=str,
        default="static",
        choices=[n.value for n in Int8Numerics],
        help=(
            "INT8 activation numerics: 'static' collects a per-tensor amax during "
            "calibration, 'per_token_dynamic' computes a scale per token at runtime"
        ),
    )
    quant_group.add_argument(
        "--ltx-protect-target",
        type=float,
        default=None,
        help=(
            "Solve an LTX-2 protection tier that retains this fraction of the achievable "
            "saving on --ltx-protect-metric (1.0 protects only the base set, 0.0 keeps "
            "everything in high precision). Solved against the live module tree, so the "
            "tier's predicted cost is reported before calibration starts."
        ),
    )
    quant_group.add_argument(
        "--ltx-protect-metric",
        type=str,
        default="vram",
        choices=list(METRICS),
        help=(
            "Cost axis the protection target refers to. 'vram' is weight bytes and is "
            "realizable; 'latency' is a projected GEMM-FLOP saving."
        ),
    )
    quant_group.add_argument(
        "--protect-from-json",
        type=str,
        default=None,
        help=(
            "Path to a JSON list of module names to keep in high precision. Takes "
            "precedence over --ltx-protect-target, and is how a solved tier is replayed "
            "or a sensitivity-derived set is applied."
        ),
    )
    quant_group.add_argument(
        "--protect-out",
        type=str,
        default=None,
        help=(
            "Write the solved protection set and its predicted cost here. The file is "
            "accepted by --protect-from-json, so a solved tier can be replayed exactly."
        ),
    )

    auto_group = parser.add_argument_group("AutoQuantize Configuration")
    auto_group.add_argument(
        "--auto-quantize",
        action="store_true",
        help="Search per-layer quantization formats instead of applying a declared protection set",
    )
    auto_group.add_argument(
        "--effective-bits",
        type=float,
        default=8.4,
        help="AutoQuantize weight-bit budget across the searched modules",
    )
    auto_group.add_argument(
        "--auto-quantize-method",
        type=str,
        default="kl_div",
        choices=["kl_div", "gradient"],
        help=(
            "Sensitivity scoring method. 'kl_div' is forward-only; 'gradient' needs a "
            "backward pass and will not fit a 22B backbone on one 80 GB device."
        ),
    )
    auto_group.add_argument(
        "--auto-quantize-checkpoint",
        type=str,
        default=None,
        help=(
            "Search-state checkpoint. Reused across runs so several effective-bits points "
            "can be re-solved from one scoring pass."
        ),
    )
    auto_group.add_argument(
        "--sensitivity-out",
        type=str,
        default=None,
        help="Path to write the AutoQuantize per-layer sensitivity ranking as JSON",
    )

    cost_group = parser.add_argument_group("Cost Reporting")
    cost_group.add_argument(
        "--cost-report",
        type=str,
        default=None,
        help="Path to write the per-variant quantization cost report as JSON",
    )
    cost_group.add_argument(
        "--cost-inventory",
        type=str,
        default=None,
        help=(
            "Token-annotated linear_inventory.json. Supplies per-module GEMM FLOPs so the "
            "cost report can add its projected latency columns."
        ),
    )
    cost_group.add_argument(
        "--gemm-time-share",
        type=float,
        default=0.75,
        help="Measured share of BF16 clip wall-time spent in linear GEMMs, for the projection",
    )

    calib_group = parser.add_argument_group("Calibration Configuration")
    calib_group.add_argument("--batch-size", type=int, default=2, help="Batch size for calibration")
    calib_group.add_argument(
        "--calib-size", type=int, default=128, help="Total number of calibration samples"
    )
    calib_group.add_argument("--n-steps", type=int, default=30, help="Number of denoising steps")
    calib_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Calibrate using prompts in the file instead of the default dataset.",
    )
    calib_group.add_argument(
        "--calib-coverage-out",
        type=str,
        default=None,
        help=(
            "Write the per-quantizer calibration coverage audit here. The audit itself "
            "always runs after calibration; this only keeps its record next to the run."
        ),
    )
    calib_group.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help=(
            "Continue when the audit finds enabled quantizers that never collected an "
            "amax. Off by default: such a layer is quantized against its initialization "
            "scale, and the saved checkpoint does not record that it was."
        ),
    )

    export_group = parser.add_argument_group("Export Configuration")
    export_group.add_argument(
        "--quantized-torch-ckpt-save-path",
        type=str,
        help="Path to save quantized PyTorch checkpoint",
    )
    export_group.add_argument("--onnx-dir", type=str, help="Directory for ONNX export")
    export_group.add_argument(
        "--hf-ckpt-dir",
        type=str,
        help="Directory for HuggingFace checkpoint export",
    )
    export_group.add_argument(
        "--restore-from", type=str, help="Path to restore from previous checkpoint"
    )
    export_group.add_argument(
        "--trt-high-precision-dtype",
        type=str,
        default="Half",
        choices=[d.value for d in DataType],
        help="Precision for TensorRT high-precision layers",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def main() -> None:
    from diffusers.models.normalization import RMSNorm as DiffuserRMSNorm

    torch.nn.RMSNorm = DiffuserRMSNorm
    torch.nn.modules.normalization.RMSNorm = DiffuserRMSNorm

    parser = create_argument_parser()
    args, unknown_args = parser.parse_known_args()

    model_type = ModelType(args.model)
    if args.backbone is None:
        args.backbone = [MODEL_DEFAULTS[model_type]["backbone"]]
    s = time.time()

    model_dtype = {"default": DataType(args.model_dtype).torch_dtype}
    for component_dtype in args.component_dtype:
        component, dtype = component_dtype.split(":")
        model_dtype[component] = DataType(dtype).torch_dtype

    logger = setup_logging(args.verbose)
    logger.info("Starting Enhanced Diffusion Model Quantization")

    try:
        extra_params = parse_extra_params(args.extra_param, unknown_args, logger)
        model_config = ModelConfig(
            model_type=model_type,
            model_dtype=model_dtype,
            backbone=args.backbone,
            trt_high_precision_dtype=DataType(args.trt_high_precision_dtype),
            override_model_path=Path(args.override_model_path)
            if args.override_model_path
            else None,
            cpu_offloading=args.cpu_offloading,
            ltx_skip_upsampler=args.ltx_skip_upsampler,
            extra_params=extra_params,
        )

        quant_config = QuantizationConfig(
            format=QuantFormat(args.format),
            algo=QuantAlgo(args.quant_algo),
            percentile=args.percentile,
            collect_method=CollectMethod(args.collect_method),
            alpha=args.alpha,
            lowrank=args.lowrank,
            quantize_mha=args.quantize_mha,
            compress=args.compress,
            int8_numerics=Int8Numerics(args.int8_numerics),
            ltx_protect_target=args.ltx_protect_target,
            ltx_protect_metric=args.ltx_protect_metric,
            protect_from_json=Path(args.protect_from_json) if args.protect_from_json else None,
            auto_quantize=args.auto_quantize,
            effective_bits=args.effective_bits,
            auto_quantize_method=args.auto_quantize_method,
            auto_quantize_checkpoint=Path(args.auto_quantize_checkpoint)
            if args.auto_quantize_checkpoint
            else None,
        )

        if args.prompts_file is not None:
            prompts_file = Path(args.prompts_file)
            assert prompts_file.exists(), (
                f"User specified prompts file {prompts_file} does not exist."
            )
            prompts_dataset = prompts_file
        else:
            prompts_dataset = MODEL_DEFAULTS[model_type]["dataset"]
        calib_config = CalibrationConfig(
            prompts_dataset=prompts_dataset,
            batch_size=args.batch_size,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
        )

        export_config = ExportConfig(
            quantized_torch_ckpt_path=Path(args.quantized_torch_ckpt_save_path)
            if args.quantized_torch_ckpt_save_path
            else None,
            onnx_dir=Path(args.onnx_dir) if args.onnx_dir else None,
            hf_ckpt_dir=Path(args.hf_ckpt_dir) if args.hf_ckpt_dir else None,
            restore_from=Path(args.restore_from) if args.restore_from else None,
        )

        logger.info("Validating configurations...")
        quant_config.validate()
        export_config.validate()
        if not export_config.restore_from:
            calib_config.validate()

        pipeline_manager = PipelineManager(model_config, logger)
        pipe = pipeline_manager.create_pipeline()
        pipeline_manager.setup_device()

        backbone = pipeline_manager.get_backbone()
        export_manager = ExportManager(export_config, logger, pipeline_manager)
        protect_report: dict[str, Any] | None = None

        if export_config.restore_from and export_config.restore_from.exists():
            export_manager.restore_checkpoint()

        else:
            logger.info("Initializing calibration...")
            calibrator = Calibrator(pipeline_manager, calib_config, model_config.model_type, logger)
            batched_prompts = calibrator.load_and_batch_prompts()

            quantizer = Quantizer(quant_config, model_config, logger)
            # Solved before calibration: the tier is a property of the model and the clip
            # geometry, not of the calibration data, and a target that lands somewhere
            # unintended should fail before a calibration pass is spent on it.
            protect_report = quantizer.solve_protection_tier(backbone, model_config.extra_params)
            if protect_report is not None and args.protect_out:
                Path(args.protect_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.protect_out).write_text(
                    json.dumps({**protect_report, "protect": quantizer.protect_names}, indent=2)
                    + "\n"
                )
                logger.info(f"Wrote the solved protection set to {args.protect_out}")

            backbone_quant_config = quantizer.get_quant_config(calib_config.n_steps, backbone)

            # Pipe loads the ckpt just before the inference.
            def forward_loop(mod):
                calibrator.run_calibration(batched_prompts)

            if quant_config.auto_quantize:
                quantizer.auto_quantize_model(
                    backbone,
                    backbone_quant_config,
                    forward_loop,
                    sensitivity_out=Path(args.sensitivity_out) if args.sensitivity_out else None,
                )
            else:
                quantizer.quantize_model(backbone, backbone_quant_config, forward_loop)

            # Before compression, which folds the scales in and makes a missing amax
            # indistinguishable from a collected one.
            check_calibration_coverage(
                backbone,
                out_path=Path(args.calib_coverage_out) if args.calib_coverage_out else None,
                logger=logger,
                fail_on_uncalibrated=not args.allow_uncalibrated,
            )

            if quant_config.compress:
                logger.info("Compressing model weights to reduce memory footprint...")
                mtq.compress(backbone)
                logger.info("Model compression completed")

            export_manager.save_checkpoint(backbone)

        # TODO (Jingyu): To update this function, as we are focusing more on the torch deployment side.
        check_conv_and_mha(
            backbone, quant_config.format == QuantFormat.FP4, quant_config.quantize_mha
        )

        pipeline_manager.print_quant_summary()

        if args.cost_report:
            write_quant_cost_report(
                backbone,
                Path(args.cost_report),
                logger=logger,
                inventory_path=Path(args.cost_inventory) if args.cost_inventory else None,
                gemm_time_share=args.gemm_time_share,
                meta={
                    "model": model_type.value,
                    "format": quant_config.format.value,
                    "algo": quant_config.algo.value,
                    "int8_numerics": quant_config.int8_numerics.value,
                    "alpha": quant_config.alpha,
                    "protect_target": quant_config.ltx_protect_target,
                    "protect_metric": quant_config.ltx_protect_metric
                    if quant_config.ltx_protect_target is not None
                    else None,
                    # The predicted cost of the solved tier, kept next to the measured cost
                    # so the two can be compared without joining files.
                    "protect_predicted": protect_report,
                    "protect_from_json": str(quant_config.protect_from_json)
                    if quant_config.protect_from_json
                    else None,
                    "auto_quantize": quant_config.auto_quantize,
                    "effective_bits_requested": quant_config.effective_bits
                    if quant_config.auto_quantize
                    else None,
                    "compressed": quant_config.compress,
                    "calib_size": calib_config.calib_size,
                    "n_steps": calib_config.n_steps,
                },
            )

        export_manager.export_onnx(
            pipe,
            backbone,
            model_config.model_type,
            quant_config.format,
        )

        export_manager.export_hf_ckpt(pipe)

        logger.info(
            f"Quantization process completed successfully! Time taken = {time.time() - s} seconds"
        )

    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
