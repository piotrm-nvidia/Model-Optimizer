from __future__ import annotations

import argparse
from pathlib import Path

from .attribution import run_tensor_attribution
from .runtime import (
    create_deploy,
    evaluate_bundle,
    export_parity,
    inference_matrix,
    load_config,
    preprocess,
    run_ptq,
    run_qad,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ModelOpt LTX QAD example workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = commands.add_parser("preprocess")
    preprocess_parser.add_argument("--config", type=Path, required=True)
    preprocess_parser.add_argument("--manifest", type=Path, required=True)
    preprocess_parser.add_argument("--output", type=Path, required=True)
    preprocess_parser.add_argument("--seed", type=int, default=180100)

    ptq_parser = commands.add_parser("ptq")
    ptq_parser.add_argument("--quant-recipe", choices=("nvfp4", "fp8"), default="nvfp4")
    ptq_parser.add_argument("--config", type=Path, required=True)
    ptq_parser.add_argument("--calibration-manifest", type=Path, required=True)
    ptq_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_commands = evaluate_parser.add_subparsers(dest="evaluation", required=True)
    step0_parser = evaluate_commands.add_parser("step0")
    step0_parser.add_argument("--checkpoint", type=Path, required=True)
    step0_parser.add_argument("--config", type=Path, required=True)
    step0_parser.add_argument("--manifest", type=Path, required=True)
    step0_parser.add_argument("--seeds", nargs="+", type=int, required=True)
    step0_parser.add_argument("--backend", choices=("native", "fake"), default="native")
    step0_parser.add_argument("--output", type=Path, required=True)

    qad_parser = commands.add_parser("qad")
    qad_parser.add_argument("--quant-recipe", choices=("nvfp4", "fp8"), default="nvfp4")
    qad_parser.add_argument("--config", type=Path, required=True)
    qad_parser.add_argument("--init", type=Path, required=True)
    qad_parser.add_argument("--output", type=Path, required=True)

    deploy_parser = commands.add_parser("create-fp8-deploy")
    deploy_parser.add_argument("--checkpoint", type=Path, required=True)
    deploy_parser.add_argument("--config", type=Path, required=True)
    deploy_parser.add_argument("--output", type=Path, required=True)

    parity_parser = commands.add_parser("export-parity")
    parity_parser.add_argument("--source", type=Path, required=True)
    parity_parser.add_argument("--deploy", type=Path, required=True)
    parity_parser.add_argument("--manifest", type=Path, required=True)
    parity_parser.add_argument("--output", type=Path, required=True)

    matrix_parser = commands.add_parser("inference-matrix")
    matrix_parser.add_argument("--matrix", type=Path, required=True)
    matrix_parser.add_argument("--config", type=Path, required=True)
    matrix_parser.add_argument("--heldout-manifest", type=Path, required=True)
    matrix_parser.add_argument("--baseline-root", type=Path, required=True)
    matrix_parser.add_argument("--ptq", type=Path, required=True)
    matrix_parser.add_argument("--qad-root", type=Path, required=True)
    matrix_parser.add_argument("--output", type=Path, required=True)
    matrix_parser.add_argument("--resume", action="store_true")

    attribution_parser = commands.add_parser("tensor-attribution")
    attribution_parser.add_argument("--checkpoint", type=Path, required=True)
    attribution_parser.add_argument("--config", type=Path, required=True)
    attribution_parser.add_argument("--manifest", type=Path, required=True)
    attribution_parser.add_argument("--output", type=Path, required=True)
    attribution_parser.add_argument("--sample-id", required=True)
    attribution_parser.add_argument("--seed", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preprocess":
        preprocess(
            config_path=args.config,
            manifest_path=args.manifest,
            output=args.output,
            backend=args.backend,
        )
    elif args.command == "ptq":
        if not args.calibration_manifest.is_file():
            raise FileNotFoundError(args.calibration_manifest)
        _, qad = load_config(args.config)
        run_ptq(
            config_path=args.config,
            output=args.output,
            quant_recipe=args.quant_recipe,
            calibration_batches=int(qad.get("calib_size", 8)),
        )
    elif args.command == "evaluate":
        if args.evaluation != "step0":
            raise ValueError(args.evaluation)
        if args.seeds != [180100, 180101]:
            raise ValueError("Step-0 evaluation requires seeds 180100 180101")
        evaluate_bundle(
            bundle=args.checkpoint,
            config_path=args.config,
            manifest_path=args.manifest,
            output=args.output,
        )
    elif args.command == "qad":
        run_qad(
            config_path=args.config,
            output=args.output,
            quant_recipe=args.quant_recipe,
            initial_checkpoint=args.init,
        )
    elif args.command == "create-fp8-deploy":
        create_deploy(
            checkpoint=args.checkpoint,
            output=args.output,
            config_path=args.config,
            quant_recipe="fp8",
        )
    elif args.command == "export-parity":
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        export_parity(source=args.source, deploy=args.deploy, output=args.output)
    elif args.command == "inference-matrix":
        if not args.baseline_root.is_dir():
            raise FileNotFoundError(args.baseline_root)
        inference_matrix(
            matrix_path=args.matrix,
            config_path=args.config,
            manifest_path=args.heldout_manifest,
            ptq_bundle=args.ptq,
            qad_root=args.qad_root,
            output=args.output,
            resume=args.resume,
        )
    elif args.command == "tensor-attribution":
        run_tensor_attribution(
            bundle=args.checkpoint,
            config_path=args.config,
            manifest_path=args.manifest,
            output=args.output,
            sample_id=args.sample_id,
            seed=args.seed,
        )
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
