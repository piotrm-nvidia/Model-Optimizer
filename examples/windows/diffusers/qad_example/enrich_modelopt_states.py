#!/usr/bin/env python3
"""Re-save legacy QAD ModelOpt states with restore inventory metadata.

Run on EOS compute (torch required). Uses corrected PTQ A0 inventory as the
enable/config template and overlays amax digests from each QAD state file.

By default writes enriched copies into ``--output-dir`` and leaves the input
legacy checkpoints untouched. Pass ``--in-place`` only if an overwrite is
explicitly required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Corrected PTQ state with quantizer_inventory (typically A0)",
    )
    parser.add_argument(
        "states",
        nargs="+",
        type=Path,
        help="Legacy QAD modelopt_state_*.pth files to enrich",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for enriched copies (required unless --in-place)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input paths (disabled by default to preserve legacy files)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing files",
    )
    args = parser.parse_args(argv)

    if args.in_place and args.output_dir is not None:
        raise SystemExit("pass either --output-dir or --in-place, not both")
    if not args.in_place and args.output_dir is None:
        raise SystemExit(
            "refusing to overwrite inputs: pass --output-dir <new_dir> "
            "(or --in-place only if overwrite is intentional)"
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        print("torch required; run on EOS compute container", file=sys.stderr)
        raise SystemExit(2) from exc

    # Import helpers from this example package directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sample_example_qad_diffusers import enrich_modelopt_state_with_template

    template = torch.load(args.template, map_location="cpu", weights_only=False)
    inventory = template.get("quantizer_inventory")
    if not inventory:
        raise SystemExit(f"template missing quantizer_inventory: {args.template}")

    if args.output_dir is not None and not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in args.states:
        if not path.is_file():
            raise FileNotFoundError(path)
        out_path = path if args.in_place else args.output_dir / path.name
        if not args.in_place and out_path.resolve() == path.resolve():
            raise SystemExit(
                f"output path equals input ({out_path}); refusing silent overwrite"
            )

        state = torch.load(path, map_location="cpu", weights_only=False)
        before = {
            "has_keys": "quantizer_state_keys" in state,
            "has_inventory": "quantizer_inventory" in state,
            "bytes": path.stat().st_size,
            "weight_n": len(state.get("modelopt_state_weights") or {}),
        }
        enrich_modelopt_state_with_template(state, inventory)
        after = {
            "has_keys": "quantizer_state_keys" in state,
            "keys_n": len(state["quantizer_state_keys"]),
            "inventory_n": len(state["quantizer_inventory"]),
            "digest": state["quantizer_inventory_digest"],
        }
        print(f"{path} -> {out_path}: before={before} after={after}")
        if args.dry_run:
            continue
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(out_path)
        print(f"  wrote bytes={out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
