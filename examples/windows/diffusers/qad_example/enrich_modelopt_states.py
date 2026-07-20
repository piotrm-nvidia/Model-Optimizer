#!/usr/bin/env python3
"""Re-save legacy QAD ModelOpt states with restore inventory metadata.

Run on EOS compute (torch required). Uses corrected PTQ A0 inventory as the
enable/config template and overlays amax digests from each QAD state file.
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
        help="Legacy QAD modelopt_state_*.pth files to enrich in place",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without rewriting files",
    )
    args = parser.parse_args(argv)

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

    for path in args.states:
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
        print(f"{path}: before={before} after={after}")
        if args.dry_run:
            continue
        tmp = path.with_suffix(".pth.tmp")
        torch.save(state, tmp)
        tmp.replace(path)
        print(f"  rewrote bytes={path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
