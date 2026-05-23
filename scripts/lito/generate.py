#!/usr/bin/env python3
"""Run Apple LiTo image-to-3DGS generation with mlx-spatial recommended defaults.

Example:
    python scripts/lito/generate.py inputs/lito/sample.png \\
      --output outputs/lito/sample.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    from mlx_spatial.lito import main as lito_main
    from mlx_spatial.lito_inference import (
        LITO_DEFAULT_PLY_STORAGE,
        LITO_DEFAULT_MEMORY_PROFILE,
        LITO_MEMORY_PROFILES,
        LITO_RECOMMENDED_CFG_SCALE,
        LITO_RECOMMENDED_NUM_STEPS,
        LITO_RECOMMENDED_RESOLUTION,
        normalize_lito_ply_storage,
    )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input RGB/RGBA image")
    parser.add_argument(
        "--root",
        "--weights-root",
        dest="root",
        default="weights/lito-mlx",
        help="converted LiTo safetensors root",
    )
    parser.add_argument("--output", type=Path, required=True, help="output 3DGS file")
    parser.add_argument("--format", choices=("ply", "splat", "safetensors"), default="ply")
    parser.add_argument(
        "--ply-storage",
        default=LITO_DEFAULT_PLY_STORAGE,
        metavar="{binary_little_endian|ascii}",
        type=normalize_lito_ply_storage,
        help="checkpoint-backed PLY storage",
    )
    parser.add_argument("--memory-profile", choices=LITO_MEMORY_PROFILES, default=LITO_DEFAULT_MEMORY_PROFILE)
    parser.add_argument("--num-steps", type=int, default=LITO_RECOMMENDED_NUM_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=LITO_RECOMMENDED_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=LITO_RECOMMENDED_RESOLUTION[0])
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--print-metrics", action="store_true")
    parser.add_argument(
        "--source-contract-smoke",
        action="store_true",
        help="run synthetic source-contract smoke instead of checkpoint-backed LiTo generation",
    )
    args = parser.parse_args(argv)

    cli_args = [
        "generate",
        str(args.image),
        "--root",
        str(args.root),
        "--output",
        str(args.output),
        "--format",
        args.format,
        "--ply-storage",
        args.ply_storage,
        "--memory-profile",
        args.memory_profile,
        "--num-steps",
        str(args.num_steps),
        "--cfg-scale",
        str(args.cfg_scale),
        "--resolution",
        str(args.resolution),
    ]
    if args.seed is not None:
        cli_args.extend(["--seed", str(args.seed)])
    if args.render_size is not None:
        cli_args.extend(["--render-size", str(args.render_size)])
    if args.print_metrics:
        cli_args.append("--print-metrics")
    if args.source_contract_smoke:
        cli_args.append("--source-contract-smoke")
    return lito_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
