#!/usr/bin/env python3
"""Run Apple LiTo image-to-3DGS generation with MLX defaults.

Model family:
    Apple LiTo checkpoint-backed image-to-3DGS through mlx-spatial.

Input:
    A single object-centric RGB/RGBA image.

Output:
    A 3D Gaussian Splat PLY at --output when --format ply is used.

Recommended settings:
    Default root weights/lito-research-mlx, format ply, binary_little_endian
    PLY storage, memory-profile balanced, 20 sampling steps, CFG scale 3.0,
    518px square preprocessing, and the runtime unseeded policy.

Smoke/debug:
    Use --memory-profile safe with --source-contract-smoke only for synthetic
    framework probes. That path does not produce checkpoint-backed LiTo output.

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

# Keep these help-safe defaults aligned with mlx_spatial.lito_inference.
# The runtime module imports MLX, so --help must not import it before argparse exits.
LITO_DEFAULT_ROOT = "weights/lito-research-mlx"
LITO_DEFAULT_PLY_STORAGE = "binary_little_endian"
LITO_PLY_STORAGES = ("binary_little_endian", "ascii")
LITO_MEMORY_PROFILES = ("safe", "balanced", "large")
LITO_DEFAULT_MEMORY_PROFILE = "balanced"
LITO_RECOMMENDED_NUM_STEPS = 20
LITO_RECOMMENDED_CFG_SCALE = 3.0
LITO_RECOMMENDED_RESOLUTION = (518, 518)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input RGB/RGBA image")
    parser.add_argument(
        "--root",
        "--weights-root",
        dest="root",
        default=LITO_DEFAULT_ROOT,
        help="converted LiTo safetensors root; default: %(default)s",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output 3DGS file, usually outputs/lito/<name>.ply",
    )
    parser.add_argument(
        "--format",
        choices=("ply", "splat", "safetensors"),
        default="ply",
        help="output artifact format; ply is the checkpoint-backed viewer default, splat is smoke-only",
    )
    parser.add_argument(
        "--ply-storage",
        default=LITO_DEFAULT_PLY_STORAGE,
        metavar="{binary_little_endian|ascii}",
        type=_normalize_lito_ply_storage,
        help="checkpoint-backed PLY storage; binary_little_endian is recommended, ascii is for text diffs",
    )
    parser.add_argument(
        "--memory-profile",
        choices=LITO_MEMORY_PROFILES,
        default=LITO_DEFAULT_MEMORY_PROFILE,
        help="balanced is recommended; safe is lower-memory smoke/debug, large is a capacity probe",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=LITO_RECOMMENDED_NUM_STEPS,
        help="sampling steps; default %(default)s follows the upstream-recorded recommendation",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=LITO_RECOMMENDED_CFG_SCALE,
        help="classifier-free guidance scale; default %(default)s is recommended",
    )
    parser.add_argument("--seed", type=int, default=None, help="optional seed; omit for the runtime unseeded policy")
    parser.add_argument(
        "--resolution",
        type=int,
        default=LITO_RECOMMENDED_RESOLUTION[0],
        help="square preprocessing resolution; default %(default)s matches the LiTo recommendation",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        help="source-contract smoke render-size override; ignored by checkpoint-backed PLY export",
    )
    parser.add_argument("--print-metrics", action="store_true", help="print per-stage timing and memory metrics")
    parser.add_argument(
        "--source-contract-smoke",
        action="store_true",
        help="run synthetic source-contract smoke instead of checkpoint-backed LiTo generation",
    )
    args = parser.parse_args(argv)

    from mlx_spatial.lito import main as lito_main

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


def _normalize_lito_ply_storage(ply_storage: str) -> str:
    normalized = str(ply_storage).strip().lower().replace("-", "_")
    if normalized == "binary":
        normalized = "binary_little_endian"
    if normalized not in LITO_PLY_STORAGES:
        allowed = ", ".join(LITO_PLY_STORAGES)
        raise argparse.ArgumentTypeError(f"expected one of: {allowed}")
    return normalized


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
