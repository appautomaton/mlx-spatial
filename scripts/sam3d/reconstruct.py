#!/usr/bin/env python3
"""Run SAM 3D Objects image+mask reconstruction with MLX defaults.

Model family:
    SAM 3D Objects, using the converted mlx-spatial safetensors bundle.

Input:
    A single RGB/RGBA image plus a binary object mask.

Output:
    A Gaussian PLY and trace JSON. With --output-dir, the defaults are
    gaussians.ply and trace.json inside that directory.

Recommended settings:
    Default root weights/sam-3d-objects-mlx, bundled MoGe root
    weights/sam-3d-objects-mlx/moge, seed 42, and memory-profile balanced.
    Use safe only for lower-memory debug runs; large is for capacity probes.

Example:
    python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \\
      --mask inputs/sam3d/living-room/mask-3.png \\
      --output-dir outputs/sam3d/living-room-script
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

SAM3D_DEFAULT_ROOT = "weights/sam-3d-objects-mlx"
SAM3D_DEFAULT_MOGE_ROOT = "weights/sam-3d-objects-mlx/moge"
SAM3D_RECOMMENDED_MEMORY_PROFILE = "balanced"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input RGB/RGBA image containing one target object")
    parser.add_argument("--mask", required=True, type=Path, help="binary object mask aligned to the input image")
    parser.add_argument(
        "--root",
        default=SAM3D_DEFAULT_ROOT,
        help="converted SAM3D safetensors root; default: %(default)s",
    )
    parser.add_argument(
        "--moge-root",
        default=SAM3D_DEFAULT_MOGE_ROOT,
        help="bundled converted MoGe safetensors root; default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for gaussians.ply and trace.json; default: outputs/sam3d/<image-stem>",
    )
    parser.add_argument("--output", type=Path, help="explicit Gaussian PLY path under outputs/")
    parser.add_argument("--trace-output", type=Path, help="explicit trace JSON path under outputs/")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed; default: %(default)s")
    parser.add_argument(
        "--memory-profile",
        choices=("safe", "balanced", "large"),
        default=SAM3D_RECOMMENDED_MEMORY_PROFILE,
        help="balanced is the recommended profile; safe is lower-memory debug, large is a capacity probe",
    )
    args = parser.parse_args(argv)

    from mlx_spatial.sam3d import main as sam3d_main

    output, trace = _resolve_outputs(args.image, args.output_dir, args.output, args.trace_output)
    cli_args = [
        "reconstruct",
        str(args.root),
        str(args.image),
        "--mask",
        str(args.mask),
        "--moge-root",
        str(args.moge_root),
        "--output",
        str(output),
        "--trace-output",
        str(trace),
        "--seed",
        str(args.seed),
        "--memory-profile",
        args.memory_profile,
    ]
    return sam3d_main(cli_args)


def _resolve_outputs(
    image: Path,
    output_dir: Path | None,
    output: Path | None,
    trace_output: Path | None,
) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = Path("outputs/sam3d") / _slug(image.stem)
    if output is None:
        output = output_dir / "gaussians.ply"
    if trace_output is None:
        trace_output = output.with_name("trace.json")
    return output, trace_output


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "sam3d-object"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
