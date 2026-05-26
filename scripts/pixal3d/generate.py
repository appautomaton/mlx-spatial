#!/usr/bin/env python3
"""Run Pixal3D image-to-3D generation with MLX defaults.

Model family:
    TencentARC Pixal3D main-branch image-to-3D generation.

Input:
    A single object-centric RGB/RGBA image.
    For a local sample from the vendored reference repo, use:
        vendors/Pixal3D/assets/images/0_img.png

Output:
    A textured GLB once the MLX model/export path is implemented. Until then,
    the script writes trace.json and, when an MLX intermediate boundary has
    completed, writes the corresponding NPZ artifact next to the trace.

Recommended settings:
    Default root weights/pixal3d, pipeline-type 1024_cascade for Apple Silicon,
    seed 42, max-num-tokens 49152, and manual FOV when avoiding MoGe auto-camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from mlx_spatial.pixal3d import main as pixal3d_main  # noqa: E402
from mlx_spatial.pixal3d_inference import (  # noqa: E402
    PIXAL3D_DEFAULT_MAX_NUM_TOKENS,
    PIXAL3D_DEFAULT_SEED,
    PIXAL3D_PIPELINE_TYPES,
    PIXAL3D_RECOMMENDED_PIPELINE_TYPE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input RGB/RGBA image")
    parser.add_argument("--root", default="weights/pixal3d", help="Pixal3D safetensors root; default: %(default)s")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for model.glb and trace.json; default: outputs/pixal3d/<image-stem>",
    )
    parser.add_argument("--output", type=Path, help="explicit GLB output path")
    parser.add_argument(
        "--pipeline-type",
        choices=PIXAL3D_PIPELINE_TYPES,
        default=PIXAL3D_RECOMMENDED_PIPELINE_TYPE,
        help="1024_cascade is recommended for Apple Silicon; 1536_cascade is explicit high-memory mode",
    )
    parser.add_argument("--manual-fov", type=float, help="manual horizontal FOV in radians, e.g. 0.2")
    parser.add_argument("--seed", type=int, default=PIXAL3D_DEFAULT_SEED, help="sampling seed; default: %(default)s")
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=PIXAL3D_DEFAULT_MAX_NUM_TOKENS,
        help="Pixal3D sparse token guard; default: %(default)s",
    )
    parser.add_argument("--trace-output", type=Path, help="trace JSON path; default: next to output")
    args = parser.parse_args(argv)

    cli_args = [
        "generate",
        str(args.image),
        "--root",
        str(args.root),
        "--pipeline-type",
        args.pipeline_type,
        "--seed",
        str(args.seed),
        "--max-num-tokens",
        str(args.max_num_tokens),
    ]
    if args.output_dir is not None:
        cli_args.extend(["--output-dir", str(args.output_dir)])
    if args.output is not None:
        cli_args.extend(["--output", str(args.output)])
    if args.manual_fov is not None:
        cli_args.extend(["--manual-fov", str(args.manual_fov)])
    if args.trace_output is not None:
        cli_args.extend(["--trace-output", str(args.trace_output)])
    return pixal3d_main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
