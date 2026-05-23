#!/usr/bin/env python3
"""Run HY-WorldMirror 2.0 scene reconstruction with verified MLX defaults.

Input:
    A single RGB/RGBA image, or a directory of image frames. HY-WorldMirror is a
    scene/world reconstruction pipeline, so it does not take object masks.

Production defaults:
    real Tencent WorldMirror safetensors, large memory profile, official 952px
    target size, and the verified camera/depth/normal/points heads.

    The optional GS head is intentionally not exposed here; use the package CLI
    directly for development probes until that export path is production-ready.

Example:
    python scripts/hyworld2/generate_scene.py inputs/sam3d/kidsroom/image.png \
      --output-dir outputs/hyworld2/kidsroom-scene-script
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

PRODUCTION_HEADS = "camera,depth,normal,points"
PRODUCTION_MEMORY_PROFILE = "large"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="input scene image or directory of image frames")
    parser.add_argument("--root", default="weights/hy-world-2", help="HY-WorldMirror 2.0 safetensors root")
    parser.add_argument("--output-dir", type=Path, help="directory for camera/depth/normal/points outputs")
    parser.add_argument(
        "--memory-profile",
        choices=("safe", "balanced", "large"),
        default=PRODUCTION_MEMORY_PROFILE,
        help="large matches the official 952px target size used by the verified run",
    )
    parser.add_argument(
        "--heads",
        default=PRODUCTION_HEADS,
        help="comma-separated heads; default is the verified release path. Do not include gs for normal runs.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or Path("outputs/hyworld2") / _slug(args.input.stem or args.input.name)
    _print_effective_settings(args, output_dir)

    from mlx_spatial.hyworld2 import main as hyworld2_main

    cli_args = [
        "reconstruct",
        str(args.root),
        str(args.input),
        "--output",
        str(output_dir),
        "--heads",
        args.heads,
        "--memory-profile",
        args.memory_profile,
    ]
    return hyworld2_main(cli_args)


def _print_effective_settings(args: argparse.Namespace, output_dir: Path) -> None:
    is_production_like = args.memory_profile == PRODUCTION_MEMORY_PROFILE and args.heads == PRODUCTION_HEADS
    print(f"profile={'production-like' if is_production_like else 'custom'}", flush=True)
    print(
        "settings="
        f"memory_profile={args.memory_profile} "
        f"heads={args.heads} "
        "fixture_tensors=false "
        "input_kind=image_or_frame_directory",
        flush=True,
    )
    print(f"root={args.root}", flush=True)
    print(f"output_dir={output_dir}", flush=True)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "hyworld2-scene"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
