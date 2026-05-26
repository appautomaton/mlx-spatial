#!/usr/bin/env python3
"""Run Meta MapAnything multi-view scene generation with MLX defaults.

Model family:
    Meta MapAnything image-only multi-view scene geometry through mlx-spatial.

Input:
    A single image or a directory of scene-view images. The standard use case is
    two or more related views of the same scene.

Output:
    A scene `.npz` bundle with images, depth, confidence, masks, intrinsics,
    camera poses, extrinsics, world points, and a `trace.json` next to it.

Recommended settings:
    Default root weights/map-anything, upstream fixed_mapping preprocessing,
    stride 1, DINOv2 normalization, config-derived patch size, and
    postprocessing with masks and edge masks.

Example:
    python scripts/mapanything/generate_scene.py inputs/map-anything/desk \\
      --output-dir outputs/mapanything/desk-script
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

MAPANYTHING_DEFAULT_ROOT = "weights/map-anything"
MAPANYTHING_RECOMMENDED_RESIZE_MODE = "fixed_mapping"
MAPANYTHING_RECOMMENDED_STRIDE = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="input image file or directory of scene views")
    parser.add_argument(
        "--root",
        "--weights-root",
        dest="root",
        default=MAPANYTHING_DEFAULT_ROOT,
        help="MapAnything HF checkpoint root; default: %(default)s",
    )
    parser.add_argument("--output", type=Path, help="output .npz scene bundle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for scene.npz and trace.json; default: outputs/mapanything/<input-stem>",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=MAPANYTHING_RECOMMENDED_STRIDE,
        help="input frame stride; default 1 keeps every image",
    )
    parser.add_argument("--trace-output", type=Path, help="trace JSON path; default: next to scene.npz")
    args = parser.parse_args(argv)

    if args.output is not None and args.output_dir is not None:
        parser.error("use either --output or --output-dir, not both")
    output_path = _resolve_scene_output(args.input, output=args.output, output_dir=args.output_dir)
    trace_path = args.trace_output or output_path.with_name("trace.json")
    _print_effective_settings(args, output_path, trace_path)

    from mlx_spatial.mapanything import main as mapanything_main

    cli_args = [
        "generate",
        str(args.input),
        "--root",
        str(args.root),
        "--output",
        str(output_path),
        "--resize-mode",
        MAPANYTHING_RECOMMENDED_RESIZE_MODE,
        "--stride",
        str(args.stride),
        "--trace-output",
        str(trace_path),
    ]
    return mapanything_main(cli_args)


def _resolve_scene_output(input_path: Path, *, output: Path | None, output_dir: Path | None) -> Path:
    if output is not None:
        if output.suffix != ".npz":
            raise SystemExit("MapAnything scene output must be a .npz file")
        return output
    directory = output_dir or Path("outputs/mapanything") / _slug(input_path.stem or input_path.name)
    return directory / "scene.npz"


def _print_effective_settings(args: argparse.Namespace, output_path: Path, trace_path: Path) -> None:
    is_production_like = args.stride == MAPANYTHING_RECOMMENDED_STRIDE
    print(f"profile={'production-like' if is_production_like else 'custom'}", flush=True)
    print(
        "settings="
        f"resize_mode={MAPANYTHING_RECOMMENDED_RESIZE_MODE} "
        f"stride={args.stride} "
        "patch_size=checkpoint_config "
        "postprocess=apply_mask,mask_edges "
        "artifact=npz "
        "runtime_depends_on_torch=false",
        flush=True,
    )
    print(f"root={args.root}", flush=True)
    print(f"input={args.input}", flush=True)
    print(f"output={output_path}", flush=True)
    print(f"trace={trace_path}", flush=True)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "mapanything-scene"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
