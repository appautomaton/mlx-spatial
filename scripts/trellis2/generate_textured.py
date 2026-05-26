#!/usr/bin/env python3
"""Run TRELLIS.2 image-to-textured-GLB generation with production-like defaults.

Input:
    A single RGB/RGBA image. RGBA images use their alpha channel directly.
    RGB images use the configured RMBG root to generate foreground alpha.

Production defaults:
    pipeline type 512, model-config SLat sampler steps, 1024 texture,
    200k GLB face target, global xatlas unwrap, and kdtree texture baking.

    Do not pass --slat-steps for quality runs. Use --slat-steps 1 only for
    smoke tests where broken-looking geometry is acceptable.

Example:
    python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \\
      --output-dir outputs/trellis2/cup-of-tea-script
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

PRODUCTION_PIPELINE_TYPE = "512"
PRODUCTION_TEXTURE_SIZE = 1024
PRODUCTION_GLB_TARGET_FACES = 200_000
PRODUCTION_XATLAS_PARALLEL_CHUNKS = 1
PRODUCTION_TEXTURE_BAKE_BACKEND = "kdtree"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input RGB/RGBA image")
    parser.add_argument("--root", default="weights/trellis2", help="TRELLIS.2 safetensors root; default: %(default)s")
    parser.add_argument(
        "--rmbg-root",
        default="weights/rmbg2",
        help="RMBG-2.0 safetensors root for RGB images; default: %(default)s",
    )
    parser.add_argument(
        "--dino-root",
        default="weights/dinov3-vitl16-pretrain-lvd1689m",
        help="DINOv3 ViT-L/16 safetensors root; default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for model.glb and trace.json; default: outputs/trellis2/<image-stem>",
    )
    parser.add_argument("--output", type=Path, help="explicit textured GLB path under outputs/")
    parser.add_argument("--trace-output", type=Path, help="explicit trace JSON path under outputs/")
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default=PRODUCTION_PIPELINE_TYPE,
        help="generation route; default 512 is the production-like Apple Silicon path",
    )
    parser.add_argument("--seed", type=int, default=42, help="sampling seed; default: %(default)s")
    parser.add_argument(
        "--slat-steps",
        type=int,
        help="override all SLat sampler steps; omit for quality runs so the model config default is used",
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=49_152,
        help="sparse sampling token budget; default: %(default)s",
    )
    parser.add_argument(
        "--decoder-token-limit",
        type=int,
        default=1_000_000,
        help="decoder token guard for textured export; default: %(default)s",
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=PRODUCTION_TEXTURE_SIZE,
        help="baked texture resolution; 1024 is the current production-like Mac default",
    )
    parser.add_argument(
        "--glb-target-faces",
        type=int,
        default=PRODUCTION_GLB_TARGET_FACES,
        help="postprocess target before GLB export; 200000 is recommended for quality runs",
    )
    parser.add_argument(
        "--xatlas-face-guard",
        type=_parse_xatlas_face_guard,
        default="auto",
        help="maximum faces allowed into xatlas unwrap, or 'auto'; default: %(default)s",
    )
    parser.add_argument(
        "--xatlas-parallel-chunks",
        type=int,
        default=PRODUCTION_XATLAS_PARALLEL_CHUNKS,
        help="split xatlas unwrap into chunks; 1 keeps a global unwrap for quality",
    )
    parser.add_argument(
        "--texture-bake-backend",
        choices=("kdtree", "trilinear"),
        default=PRODUCTION_TEXTURE_BAKE_BACKEND,
        help="texture voxel sampling backend; kdtree is the recommended visual default, trilinear is parity-oriented",
    )
    args = parser.parse_args(argv)

    from mlx_spatial.trellis2_inference import Trellis2InferencePipeline

    output, trace_output = _resolve_outputs(args.image, args.output_dir, args.output, args.trace_output, suffix=".glb")
    _print_effective_settings(args, output, trace_output)
    result = Trellis2InferencePipeline(args.root, rmbg_root=args.rmbg_root).generate_textured_glb(
        args.image,
        output_path=output,
        dino_root=args.dino_root,
        slat_steps=args.slat_steps,
        pipeline_type=args.pipeline_type,
        seed=args.seed,
        max_num_tokens=args.max_num_tokens,
        decoder_token_limit=args.decoder_token_limit,
        texture_size=args.texture_size,
        glb_target_faces=args.glb_target_faces,
        xatlas_face_guard=args.xatlas_face_guard,
        xatlas_parallel_chunks=args.xatlas_parallel_chunks,
        texture_bake_backend=args.texture_bake_backend,
        retain_trace_payloads=False,
    )
    _write_trace(trace_output, result.trace)

    print(f"completed={result.trace.completed_stages}")
    print(f"outputs={tuple(output.name for output in result.trace.outputs)}")
    print(f"trace={trace_output}")
    if result.artifact is not None:
        print(f"artifact={result.artifact.path}")
        print(f"bytes={result.artifact.bytes_written}")
        return 0
    print(f"blocker_stage={result.trace.blocker.stage if result.trace.blocker else None}")
    print(f"operation={result.trace.blocker.operation if result.trace.blocker else None}")
    print(f"reason={result.trace.blocker.reason if result.trace.blocker else None}")
    return 2


def _print_effective_settings(args: argparse.Namespace, output: Path, trace_output: Path) -> None:
    slat_steps = args.slat_steps if args.slat_steps is not None else "model-config"
    is_production_like = (
        args.pipeline_type == PRODUCTION_PIPELINE_TYPE
        and args.slat_steps is None
        and args.texture_size == PRODUCTION_TEXTURE_SIZE
        and args.glb_target_faces == PRODUCTION_GLB_TARGET_FACES
        and args.xatlas_parallel_chunks == PRODUCTION_XATLAS_PARALLEL_CHUNKS
        and args.texture_bake_backend == PRODUCTION_TEXTURE_BAKE_BACKEND
    )
    print(f"profile={'production-like' if is_production_like else 'custom'}", flush=True)
    print(
        "settings="
        f"pipeline_type={args.pipeline_type} "
        f"slat_steps={slat_steps} "
        f"texture_size={args.texture_size} "
        f"glb_target_faces={args.glb_target_faces} "
        f"xatlas_parallel_chunks={args.xatlas_parallel_chunks} "
        f"texture_bake_backend={args.texture_bake_backend}",
        flush=True,
    )
    print(f"output={output}", flush=True)
    print(f"trace={trace_output}", flush=True)


def _resolve_outputs(
    image: Path,
    output_dir: Path | None,
    output: Path | None,
    trace_output: Path | None,
    *,
    suffix: str,
) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = Path("outputs/trellis2") / _slug(image.stem)
    if output is None:
        output = output_dir / f"model{suffix}"
    if trace_output is None:
        trace_output = output.with_name("trace.json")
    return output, trace_output


def _parse_xatlas_face_guard(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    parsed = int(normalized)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("xatlas-face-guard must be 'auto' or a positive integer")
    return parsed


def _write_trace(path: Path, trace: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(trace), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "trellis2-object"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
