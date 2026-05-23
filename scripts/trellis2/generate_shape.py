#!/usr/bin/env python3
"""Run TRELLIS.2 image-to-shape-OBJ generation with production-like defaults.

Input:
    A single RGB/RGBA image. RGBA images use their alpha channel directly.
    RGB images use the configured RMBG root to generate foreground alpha.

Production defaults:
    pipeline type 512 and model-config SLat sampler steps. Do not pass
    --slat-steps for quality runs. Use --slat-steps 1 only for smoke tests.

Example:
    python scripts/trellis2/generate_shape.py inputs/trellis2/cup-of-tea.jpg \\
      --output-dir outputs/trellis2/cup-of-tea-shape-script
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="input RGB/RGBA image")
    parser.add_argument("--root", default="weights/trellis2", help="TRELLIS.2 safetensors root")
    parser.add_argument("--rmbg-root", default="weights/rmbg2", help="RMBG-2.0 safetensors root for RGB images")
    parser.add_argument(
        "--dino-root",
        default="weights/dinov3-vitl16-pretrain-lvd1689m",
        help="DINOv3 ViT-L/16 safetensors root",
    )
    parser.add_argument("--output-dir", type=Path, help="directory for model.obj and trace.json")
    parser.add_argument("--output", type=Path, help="explicit OBJ path under outputs/")
    parser.add_argument("--trace-output", type=Path, help="explicit trace JSON path under outputs/")
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default=PRODUCTION_PIPELINE_TYPE,
        help="generation route; default 512 is the production-like Apple Silicon path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--slat-steps",
        type=int,
        help="override all SLat sampler steps; omit for quality runs so the model config default is used",
    )
    parser.add_argument("--max-num-tokens", type=int, default=49_152)
    parser.add_argument("--decoder-token-limit", type=int, default=1_000_000)
    args = parser.parse_args(argv)

    from mlx_spatial.trellis2_inference import Trellis2InferencePipeline

    output, trace_output = _resolve_outputs(args.image, args.output_dir, args.output, args.trace_output)
    _print_effective_settings(args, output, trace_output)
    result = Trellis2InferencePipeline(args.root, rmbg_root=args.rmbg_root).generate_shape_obj(
        args.image,
        output_path=output,
        dino_root=args.dino_root,
        slat_steps=args.slat_steps,
        pipeline_type=args.pipeline_type,
        seed=args.seed,
        max_num_tokens=args.max_num_tokens,
        decoder_token_limit=args.decoder_token_limit,
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
    is_production_like = args.pipeline_type == PRODUCTION_PIPELINE_TYPE and args.slat_steps is None
    print(f"profile={'production-like' if is_production_like else 'custom'}", flush=True)
    print(
        "settings="
        f"pipeline_type={args.pipeline_type} "
        f"slat_steps={slat_steps} "
        f"max_num_tokens={args.max_num_tokens} "
        f"decoder_token_limit={args.decoder_token_limit}",
        flush=True,
    )
    print(f"output={output}", flush=True)
    print(f"trace={trace_output}", flush=True)


def _resolve_outputs(
    image: Path,
    output_dir: Path | None,
    output: Path | None,
    trace_output: Path | None,
) -> tuple[Path, Path]:
    if output_dir is None:
        output_dir = Path("outputs/trellis2") / f"{_slug(image.stem)}-shape"
    if output is None:
        output = output_dir / "model.obj"
    if trace_output is None:
        trace_output = output.with_name("trace.json")
    return output, trace_output


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
