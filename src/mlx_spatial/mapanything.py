"""MapAnything local asset and inference tooling."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .mapanything_assets import (
    MAPANYTHING_DEFAULT_ROOT,
    inspect_mapanything_checkpoint,
    inspect_mapanything_model_assets,
    mapanything_download_command,
    validate_mapanything_assets,
)


MAPANYTHING_RECOMMENDED_RESIZE_MODE = "fixed_mapping"
MAPANYTHING_RECOMMENDED_STRIDE = 1
MAPANYTHING_RESIZE_MODES = ("fixed_mapping", "longest_side", "square", "fixed_size")
MAPANYTHING_SCENE_REQUIRED_GROUPS = (
    "encoder",
    "info_sharing",
    "dense_head",
    "pose_head",
    "scale_head",
    "fusion_norm_layer",
    "scale_token",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Meta MapAnything MLX scene generation")
    parser.add_argument("--root", dest="global_root", default=MAPANYTHING_DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the local MapAnything asset layout")
    validate_parser.add_argument("root_path", nargs="?")
    validate_parser.add_argument("--root", dest="command_root")

    inspect_parser = subparsers.add_parser("inspect", help="inspect local MapAnything checkpoint tensors")
    inspect_parser.add_argument("root_path", nargs="?")
    inspect_parser.add_argument("--root", dest="command_root")
    inspect_parser.add_argument("--prefix", action="append", dest="prefixes")
    inspect_parser.add_argument("--limit", type=int, default=20)

    download_parser = subparsers.add_parser("download-command", help="print the manual HF CLI command")
    download_parser.add_argument("root_path", nargs="?")
    download_parser.add_argument("--root", dest="command_root")

    generate_parser = subparsers.add_parser("generate", help="run MapAnything multi-view scene generation")
    generate_parser.add_argument("input", help="input image file or directory of scene views")
    generate_parser.add_argument("--root", "--weights-root", dest="command_root")
    generate_parser.add_argument("--output", type=Path, help="output .npz scene bundle")
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for scene.npz and trace.json; default: outputs/mapanything/<input-stem>",
    )
    generate_parser.add_argument(
        "--resize-mode",
        choices=MAPANYTHING_RESIZE_MODES,
        default=MAPANYTHING_RECOMMENDED_RESIZE_MODE,
        help="fixed_mapping is the recommended upstream image-only inference setting",
    )
    generate_parser.add_argument(
        "--size",
        type=_parse_size,
        help="optional size for square/longest_side/fixed_size debug runs, e.g. 518 or 518x392",
    )
    generate_parser.add_argument(
        "--stride",
        type=int,
        default=MAPANYTHING_RECOMMENDED_STRIDE,
        help="input frame stride; default 1 keeps every image",
    )
    generate_parser.add_argument("--trace-output", type=Path, help="trace JSON path; default: next to scene.npz")

    args = parser.parse_args(argv)
    root = _root_arg(args, fallback=args.global_root)

    if args.command == "validate":
        validation = validate_mapanything_assets(root)
        inspection = inspect_mapanything_model_assets(root, required_groups=MAPANYTHING_SCENE_REQUIRED_GROUPS)
        print(f"ready={inspection.ready}")
        print(f"root={validation.root}")
        print(f"present={len(validation.present)}")
        print(f"missing={len(validation.missing)}")
        for missing in validation.missing:
            print(f"missing {missing}")
        if inspection.config is not None:
            print(f"encoder={inspection.config.encoder_size}")
            print(f"info_sharing_dim={inspection.config.info_sharing_dim}")
            print(f"patch_size={inspection.config.patch_size}")
        if inspection.blocker is not None:
            print(f"blocker_stage={inspection.blocker.stage}")
            print(f"operation={inspection.blocker.operation}")
            print(f"reason={inspection.blocker.reason}")
        return 0 if inspection.ready else 1

    if args.command == "download-command":
        print(" ".join(mapanything_download_command(root)))
        return 0

    if args.command == "inspect":
        validation = validate_mapanything_assets(root)
        print(f"ready={validation.ready}")
        print(f"root={validation.root}")
        for missing in validation.missing:
            print(f"missing {missing}")
        if not validation.checkpoint_path.is_file():
            return 1
        infos = inspect_mapanything_checkpoint(root, prefixes=args.prefixes)
        limit = max(args.limit, 0)
        selected = infos[:limit] if limit else infos
        for info in selected:
            print(f"tensor {info.name} shape={info.shape} dtype={info.dtype}")
        if limit and len(infos) > limit:
            print(f"truncated={len(infos) - limit}")
        return 0

    if args.command == "generate":
        if args.output is not None and args.output_dir is not None:
            parser.error("use either --output or --output-dir, not both")
        output_path = _resolve_scene_output(args.input, output=args.output, output_dir=args.output_dir)
        trace_path = args.trace_output or output_path.with_name("trace.json")

        from .mapanything_scene import MapAnythingScenePipeline, write_mapanything_scene_npz

        result = MapAnythingScenePipeline(root).generate(
            args.input,
            resize_mode=args.resize_mode,
            size=args.size,
            stride=args.stride,
        )
        written: Path | None = None
        if result.ready and result.predictions is not None:
            written = write_mapanything_scene_npz(
                output_path,
                result.predictions,
                metadata={
                    **result.predictions.metadata,
                    "completed_stages": list(result.trace.completed_stages),
                },
            )

        _write_trace(trace_path, _trace_payload(result.trace, output_path=written or output_path))
        print(f"completed={result.trace.completed_stages}")
        print(f"frames={result.trace.frame_count}")
        print(f"target_size={result.trace.target_size}")
        print(f"output={written or output_path}")
        print(f"trace={trace_path}")
        if result.trace.blocker is not None:
            print(f"blocker_stage={result.trace.blocker.stage}")
            print(f"operation={result.trace.blocker.operation}")
            print(f"reason={result.trace.blocker.reason}")
            return 2
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


def _root_arg(args: argparse.Namespace, *, fallback: str | Path) -> str | Path:
    return getattr(args, "command_root", None) or getattr(args, "root_path", None) or fallback


def _parse_size(value: str) -> int | tuple[int, int]:
    normalized = value.strip().lower().replace(",", "x")
    if "x" not in normalized:
        try:
            return int(normalized)
        except ValueError as error:
            raise argparse.ArgumentTypeError("size must be an int or WIDTHxHEIGHT") from error
    width, separator, height = normalized.partition("x")
    if not separator:
        raise argparse.ArgumentTypeError("size must be an int or WIDTHxHEIGHT")
    try:
        parsed = (int(width), int(height))
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must be an int or WIDTHxHEIGHT") from error
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return parsed


def _resolve_scene_output(input_path: str | Path, *, output: Path | None, output_dir: Path | None) -> Path:
    if output is not None:
        if output.suffix != ".npz":
            raise SystemExit("MapAnything scene output must be a .npz file")
        return output
    directory = output_dir or Path("outputs/mapanything") / _slug(Path(input_path).stem or Path(input_path).name)
    return directory / "scene.npz"


def _write_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _trace_payload(trace: Any, *, output_path: Path) -> dict[str, Any]:
    blocker = trace.blocker
    return {
        "ready": blocker is None,
        "completed_stages": list(trace.completed_stages),
        "model_root": str(trace.model_root),
        "input_path": str(trace.input_path),
        "frame_count": trace.frame_count,
        "target_size": list(trace.target_size) if trace.target_size is not None else None,
        "output_keys": list(trace.output_keys),
        "output": str(output_path),
        "metadata": _jsonable(trace.metadata),
        "blocker": None
        if blocker is None
        else {
            "stage": blocker.stage,
            "operation": blocker.operation,
            "reason": blocker.reason,
            "metadata": _jsonable(blocker.metadata),
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "mapanything-scene"


if __name__ == "__main__":
    raise SystemExit(main())
