"""Pixal3D local asset and inference tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from .pixal3d_assets import (
    PIXAL3D_DEFAULT_ROOT,
    PIXAL3D_LICENSE_NOTE,
    PIXAL3D_PROBE_GROUPS,
    inspect_pixal3d_checkpoints,
    inspect_pixal3d_probe,
    pixal3d_download_command,
    read_pixal3d_pipeline_config,
    validate_pixal3d_assets,
)
from .pixal3d_inference import (
    PIXAL3D_DEFAULT_DINO_ROOT,
    PIXAL3D_DEFAULT_GLB_TARGET_FACES,
    PIXAL3D_DEFAULT_MAX_NUM_TOKENS,
    PIXAL3D_DEFAULT_MOGE_MEMORY_PROFILE,
    PIXAL3D_DEFAULT_MOGE_ROOT,
    PIXAL3D_DEFAULT_NAF_COORDINATE_CHUNK_SIZE,
    PIXAL3D_DEFAULT_NAF_ROOT,
    PIXAL3D_DEFAULT_SEED,
    PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT,
    PIXAL3D_DEFAULT_TEXTURE_BAKE_BACKEND,
    PIXAL3D_DEFAULT_TEXTURE_SIZE,
    PIXAL3D_PIPELINE_TYPES,
    PIXAL3D_RECOMMENDED_PIPELINE_TYPE,
    Pixal3DInferencePipeline,
)
from .sam3d_moge import SAM3D_MOGE_MEMORY_PROFILES
from .trellis2_export import TRELLIS2_TEXTURE_BAKE_BACKENDS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TencentARC Pixal3D MLX tooling")
    parser.add_argument("--root", dest="global_root", default=PIXAL3D_DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the local Pixal3D asset layout")
    validate_parser.add_argument("root_path", nargs="?")
    validate_parser.add_argument("--root", dest="command_root")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    inspect_parser = subparsers.add_parser("inspect", help="inspect local Pixal3D checkpoints")
    inspect_parser.add_argument("root_path", nargs="?")
    inspect_parser.add_argument("--root", dest="command_root")
    inspect_parser.add_argument("--checkpoint", action="append", dest="checkpoints")
    inspect_parser.add_argument("--limit", type=int, default=20)

    probe_parser = subparsers.add_parser("probe", help="inspect a named Pixal3D probe group")
    probe_parser.add_argument("group", choices=tuple(group.name for group in PIXAL3D_PROBE_GROUPS))
    probe_parser.add_argument("root_path", nargs="?")
    probe_parser.add_argument("--root", dest="command_root")

    download_parser = subparsers.add_parser("download-command", help="print the manual HF CLI command")
    download_parser.add_argument("root_path", nargs="?")
    download_parser.add_argument("--root", dest="command_root")

    generate_parser = subparsers.add_parser("generate", help="run Pixal3D generation or report the current blocker")
    generate_parser.add_argument("image", help="input RGB/RGBA image")
    generate_parser.add_argument("--root", "--weights-root", dest="command_root")
    generate_parser.add_argument("--output", type=Path, help="explicit output GLB path")
    generate_parser.add_argument("--output-dir", type=Path, help="directory for generated artifacts")
    generate_parser.add_argument(
        "--pipeline-type",
        choices=PIXAL3D_PIPELINE_TYPES,
        default=PIXAL3D_RECOMMENDED_PIPELINE_TYPE,
        help="1024_cascade is the recommended Apple Silicon default",
    )
    generate_parser.add_argument(
        "--manual-fov",
        type=float,
        help="manual horizontal FOV in radians; overrides MoGe auto-camera",
    )
    generate_parser.add_argument("--seed", type=int, default=PIXAL3D_DEFAULT_SEED)
    generate_parser.add_argument("--max-num-tokens", type=int, default=PIXAL3D_DEFAULT_MAX_NUM_TOKENS)
    generate_parser.add_argument(
        "--shape-upsample-token-limit",
        type=int,
        default=PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT,
        help="compute token guard for Pixal3D shape decoder upsample before HR coordinate selection; default: %(default)s",
    )
    generate_parser.add_argument(
        "--texture-size",
        type=int,
        default=PIXAL3D_DEFAULT_TEXTURE_SIZE,
        help="baked GLB texture resolution; default: %(default)s",
    )
    generate_parser.add_argument(
        "--glb-target-faces",
        type=int,
        default=PIXAL3D_DEFAULT_GLB_TARGET_FACES,
        help="mesh postprocess face target before GLB export; default: %(default)s",
    )
    generate_parser.add_argument(
        "--xatlas-face-guard",
        type=_parse_xatlas_face_guard,
        default="auto",
        help="maximum faces allowed into xatlas unwrap, or 'auto'; default: %(default)s",
    )
    generate_parser.add_argument(
        "--xatlas-parallel-chunks",
        type=int,
        default=0,
        help="split xatlas unwrap into chunks; default: %(default)s",
    )
    generate_parser.add_argument(
        "--texture-bake-backend",
        choices=TRELLIS2_TEXTURE_BAKE_BACKENDS,
        default=PIXAL3D_DEFAULT_TEXTURE_BAKE_BACKEND,
        help="texture voxel sampling backend for GLB export; default: %(default)s",
    )
    generate_parser.add_argument(
        "--dino-root",
        default=PIXAL3D_DEFAULT_DINO_ROOT,
        help="local DINOv3 ViT-L/16 root for Pixal3D image conditioning",
    )
    generate_parser.add_argument(
        "--moge-root",
        default=PIXAL3D_DEFAULT_MOGE_ROOT,
        help="local converted MoGe safetensors root for Pixal3D auto-camera",
    )
    generate_parser.add_argument(
        "--moge-memory-profile",
        choices=tuple(SAM3D_MOGE_MEMORY_PROFILES),
        default=PIXAL3D_DEFAULT_MOGE_MEMORY_PROFILE,
        help="MLX MoGe memory profile used when --manual-fov is omitted; default: %(default)s",
    )
    generate_parser.add_argument(
        "--naf-root",
        default=PIXAL3D_DEFAULT_NAF_ROOT,
        help="local converted NAF safetensors root for high-resolution projection features",
    )
    generate_parser.add_argument(
        "--naf-coordinate-chunk-size",
        type=int,
        default=PIXAL3D_DEFAULT_NAF_COORDINATE_CHUNK_SIZE,
        help="coordinate chunk size for MLX NAF projected-feature sampling; default: %(default)s",
    )
    generate_parser.add_argument("--trace-output", type=Path, help="trace JSON path; default: <output-dir>/trace.json")

    args = parser.parse_args(argv)
    root = _root_arg(args, fallback=args.global_root)

    if args.command == "validate":
        validation = validate_pixal3d_assets(root)
        payload: dict[str, Any] = {
            "ready": validation.ready,
            "name": validation.name,
            "root": str(validation.root),
            "present": list(validation.present),
            "missing": list(validation.missing),
            "license_note": PIXAL3D_LICENSE_NOTE,
        }
        if validation.ready:
            try:
                payload["pipeline"] = _jsonable(read_pixal3d_pipeline_config(root))
            except ValueError as error:
                payload["ready"] = False
                payload["config_error"] = str(error)
        _print_payload(payload, as_json=args.as_json)
        return 0 if payload["ready"] else 1

    if args.command == "download-command":
        print(" ".join(pixal3d_download_command(root)))
        return 0

    if args.command == "inspect":
        validation = validate_pixal3d_assets(root)
        print(f"ready={validation.ready}")
        print(f"root={validation.root}")
        for missing in validation.missing:
            print(f"missing {missing}")
        if not validation.ready:
            return 1
        infos_by_path = inspect_pixal3d_checkpoints(root, checkpoint_paths=args.checkpoints)
        limit = max(args.limit, 0)
        for checkpoint_path, infos in infos_by_path.items():
            print(f"checkpoint {checkpoint_path} tensors={len(infos)}")
            selected = infos[:limit] if limit else infos
            for info in selected:
                print(f"tensor {info.name} shape={info.shape} dtype={info.dtype}")
            if limit and len(infos) > limit:
                print(f"truncated={len(infos) - limit}")
        return 0

    if args.command == "probe":
        try:
            infos = inspect_pixal3d_probe(root, args.group)
        except FileNotFoundError as error:
            print(error)
            return 1
        for info in infos:
            print(f"tensor {info.name} shape={info.shape} dtype={info.dtype}")
        return 0

    if args.command == "generate":
        if args.output is not None and args.output_dir is not None:
            parser.error("use either --output or --output-dir, not both")
        result = Pixal3DInferencePipeline(root).generate(
            args.image,
            output=args.output,
            output_dir=args.output_dir,
            pipeline_type=args.pipeline_type,
            manual_fov=args.manual_fov,
            seed=args.seed,
            max_num_tokens=args.max_num_tokens,
            shape_upsample_token_limit=args.shape_upsample_token_limit,
            dino_root=args.dino_root,
            texture_size=args.texture_size,
            glb_target_faces=args.glb_target_faces,
            xatlas_face_guard=args.xatlas_face_guard,
            xatlas_parallel_chunks=args.xatlas_parallel_chunks,
            texture_bake_backend=args.texture_bake_backend,
            naf_root=args.naf_root,
            naf_coordinate_chunk_size=args.naf_coordinate_chunk_size,
            moge_root=args.moge_root,
            moge_memory_profile=args.moge_memory_profile,
        )
        output_path = result.trace.output_path or Path("outputs/pixal3d/model.glb")
        trace_path = args.trace_output or output_path.with_name("trace.json")
        _write_trace(trace_path, result)
        print(f"completed={result.trace.completed_stages}")
        print(f"pipeline_type={result.trace.pipeline_type}")
        print(f"output={output_path}")
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


def _parse_xatlas_face_guard(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    parsed = int(normalized)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--xatlas-face-guard must be 'auto' or a positive integer")
    return parsed


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"ready={payload['ready']}")
    print(f"root={payload['root']}")
    print(f"present={len(payload['present'])}")
    print(f"missing={len(payload['missing'])}")
    for missing in payload["missing"]:
        print(f"missing {missing}")
    if "pipeline" in payload:
        pipeline = payload["pipeline"]
        print(f"default_pipeline_type={pipeline['default_pipeline_type']}")
        print(f"models={len(pipeline['models'])}")
    if "config_error" in payload:
        print(f"config_error={payload['config_error']}")


def _write_trace(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_trace_payload(result), indent=2, sort_keys=True, default=str), encoding="utf-8")


def _trace_payload(result: Any) -> dict[str, Any]:
    trace = result.trace
    blocker = trace.blocker
    return {
        "ready": result.ready,
        "root": str(trace.root),
        "image_path": str(trace.image_path),
        "completed_stages": list(trace.completed_stages),
        "pipeline_type": trace.pipeline_type,
        "manual_fov": trace.manual_fov,
        "seed": trace.seed,
        "max_num_tokens": trace.max_num_tokens,
        "output_path": str(trace.output_path) if trace.output_path is not None else None,
        "artifacts": [str(path) for path in getattr(result, "artifacts", ())],
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
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
