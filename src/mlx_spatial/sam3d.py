"""SAM 3D Objects local asset and inference tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Sequence

from .sam3d_assets import (
    SAM3D_OBJECTS_ACCESS_NOTE,
    SAM3D_OBJECTS_DEFAULT_ROOT,
    SAM3D_OBJECTS_MLX_DEFAULT_ROOT,
    SAM3D_OBJECTS_REPO_ID,
    convert_sam3d_assets_to_safetensors,
    convert_torch_checkpoint_to_safetensors,
    download_sam3d_assets,
    inspect_sam3d_model_assets,
    sam3d_download_command,
    validate_sam3d_assets,
)
from .sam3d_export import (
    SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACE_FRACTION,
    SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACES,
    SAM3D_GLB_DEFAULT_TARGET_FACES,
    SAM3D_XATLAS_FACE_GUARD,
)
from .sam3d_inference import Sam3dInferencePipeline
from .sam3d_moge import SAM3D_MOGE_DEFAULT_ROOT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and run SAM 3D Objects MLX parity tooling")
    parser.add_argument("--root", default=SAM3D_OBJECTS_DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the local SAM3D asset layout")
    validate_parser.add_argument("root_path", nargs="?")
    validate_parser.add_argument("--root", dest="command_root")

    inspect_parser = subparsers.add_parser("inspect", help="inspect SAM3D pipeline config and checkpoints")
    inspect_parser.add_argument("root_path", nargs="?")
    inspect_parser.add_argument("--root", dest="command_root")

    download_command_parser = subparsers.add_parser("download-command", help="print the gated HF download command")
    download_command_parser.add_argument("root_path", nargs="?")
    download_command_parser.add_argument("--root", dest="command_root")

    download_parser = subparsers.add_parser("download", help="download gated SAM3D assets with huggingface_hub")
    download_parser.add_argument("root_path", nargs="?")
    download_parser.add_argument("--root", dest="command_root")
    download_parser.add_argument("--max-workers", type=int, default=1)

    convert_parser = subparsers.add_parser("convert", help="convert official SAM3D checkpoints to safetensors")
    convert_parser.add_argument("source_root", nargs="?")
    convert_parser.add_argument("--root", dest="command_root")
    convert_parser.add_argument("--output-root", default=SAM3D_OBJECTS_MLX_DEFAULT_ROOT)
    convert_parser.add_argument("--moge-root")
    convert_parser.add_argument("--moge-output-root", default="weights/moge-vitl-mlx")
    convert_parser.add_argument("--max-archive-gb", type=float, default=16.0)
    convert_parser.add_argument("--max-tensor-gb", type=float, default=16.0)
    convert_parser.add_argument("--overwrite", action="store_true")

    reconstruct_parser = subparsers.add_parser("reconstruct", help="run exact SAM3D image+mask to Gaussian PLY and optional GLB")
    reconstruct_parser.add_argument("reconstruct_root")
    reconstruct_parser.add_argument("image")
    reconstruct_parser.add_argument("--mask", required=True)
    reconstruct_parser.add_argument("--moge-root", default=SAM3D_MOGE_DEFAULT_ROOT)
    reconstruct_parser.add_argument("--output", required=True)
    reconstruct_parser.add_argument("--glb-output")
    reconstruct_parser.add_argument("--seed", type=int, default=42)
    reconstruct_parser.add_argument("--stage1-steps", type=int, default=2)
    reconstruct_parser.add_argument("--stage2-steps", type=int, default=12)
    reconstruct_parser.add_argument("--memory-profile", choices=("safe", "balanced", "large"), default="balanced")
    reconstruct_parser.add_argument("--glb-postprocess", choices=("cleaned", "basic"), default="cleaned")
    reconstruct_parser.add_argument("--glb-target-faces", type=int, default=SAM3D_GLB_DEFAULT_TARGET_FACES)
    reconstruct_parser.add_argument("--glb-min-component-faces", type=int, default=SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACES)
    reconstruct_parser.add_argument(
        "--glb-min-component-face-fraction",
        type=float,
        default=SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACE_FRACTION,
    )
    reconstruct_parser.add_argument("--no-glb-simplify", action="store_true")
    reconstruct_parser.add_argument("--glb-smooth-iterations", type=int, default=0)
    reconstruct_parser.add_argument("--glb-texture", choices=("gaussian", "none"), default="gaussian")
    reconstruct_parser.add_argument("--glb-texture-size", type=int, default=1024)
    reconstruct_parser.add_argument("--glb-gaussian-k", type=int, default=8)
    reconstruct_parser.add_argument("--glb-texel-chunk-size", type=int, default=262_144)
    reconstruct_parser.add_argument("--glb-xatlas-face-guard", type=int, default=SAM3D_XATLAS_FACE_GUARD)
    reconstruct_parser.add_argument("--trace-output")

    args = parser.parse_args(argv)
    root = (
        getattr(args, "command_root", None)
        or getattr(args, "root_path", None)
        or getattr(args, "source_root", None)
        or args.root
    )

    if args.command == "validate":
        _print_validation(validate_sam3d_assets(root))
        return 0

    if args.command == "download-command":
        print(SAM3D_OBJECTS_ACCESS_NOTE)
        print(" ".join(sam3d_download_command(root)))
        return 0

    if args.command == "download":
        result = download_sam3d_assets(root, max_workers=max(1, args.max_workers))
        _print_validation(result.validation)
        if result.blocker is not None:
            print(f"blocker_stage={result.blocker.stage}")
            print(f"operation={result.blocker.operation}")
            print(f"reason={result.blocker.reason}")
            return 2
        return 0

    if args.command == "convert":
        result = convert_sam3d_assets_to_safetensors(
            root,
            output_root=args.output_root,
            overwrite=args.overwrite,
            max_archive_bytes=_gb_to_bytes(args.max_archive_gb),
            max_tensor_bytes=_gb_to_bytes(args.max_tensor_gb),
        )
        _print_validation(result.validation)
        print(f"output_root={result.output_root}")
        if result.output_pipeline_path is not None:
            print(f"output_pipeline={result.output_pipeline_path}")
        for item in result.items:
            detail = f" tensors={item.tensor_count}" if item.tensor_count is not None else ""
            print(f"{item.status} {item.kind} {item.role} {item.output_path}{detail}")
        if result.blocker is not None:
            print(f"blocker_stage={result.blocker.stage}")
            print(f"operation={result.blocker.operation}")
            print(f"reason={result.blocker.reason}")
            return 2
        if args.moge_root:
            moge_source = Path(args.moge_root) / "model.pt"
            moge_output = Path(args.moge_output_root) / "model.safetensors"
            try:
                moge_item = convert_torch_checkpoint_to_safetensors(
                    moge_source,
                    moge_output,
                    role="moge",
                    overwrite=args.overwrite,
                    max_archive_bytes=_gb_to_bytes(args.max_archive_gb),
                    max_tensor_bytes=_gb_to_bytes(args.max_tensor_gb),
                )
            except Exception as error:
                print("blocker_stage=moge-checkpoint-conversion")
                print("operation=convert MoGe PyTorch checkpoint to safetensors without torch")
                print(f"reason={error}")
                return 2
            print(f"{moge_item.status} checkpoint {moge_item.role} {moge_item.output_path} tensors={moge_item.tensor_count}")
        return 0

    if args.command == "inspect":
        inspection = inspect_sam3d_model_assets(root)
        _print_validation(inspection.validation)
        if inspection.config is not None:
            print(f"target={inspection.config.target}")
            print(f"dtype={inspection.config.dtype}")
            print(f"rendering_engine={inspection.config.rendering_engine}")
            print(f"decode_formats={inspection.config.decode_formats}")
        for item in inspection.paths:
            status = "present" if item.exists else "missing"
            print(f"{status} {item.kind} {item.field} {item.relative_path}")
        for checkpoint in inspection.checkpoints:
            print(
                f"checkpoint {checkpoint.role} {checkpoint.relative_path} "
                f"tensors={checkpoint.tensor_count} prefixes={checkpoint.prefixes}"
            )
        if inspection.blocker is not None:
            print(f"blocker_stage={inspection.blocker.stage}")
            print(f"operation={inspection.blocker.operation}")
            print(f"reason={inspection.blocker.reason}")
            return 2
        return 0

    if args.command == "reconstruct":
        try:
            result = Sam3dInferencePipeline(args.reconstruct_root).generate_gaussians_ply(
                args.image,
                mask_path=args.mask,
                output_path=args.output,
                glb_output_path=args.glb_output,
                moge_root=args.moge_root,
                seed=args.seed,
                stage1_steps=args.stage1_steps,
                stage2_steps=args.stage2_steps,
                memory_profile=args.memory_profile,
                glb_postprocess=args.glb_postprocess,
                glb_target_faces=0 if args.no_glb_simplify else args.glb_target_faces,
                glb_min_component_faces=args.glb_min_component_faces,
                glb_min_component_face_fraction=args.glb_min_component_face_fraction,
                glb_smooth_iterations=args.glb_smooth_iterations,
                glb_texture=args.glb_texture,
                glb_texture_size=args.glb_texture_size,
                glb_gaussian_k=args.glb_gaussian_k,
                glb_texel_chunk_size=args.glb_texel_chunk_size,
                glb_xatlas_face_guard=args.glb_xatlas_face_guard,
            )
        except ValueError as error:
            print("blocker_stage=argument-validation")
            print("operation=validate SAM3D reconstruct arguments")
            print(f"reason={error}")
            return 2
        trace = result.trace
        payload = _jsonable(trace)
        if args.trace_output:
            trace_output = Path(args.trace_output)
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            trace_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"completed={trace.completed_stages}")
        print(f"outputs={tuple(output.name for output in trace.outputs)}")
        if trace.blocker is not None:
            print(f"blocker_stage={trace.blocker.stage}")
            print(f"operation={trace.blocker.operation}")
            print(f"reason={trace.blocker.reason}")
            return 2
        if result.artifact is not None:
            print(f"artifact={result.artifact.path}")
        return 0

    parser.error(f"unsupported command: {args.command}")


def _print_validation(validation) -> None:
    print(f"ready={validation.ready}")
    print(f"repo={SAM3D_OBJECTS_REPO_ID}")
    print(f"root={validation.root}")
    print(f"model_dir={validation.model_dir}")
    print(f"pipeline={validation.pipeline_path if validation.pipeline_path is not None else '<missing>'}")
    print(f"present={len(validation.present)}")
    print(f"missing={len(validation.missing)}")
    for path in validation.missing:
        print(f"missing {path}")


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _gb_to_bytes(value: float) -> int | None:
    if value <= 0:
        return None
    return int(value * 1024**3)


if __name__ == "__main__":
    raise SystemExit(main())
