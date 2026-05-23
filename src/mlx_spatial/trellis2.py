"""TRELLIS.2 local asset and checkpoint tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mlx.core as mx

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint, load_checkpoint_tensors
from .model_assets import DINOv3_VITL16_ASSETS, RMBG2_ASSETS, TRELLIS2_ASSETS, ModelAssetValidation, validate_model_assets
from .trellis2_dinov3 import DINOv3_ACCESS_NOTE, DINOv3_VITL16_REPO_ID, dinov3_download_command


@dataclass(frozen=True)
class Trellis2ProbeGroup:
    """Named tensor selection for one TRELLIS.2 checkpoint file."""

    name: str
    checkpoint_path: str
    names: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()
    reference: str = ""


@dataclass(frozen=True)
class Trellis2TensorProbe:
    """Loaded TRELLIS.2 tensor and its probe context."""

    group: str
    checkpoint_path: str
    name: str
    array: mx.array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> str:
        return str(self.array.dtype).removeprefix("mlx.core.")


TRELLIS2_REPO_ID = "microsoft/TRELLIS.2-4B"
RMBG2_REPO_ID = "briaai/RMBG-2.0"
RMBG2_LICENSE_NOTE = (
    "briaai/RMBG-2.0 is gated on Hugging Face and released for non-commercial use; "
    "authenticate and review the license before downloading."
)
TRELLIS2_SHAPE_DEFAULT_SEED = 42
TRELLIS2_SHAPE_DEFAULT_MAX_NUM_TOKENS = 49_152
TRELLIS2_SHAPE_DEFAULT_DECODER_TOKEN_LIMIT = 1_000_000
TRELLIS2_TEXTURE_DEFAULT_SIZE = 1024
TRELLIS2_GLB_DEFAULT_FACE_TARGET = 50_000
TRELLIS2_XATLAS_DEFAULT_FACE_GUARD = "auto"
TRELLIS2_XATLAS_DEFAULT_PARALLEL_CHUNKS = 0
TRELLIS2_TEXTURE_BAKE_BACKENDS = ("trilinear", "kdtree")

TRELLIS2_PROBE_GROUPS = (
    Trellis2ProbeGroup(
        name="sparse-structure-flow",
        checkpoint_path="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
        names=("blocks.0.norm2.weight",),
        reference="trellis-mac generation path loads microsoft/TRELLIS.2-4B; original TRELLIS.2 uses ckpts/ss_flow_img_dit_1_3B_64_bf16 for sparse structure sampling.",
    ),
    Trellis2ProbeGroup(
        name="shape-slat-flow",
        checkpoint_path="ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
        names=("blocks.0.norm2.weight",),
        reference="trellis-mac pipeline type 512 includes shape SLat sampling; original TRELLIS.2 names slat_flow_img2shape_dit_1_3B_512_bf16.",
    ),
    Trellis2ProbeGroup(
        name="texture-slat-flow",
        checkpoint_path="ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
        names=("blocks.0.norm2.weight",),
        reference="trellis-mac texture path includes texture SLat sampling; original TRELLIS.2 names slat_flow_imgshape2tex_dit_1_3B_512_bf16.",
    ),
    Trellis2ProbeGroup(
        name="shape-decoder",
        checkpoint_path="ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        names=("blocks.0.0.norm.weight",),
        reference="trellis-mac shape decoder timing maps to original TRELLIS.2 shape_dec_next_dc_f16c32_fp16 checkpoint.",
    ),
    Trellis2ProbeGroup(
        name="texture-decoder",
        checkpoint_path="ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        names=("blocks.0.0.norm.weight",),
        reference="trellis-mac texture decoder timing maps to original TRELLIS.2 tex_dec_next_dc_f16c32_fp16 checkpoint.",
    ),
)


def validate_trellis2_assets(root: str | Path = TRELLIS2_ASSETS.root_hint) -> ModelAssetValidation:
    """Validate a local TRELLIS.2 asset root without downloading files."""

    return validate_model_assets(_validate_root(root), TRELLIS2_ASSETS)


def validate_rmbg2_assets(root: str | Path = RMBG2_ASSETS.root_hint) -> ModelAssetValidation:
    """Validate a local RMBG-2.0 asset root without downloading files."""

    return validate_model_assets(root, RMBG2_ASSETS)


def validate_dinov3_assets(root: str | Path = DINOv3_VITL16_ASSETS.root_hint) -> ModelAssetValidation:
    """Validate a local DINOv3 asset root without downloading files."""

    return validate_model_assets(root, DINOv3_VITL16_ASSETS)


def trellis2_probe_group(name: str) -> Trellis2ProbeGroup:
    """Return a named TRELLIS.2 probe group."""

    for group in TRELLIS2_PROBE_GROUPS:
        if group.name == name:
            return group
    raise ValueError(f"unknown TRELLIS.2 probe group: {name!r}")


def inspect_trellis2_checkpoints(
    root: str | Path = TRELLIS2_ASSETS.root_hint,
    *,
    checkpoint_paths: Iterable[str] | None = None,
) -> dict[str, tuple[CheckpointTensorInfo, ...]]:
    """Inspect configured TRELLIS.2 safetensors checkpoints under a local root."""

    root_path = _validate_root(root)
    paths = _checkpoint_paths(checkpoint_paths)
    return {relative_path: inspect_checkpoint(root_path / relative_path) for relative_path in paths}


def inspect_trellis2_probe(
    root: str | Path,
    group: str | Trellis2ProbeGroup,
) -> tuple[CheckpointTensorInfo, ...]:
    """Inspect tensors matched by a named TRELLIS.2 probe group."""

    root_path = _validate_root(root)
    probe_group = _coerce_group(group)
    return inspect_checkpoint(
        root_path / probe_group.checkpoint_path,
        names=probe_group.names or None,
        prefixes=probe_group.prefixes or None,
    )


def load_trellis2_probe(
    root: str | Path,
    group: str | Trellis2ProbeGroup,
) -> tuple[Trellis2TensorProbe, ...]:
    """Load tensors matched by a named TRELLIS.2 probe group as MLX arrays."""

    root_path = _validate_root(root)
    probe_group = _coerce_group(group)
    if not probe_group.names and not probe_group.prefixes:
        raise ValueError("TRELLIS.2 probe group must define names or prefixes")

    tensors = load_checkpoint_tensors(
        root_path / probe_group.checkpoint_path,
        names=probe_group.names or None,
        prefixes=probe_group.prefixes or None,
    )
    return tuple(
        Trellis2TensorProbe(
            group=probe_group.name,
            checkpoint_path=probe_group.checkpoint_path,
            name=name,
            array=tensors[name],
        )
        for name in sorted(tensors)
    )


def trellis2_download_command(root: str | Path = TRELLIS2_ASSETS.root_hint) -> tuple[str, ...]:
    """Return the dev-environment HF command for downloading TRELLIS.2 assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        TRELLIS2_REPO_ID,
        "--local-dir",
        str(root),
    )


def rmbg2_download_command(root: str | Path = RMBG2_ASSETS.root_hint) -> tuple[str, ...]:
    """Return the dev-environment HF command for manually downloading RMBG-2.0 assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        RMBG2_REPO_ID,
        "--local-dir",
        str(root),
    )


def _parse_xatlas_face_guard(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        parsed = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local TRELLIS.2 assets and checkpoints")
    parser.add_argument("--root", default=TRELLIS2_ASSETS.root_hint)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the local TRELLIS.2 asset layout")
    validate_parser.add_argument("--root", dest="command_root")

    rmbg_validate_parser = subparsers.add_parser("rmbg-validate", help="validate the local RMBG-2.0 asset layout")
    rmbg_validate_parser.add_argument("--root", dest="command_root")

    dinov3_validate_parser = subparsers.add_parser("dinov3-validate", help="validate the local DINOv3 asset layout")
    dinov3_validate_parser.add_argument("root_path", nargs="?")
    dinov3_validate_parser.add_argument("--root", dest="command_root")

    download_parser = subparsers.add_parser("download-command", help="print the manual HF CLI command")
    download_parser.add_argument("--root", dest="command_root")

    rmbg_download_parser = subparsers.add_parser(
        "rmbg-download-command",
        help="print the manual HF CLI command for gated RMBG-2.0 assets",
    )
    rmbg_download_parser.add_argument("--root", dest="command_root")

    dinov3_download_parser = subparsers.add_parser(
        "dinov3-download-command",
        help="print the manual HF CLI command for local DINOv3 assets",
    )
    dinov3_download_parser.add_argument("root_path", nargs="?")
    dinov3_download_parser.add_argument("--root", dest="command_root")

    inspect_parser = subparsers.add_parser("inspect", help="inspect local TRELLIS.2 checkpoints")
    inspect_parser.add_argument("--root", dest="command_root")
    inspect_parser.add_argument("--checkpoint", action="append", dest="checkpoints")

    probe_parser = subparsers.add_parser("probe", help="inspect or load a named probe group")
    probe_parser.add_argument("--root", dest="command_root")
    probe_parser.add_argument("group")
    probe_parser.add_argument("--load", action="store_true")

    forward_parser = subparsers.add_parser("attempt-forward-trace", help="run the staged TRELLIS.2 forward trace")
    forward_parser.add_argument("trace_root")
    forward_parser.add_argument("image")
    forward_parser.add_argument("--dino-root")
    forward_parser.add_argument("--rmbg-root")
    forward_parser.add_argument("--slat-steps", type=int)
    forward_parser.add_argument("--decoder-token-limit", type=int)
    forward_parser.add_argument("--output")

    generate_shape_parser = subparsers.add_parser("generate-shape", help="run shape-only TRELLIS.2 MLX mesh generation")
    generate_shape_parser.add_argument("shape_root")
    generate_shape_parser.add_argument("image")
    generate_shape_parser.add_argument("--output", required=True)
    generate_shape_parser.add_argument("--dino-root")
    generate_shape_parser.add_argument("--rmbg-root")
    generate_shape_parser.add_argument("--slat-steps", type=int)
    generate_shape_parser.add_argument("--pipeline-type", choices=("512", "1024", "1024_cascade", "1536_cascade"))
    generate_shape_parser.add_argument("--seed", type=int, default=TRELLIS2_SHAPE_DEFAULT_SEED)
    generate_shape_parser.add_argument("--max-num-tokens", type=int, default=TRELLIS2_SHAPE_DEFAULT_MAX_NUM_TOKENS)
    generate_shape_parser.add_argument("--decoder-token-limit", type=int, default=TRELLIS2_SHAPE_DEFAULT_DECODER_TOKEN_LIMIT)

    generate_textured_parser = subparsers.add_parser("generate-textured", help="run TRELLIS.2 textured GLB generation")
    generate_textured_parser.add_argument("textured_root")
    generate_textured_parser.add_argument("image")
    generate_textured_parser.add_argument("--output", required=True)
    generate_textured_parser.add_argument("--dino-root")
    generate_textured_parser.add_argument("--rmbg-root")
    generate_textured_parser.add_argument("--slat-steps", type=int)
    generate_textured_parser.add_argument("--pipeline-type", choices=("512", "1024", "1024_cascade", "1536_cascade"))
    generate_textured_parser.add_argument("--seed", type=int, default=TRELLIS2_SHAPE_DEFAULT_SEED)
    generate_textured_parser.add_argument("--max-num-tokens", type=int, default=TRELLIS2_SHAPE_DEFAULT_MAX_NUM_TOKENS)
    generate_textured_parser.add_argument("--decoder-token-limit", type=int, default=TRELLIS2_SHAPE_DEFAULT_DECODER_TOKEN_LIMIT)
    generate_textured_parser.add_argument("--texture-size", type=int, default=TRELLIS2_TEXTURE_DEFAULT_SIZE)
    generate_textured_parser.add_argument("--glb-target-faces", type=int, default=TRELLIS2_GLB_DEFAULT_FACE_TARGET)
    generate_textured_parser.add_argument(
        "--xatlas-face-guard",
        type=_parse_xatlas_face_guard,
        default=TRELLIS2_XATLAS_DEFAULT_FACE_GUARD,
        help="maximum faces allowed into xatlas unwrap; use 'auto' for adaptive headroom",
    )
    generate_textured_parser.add_argument("--texture-bake-backend", choices=TRELLIS2_TEXTURE_BAKE_BACKENDS, default="kdtree")
    generate_textured_parser.add_argument(
        "--xatlas-parallel-chunks",
        type=int,
        default=TRELLIS2_XATLAS_DEFAULT_PARALLEL_CHUNKS,
        help="split large xatlas unwraps into this many spatial chunks; 0 auto-selects by face count",
    )

    args = parser.parse_args(argv)
    root = getattr(args, "command_root", None) or getattr(args, "root_path", None) or args.root
    if args.command == "validate":
        validation = validate_trellis2_assets(root)
        print(f"ready={validation.ready}")
        print(f"present={len(validation.present)}")
        print(f"missing={len(validation.missing)}")
        for path in validation.missing:
            print(f"missing {path}")
        return 0
    if args.command == "rmbg-validate":
        validation = validate_rmbg2_assets(root)
        print(f"ready={validation.ready}")
        print(f"present={len(validation.present)}")
        print(f"missing={len(validation.missing)}")
        for path in validation.missing:
            print(f"missing {path}")
        return 0
    if args.command == "dinov3-validate":
        validation = validate_dinov3_assets(root)
        print(f"ready={validation.ready}")
        print(f"present={len(validation.present)}")
        print(f"missing={len(validation.missing)}")
        for path in validation.missing:
            print(f"missing {path}")
        return 0
    if args.command == "download-command":
        print(" ".join(trellis2_download_command(root)))
        return 0
    if args.command == "rmbg-download-command":
        print(RMBG2_LICENSE_NOTE)
        print(" ".join(rmbg2_download_command(root)))
        return 0
    if args.command == "dinov3-download-command":
        print(DINOv3_ACCESS_NOTE)
        print(" ".join(dinov3_download_command(root)))
        return 0
    if args.command == "inspect":
        for checkpoint_path, infos in inspect_trellis2_checkpoints(root, checkpoint_paths=args.checkpoints).items():
            print(f"checkpoint {checkpoint_path}")
            for info in infos:
                print(f"tensor {info.name} shape={info.shape} dtype={info.dtype}")
        return 0
    if args.command == "probe":
        if args.load:
            for tensor in load_trellis2_probe(root, args.group):
                print(f"tensor {tensor.name} shape={tensor.shape} dtype={tensor.dtype} group={tensor.group}")
        else:
            for info in inspect_trellis2_probe(root, args.group):
                print(f"tensor {info.name} shape={info.shape} dtype={info.dtype}")
        return 0
    if args.command == "attempt-forward-trace":
        from .trellis2_inference import Trellis2InferencePipeline

        report = Trellis2InferencePipeline(
            args.trace_root,
            rmbg_root=args.rmbg_root,
        ).attempt_forward_trace(
            args.image,
            dino_root=args.dino_root,
            slat_steps=args.slat_steps,
            decoder_token_limit=args.decoder_token_limit,
        )
        payload = _jsonable(report)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"completed={report.completed_stages}")
        print(f"outputs={tuple(output.name for output in report.outputs)}")
        print(f"blocker_stage={report.blocker.stage if report.blocker else None}")
        print(f"operation={report.blocker.operation if report.blocker else None}")
        return 0
    if args.command == "generate-shape":
        from .trellis2_inference import Trellis2InferencePipeline

        result = Trellis2InferencePipeline(
            args.shape_root,
            rmbg_root=args.rmbg_root,
        ).generate_shape_obj(
            args.image,
            output_path=args.output,
            dino_root=args.dino_root,
            slat_steps=args.slat_steps,
            pipeline_type=args.pipeline_type,
            seed=args.seed,
            max_num_tokens=args.max_num_tokens,
            decoder_token_limit=args.decoder_token_limit,
            retain_trace_payloads=False,
        )
        report = result.trace
        print(f"completed={report.completed_stages}")
        print(f"outputs={tuple(output.name for output in report.outputs)}")
        if result.artifact is not None:
            print(f"artifact={result.artifact.path}")
            print(f"bytes={result.artifact.bytes_written}")
            return 0
        print(f"blocker_stage={report.blocker.stage if report.blocker else None}")
        print(f"operation={report.blocker.operation if report.blocker else None}")
        print(f"reason={report.blocker.reason if report.blocker else None}")
        return 2
    if args.command == "generate-textured":
        from .trellis2_inference import Trellis2InferencePipeline

        result = Trellis2InferencePipeline(
            args.textured_root,
            rmbg_root=args.rmbg_root,
        ).generate_textured_glb(
            args.image,
            output_path=args.output,
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
        report = result.trace
        print(f"completed={report.completed_stages}")
        print(f"outputs={tuple(output.name for output in report.outputs)}")
        if result.artifact is not None:
            print(f"artifact={result.artifact.path}")
            print(f"bytes={result.artifact.bytes_written}")
            return 0
        print(f"blocker_stage={report.blocker.stage if report.blocker else None}")
        print(f"operation={report.blocker.operation if report.blocker else None}")
        print(f"reason={report.blocker.reason if report.blocker else None}")
        return 2
    return 1


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, mx.array):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype).removeprefix("mlx.core."),
        }
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _validate_root(root: str | Path) -> Path:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"TRELLIS.2 asset root not found: {root_path}")
    return root_path


def _checkpoint_paths(checkpoint_paths: Iterable[str] | None) -> tuple[str, ...]:
    paths = tuple(checkpoint_paths) if checkpoint_paths is not None else tuple(
        path for path in TRELLIS2_ASSETS.required_paths if path.endswith(".safetensors")
    )
    if not paths:
        raise ValueError("checkpoint selection must not be empty")
    for path in paths:
        if not isinstance(path, str) or not path:
            raise ValueError("checkpoint paths must be non-empty strings")
        if not path.endswith(".safetensors"):
            raise ValueError(f"unsupported TRELLIS.2 checkpoint format: {path!r}")
    return paths


def _coerce_group(group: str | Trellis2ProbeGroup) -> Trellis2ProbeGroup:
    if isinstance(group, Trellis2ProbeGroup):
        return group
    if isinstance(group, str):
        return trellis2_probe_group(group)
    raise ValueError("TRELLIS.2 probe group must be a name or Trellis2ProbeGroup")


if __name__ == "__main__":
    raise SystemExit(main())
