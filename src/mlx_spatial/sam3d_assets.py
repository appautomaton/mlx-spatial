"""SAM 3D Objects asset validation and official pipeline inventory."""

from __future__ import annotations

import json
import pickle
import shutil
import tempfile
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from safetensors import SafetensorError
from safetensors.numpy import save_file as save_numpy_safetensors

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint


SAM3D_OBJECTS_REPO_ID = "facebook/sam-3d-objects"
SAM3D_OBJECTS_DEFAULT_ROOT = "weights/sam-3d-objects"
SAM3D_OBJECTS_MLX_DEFAULT_ROOT = "weights/sam-3d-objects-mlx"
SAM3D_OBJECTS_ACCESS_NOTE = (
    "SAM 3D Objects checkpoints are gated on Hugging Face. Request access to "
    "facebook/sam-3d-objects and authenticate with `uv run hf auth login` before downloading."
)

SAM3D_REQUIRED_PIPELINE_PATHS = (
    "pipeline.yaml",
    "checkpoints/pipeline.yaml",
    "checkpoints/hf/pipeline.yaml",
    "hf/pipeline.yaml",
)

SAM3D_REQUIRED_CONFIG_FIELDS = (
    "ss_generator_config_path",
    "slat_generator_config_path",
    "ss_decoder_config_path",
    "slat_decoder_gs_config_path",
    "slat_decoder_mesh_config_path",
)
SAM3D_REQUIRED_CHECKPOINT_FIELDS = (
    "ss_generator_ckpt_path",
    "slat_generator_ckpt_path",
    "ss_decoder_ckpt_path",
    "slat_decoder_gs_ckpt_path",
    "slat_decoder_mesh_ckpt_path",
)
SAM3D_OPTIONAL_PATH_FIELDS = (
    "ss_encoder_config_path",
    "ss_encoder_ckpt_path",
    "slat_decoder_gs_4_config_path",
    "slat_decoder_gs_4_ckpt_path",
)
SAM3D_REQUIRED_RUNTIME_COMPONENTS = (
    "depth_model",
    "ss_generator",
    "ss_decoder",
    "slat_generator",
    "slat_decoder_gs",
)


@dataclass(frozen=True)
class Sam3dAssetValidation:
    """Presence report for a local SAM 3D Objects checkpoint bundle."""

    root: Path
    model_dir: Path
    pipeline_path: Path | None
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class Sam3dAssetBlocker:
    """Structured setup/runtime blocker for SAM3D exact-mode execution."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class Sam3dPipelinePath:
    """Resolved path declared by the official pipeline YAML."""

    role: str
    field: str
    relative_path: str
    path: Path
    exists: bool
    required: bool
    kind: str


@dataclass(frozen=True)
class Sam3dCheckpointInventory:
    """Safetensors metadata for one checkpoint referenced by the pipeline."""

    role: str
    relative_path: str
    path: Path
    format: str
    tensor_count: int
    prefixes: tuple[str, ...]
    sample_tensors: tuple[CheckpointTensorInfo, ...]


@dataclass(frozen=True)
class Sam3dPipelineConfig:
    """Normalized fields from SAM3D's official pipeline.yaml."""

    target: str | None
    dtype: str | None
    rendering_engine: str | None
    decode_formats: tuple[str, ...]
    raw: dict[str, object]


@dataclass(frozen=True)
class Sam3dPipelineInspection:
    """Pipeline/config/checkpoint inventory without constructing PyTorch modules."""

    validation: Sam3dAssetValidation
    config: Sam3dPipelineConfig | None
    paths: tuple[Sam3dPipelinePath, ...]
    checkpoints: tuple[Sam3dCheckpointInventory, ...]
    blocker: Sam3dAssetBlocker | None

    @property
    def ready(self) -> bool:
        return self.blocker is None

    @property
    def missing_paths(self) -> tuple[str, ...]:
        return tuple(item.relative_path for item in self.paths if item.required and not item.exists)


@dataclass(frozen=True)
class Sam3dDownloadResult:
    """Result of an explicit gated Hugging Face checkpoint download attempt."""

    root: Path
    repo_id: str
    local_dir: Path | None
    validation: Sam3dAssetValidation
    blocker: Sam3dAssetBlocker | None

    @property
    def ready(self) -> bool:
        return self.blocker is None and self.validation.ready


@dataclass(frozen=True)
class Sam3dConversionItem:
    """One file copied or converted for the local MLX checkpoint mirror."""

    role: str
    kind: str
    source_path: Path
    output_path: Path
    status: str
    tensor_count: int | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class Sam3dConversionResult:
    """Result of converting official SAM3D assets to a safetensors mirror."""

    source_root: Path
    output_root: Path
    output_pipeline_path: Path | None
    items: tuple[Sam3dConversionItem, ...]
    validation: Sam3dAssetValidation
    blocker: Sam3dAssetBlocker | None

    @property
    def ready(self) -> bool:
        return self.blocker is None and self.validation.ready


def validate_sam3d_assets(root: str | Path = SAM3D_OBJECTS_DEFAULT_ROOT) -> Sam3dAssetValidation:
    """Validate a local SAM 3D Objects root without loading tensors."""

    root_path = Path(root)
    pipeline_path = resolve_sam3d_pipeline_path(root_path)
    if pipeline_path is None:
        return Sam3dAssetValidation(
            root=root_path,
            model_dir=root_path,
            pipeline_path=None,
            present=(),
            missing=("pipeline.yaml or checkpoints/pipeline.yaml or checkpoints/hf/pipeline.yaml",),
        )

    return Sam3dAssetValidation(
        root=root_path,
        model_dir=pipeline_path.parent,
        pipeline_path=pipeline_path,
        present=(_relative_report_path(root_path, pipeline_path),),
        missing=(),
    )


def resolve_sam3d_pipeline_path(root: str | Path) -> Path | None:
    """Resolve supported local SAM3D checkpoint layouts to a pipeline.yaml."""

    root_path = Path(root)
    for relative_path in SAM3D_REQUIRED_PIPELINE_PATHS:
        candidate = root_path / relative_path
        if candidate.is_file():
            return candidate
    return None


def read_sam3d_pipeline_config(path: str | Path) -> Sam3dPipelineConfig:
    """Read official SAM3D pipeline.yaml into the subset needed for MLX routing."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"SAM3D pipeline config not found: {config_path}")
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"unsupported SAM3D pipeline config format: {config_path.suffix or '<none>'}")

    raw = _read_yaml_config(config_path)
    if not isinstance(raw, dict):
        raise ValueError("SAM3D pipeline config must be a mapping")

    decode_formats = raw.get("decode_formats", ("gaussian", "mesh"))
    if isinstance(decode_formats, str):
        formats = (decode_formats,)
    elif isinstance(decode_formats, Sequence):
        formats = tuple(str(item) for item in decode_formats)
    else:
        raise ValueError("SAM3D pipeline decode_formats must be a string or list")

    return Sam3dPipelineConfig(
        target=_optional_str(raw.get("_target_")),
        dtype=_optional_str(raw.get("dtype")),
        rendering_engine=_optional_str(raw.get("rendering_engine")),
        decode_formats=formats,
        raw=raw,
    )


def inspect_sam3d_model_assets(
    root: str | Path = SAM3D_OBJECTS_DEFAULT_ROOT,
    *,
    required_roles: Iterable[str] | None = None,
) -> Sam3dPipelineInspection:
    """Inspect official SAM3D pipeline config and referenced checkpoints before execution."""

    validation = validate_sam3d_assets(root)
    if not validation.ready or validation.pipeline_path is None:
        return Sam3dPipelineInspection(
            validation=validation,
            config=None,
            paths=(),
            checkpoints=(),
            blocker=Sam3dAssetBlocker(
                stage="asset-validation",
                operation="validate SAM 3D Objects checkpoint layout",
                reason="missing required SAM3D pipeline.yaml",
                metadata={"missing": validation.missing, "root": str(validation.root)},
            ),
        )

    try:
        config = read_sam3d_pipeline_config(validation.pipeline_path)
    except ValueError as error:
        return Sam3dPipelineInspection(
            validation=validation,
            config=None,
            paths=(),
            checkpoints=(),
            blocker=Sam3dAssetBlocker(
                stage="pipeline-config",
                operation="parse SAM3D pipeline.yaml",
                reason=str(error),
                metadata={"pipeline": str(validation.pipeline_path)},
            ),
        )

    paths = resolve_sam3d_pipeline_paths(config, validation.model_dir, required_roles=required_roles)
    missing = tuple(item.relative_path for item in paths if item.required and not item.exists)
    if missing:
        return Sam3dPipelineInspection(
            validation=validation,
            config=config,
            paths=paths,
            checkpoints=(),
            blocker=Sam3dAssetBlocker(
                stage="pipeline-config",
                operation="resolve SAM3D pipeline config and checkpoint paths",
                reason="pipeline.yaml references files that are not present under the checkpoint root",
                metadata={"missing": missing, "pipeline": str(validation.pipeline_path)},
            ),
        )

    checkpoint_paths = tuple(
        item
        for item in paths
        if item.kind == "checkpoint" and item.exists and (required_roles is None or item.required)
    )
    unsupported = tuple(item.relative_path for item in checkpoint_paths if item.path.suffix != ".safetensors")
    if unsupported:
        return Sam3dPipelineInspection(
            validation=validation,
            config=config,
            paths=paths,
            checkpoints=(),
            blocker=Sam3dAssetBlocker(
                stage="checkpoint-inspection",
                operation="inspect SAM3D checkpoint metadata without PyTorch",
                reason="SAM3D MLX currently supports safetensors checkpoints only",
                metadata={"unsupported": unsupported},
            ),
        )

    try:
        checkpoints = tuple(_inspect_checkpoint_path(item) for item in checkpoint_paths)
    except (SafetensorError, OSError, ValueError) as error:
        return Sam3dPipelineInspection(
            validation=validation,
            config=config,
            paths=paths,
            checkpoints=(),
            blocker=Sam3dAssetBlocker(
                stage="checkpoint-inspection",
                operation="inspect SAM3D safetensors checkpoint metadata",
                reason=f"checkpoint metadata could not be inspected: {error}",
                metadata={"error_type": type(error).__name__, "error": str(error)},
            ),
        )

    return Sam3dPipelineInspection(
        validation=validation,
        config=config,
        paths=paths,
        checkpoints=checkpoints,
        blocker=None,
    )


def resolve_sam3d_pipeline_paths(
    config: Sam3dPipelineConfig,
    model_dir: str | Path,
    *,
    required_roles: Iterable[str] | None = None,
) -> tuple[Sam3dPipelinePath, ...]:
    """Resolve config/checkpoint paths declared by the official SAM3D pipeline config."""

    root = Path(model_dir)
    required_role_set = _required_role_set(required_roles)
    paths: list[Sam3dPipelinePath] = []
    for field in SAM3D_REQUIRED_CONFIG_FIELDS:
        paths.append(_path_item(config.raw, root, field, required=_field_role(field) in required_role_set, kind="config"))
    for field in SAM3D_REQUIRED_CHECKPOINT_FIELDS:
        paths.append(_path_item(config.raw, root, field, required=_field_role(field) in required_role_set, kind="checkpoint"))
    for field in SAM3D_OPTIONAL_PATH_FIELDS:
        value = config.raw.get(field)
        if value not in {None, ""}:
            paths.append(_path_item(config.raw, root, field, required=False, kind=_field_kind(field)))
    return tuple(paths)


def sam3d_download_command(root: str | Path = SAM3D_OBJECTS_DEFAULT_ROOT) -> tuple[str, ...]:
    """Return the dev-environment HF command for downloading gated SAM3D assets."""

    return (
        "uv",
        "run",
        "hf",
        "download",
        "--repo-type",
        "model",
        "--local-dir",
        str(root),
        "--max-workers",
        "1",
        SAM3D_OBJECTS_REPO_ID,
    )


def download_sam3d_assets(
    root: str | Path = SAM3D_OBJECTS_DEFAULT_ROOT,
    *,
    repo_id: str = SAM3D_OBJECTS_REPO_ID,
    max_workers: int = 1,
) -> Sam3dDownloadResult:
    """Explicitly download gated SAM3D assets, then validate the local layout."""

    root_path = Path(root)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        validation = validate_sam3d_assets(root_path)
        return Sam3dDownloadResult(
            root=root_path,
            repo_id=repo_id,
            local_dir=None,
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="asset-download",
                operation="download gated SAM3D checkpoints from Hugging Face",
                reason="huggingface-hub is not installed; run `uv sync --dev` or use `uv run hf download`",
                metadata={"error": str(error), "command": sam3d_download_command(root_path)},
            ),
        )

    try:
        local_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(root_path),
                max_workers=max_workers,
            )
        )
    except Exception as error:  # pragma: no cover - exact HF exception depends on auth/network state.
        validation = validate_sam3d_assets(root_path)
        return Sam3dDownloadResult(
            root=root_path,
            repo_id=repo_id,
            local_dir=None,
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="asset-download",
                operation="download gated SAM3D checkpoints from Hugging Face",
                reason=f"Hugging Face download failed: {error}",
                metadata={"error_type": type(error).__name__, "error": str(error), "repo_id": repo_id},
            ),
        )

    validation = validate_sam3d_assets(root_path)
    blocker = None
    if not validation.ready:
        blocker = Sam3dAssetBlocker(
            stage="asset-validation",
            operation="validate downloaded SAM3D checkpoint layout",
            reason="download completed but required SAM3D files are still missing",
            metadata={"missing": validation.missing, "local_dir": str(local_dir)},
        )
    return Sam3dDownloadResult(
        root=root_path,
        repo_id=repo_id,
        local_dir=local_dir,
        validation=validation,
        blocker=blocker,
    )


def convert_sam3d_assets_to_safetensors(
    root: str | Path = SAM3D_OBJECTS_DEFAULT_ROOT,
    *,
    output_root: str | Path = SAM3D_OBJECTS_MLX_DEFAULT_ROOT,
    overwrite: bool = False,
    max_archive_bytes: int | None = 16 * 1024**3,
    max_tensor_bytes: int | None = 16 * 1024**3,
) -> Sam3dConversionResult:
    """Convert official SAM3D `.ckpt`/`.pt` checkpoints to a safetensors mirror.

    The official checkpoint bundle is PyTorch-zip based. This function keeps that
    format out of the runtime path by creating a separate local tree whose
    pipeline.yaml references only `.safetensors` checkpoints.
    """

    source_root = Path(root)
    target_root = Path(output_root)
    source_validation = validate_sam3d_assets(source_root)
    if not source_validation.ready or source_validation.pipeline_path is None:
        validation = validate_sam3d_assets(target_root)
        return Sam3dConversionResult(
            source_root=source_root,
            output_root=target_root,
            output_pipeline_path=None,
            items=(),
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="asset-validation",
                operation="convert SAM3D checkpoint layout to safetensors",
                reason="missing required source SAM3D pipeline.yaml",
                metadata={"missing": source_validation.missing, "source_root": str(source_root)},
            ),
        )

    try:
        config = read_sam3d_pipeline_config(source_validation.pipeline_path)
    except ValueError as error:
        validation = validate_sam3d_assets(target_root)
        return Sam3dConversionResult(
            source_root=source_root,
            output_root=target_root,
            output_pipeline_path=None,
            items=(),
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="pipeline-config",
                operation="parse source SAM3D pipeline.yaml for safetensors conversion",
                reason=str(error),
                metadata={"pipeline": str(source_validation.pipeline_path)},
            ),
        )

    paths = resolve_sam3d_pipeline_paths(config, source_validation.model_dir)
    missing = tuple(item.relative_path for item in paths if item.required and not item.exists)
    if missing:
        validation = validate_sam3d_assets(target_root)
        return Sam3dConversionResult(
            source_root=source_root,
            output_root=target_root,
            output_pipeline_path=None,
            items=(),
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="pipeline-config",
                operation="resolve source SAM3D files for safetensors conversion",
                reason="pipeline.yaml references files that are not present under the source checkpoint root",
                metadata={"missing": missing, "pipeline": str(source_validation.pipeline_path)},
            ),
        )

    try:
        from pt_loader import PtCheckpoint  # type: ignore
    except ModuleNotFoundError as error:
        validation = validate_sam3d_assets(target_root)
        return Sam3dConversionResult(
            source_root=source_root,
            output_root=target_root,
            output_pipeline_path=None,
            items=(),
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="checkpoint-conversion",
                operation="convert PyTorch zip checkpoints to safetensors without torch",
                reason="pt-safe-loader is not installed; run `uv sync --dev` or `uv add --dev pt-safe-loader`",
                metadata={"error": str(error)},
            ),
        )

    output_model_dir = _mirrored_model_dir(source_validation, target_root)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    converted_raw = dict(config.raw)
    items: list[Sam3dConversionItem] = []

    try:
        for item in paths:
            if item.kind == "config":
                output_path = output_model_dir / item.relative_path
                _copy_file(item.path, output_path, overwrite=overwrite)
                items.append(
                    Sam3dConversionItem(
                        role=item.role,
                        kind=item.kind,
                        source_path=item.path,
                        output_path=output_path,
                        status="copied",
                    )
                )
                continue

            output_relative = Path(item.relative_path).with_suffix(".safetensors").as_posix()
            output_path = output_model_dir / output_relative
            converted_raw[item.field] = output_relative
            conversion = _convert_or_copy_checkpoint(
                item.path,
                output_path,
                role=item.role,
                overwrite=overwrite,
                pt_checkpoint_cls=PtCheckpoint,
                max_archive_bytes=max_archive_bytes,
                max_tensor_bytes=max_tensor_bytes,
            )
            items.append(conversion)

        output_pipeline_path = output_model_dir / "pipeline.yaml"
        _write_yaml_config(output_pipeline_path, converted_raw)
        items.append(
            Sam3dConversionItem(
                role="pipeline",
                kind="config",
                source_path=source_validation.pipeline_path,
                output_path=output_pipeline_path,
                status="rewritten",
            )
        )
        _write_conversion_manifest(target_root, source_root, output_pipeline_path, items)
    except Exception as error:  # pragma: no cover - exact converter errors are format/version dependent.
        validation = validate_sam3d_assets(target_root)
        return Sam3dConversionResult(
            source_root=source_root,
            output_root=target_root,
            output_pipeline_path=None,
            items=tuple(items),
            validation=validation,
            blocker=Sam3dAssetBlocker(
                stage="checkpoint-conversion",
                operation="convert SAM3D checkpoints to safetensors",
                reason=f"checkpoint conversion failed: {error}",
                metadata={"error_type": type(error).__name__, "error": str(error)},
            ),
        )

    validation = validate_sam3d_assets(target_root)
    return Sam3dConversionResult(
        source_root=source_root,
        output_root=target_root,
        output_pipeline_path=output_pipeline_path,
        items=tuple(items),
        validation=validation,
        blocker=None if validation.ready else Sam3dAssetBlocker(
            stage="asset-validation",
            operation="validate converted SAM3D safetensors mirror",
            reason="converted tree is missing required SAM3D files",
            metadata={"missing": validation.missing, "output_root": str(target_root)},
        ),
    )


def convert_torch_checkpoint_to_safetensors(
    source_path: str | Path,
    output_path: str | Path,
    *,
    role: str = "checkpoint",
    overwrite: bool = False,
    max_archive_bytes: int | None = 16 * 1024**3,
    max_tensor_bytes: int | None = 16 * 1024**3,
) -> Sam3dConversionItem:
    """Convert a standalone PyTorch zip checkpoint to safetensors without torch."""

    try:
        from pt_loader import PtCheckpoint  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "pt-safe-loader is required for PyTorch checkpoint conversion; run `uv sync --dev`"
        ) from error
    return _convert_or_copy_checkpoint(
        Path(source_path),
        Path(output_path),
        role=role,
        overwrite=overwrite,
        pt_checkpoint_cls=PtCheckpoint,
        max_archive_bytes=max_archive_bytes,
        max_tensor_bytes=max_tensor_bytes,
    )


def _inspect_checkpoint_path(path_item: Sam3dPipelinePath) -> Sam3dCheckpointInventory:
    infos = inspect_checkpoint(path_item.path)
    prefixes = _top_level_prefixes(info.name for info in infos)
    return Sam3dCheckpointInventory(
        role=path_item.role,
        relative_path=path_item.relative_path,
        path=path_item.path,
        format=path_item.path.suffix.removeprefix("."),
        tensor_count=len(infos),
        prefixes=prefixes,
        sample_tensors=infos[:10],
    )


def _mirrored_model_dir(source_validation: Sam3dAssetValidation, output_root: Path) -> Path:
    try:
        relative_parent = source_validation.pipeline_path.parent.relative_to(source_validation.root)  # type: ignore[union-attr]
    except ValueError:
        relative_parent = Path()
    return output_root / relative_parent


def _copy_file(source: Path, output: Path, *, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)


def _convert_or_copy_checkpoint(
    source: Path,
    output: Path,
    *,
    role: str,
    overwrite: bool,
    pt_checkpoint_cls: object,
    max_archive_bytes: int | None,
    max_tensor_bytes: int | None,
) -> Sam3dConversionItem:
    if output.exists() and output.stat().st_size > 0 and not overwrite:
        try:
            tensor_count = len(inspect_checkpoint(output))
        except Exception:
            tensor_count = None
        return Sam3dConversionItem(role=role, kind="checkpoint", source_path=source, output_path=output, status="exists", tensor_count=tensor_count)

    if source.suffix == ".safetensors":
        _copy_file(source, output, overwrite=overwrite)
        return Sam3dConversionItem(
            role=role,
            kind="checkpoint",
            source_path=source,
            output_path=output,
            status="copied",
            tensor_count=len(inspect_checkpoint(output)),
        )

    if source.suffix not in {".ckpt", ".pt", ".pth", ".bin"}:
        raise ValueError(f"unsupported SAM3D checkpoint conversion format: {source.suffix or '<none>'}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlx-spatial-sam3d-convert-") as tmp:
        try:
            checkpoint = pt_checkpoint_cls.load(  # type: ignore[attr-defined]
                str(source),
                max_archive_bytes=max_archive_bytes,
                max_tensor_bytes=max_tensor_bytes,
            )
            result = checkpoint.export(format="safetensors", dir=tmp)
            produced_weights = Path(result["weights_path"])
            if output.exists():
                output.unlink()
            shutil.move(str(produced_weights), output)
            metadata_path = result.get("metadata_path")
            if metadata_path is not None:
                metadata_output = output.parent / "conversion_metadata" / f"{output.stem}.yaml"
                metadata_output.parent.mkdir(parents=True, exist_ok=True)
                if metadata_output.exists() and overwrite:
                    metadata_output.unlink()
                if not metadata_output.exists():
                    shutil.move(str(metadata_path), metadata_output)
        except Exception:
            arrays = _load_restricted_torch_zip_state_dict(source, max_archive_bytes=max_archive_bytes)
            if output.exists():
                output.unlink()
            save_numpy_safetensors(arrays, output)
            metadata_output = output.parent / "conversion_metadata" / f"{output.stem}.yaml"
            metadata_output.parent.mkdir(parents=True, exist_ok=True)
            metadata_output.write_text(
                f"source_sha256: {_sha256_file(source)}\n"
                "converter: mlx-spatial-restricted-torch-zip\n"
                f"tensor_count: {len(arrays)}\n",
                encoding="utf-8",
            )
    return Sam3dConversionItem(
        role=role,
        kind="checkpoint",
        source_path=source,
        output_path=output,
        status="converted",
        tensor_count=len(inspect_checkpoint(output)),
        source_sha256=_safe_source_sha256_from_metadata(output.parent / "conversion_metadata" / f"{output.stem}.yaml"),
    )


def _safe_source_sha256_from_metadata(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("source_sha256:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return None


class _TorchStorageType:
    def __init__(self, name: str):
        self.name = name


@dataclass(frozen=True)
class _TorchStorageRef:
    storage_type: _TorchStorageType
    key: str
    location: str
    size: int


@dataclass(frozen=True)
class _TorchTensorRef:
    storage: _TorchStorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor_v2(
    storage: _TorchStorageRef,
    storage_offset: int,
    size: Sequence[int],
    stride: Sequence[int],
    requires_grad: bool,
    backward_hooks: object,
) -> _TorchTensorRef:
    del requires_grad, backward_hooks
    return _TorchTensorRef(
        storage=storage,
        storage_offset=int(storage_offset),
        size=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
    )


class _RestrictedTorchZipUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _TorchStorageType(name)
        raise pickle.UnpicklingError(f"unsupported PyTorch checkpoint global: {module}.{name}")

    def persistent_load(self, persistent_id: object) -> _TorchStorageRef:
        if not isinstance(persistent_id, tuple) or len(persistent_id) < 5:
            raise pickle.UnpicklingError(f"unsupported persistent id: {persistent_id!r}")
        tag, storage_type, key, location, size = persistent_id[:5]
        if tag != "storage" or not isinstance(storage_type, _TorchStorageType):
            raise pickle.UnpicklingError(f"unsupported storage persistent id: {persistent_id!r}")
        return _TorchStorageRef(
            storage_type=storage_type,
            key=str(key),
            location=str(location),
            size=int(size),
        )


_TORCH_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


def _load_restricted_torch_zip_state_dict(
    source: Path,
    *,
    max_archive_bytes: int | None,
) -> dict[str, np.ndarray]:
    if max_archive_bytes is not None and source.stat().st_size > max_archive_bytes:
        raise ValueError(f"archive is {source.stat().st_size} bytes, limit is {max_archive_bytes}")
    with zipfile.ZipFile(source) as archive:
        data_pkl = _torch_zip_data_pickle_name(archive)
        prefix = data_pkl[: -len("/data.pkl")] if data_pkl.endswith("/data.pkl") else ""
        root = _RestrictedTorchZipUnpickler(archive.open(data_pkl)).load()
        state_dict = _extract_state_dict(root)
        arrays: dict[str, np.ndarray] = {}
        storage_cache: dict[tuple[str, str], np.ndarray] = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, _TorchTensorRef):
                arrays[str(key)] = _tensor_ref_to_numpy(archive, prefix, tensor, storage_cache)
        if not arrays:
            raise ValueError("restricted PyTorch checkpoint parser found no tensor state_dict entries")
        return arrays


def _torch_zip_data_pickle_name(archive: zipfile.ZipFile) -> str:
    candidates = sorted(name for name in archive.namelist() if name.endswith("data.pkl"))
    if not candidates:
        raise ValueError("PyTorch zip checkpoint does not contain data.pkl")
    return candidates[0]


def _extract_state_dict(root: object) -> OrderedDict[str, object] | dict[str, object]:
    if isinstance(root, OrderedDict):
        return root
    if isinstance(root, dict):
        state_dict = root.get("state_dict")
        if isinstance(state_dict, OrderedDict | dict):
            return state_dict
        model = root.get("model")
        if isinstance(model, OrderedDict | dict):
            return model
        if all(isinstance(value, _TorchTensorRef) for value in root.values()):
            return root
    raise ValueError(f"unsupported PyTorch checkpoint root object: {type(root).__name__}")


def _tensor_ref_to_numpy(
    archive: zipfile.ZipFile,
    prefix: str,
    tensor: _TorchTensorRef,
    storage_cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    dtype = _TORCH_STORAGE_DTYPES.get(tensor.storage.storage_type.name)
    if dtype is None:
        raise ValueError(f"unsupported PyTorch storage type: {tensor.storage.storage_type.name}")
    cache_key = (tensor.storage.storage_type.name, tensor.storage.key)
    if cache_key not in storage_cache:
        member = f"{prefix}/data/{tensor.storage.key}" if prefix else f"data/{tensor.storage.key}"
        raw = archive.read(member)
        storage_cache[cache_key] = np.frombuffer(raw, dtype=dtype)
    storage = storage_cache[cache_key]
    offset = tensor.storage_offset
    if not tensor.size:
        return np.array(storage[offset], dtype=dtype)
    byte_strides = tuple(stride * dtype.itemsize for stride in tensor.stride)
    view = np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=tensor.size,
        strides=byte_strides,
        writeable=False,
    )
    return np.array(view, copy=True)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_conversion_manifest(
    output_root: Path,
    source_root: Path,
    output_pipeline_path: Path,
    items: Sequence[Sam3dConversionItem],
) -> None:
    payload = {
        "source_root": str(source_root),
        "output_pipeline_path": str(output_pipeline_path),
        "items": [
            {
                "role": item.role,
                "kind": item.kind,
                "source_path": str(item.source_path),
                "output_path": str(item.output_path),
                "status": item.status,
                "tensor_count": item.tensor_count,
                "source_sha256": item.source_sha256,
            }
            for item in items
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _path_item(
    raw: dict[str, object],
    root: Path,
    field: str,
    *,
    required: bool,
    kind: str,
) -> Sam3dPipelinePath:
    value = raw.get(field)
    if value is None:
        relative = ""
        path = root / f"<missing:{field}>"
        exists = False
    else:
        relative = str(value)
        path = root / relative
        exists = path.is_file()
    role = field.removesuffix("_config_path").removesuffix("_ckpt_path")
    return Sam3dPipelinePath(
        role=role,
        field=field,
        relative_path=relative or field,
        path=path,
        exists=exists,
        required=required,
        kind=kind,
    )


def _field_kind(field: str) -> str:
    if field.endswith("_ckpt_path"):
        return "checkpoint"
    return "config"


def _field_role(field: str) -> str:
    return field.removesuffix("_config_path").removesuffix("_ckpt_path")


def _required_role_set(required_roles: Iterable[str] | None) -> set[str]:
    if required_roles is None:
        return {
            _field_role(field)
            for field in (*SAM3D_REQUIRED_CONFIG_FIELDS, *SAM3D_REQUIRED_CHECKPOINT_FIELDS)
        }
    return {str(role) for role in required_roles}


def _top_level_prefixes(keys: Iterable[str]) -> tuple[str, ...]:
    prefixes = sorted({key.split(".", 1)[0] for key in keys if key})
    return tuple(prefixes)


def _relative_report_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _read_yaml_config(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as error:  # pragma: no cover - PyYAML is present in uv dev env.
        raise ValueError("PyYAML is required to parse official SAM3D pipeline.yaml") from error

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover - exact PyYAML exception varies by version.
        raise ValueError(f"SAM3D YAML config is invalid: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("SAM3D YAML config must be a mapping")
    return raw


def _write_yaml_config(path: Path, raw: dict[str, object]) -> None:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as error:  # pragma: no cover - PyYAML is present in uv dev env.
        raise ValueError("PyYAML is required to write converted SAM3D pipeline.yaml") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
