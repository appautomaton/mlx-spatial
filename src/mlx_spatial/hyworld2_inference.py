"""Staged HY-World-2.0 WorldMirror inference orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from .checkpoint import load_checkpoint_tensors
from .hyworld2_assets import HYWORLD2_DEFAULT_ROOT, inspect_hyworld2_model_assets, validate_hyworld2_assets
from .hyworld2_export import (
    export_hyworld2_cameras,
    export_hyworld2_depth,
    export_hyworld2_gaussian_attributes,
    export_hyworld2_normals,
    export_hyworld2_points_ply,
    write_hyworld2_trace,
)
from .hyworld2_heads import (
    CameraHeadConfig,
    DPTHeadConfig,
    GaussianAttributeConfig,
    run_camera_head,
    run_dpt_head,
    run_gaussian_attribute_head,
)
from .hyworld2_preprocess import memory_profile_config, preprocess_hyworld2_images
from .hyworld2_worldmirror import VisualGeometryTransformerConfig, run_visual_geometry_transformer
from .mlx_memory import clear_mlx_cache, mlx_memory_snapshot, reset_mlx_peak_memory


HYWORLD2_SUPPORTED_HEADS = ("camera", "depth", "normal", "points", "gs")
HYWORLD2_DEFAULT_HEADS = ("depth", "normal", "points")
HYWORLD2_MEMORY_PROFILES = ("safe", "balanced", "large")
HYWORLD2_FIXTURE_EXPORT_ARTIFACTS = ("camera", "depth", "normal", "points", "gaussian", "trace.json")
HYWORLD2_ROOT_EXPORT_ARTIFACTS = ("camera_params.json", "gaussians.ply")
HYWORLD2_REAL_HEAD_PREFIXES = {
    "camera": "cam_head.",
    "depth": "depth_head.",
    "normal": "norm_head.",
    "points": "pts_head.",
    "gs": "gs_head.",
}


@dataclass(frozen=True)
class HyWorld2Blocker:
    """Structured blocker for exact HY-World execution."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HyWorld2StageOutput:
    """Concrete output artifact metadata."""

    name: str
    path: Path
    kind: str


@dataclass(frozen=True)
class HyWorld2ReconstructionTrace:
    """Trace metadata for one staged reconstruction attempt."""

    completed_stages: tuple[str, ...]
    outputs: tuple[HyWorld2StageOutput, ...]
    blocker: HyWorld2Blocker | None
    requested_heads: tuple[str, ...]
    enabled_heads: tuple[str, ...]
    memory_profile: str
    model_root: Path
    model_dir: Path
    checkpoint_ready: bool
    config_kind: str | None
    input_path: Path
    output_path: Path
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class HyWorld2ReconstructionResult:
    """Result wrapper for a reconstruction attempt."""

    trace: HyWorld2ReconstructionTrace
    output_dir: Path | None = None


class HyWorld2InferencePipeline:
    """Small staged pipeline shell for exact WorldMirror MLX inference."""

    def __init__(self, root: str | Path = HYWORLD2_DEFAULT_ROOT):
        self.root = Path(root)

    def reconstruct(
        self,
        input_path: str | Path,
        *,
        output_path: str | Path,
        heads: Sequence[str] = HYWORLD2_DEFAULT_HEADS,
        memory_profile: str = "balanced",
        fixture_tensors: bool = False,
    ) -> HyWorld2ReconstructionResult:
        requested_heads = normalize_hyworld2_heads(heads)
        profile = memory_profile_config(memory_profile)
        reset_mlx_peak_memory()
        memory_snapshots: dict[str, dict[str, int | None]] = {"start": mlx_memory_snapshot().as_dict()}
        head_status = _initial_head_status(requested_heads)

        input_root = Path(input_path)
        output_dir = Path(output_path)
        validation = validate_hyworld2_assets(self.root)
        completed: list[str] = []

        path_blocker = validate_hyworld2_output_path(output_dir)
        if path_blocker is not None:
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=path_blocker,
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={"memory_profile": _profile_metadata(profile), "heads": head_status},
                )
            )

        if not validation.ready:
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=HyWorld2Blocker(
                        stage="asset-validation",
                        operation="validate HY-World-2.0 WorldMirror checkpoint layout",
                        reason="missing required WorldMirror checkpoint assets",
                        metadata={"missing": validation.missing},
                    ),
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={"memory_profile": _profile_metadata(profile), "heads": head_status},
                )
            )
        completed.append("asset-validation")

        if not input_root.exists():
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=HyWorld2Blocker(
                        stage="input-discovery",
                        operation="discover HY-World input images or frames",
                        reason=f"input path not found: {input_root}",
                    ),
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={"memory_profile": _profile_metadata(profile), "heads": head_status},
                )
            )
        completed.append("input-discovery")

        try:
            preprocessed = preprocess_hyworld2_images(input_root, memory_profile=memory_profile)
        except (FileNotFoundError, ValueError) as error:
            stage = "input-discovery" if isinstance(error, FileNotFoundError) or "no supported images" in str(error) else "image-preprocessing"
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=HyWorld2Blocker(
                        stage=stage,
                        operation="prepare HY-World images as MLX tensor",
                        reason=str(error),
                    ),
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={"memory_profile": _profile_metadata(profile), "heads": head_status},
                )
            )
        completed.append("image-preprocessing")
        _release_hyworld_mlx_stage(memory_snapshots, "after-image-preprocessing")

        inspection = inspect_hyworld2_model_assets(self.root, requested_heads=requested_heads)
        inspection_metadata = _inspection_metadata(inspection)
        if inspection.blocker is not None:
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=HyWorld2Blocker(
                        stage=inspection.blocker.stage,
                        operation=inspection.blocker.operation,
                        reason=inspection.blocker.reason,
                        metadata=inspection.blocker.metadata,
                    ),
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={
                        "memory_profile": _profile_metadata(profile),
                        "input": _preprocess_metadata(preprocessed),
                        "checkpoint": inspection_metadata,
                        "heads": head_status,
                    },
                )
            )
        completed.append("checkpoint-inspection")

        if "gs" in requested_heads:
            head_status["gs"] = {
                "requested": True,
                "enabled": False,
                "export": False,
                "reason": "pending official GS DPT and Gaussian PLY export",
            }

        model_config = inspection.config
        if model_config is None:
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=HyWorld2Blocker(
                        stage="model-construction",
                        operation="resolve HY-World model config",
                        reason="checkpoint inspection did not return a model config",
                    ),
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={
                        "memory_profile": _profile_metadata(profile),
                        "input": _preprocess_metadata(preprocessed),
                        "checkpoint": inspection_metadata,
                        "heads": head_status,
                        "fixture_tensors": fixture_tensors,
                    },
                )
            )

        real_tensors = None
        if not fixture_tensors:
            try:
                real_tensors = _load_real_hyworld2_tensors(validation.checkpoint_path, requested_heads)
                _release_hyworld_mlx_stage(memory_snapshots, "after-checkpoint-load")
            except (OSError, ValueError) as error:
                return HyWorld2ReconstructionResult(
                    trace=self._trace(
                        completed,
                        blocker=HyWorld2Blocker(
                            stage="model-construction",
                            operation="load HY-World real checkpoint tensors into MLX",
                            reason=str(error),
                            metadata={"checkpoint": str(validation.checkpoint_path)},
                        ),
                        requested_heads=requested_heads,
                        enabled_heads=(),
                        memory_profile=memory_profile,
                        validation=validation,
                        input_path=input_root,
                        output_path=output_dir,
                        metadata={
                            "memory_profile": _profile_metadata(profile),
                            "input": _preprocess_metadata(preprocessed),
                            "checkpoint": inspection_metadata,
                            "heads": head_status,
                            "fixture_tensors": False,
                        },
                    )
                )
        completed.append("model-construction")

        intermediate_layers = (
            tuple(index for index in (4, 11, 17, 23) if index < int(model_config.depth))
            if not fixture_tensors
            else tuple(range(min(int(model_config.depth), 4)))
        )
        transformer_config = VisualGeometryTransformerConfig.from_model_config(
            model_config,
            max_tokens=_fixture_max_tokens(preprocessed),
            max_attention_bytes=profile.activation_guard_bytes,
            max_fixture_bytes=64_000_000,
            intermediate_layers=intermediate_layers,
        )
        transformer_tensors = None if fixture_tensors else real_tensors["visual_transformer"]
        backbone = run_visual_geometry_transformer(preprocessed.tensor, transformer_config, transformer_tensors)
        if backbone.blocker is not None:
            return HyWorld2ReconstructionResult(
                trace=self._trace(
                    completed,
                    blocker=_pipeline_blocker(backbone.blocker),
                    requested_heads=requested_heads,
                    enabled_heads=(),
                    memory_profile=memory_profile,
                    validation=validation,
                    input_path=input_root,
                    output_path=output_dir,
                    metadata={
                        "memory_profile": _profile_metadata(profile),
                        "input": _preprocess_metadata(preprocessed),
                        "checkpoint": inspection_metadata,
                        "heads": head_status,
                        "fixture_tensors": fixture_tensors,
                    },
                )
            )
        completed.append("visual-transformer")
        if not fixture_tensors and real_tensors is not None:
            real_tensors["visual_transformer"] = {}
        del transformer_tensors
        _release_hyworld_mlx_stage(memory_snapshots, "after-visual-transformer")

        head_outputs: dict[str, object] = {}
        enabled_heads: list[str] = []
        real_head_tensors = {} if fixture_tensors else real_tensors["heads"]
        for head in requested_heads:
            head_status[head]["enabled"] = True
            if head == "camera":
                head_tensors = real_head_tensors.get("camera") if not fixture_tensors else None
                output = run_camera_head(
                    backbone,
                    CameraHeadConfig(),
                    tensors=head_tensors,
                )
            elif head == "depth":
                head_tensors = real_head_tensors.get("depth") if not fixture_tensors else None
                output = run_dpt_head(
                    backbone,
                    preprocessed.tensor,
                    DPTHeadConfig(
                        head_type="depth",
                        attr_channels=1,
                        activation="exp+expp1+linear",
                        enable_depth_mask=bool(model_config.enable_depth_mask),
                    ),
                    tensors=head_tensors,
                    frames_chunk_size=1,
                )
            elif head == "normal":
                head_tensors = real_head_tensors.get("normal") if not fixture_tensors else None
                output = run_dpt_head(
                    backbone,
                    preprocessed.tensor,
                    DPTHeadConfig(head_type="normal", attr_channels=3, activation="norm+expp1"),
                    tensors=head_tensors,
                    frames_chunk_size=1,
                )
            elif head == "points":
                head_tensors = real_head_tensors.get("points") if not fixture_tensors else None
                output = run_dpt_head(
                    backbone,
                    preprocessed.tensor,
                    DPTHeadConfig(head_type="points", attr_channels=3, activation="inv_log+expp1"),
                    tensors=head_tensors,
                    frames_chunk_size=1,
                )
            elif head == "gs":
                head_tensors = real_head_tensors.get("gs") if not fixture_tensors else None
                renderer_tensors = real_head_tensors.get("gs_renderer") if not fixture_tensors else None
                output = run_gaussian_attribute_head(
                    backbone,
                    preprocessed.tensor,
                    GaussianAttributeConfig(enable_depth_mask=bool(model_config.enable_depth_mask)),
                    tensors=head_tensors,
                    renderer_tensors=renderer_tensors,
                    frames_chunk_size=1,
                )
            else:
                continue
            blocker = getattr(output, "blocker", None)
            if blocker is not None:
                head_status[head]["enabled"] = False
                head_status[head]["reason"] = getattr(blocker, "reason", "head execution blocked")
                return HyWorld2ReconstructionResult(
                    trace=self._trace(
                        completed,
                        blocker=_pipeline_blocker(blocker),
                        requested_heads=requested_heads,
                        enabled_heads=tuple(enabled_heads),
                        memory_profile=memory_profile,
                        validation=validation,
                        input_path=input_root,
                        output_path=output_dir,
                        metadata={
                            "memory_profile": _profile_metadata(profile),
                            "input": _preprocess_metadata(preprocessed),
                            "checkpoint": inspection_metadata,
                            "heads": head_status,
                            "fixture_tensors": fixture_tensors,
                        },
                    )
                )
            head_outputs[head] = output
            enabled_heads.append(head)
            if not fixture_tensors:
                real_head_tensors.pop(head, None)
                if head == "gs":
                    real_head_tensors.pop("gs_renderer", None)
                del head_tensors
                if head == "gs":
                    del renderer_tensors
            _release_hyworld_mlx_stage(memory_snapshots, f"after-head-{head}")
        completed.append("head-execution")
        if not fixture_tensors:
            del real_head_tensors, real_tensors
        _release_hyworld_mlx_stage(memory_snapshots, "after-head-execution")

        _clean_fixture_export_outputs(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_records: list[HyWorld2StageOutput] = []
        for head in requested_heads:
            if head == "depth" and head in head_outputs:
                depth = head_outputs[head]
                output_records.extend(
                    _stage_outputs(export_hyworld2_depth(output_dir, depth.values, depth.confidence))
                )
            elif head == "normal" and head in head_outputs:
                normal = head_outputs[head]
                output_records.extend(
                    _stage_outputs(export_hyworld2_normals(output_dir, normal.values, normal.confidence))
                )
            elif head == "camera" and head in head_outputs:
                camera = head_outputs[head]
                output_records.extend(
                    _stage_outputs(
                        export_hyworld2_cameras(
                            output_dir,
                            camera.camera_params,
                            image_size=(int(preprocessed.tensor.shape[3]), int(preprocessed.tensor.shape[4])),
                            image_paths=preprocessed.image_paths,
                        )
                    )
                )
            elif head == "points" and head in head_outputs:
                points = head_outputs[head]
                output_records.extend(
                    _stage_outputs(export_hyworld2_points_ply(output_dir, points.values, preprocessed.tensor))
                )
            elif head == "gs" and head in head_outputs:
                gaussian = head_outputs[head]
                output_records.extend(
                    _stage_outputs(
                        export_hyworld2_gaussian_attributes(
                            output_dir,
                            features=gaussian.features,
                            depth=gaussian.depth,
                            confidence=gaussian.confidence,
                            raw_params=gaussian.raw_params,
                            image_tensor=preprocessed.tensor,
                            camera_params=head_outputs["camera"].camera_params
                            if "camera" in head_outputs
                            else None,
                            points=head_outputs["points"].values
                            if "points" in head_outputs
                            else None,
                            depth_mask_logits=gaussian.depth_mask_logits,
                        )
                    )
                )
            if head in head_outputs:
                head_status[head]["export"] = True
                head_status[head]["reason"] = "exported"
        completed.append("export")

        blocker = None

        trace = self._trace(
            completed,
            blocker=blocker,
            requested_heads=requested_heads,
            enabled_heads=tuple(enabled_heads),
            memory_profile=memory_profile,
            validation=validation,
            input_path=input_root,
            output_path=output_dir,
            outputs=tuple(output_records),
            metadata={
                "memory_profile": _profile_metadata(profile),
                "input": _preprocess_metadata(preprocessed),
                "checkpoint": inspection_metadata,
                "heads": head_status,
                "fixture_tensors": fixture_tensors,
                "visual_transformer": {
                    "intermediate_layers": intermediate_layers,
                    "attention_modes": backbone.attention_modes,
                    "real_tensors": not fixture_tensors,
                },
                "gaussian": _gaussian_metadata(requested_heads, head_outputs),
                "mlx_memory": memory_snapshots,
            },
        )
        trace_record = write_hyworld2_trace(output_dir, _trace_payload(trace))
        output_records.append(_stage_output(trace_record))
        trace = self._trace(
            completed,
            blocker=blocker,
            requested_heads=requested_heads,
            enabled_heads=tuple(enabled_heads),
            memory_profile=memory_profile,
            validation=validation,
            input_path=input_root,
            output_path=output_dir,
            outputs=tuple(output_records),
            metadata=trace.metadata,
        )
        return HyWorld2ReconstructionResult(
            trace=trace,
            output_dir=output_dir,
        )

    def _trace(
        self,
        completed_stages: Sequence[str],
        *,
        blocker: HyWorld2Blocker | None,
        requested_heads: tuple[str, ...],
        enabled_heads: tuple[str, ...],
        memory_profile: str,
        validation,
        input_path: Path,
        output_path: Path,
        outputs: Sequence[HyWorld2StageOutput] = (),
        metadata: dict[str, object] | None = None,
    ) -> HyWorld2ReconstructionTrace:
        return HyWorld2ReconstructionTrace(
            completed_stages=tuple(completed_stages),
            outputs=tuple(outputs),
            blocker=blocker,
            requested_heads=requested_heads,
            enabled_heads=enabled_heads,
            memory_profile=memory_profile,
            model_root=validation.root,
            model_dir=validation.model_dir,
            checkpoint_ready=validation.ready,
            config_kind=validation.config_kind,
            input_path=input_path,
            output_path=output_path,
            metadata=metadata or {},
        )


def normalize_hyworld2_heads(heads: Sequence[str] | str) -> tuple[str, ...]:
    """Normalize comma-separated or sequence head selection."""

    raw_values = [heads] if isinstance(heads, str) else list(heads)
    normalized: list[str] = []
    for value in raw_values:
        for item in str(value).split(","):
            head = item.strip().lower()
            if not head:
                continue
            if head not in HYWORLD2_SUPPORTED_HEADS:
                raise ValueError(f"unsupported HY-World head: {head!r}")
            if head not in normalized:
                normalized.append(head)
    if not normalized:
        raise ValueError("at least one HY-World head must be requested")
    return tuple(normalized)


def validate_hyworld2_output_path(output_path: str | Path) -> HyWorld2Blocker | None:
    """Return a blocker when an output directory escapes the ignored outputs tree."""

    candidate = Path(output_path)
    resolved = candidate.resolve()
    outputs_root = Path("outputs").resolve()
    try:
        resolved.relative_to(outputs_root)
    except ValueError:
        return HyWorld2Blocker(
            stage="output-path",
            operation="validate HY-World output path",
            reason=f"output path must stay under outputs/: {candidate}",
            metadata={"output": str(candidate)},
        )
    return None


def _profile_metadata(profile) -> dict[str, int | str]:
    return {
        "name": profile.name,
        "target_size": profile.target_size,
        "max_frames": profile.max_frames,
        "activation_guard_bytes": profile.activation_guard_bytes,
    }


def _release_hyworld_mlx_stage(memory_snapshots: dict[str, dict[str, int | None]], label: str) -> None:
    clear_mlx_cache()
    memory_snapshots[label] = mlx_memory_snapshot().as_dict()


def _preprocess_metadata(preprocessed) -> dict[str, object]:
    return {
        "frame_count": preprocessed.frame_count,
        "target_size": preprocessed.target_size,
        "processed_size": preprocessed.processed_size,
        "patch_grid": preprocessed.patch_grid,
        "token_count": preprocessed.token_count,
        "image_paths": tuple(str(path) for path in preprocessed.image_paths),
        "original_sizes": preprocessed.original_sizes,
    }


def _inspection_metadata(inspection) -> dict[str, object]:
    groups = {
        group.name: {
            "present": group.present,
            "key_count": len(group.keys),
            "sample_keys": group.keys[:3],
        }
        for group in inspection.groups
    }
    metadata: dict[str, object] = {
        "ready": inspection.ready,
        "missing_groups": inspection.missing_groups,
        "component_groups": groups,
    }
    if inspection.config is not None:
        metadata["model_config"] = {
            "model_size": inspection.config.model_size,
            "img_size": inspection.config.img_size,
            "patch_size": inspection.config.patch_size,
            "embed_dim": inspection.config.embed_dim,
            "depth": inspection.config.depth,
            "num_heads": inspection.config.num_heads,
            "num_register_tokens": inspection.config.num_register_tokens,
            "visual_geometry_transformer": inspection.config.visual_geometry_transformer,
            "camera_head": inspection.config.camera_head,
            "dpt_heads": inspection.config.dpt_heads,
        }
    return metadata


def _clean_fixture_export_outputs(output_dir: Path) -> None:
    for name in (*HYWORLD2_FIXTURE_EXPORT_ARTIFACTS, *HYWORLD2_ROOT_EXPORT_ARTIFACTS):
        candidate = output_dir / name
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink()
        elif candidate.is_dir():
            shutil.rmtree(candidate)


def _initial_head_status(requested_heads: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {
        head: {
            "requested": head in requested_heads,
            "enabled": False,
            "export": False,
            "reason": "pending" if head in requested_heads else "not requested",
        }
        for head in HYWORLD2_SUPPORTED_HEADS
    }


def _fixture_max_tokens(preprocessed) -> int:
    batch, frames, _, _, _ = tuple(int(dim) for dim in preprocessed.tensor.shape)
    patch_count = preprocessed.patch_grid[0] * preprocessed.patch_grid[1]
    return max(batch * frames * (patch_count + 16), 64)


def _load_real_hyworld2_tensors(
    checkpoint_path: Path,
    requested_heads: tuple[str, ...],
) -> dict[str, object]:
    prefixes = ["visual_geometry_transformer."]
    for head in requested_heads:
        prefix = HYWORLD2_REAL_HEAD_PREFIXES.get(head)
        if prefix is not None:
            prefixes.append(prefix)
        if head == "gs":
            prefixes.append("gs_renderer.")
    tensors = load_checkpoint_tensors(checkpoint_path, prefixes=prefixes)
    visual = _strip_tensor_prefix(tensors, "visual_geometry_transformer.")
    heads = {
        head: _strip_tensor_prefix(tensors, HYWORLD2_REAL_HEAD_PREFIXES[head])
        for head in requested_heads
        if head in HYWORLD2_REAL_HEAD_PREFIXES
    }
    if "gs" in requested_heads:
        heads["gs_renderer"] = _strip_tensor_prefix(tensors, "gs_renderer.")
    _require_real_tensor_keys(
        visual,
        (
            "cam_token",
            "reg_token",
            "patch_embed.patch_embed.proj.weight",
            "patch_embed.patch_embed.proj.bias",
            "patch_embed.cls_token",
            "patch_embed.pos_embed",
            "patch_embed.register_tokens",
            "patch_embed.norm.weight",
            "patch_embed.norm.bias",
            "frame_blocks.0.attn.qkv.weight",
            "global_blocks.0.attn.qkv.weight",
        ),
        "VisualGeometryTransformer",
    )
    for head, head_tensors in heads.items():
        if head in {"depth", "normal", "points"}:
            _require_real_tensor_keys(
                head_tensors,
                (
                    "norm.weight",
                    "norm.bias",
                    "projects.0.weight",
                    "projects.1.weight",
                    "projects.2.weight",
                    "projects.3.weight",
                    "scratch.output_conv2.2.weight",
                    "scratch.output_conv2.2.bias",
                ),
                f"{head} head",
            )
        if head == "camera":
            _require_real_tensor_keys(
                head_tensors,
                (
                    "token_norm.weight",
                    "token_norm.bias",
                    "init_token",
                    "param_embed.weight",
                    "adapt_norm_gen.1.weight",
                    "refine_net.0.attn.qkv.weight",
                    "out_norm.weight",
                    "param_predictor.fc2.weight",
                ),
                "camera head",
            )
        if head == "gs":
            _require_real_tensor_keys(
                head_tensors,
                (
                    "norm.weight",
                    "projects.0.weight",
                    "input_merger.0.weight",
                    "scratch.output_conv2.2.weight",
                    "scratch.output_conv2.2.bias",
                ),
                "Gaussian DPT head",
            )
            _require_real_tensor_keys(
                heads["gs_renderer"],
                ("gs_head.0.weight", "gs_head.2.weight", "gs_head.2.bias"),
                "Gaussian renderer",
            )
    return {"visual_transformer": visual, "heads": heads}


def _strip_tensor_prefix(tensors: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    return {
        name[len(prefix) :]: value
        for name, value in tensors.items()
        if name.startswith(prefix)
    }


def _require_real_tensor_keys(
    tensors: dict[str, mx.array],
    required: tuple[str, ...],
    label: str,
) -> None:
    missing = tuple(name for name in required if name not in tensors)
    if missing:
        raise ValueError(f"checkpoint is missing real {label} tensors: {missing[0]}")


def _gaussian_metadata(requested_heads: tuple[str, ...], head_outputs: dict[str, object]) -> dict[str, object]:
    if "gs" not in requested_heads:
        return {"gaussians_ply": "not requested", "point_cloud_ply": "separate optional output"}
    return {
        "gaussians_ply": "exported" if "gs" in head_outputs else "blocked",
        "point_cloud_ply": "points.ply is a point-cloud artifact, not 3DGS",
        "renderer": "MLX gs_renderer attribute conv; no CUDA rasterization",
        "requires_cuda_gsplat": False,
    }


def _pipeline_blocker(blocker) -> HyWorld2Blocker:
    return HyWorld2Blocker(
        stage=blocker.stage,
        operation=blocker.operation,
        reason=blocker.reason,
        metadata=blocker.metadata,
    )


def _stage_outputs(records: Sequence[dict[str, object]]) -> tuple[HyWorld2StageOutput, ...]:
    return tuple(_stage_output(record) for record in records)


def _stage_output(record: dict[str, object]) -> HyWorld2StageOutput:
    return HyWorld2StageOutput(
        name=str(record["name"]),
        path=Path(record["path"]),
        kind=str(record["kind"]),
    )


def _trace_payload(trace: HyWorld2ReconstructionTrace) -> dict[str, object]:
    return {
        "completed_stages": list(trace.completed_stages),
        "outputs": [
            {"name": output.name, "path": str(output.path), "kind": output.kind}
            for output in trace.outputs
        ],
        "blocker": None
        if trace.blocker is None
        else {
            "stage": trace.blocker.stage,
            "operation": trace.blocker.operation,
            "reason": trace.blocker.reason,
            "metadata": trace.blocker.metadata,
        },
        "requested_heads": list(trace.requested_heads),
        "enabled_heads": list(trace.enabled_heads),
        "memory_profile": trace.memory_profile,
        "model_root": str(trace.model_root),
        "model_dir": str(trace.model_dir),
        "checkpoint_ready": trace.checkpoint_ready,
        "config_kind": trace.config_kind,
        "input_path": str(trace.input_path),
        "output_path": str(trace.output_path),
        "metadata": _jsonable(trace.metadata),
    }


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, mx.array):
        return str(value)
    return value
