"""Pixal3D inference orchestration skeleton."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mlx.core as mx

from .mlx_memory import mlx_memory_snapshot
from .pixal3d_assets import PIXAL3D_DEFAULT_ROOT, read_pixal3d_pipeline_config, validate_pixal3d_assets
from .pixal3d_camera import pixal3d_manual_camera_params, pixal3d_stage_plan
from .pixal3d_export import write_pixal3d_projection_npz
from .pixal3d_projection import build_pixal3d_projection_conditioning


PIXAL3D_RECOMMENDED_PIPELINE_TYPE = "1024_cascade"
PIXAL3D_PIPELINE_TYPES = ("1024_cascade", "1536_cascade")
PIXAL3D_DEFAULT_SEED = 42
PIXAL3D_DEFAULT_MAX_NUM_TOKENS = 49_152

Pixal3DStageStatus = Literal["ready", "blocked", "unimplemented"]


@dataclass(frozen=True)
class Pixal3DInferenceBlocker:
    """Structured blocker returned before unsupported Pixal3D compute/export boundaries."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Pixal3DInferenceTrace:
    """Trace metadata for a Pixal3D generation attempt."""

    root: Path
    image_path: Path
    completed_stages: tuple[str, ...]
    pipeline_type: str
    manual_fov: float | None
    seed: int
    max_num_tokens: int
    output_path: Path | None = None
    blocker: Pixal3DInferenceBlocker | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.blocker is None


@dataclass(frozen=True)
class Pixal3DGenerationResult:
    """Result of a Pixal3D generation attempt."""

    trace: Pixal3DInferenceTrace
    artifacts: tuple[Path, ...] = ()

    @property
    def ready(self) -> bool:
        return self.trace.ready


class Pixal3DInferencePipeline:
    """Pixal3D runtime skeleton that validates setup before MLX model execution."""

    def __init__(self, root: str | Path = PIXAL3D_DEFAULT_ROOT):
        self.root = Path(root)

    def generate(
        self,
        image: str | Path,
        *,
        output_dir: str | Path | None = None,
        output: str | Path | None = None,
        pipeline_type: str = PIXAL3D_RECOMMENDED_PIPELINE_TYPE,
        manual_fov: float | None = None,
        seed: int = PIXAL3D_DEFAULT_SEED,
        max_num_tokens: int = PIXAL3D_DEFAULT_MAX_NUM_TOKENS,
        projection_hidden_states: mx.array | None = None,
    ) -> Pixal3DGenerationResult:
        """Validate Pixal3D inputs and return the current execution boundary."""

        image_path = Path(image)
        completed: list[str] = []
        output_path = _resolve_output_path(image_path, output=output, output_dir=output_dir)
        artifact_dir = output_path.parent
        timings: dict[str, float] = {}
        metadata = {
            "memory_before": mlx_memory_snapshot().as_dict(),
        }
        started = time.perf_counter()

        if pipeline_type not in PIXAL3D_PIPELINE_TYPES:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "input-validation",
                "validate Pixal3D pipeline type",
                f"unsupported pipeline_type={pipeline_type!r}",
                {"supported": PIXAL3D_PIPELINE_TYPES},
            )

        if not image_path.is_file():
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "input-validation",
                "validate Pixal3D input image",
                f"image file not found: {image_path}",
                {},
            )
        completed.append("input-image")

        validation = validate_pixal3d_assets(self.root)
        metadata["asset_present"] = list(validation.present)
        metadata["asset_missing"] = list(validation.missing)
        if not validation.ready:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "asset-validation",
                "validate Pixal3D checkpoint layout",
                "missing required Pixal3D checkpoint assets",
                {"missing": validation.missing},
                metadata=metadata,
            )
        completed.append("asset-validation")

        try:
            config = read_pixal3d_pipeline_config(self.root)
        except (FileNotFoundError, ValueError) as error:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "pipeline-config",
                "parse Pixal3D pipeline config",
                str(error),
                {"root": str(self.root)},
                metadata=metadata,
            )
        completed.append("pipeline-config")
        timings["pipeline-config"] = time.perf_counter() - started
        metadata["default_pipeline_type"] = config.default_pipeline_type
        metadata["model_keys"] = [asset.key for asset in config.models]
        plan = pixal3d_stage_plan(pipeline_type, max_num_tokens=max_num_tokens)
        metadata["stage_plan"] = plan
        metadata["samplers"] = {
            "sparse_structure": config.sparse_structure_sampler,
            "shape_slat": config.shape_slat_sampler,
            "tex_slat": config.texture_slat_sampler,
        }

        if manual_fov is None:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "camera-setup",
                "estimate Pixal3D camera parameters",
                "auto-camera through MoGe is not implemented for Pixal3D yet; pass --manual-fov",
                {
                    "manual_fov_required": True,
                    "moge_model": "Ruicheng/moge-2-vitl",
                    "upstream_function": "get_camera_params_wild_moge",
                },
                metadata=metadata,
            )
        if manual_fov <= 0:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "camera-setup",
                "validate manual Pixal3D FOV",
                "manual_fov must be positive radians",
                {"manual_fov": manual_fov},
                metadata=metadata,
            )
        try:
            camera = pixal3d_manual_camera_params(manual_fov)
        except ValueError as error:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "camera-setup",
                "compute Pixal3D manual-FOV camera parameters",
                str(error),
                {"manual_fov": manual_fov},
                metadata=metadata,
            )
        completed.append("camera-setup")
        timings["camera-setup"] = time.perf_counter() - started
        metadata["camera"] = camera

        if projection_hidden_states is None:
            metadata["memory_after"] = mlx_memory_snapshot().as_dict()
            metadata["timings_sec"] = timings
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "image-conditioning",
                "run Pixal3D DINOv3 hidden-state extraction",
                "Pixal3D MLX DINOv3 hidden-state extraction is not wired into the runtime yet",
                {
                    "accepted_manual_boundary": "pass projection_hidden_states to exercise projection-conditioned model boundaries",
                    "expected_hidden_state_layout": "[batch, cls+register+patch_tokens, channels]",
                },
                metadata=metadata,
            )

        ss_conditioning = build_pixal3d_projection_conditioning(
            projection_hidden_states,
            "ss",
            camera_angle_x=camera.camera_angle_x,
            distance=camera.distance,
            mesh_scale=camera.mesh_scale,
        )
        metadata["ss_projection"] = {
            "ready": ss_conditioning.ready,
            "global_shape": tuple(int(dim) for dim in ss_conditioning.global_tokens.shape)
            if ss_conditioning.global_tokens is not None
            else None,
            "projected_shape": tuple(int(dim) for dim in ss_conditioning.projected_features.shape)
            if ss_conditioning.projected_features is not None
            else None,
            "blocker": ss_conditioning.blocker,
        }
        if ss_conditioning.blocker is not None:
            metadata["memory_after"] = mlx_memory_snapshot().as_dict()
            metadata["timings_sec"] = timings
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "projection-conditioning",
                ss_conditioning.blocker.operation,
                ss_conditioning.blocker.reason,
                ss_conditioning.blocker.metadata,
                metadata=metadata,
            )
        completed.append("projection-conditioning:ss")
        timings["projection-conditioning:ss"] = time.perf_counter() - started
        projection_artifact = write_pixal3d_projection_npz(
            artifact_dir / "sparse_projection.npz",
            ss_conditioning,
            metadata={
                "pipeline_type": pipeline_type,
                "manual_fov": manual_fov,
                "camera_distance": camera.distance,
                "mesh_scale": camera.mesh_scale,
                "seed": seed,
            },
        )
        completed.append("artifact:sparse_projection")
        metadata["artifact_paths"] = [projection_artifact.path]
        metadata["projection_artifact"] = projection_artifact
        metadata["memory_after"] = mlx_memory_snapshot().as_dict()
        metadata["timings_sec"] = timings

        return self._blocked(
            image_path,
            completed,
            pipeline_type,
            manual_fov,
            seed,
            max_num_tokens,
            output_path,
            "sparse-structure-flow",
            "run Pixal3D sparse-structure FlowEuler cascade",
            "Pixal3D orchestration reached the first model block boundary; full checkpoint execution and decoder handoff continue in later slices",
            {
                "next_target": "wire real Pixal3D checkpoint execution and sparse decoder handoff",
                "implemented_boundary": "sparse structure projection conditioning",
            },
            metadata=metadata,
            artifacts=(projection_artifact.path,),
        )

    def _blocked(
        self,
        image_path: Path,
        completed: list[str],
        pipeline_type: str,
        manual_fov: float | None,
        seed: int,
        max_num_tokens: int,
        output_path: Path,
        stage: str,
        operation: str,
        reason: str,
        blocker_metadata: dict[str, object],
        *,
        metadata: dict[str, object] | None = None,
        artifacts: tuple[Path, ...] = (),
    ) -> Pixal3DGenerationResult:
        trace = Pixal3DInferenceTrace(
            root=self.root,
            image_path=image_path,
            completed_stages=tuple(completed),
            pipeline_type=pipeline_type,
            manual_fov=manual_fov,
            seed=seed,
            max_num_tokens=max_num_tokens,
            output_path=output_path,
            blocker=Pixal3DInferenceBlocker(
                stage=stage,
                operation=operation,
                reason=reason,
                metadata=blocker_metadata,
            ),
            metadata=metadata or {},
        )
        return Pixal3DGenerationResult(trace=trace, artifacts=artifacts)


def _resolve_output_path(
    image_path: Path,
    *,
    output: str | Path | None,
    output_dir: str | Path | None,
) -> Path:
    if output is not None:
        return Path(output)
    directory = Path(output_dir) if output_dir is not None else Path("outputs/pixal3d") / _slug(image_path.stem)
    return directory / "model.glb"


def _slug(value: str) -> str:
    normalized = "".join(char if char.isalnum() or char in "._-" else "-" for char in value.strip())
    return normalized.strip("-._") or "pixal3d"
