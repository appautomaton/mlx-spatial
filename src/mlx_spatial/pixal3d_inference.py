"""Pixal3D inference orchestration skeleton."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from PIL import Image, UnidentifiedImageError

from .model_assets import DINOv3_VITL16_ASSETS
from .mlx_memory import clear_mlx_cache, mlx_memory_snapshot
from .naf import NAF_DEFAULT_ROOT, load_naf_tensors, naf_conversion_command, prepare_naf_image_tensor, project_naf_features_at_points
from .ovoxel import flexi_dual_grid_fields_to_mesh
from .pixal3d_assets import PIXAL3D_DEFAULT_ROOT, read_pixal3d_pipeline_config, validate_pixal3d_assets
from .pixal3d_camera import (
    pixal3d_camera_params_from_moge_intrinsics,
    pixal3d_manual_camera_params,
    pixal3d_select_hr_coordinates,
    pixal3d_stage_plan,
)
from .pixal3d_export import (
    write_pixal3d_projection_npz,
    write_pixal3d_shape_decoder_npz,
    write_pixal3d_shape_hr_coordinates_npz,
    write_pixal3d_shape_slat_npz,
    write_pixal3d_sparse_structure_npz,
    write_pixal3d_texture_decoder_npz,
    write_pixal3d_texture_slat_npz,
    write_pixal3d_textured_glb,
)
from .pixal3d_projection import (
    PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS,
    PIXAL3D_DINOV3_EMBED_DIM,
    build_pixal3d_projection_conditioning,
    pixal3d_projection_grid_points,
    pixal3d_projection_stage_config,
    pixal3d_stage_with_grid_resolution,
    project_pixal3d_points_to_image,
    select_pixal3d_projected_features_at_coordinates,
)
from .sam3d_moge import SAM3D_MOGE_DEFAULT_ROOT, SAM3D_MOGE_MEMORY_PROFILES, run_sam3d_moge_pointmap
from .trellis2_decode import (
    read_structured_latent_decoder_config,
    run_shape_decoder_to_fields,
    run_shape_decoder_upsample_coordinates,
    run_texture_decoder_to_representation,
)
from .trellis2_dinov3 import (
    DINOv3_ACCESS_NOTE,
    DINOv3_VITL16_REPO_ID,
    assess_dinov3_mlx_conditioning,
    dinov3_download_command,
    inspect_dinov3_assets,
)
from .trellis2_forward import prepare_dinov3_image_tensor, sparse_structure_target_resolution
from .trellis2_export import (
    TRELLIS2_GLB_DEFAULT_FACE_TARGET,
    TRELLIS2_TEXTURE_BAKE_BACKENDS,
    TRELLIS2_XATLAS_AUTO_FACE_GUARD,
    bake_trellis2_texture_fields_mac_native,
    postprocess_trellis2_mesh_for_glb,
)
from .trellis2_sparse_structure import (
    probe_sparse_structure_decoder_boundary,
    probe_sparse_structure_forward_boundary,
    read_sparse_structure_decoder_config,
    read_sparse_structure_flow_config,
)
from .trellis2_slat import probe_shape_slat_forward_boundary, probe_texture_slat_forward_boundary, read_slat_flow_config


PIXAL3D_RECOMMENDED_PIPELINE_TYPE = "1024_cascade"
PIXAL3D_PIPELINE_TYPES = ("1024_cascade", "1536_cascade")
PIXAL3D_DEFAULT_DINO_ROOT = DINOv3_VITL16_ASSETS.root_hint
PIXAL3D_DEFAULT_SEED = 42
PIXAL3D_DEFAULT_MAX_NUM_TOKENS = 49_152
PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT = 1_000_000
PIXAL3D_DEFAULT_TEXTURE_SIZE = 1024
PIXAL3D_DEFAULT_GLB_TARGET_FACES = TRELLIS2_GLB_DEFAULT_FACE_TARGET
PIXAL3D_DEFAULT_TEXTURE_BAKE_BACKEND = "kdtree"
PIXAL3D_DEFAULT_NAF_ROOT = NAF_DEFAULT_ROOT
PIXAL3D_DEFAULT_NAF_COORDINATE_CHUNK_SIZE = 8192
PIXAL3D_DEFAULT_MOGE_ROOT = SAM3D_MOGE_DEFAULT_ROOT
PIXAL3D_DEFAULT_MOGE_MEMORY_PROFILE = "balanced"

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
        shape_upsample_token_limit: int = PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT,
        dino_root: str | Path | None = None,
        texture_size: int = PIXAL3D_DEFAULT_TEXTURE_SIZE,
        glb_target_faces: int = PIXAL3D_DEFAULT_GLB_TARGET_FACES,
        xatlas_face_guard: int | str = TRELLIS2_XATLAS_AUTO_FACE_GUARD,
        xatlas_parallel_chunks: int = 0,
        texture_bake_backend: str = PIXAL3D_DEFAULT_TEXTURE_BAKE_BACKEND,
        naf_root: str | Path | None = PIXAL3D_DEFAULT_NAF_ROOT,
        naf_coordinate_chunk_size: int = PIXAL3D_DEFAULT_NAF_COORDINATE_CHUNK_SIZE,
        moge_root: str | Path | None = PIXAL3D_DEFAULT_MOGE_ROOT,
        moge_memory_profile: str = PIXAL3D_DEFAULT_MOGE_MEMORY_PROFILE,
        projection_hidden_states: mx.array | None = None,
        shape_lr_naf_feature_map: mx.array | None = None,
        shape_hr_naf_feature_map: mx.array | None = None,
        texture_naf_feature_map: mx.array | None = None,
    ) -> Pixal3DGenerationResult:
        """Validate Pixal3D inputs and return the current execution boundary."""

        image_path = Path(image)
        completed: list[str] = []
        output_path = _resolve_output_path(image_path, output=output, output_dir=output_dir)
        artifact_dir = output_path.parent
        timings: dict[str, float] = {}
        metadata = {
            "memory_before": mlx_memory_snapshot().as_dict(),
            "export_options": {
                "texture_size": int(texture_size),
                "glb_target_faces": int(glb_target_faces),
                "xatlas_face_guard": xatlas_face_guard,
                "xatlas_parallel_chunks": int(xatlas_parallel_chunks),
                "texture_bake_backend": texture_bake_backend,
            },
            "naf_options": {
                "naf_root": str(naf_root) if naf_root is not None else None,
                "coordinate_chunk_size": int(naf_coordinate_chunk_size),
                "full_map_avoidance": True,
            },
            "moge_options": {
                "moge_root": str(moge_root) if moge_root is not None else str(PIXAL3D_DEFAULT_MOGE_ROOT),
                "memory_profile": moge_memory_profile,
                "source": "existing MLX SAM3D MoGe pointmap/intrinsics runtime",
                "upstream_pixal3d_model": "Ruicheng/moge-2-vitl",
                "exact_upstream_v2_parity": False,
            },
            "shape_upsample_options": {
                "token_limit": int(shape_upsample_token_limit),
                "hr_selection_max_num_tokens": int(max_num_tokens),
            },
        }
        started = time.perf_counter()
        mx.random.seed(seed)

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

        export_guard_error = _validate_pixal3d_export_guards(
            texture_size=texture_size,
            glb_target_faces=glb_target_faces,
            xatlas_parallel_chunks=xatlas_parallel_chunks,
            texture_bake_backend=texture_bake_backend,
        )
        if export_guard_error is not None:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "input-validation",
                "validate Pixal3D export options",
                export_guard_error,
                {
                    "texture_size": texture_size,
                    "glb_target_faces": glb_target_faces,
                    "xatlas_parallel_chunks": xatlas_parallel_chunks,
                    "texture_bake_backend": texture_bake_backend,
                    "supported_texture_bake_backends": TRELLIS2_TEXTURE_BAKE_BACKENDS,
                },
                metadata=metadata,
            )

        if naf_coordinate_chunk_size <= 0:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "input-validation",
                "validate Pixal3D NAF options",
                f"naf_coordinate_chunk_size must be positive, got {naf_coordinate_chunk_size}",
                {"naf_coordinate_chunk_size": naf_coordinate_chunk_size},
                metadata=metadata,
            )

        if moge_memory_profile not in SAM3D_MOGE_MEMORY_PROFILES:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "input-validation",
                "validate Pixal3D MoGe memory profile",
                f"unsupported MoGe memory profile: {moge_memory_profile}",
                {"memory_profile": moge_memory_profile, "supported": tuple(SAM3D_MOGE_MEMORY_PROFILES)},
                metadata=metadata,
            )

        if shape_upsample_token_limit <= 0:
            return self._blocked(
                image_path,
                completed,
                pipeline_type,
                manual_fov,
                seed,
                max_num_tokens,
                output_path,
                "input-validation",
                "validate Pixal3D shape upsample token limit",
                f"shape_upsample_token_limit must be positive, got {shape_upsample_token_limit}",
                {"shape_upsample_token_limit": shape_upsample_token_limit},
                metadata=metadata,
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

        if manual_fov is not None and manual_fov <= 0:
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

        if manual_fov is None:
            resolved_moge_root = Path(moge_root) if moge_root is not None else Path(PIXAL3D_DEFAULT_MOGE_ROOT)
            try:
                with Image.open(image_path) as pil_image:
                    image_rgb_pil = pil_image.convert("RGB")
                    image_rgb = np.array(image_rgb_pil, dtype=np.uint8)
                    image_width = int(image_rgb_pil.width)
            except (OSError, UnidentifiedImageError, ValueError) as error:
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
                    "camera-setup",
                    "prepare Pixal3D MoGe RGB input",
                    f"failed to decode input image for MoGe auto-camera: {error}",
                    {"image_path": str(image_path), "moge_root": str(resolved_moge_root)},
                    metadata=metadata,
                )
            try:
                moge_result = run_sam3d_moge_pointmap(
                    image_rgb,
                    root=resolved_moge_root,
                    memory_profile=moge_memory_profile,
                )
            except (OSError, RuntimeError, ValueError) as error:
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
                    "camera-setup",
                    "run MLX MoGe auto-camera",
                    str(error),
                    {"moge_root": str(resolved_moge_root), "memory_profile": moge_memory_profile},
                    metadata=metadata,
                )
            if moge_result.blocker is not None or moge_result.pointmap is None:
                metadata["memory_after"] = mlx_memory_snapshot().as_dict()
                metadata["timings_sec"] = timings
                blocker = moge_result.blocker
                return self._blocked(
                    image_path,
                    completed,
                    pipeline_type,
                    manual_fov,
                    seed,
                    max_num_tokens,
                    output_path,
                    "camera-setup",
                    blocker.operation if blocker is not None else "run MLX MoGe auto-camera",
                    blocker.reason if blocker is not None else "MLX MoGe completed without returning pointmap/intrinsics",
                    {
                        "moge_root": str(resolved_moge_root),
                        "memory_profile": moge_memory_profile,
                        "upstream_function": "get_camera_params_wild_moge",
                        "upstream_pixal3d_model": "Ruicheng/moge-2-vitl",
                        "mlx_source": "SAM3D MoGe converted pointmap/intrinsics runtime",
                        **(blocker.metadata if blocker is not None else {}),
                    },
                    metadata=metadata,
                )
            try:
                camera = pixal3d_camera_params_from_moge_intrinsics(
                    moge_result.pointmap.intrinsics,
                    image_width=image_width,
                )
            except ValueError as error:
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
                    "camera-setup",
                    "compute Pixal3D camera parameters from MoGe intrinsics",
                    str(error),
                    {"image_width": image_width, "moge_root": str(resolved_moge_root), "memory_profile": moge_memory_profile},
                    metadata=metadata,
                )
            metadata["camera_source"] = "moge"
            metadata["moge_camera"] = {
                "root": str(resolved_moge_root),
                "memory_profile": moge_memory_profile,
                "image_size": tuple(int(value) for value in image_rgb.shape[:2]),
                "intrinsics": np.asarray(moge_result.pointmap.intrinsics, dtype=np.float32).tolist(),
                "pointmap_metadata": dict(moge_result.pointmap.metadata),
                "upstream_function": "get_camera_params_wild_moge",
                "upstream_pixal3d_model": "Ruicheng/moge-2-vitl",
                "exact_upstream_v2_parity": False,
            }
            clear_mlx_cache()
        else:
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
            metadata["camera_source"] = "manual_fov"
        completed.append("camera-setup")
        timings["camera-setup"] = time.perf_counter() - started
        metadata["camera"] = camera

        if projection_hidden_states is None:
            resolved_dino_root = Path(dino_root) if dino_root is not None else Path(PIXAL3D_DEFAULT_DINO_ROOT)
            metadata["dino_root"] = resolved_dino_root
            dino_inspection = inspect_dinov3_assets(resolved_dino_root)
            if dino_inspection.blocker is not None:
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
                    dino_inspection.blocker.operation,
                    _dino_blocker_reason(dino_inspection.blocker.reason, resolved_dino_root),
                    _dino_blocker_metadata(dino_inspection.blocker, resolved_dino_root),
                    metadata=metadata,
                )
            ss_stage = pixal3d_projection_stage_config("ss")
            try:
                with Image.open(image_path) as pil_image:
                    image_tensor = prepare_dinov3_image_tensor(pil_image, image_size=ss_stage.image_size)
            except (OSError, UnidentifiedImageError, ValueError) as error:
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
                    "prepare Pixal3D DINOv3 image tensor",
                    f"failed to decode or normalize input image for DINOv3: {error}",
                    {"image_path": str(image_path), "image_size": ss_stage.image_size},
                    metadata=metadata,
                )
            dino_result = assess_dinov3_mlx_conditioning(
                resolved_dino_root,
                expected_feature_width=PIXAL3D_DINOV3_EMBED_DIM,
                image_tensor=image_tensor,
            )
            if dino_result.blocker is not None:
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
                    dino_result.blocker.operation,
                    _dino_blocker_reason(dino_result.blocker.reason, resolved_dino_root),
                    _dino_blocker_metadata(dino_result.blocker, resolved_dino_root),
                    metadata=metadata,
                )
            if dino_result.hidden_states is None:
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
                    "DINOv3 conditioning completed without returning hidden states",
                    {"dino_root": str(resolved_dino_root)},
                    metadata=metadata,
                )
            projection_hidden_states = dino_result.hidden_states
            completed.append("image-conditioning")
            timings["image-conditioning"] = time.perf_counter() - started
            metadata["dino_conditioning"] = {
                "root": str(resolved_dino_root),
                "shape": dino_result.shape,
                "dtype": dino_result.dtype,
                "detail": dino_result.detail,
                "image_size": ss_stage.image_size,
            }
        else:
            metadata["dino_conditioning"] = {
                "source": "caller-supplied projection_hidden_states",
                "shape": tuple(int(dim) for dim in projection_hidden_states.shape),
                "dtype": str(projection_hidden_states.dtype).removeprefix("mlx.core."),
            }

        naf_tensors: dict[str, mx.array] | None = None
        naf_image_cache: dict[int, mx.array] = {}

        def runtime_naf_projected_features(
            *,
            stage,
            coordinates: mx.array,
            lr_projected_features: mx.array | None,
            stage_label: str,
        ) -> tuple[mx.array | None, dict[str, object], Pixal3DInferenceBlocker | None]:
            nonlocal naf_tensors
            naf_root_path = Path(naf_root) if naf_root is not None else Path(PIXAL3D_DEFAULT_NAF_ROOT)
            if lr_projected_features is None:
                return (
                    None,
                    {"source": "unavailable", "stage": stage.name},
                    Pixal3DInferenceBlocker(
                        stage="naf-projection",
                        operation="build Pixal3D LR projected features before NAF",
                        reason="LR projected features were not available for NAF concatenation",
                        metadata={"stage": stage.name},
                    ),
                )
            try:
                if naf_tensors is None:
                    naf_tensors = load_naf_tensors(naf_root_path)
                if stage.image_size not in naf_image_cache:
                    with Image.open(image_path) as pil_image:
                        naf_image_cache[stage.image_size] = prepare_naf_image_tensor(pil_image, image_size=stage.image_size)
                image_tensor = naf_image_cache[stage.image_size]
                patch_features = _pixal3d_patch_feature_map_bchw(projection_hidden_states)
                projected_points = _pixal3d_project_sparse_coordinates(
                    coordinates,
                    stage=stage,
                    camera_angle_x=camera.camera_angle_x,
                    distance=camera.distance,
                    mesh_scale=camera.mesh_scale,
                )
                target_size = int(stage.naf_target_size or stage.image_size)
                naf_projection = project_naf_features_at_points(
                    image_tensor,
                    patch_features,
                    projected_points,
                    image_resolution=stage.image_size,
                    output_size=target_size,
                    tensors=naf_tensors,
                    chunk_size=naf_coordinate_chunk_size,
                )
                selected_lr = select_pixal3d_projected_features_at_coordinates(
                    lr_projected_features,
                    coordinates,
                    grid_resolution=stage.grid_resolution,
                )
                selected = mx.concatenate((selected_lr, naf_projection.features[0]), axis=-1)
                return (
                    selected,
                    {
                        "source": "mlx-naf",
                        "stage": stage.name,
                        "stage_label": stage_label,
                        "naf_root": str(naf_root_path),
                        "target_size": naf_projection.target_size,
                        "point_count": naf_projection.point_count,
                        "coordinate_chunk_size": naf_projection.chunk_size,
                        "kernel_size": naf_projection.kernel_size,
                        "lr_projected_shape": tuple(int(dim) for dim in selected_lr.shape),
                        "hr_projected_shape": tuple(int(dim) for dim in naf_projection.features.shape),
                        "selected_projected_shape": tuple(int(dim) for dim in selected.shape),
                        "full_map_avoidance": True,
                    },
                    None,
                )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
                operation = "load converted NAF safetensors" if isinstance(error, FileNotFoundError) else "run MLX NAF coordinate projection"
                stage_name = "naf-assets" if isinstance(error, FileNotFoundError) else "naf-projection"
                return (
                    None,
                    {
                        "source": "mlx-naf",
                        "stage": stage.name,
                        "stage_label": stage_label,
                        "naf_root": str(naf_root_path),
                    },
                    Pixal3DInferenceBlocker(
                        stage=stage_name,
                        operation=operation,
                        reason=str(error),
                        metadata={
                            "stage": stage.name,
                            "stage_label": stage_label,
                            "naf_root": str(naf_root_path),
                            "expected_weights": str(naf_root_path / "naf_release.safetensors"),
                            "conversion_command": " ".join(naf_conversion_command(naf_root_path)),
                            "coordinate_chunk_size": naf_coordinate_chunk_size,
                        },
                    ),
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

        sparse_flow_model = _pixal3d_model_asset(config.models, "sparse_structure_flow_model")
        sparse_decoder_model = _pixal3d_model_asset(config.models, "sparse_structure_decoder")
        try:
            sparse_config = read_sparse_structure_flow_config(self.root, sparse_flow_model.config_path)
            sparse_probe = probe_sparse_structure_forward_boundary(
                self.root / sparse_flow_model.checkpoint_path,
                sparse_config,
                conditioning={
                    "global": ss_conditioning.global_tokens,
                    "proj": ss_conditioning.projected_features,
                },
                steps=config.sparse_structure_sampler.steps,
                rescale_t=config.sparse_structure_sampler.rescale_t,
                guidance_strength=config.sparse_structure_sampler.guidance_strength,
                guidance_rescale=config.sparse_structure_sampler.guidance_rescale,
                guidance_interval=config.sparse_structure_sampler.guidance_interval,
                sigma_min=config.sparse_structure_sampler.sigma_min,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                str(error),
                {
                    "config_path": sparse_flow_model.config_path,
                    "checkpoint_path": sparse_flow_model.checkpoint_path,
                },
                metadata=metadata,
                artifacts=(projection_artifact.path,),
            )

        metadata["sparse_flow"] = {
            "sampled_latent_shape": sparse_probe.sampled_latent_shape,
            "completed_blocks": sparse_probe.completed_blocks,
            "blocker_operation": sparse_probe.blocker_operation,
            "blocker_detail": sparse_probe.blocker_detail,
        }
        if sparse_probe.sampled_latent is None:
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
                sparse_probe.blocker_operation,
                sparse_probe.blocker_detail,
                {
                    "config_path": sparse_flow_model.config_path,
                    "checkpoint_path": sparse_flow_model.checkpoint_path,
                },
                metadata=metadata,
                artifacts=(projection_artifact.path,),
            )
        completed.append("sparse-structure-flow")
        timings["sparse-structure-flow"] = time.perf_counter() - started

        try:
            decoder_config = read_sparse_structure_decoder_config(self.root, sparse_decoder_model.config_path)
            decoder_probe = probe_sparse_structure_decoder_boundary(
                self.root / sparse_decoder_model.checkpoint_path,
                decoder_config,
                sparse_latent=sparse_probe.sampled_latent,
                target_resolution=sparse_structure_target_resolution(pipeline_type),
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                "sparse-structure-decoding",
                "probe Pixal3D sparse decoder",
                str(error),
                {
                    "config_path": sparse_decoder_model.config_path,
                    "checkpoint_path": sparse_decoder_model.checkpoint_path,
                },
                metadata=metadata,
                artifacts=(projection_artifact.path,),
            )

        metadata["sparse_decoder"] = {
            "latent_shape": decoder_probe.latent_shape,
            "decoded_shape": decoder_probe.decoded_shape,
            "coordinates_shape": decoder_probe.coordinates_shape,
            "target_resolution": decoder_probe.target_resolution,
            "blocker_operation": decoder_probe.blocker_operation,
            "blocker_detail": decoder_probe.blocker_detail,
        }
        if decoder_probe.coordinates is not None and decoder_probe.coordinates_shape is not None and decoder_probe.coordinates_shape[0] > 0:
            completed.append("sparse-structure-decoding")
            sparse_structure_artifact = write_pixal3d_sparse_structure_npz(
                artifact_dir / "sparse_structure.npz",
                decoder_probe.coordinates,
                decoded_shape=decoder_probe.decoded_shape or (),
                target_resolution=decoder_probe.target_resolution or sparse_structure_target_resolution(pipeline_type),
                metadata={
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "sparse_flow_model": sparse_flow_model.key,
                    "sparse_decoder_model": sparse_decoder_model.key,
                    "sparse_latent_shape": decoder_probe.latent_shape,
                    "blocker_next_target": "shape-projection-conditioning",
                },
            )
            completed.append("artifact:sparse_structure")
            metadata["artifact_paths"] = [projection_artifact.path, sparse_structure_artifact.path]
            metadata["sparse_structure_artifact"] = sparse_structure_artifact

            shape_conditioning = build_pixal3d_projection_conditioning(
                projection_hidden_states,
                "shape_512",
                camera_angle_x=camera.camera_angle_x,
                distance=camera.distance,
                mesh_scale=camera.mesh_scale,
                naf_feature_map=shape_lr_naf_feature_map,
            )
            metadata["shape_lr_projection"] = {
                "ready": shape_conditioning.ready,
                "global_shape": tuple(int(dim) for dim in shape_conditioning.global_tokens.shape)
                if shape_conditioning.global_tokens is not None
                else None,
                "projected_shape": tuple(int(dim) for dim in shape_conditioning.projected_features.shape)
                if shape_conditioning.projected_features is not None
                else None,
                "projected_lr_shape": tuple(int(dim) for dim in shape_conditioning.projected_lr_features.shape)
                if shape_conditioning.projected_lr_features is not None
                else None,
                "blocker": shape_conditioning.blocker,
            }
            shape_projected = None
            if shape_conditioning.blocker is not None:
                if shape_lr_naf_feature_map is None:
                    shape_projected, naf_projection_metadata, naf_blocker = runtime_naf_projected_features(
                        stage=shape_conditioning.stage,
                        coordinates=decoder_probe.coordinates,
                        lr_projected_features=shape_conditioning.projected_lr_features,
                        stage_label="shape_lr",
                    )
                    metadata["shape_lr_projection"].update(naf_projection_metadata)
                    if naf_blocker is None:
                        metadata["shape_lr_projection"]["ready"] = True
                        metadata["shape_lr_projection"]["blocker"] = None
                    else:
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
                            naf_blocker.stage,
                            naf_blocker.operation,
                            naf_blocker.reason,
                            naf_blocker.metadata,
                            metadata=metadata,
                            artifacts=(projection_artifact.path, sparse_structure_artifact.path),
                        )
                else:
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
                        "shape-projection-conditioning",
                        shape_conditioning.blocker.operation,
                        shape_conditioning.blocker.reason,
                        shape_conditioning.blocker.metadata,
                        metadata=metadata,
                        artifacts=(projection_artifact.path, sparse_structure_artifact.path),
                    )
            if shape_conditioning.blocker is not None and shape_projected is None:
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
                    "shape-projection-conditioning",
                    shape_conditioning.blocker.operation,
                    shape_conditioning.blocker.reason,
                    shape_conditioning.blocker.metadata,
                    metadata=metadata,
                    artifacts=(projection_artifact.path, sparse_structure_artifact.path),
                )
            completed.append("projection-conditioning:shape_512")
            timings["projection-conditioning:shape_512"] = time.perf_counter() - started

            assert shape_conditioning.global_tokens is not None
            if shape_projected is None:
                assert shape_conditioning.projected_features is not None
                shape_projected = select_pixal3d_projected_features_at_coordinates(
                    shape_conditioning.projected_features,
                    decoder_probe.coordinates,
                    grid_resolution=shape_conditioning.stage.grid_resolution,
                )
            metadata["shape_lr_projection"]["selected_projected_shape"] = tuple(int(dim) for dim in shape_projected.shape)

            shape_slat_model = _pixal3d_model_asset(config.models, "shape_slat_flow_model_512")
            try:
                shape_slat_config = read_slat_flow_config(self.root, shape_slat_model.config_path)
                shape_probe = probe_shape_slat_forward_boundary(
                    self.root / shape_slat_model.checkpoint_path,
                    shape_slat_config,
                    decoder_probe.coordinates,
                    conditioning={
                        "global": shape_conditioning.global_tokens,
                        "proj": shape_projected,
                    },
                    steps=config.shape_slat_sampler.steps,
                    rescale_t=config.shape_slat_sampler.rescale_t,
                    guidance_strength=config.shape_slat_sampler.guidance_strength,
                    guidance_rescale=config.shape_slat_sampler.guidance_rescale,
                    guidance_interval=config.shape_slat_sampler.guidance_interval,
                    sigma_min=config.shape_slat_sampler.sigma_min,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                    "shape-slat-sampling",
                    "run Pixal3D 512 shape SLat FlowEuler cascade",
                    str(error),
                    {
                        "config_path": shape_slat_model.config_path,
                        "checkpoint_path": shape_slat_model.checkpoint_path,
                    },
                    metadata=metadata,
                    artifacts=(projection_artifact.path, sparse_structure_artifact.path),
                )

            metadata["shape_slat_lr"] = {
                "coordinate_shape": shape_probe.coordinate_shape,
                "feature_shape": shape_probe.feature_shape,
                "sampled_feature_shape": shape_probe.sampled_feature_shape,
                "completed_blocks": shape_probe.completed_blocks,
                "blocker_operation": shape_probe.blocker_operation,
                "blocker_detail": shape_probe.blocker_detail,
            }
            if shape_probe.sampled_features is None:
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
                    "shape-slat-sampling",
                    shape_probe.blocker_operation,
                    shape_probe.blocker_detail,
                    {
                        "config_path": shape_slat_model.config_path,
                        "checkpoint_path": shape_slat_model.checkpoint_path,
                    },
                    metadata=metadata,
                    artifacts=(projection_artifact.path, sparse_structure_artifact.path),
                )

            shape_features = _apply_pixal3d_slat_normalization(
                shape_probe.sampled_features,
                config.shape_slat_normalization,
                name="shape_slat",
            )
            shape_slat_artifact = write_pixal3d_shape_slat_npz(
                artifact_dir / "shape_slat_lr.npz",
                decoder_probe.coordinates,
                shape_features,
                metadata={
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "shape_slat_model": shape_slat_model.key,
                    "shape_slat_config_path": shape_slat_model.config_path,
                    "shape_slat_checkpoint_path": shape_slat_model.checkpoint_path,
                    "sparse_structure_artifact": str(sparse_structure_artifact.path),
                    "blocker_next_target": "shape-slat-cascade",
                },
            )
            completed.append("shape-slat-sampling:512")
            completed.append("artifact:shape_slat_lr")
            metadata["artifact_paths"] = [projection_artifact.path, sparse_structure_artifact.path, shape_slat_artifact.path]
            metadata["shape_slat_lr_artifact"] = shape_slat_artifact
            shape_decoder_model = _pixal3d_model_asset(config.models, "shape_slat_decoder")
            try:
                shape_decoder_config = read_structured_latent_decoder_config(self.root, shape_decoder_model.config_path)
                shape_upsample = run_shape_decoder_upsample_coordinates(
                    self.root / shape_decoder_model.checkpoint_path,
                    shape_decoder_config,
                    decoder_probe.coordinates,
                    shape_features,
                    upsample_times=4,
                    decoder_token_limit=shape_upsample_token_limit,
                )
                hr_selection = pixal3d_select_hr_coordinates(
                    shape_upsample.coordinates,
                    requested_hr_resolution=plan.requested_hr_resolution,
                    max_num_tokens=max_num_tokens,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                    "shape-slat-cascade",
                    "upsample Pixal3D LR shape SLat coordinates for HR cascade",
                    str(error),
                    {
                        "config_path": shape_decoder_model.config_path,
                        "checkpoint_path": shape_decoder_model.checkpoint_path,
                        "upsample_times": 4,
                        "shape_upsample_token_limit": shape_upsample_token_limit,
                        "hr_selection_max_num_tokens": max_num_tokens,
                    },
                    metadata=metadata,
                    artifacts=(projection_artifact.path, sparse_structure_artifact.path, shape_slat_artifact.path),
                )

            metadata["shape_hr_cascade"] = {
                "input_coordinate_shape": shape_upsample.input_coordinate_shape,
                "raw_upsampled_shape": shape_upsample.output_coordinate_shape,
                "completed_upsamples": shape_upsample.completed_upsamples,
                "subdivision_shapes": tuple(tuple(int(dim) for dim in subdiv.shape) for subdiv in shape_upsample.subdivisions),
                "requested_hr_resolution": hr_selection.requested_hr_resolution,
                "actual_hr_resolution": hr_selection.actual_hr_resolution,
                "actual_hr_grid_resolution": hr_selection.actual_hr_grid_resolution,
                "token_count": hr_selection.token_count,
                "max_num_tokens": hr_selection.max_num_tokens,
                "shape_upsample_token_limit": int(shape_upsample_token_limit),
            }
            shape_hr_coordinates_artifact = write_pixal3d_shape_hr_coordinates_npz(
                artifact_dir / "shape_slat_hr_coordinates.npz",
                hr_selection.coordinates,
                requested_hr_resolution=hr_selection.requested_hr_resolution,
                actual_hr_resolution=hr_selection.actual_hr_resolution,
                actual_hr_grid_resolution=hr_selection.actual_hr_grid_resolution,
                max_num_tokens=hr_selection.max_num_tokens,
                raw_upsampled_shape=shape_upsample.output_coordinate_shape,
                metadata={
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "shape_decoder_model": shape_decoder_model.key,
                    "shape_decoder_config_path": shape_decoder_model.config_path,
                    "shape_decoder_checkpoint_path": shape_decoder_model.checkpoint_path,
                    "shape_slat_lr_artifact": str(shape_slat_artifact.path),
                    "blocker_next_target": "shape-hr-projection-conditioning",
                },
            )
            completed.append("shape-slat-cascade:upsample")
            completed.append("artifact:shape_slat_hr_coordinates")
            metadata["artifact_paths"] = [
                projection_artifact.path,
                sparse_structure_artifact.path,
                shape_slat_artifact.path,
                shape_hr_coordinates_artifact.path,
            ]
            metadata["shape_slat_hr_coordinates_artifact"] = shape_hr_coordinates_artifact

            shape_hr_stage = pixal3d_stage_with_grid_resolution(
                pixal3d_projection_stage_config("shape_1024"),
                hr_selection.actual_hr_grid_resolution,
            )
            shape_hr_conditioning = build_pixal3d_projection_conditioning(
                projection_hidden_states,
                shape_hr_stage,
                camera_angle_x=camera.camera_angle_x,
                distance=camera.distance,
                mesh_scale=camera.mesh_scale,
                naf_feature_map=shape_hr_naf_feature_map,
            )
            metadata["shape_hr_projection"] = {
                "ready": shape_hr_conditioning.ready,
                "global_shape": tuple(int(dim) for dim in shape_hr_conditioning.global_tokens.shape)
                if shape_hr_conditioning.global_tokens is not None
                else None,
                "projected_shape": tuple(int(dim) for dim in shape_hr_conditioning.projected_features.shape)
                if shape_hr_conditioning.projected_features is not None
                else None,
                "projected_lr_shape": tuple(int(dim) for dim in shape_hr_conditioning.projected_lr_features.shape)
                if shape_hr_conditioning.projected_lr_features is not None
                else None,
                "blocker": shape_hr_conditioning.blocker,
            }
            shape_hr_projected = None
            if shape_hr_conditioning.blocker is not None:
                if shape_hr_naf_feature_map is None:
                    shape_hr_projected, naf_projection_metadata, naf_blocker = runtime_naf_projected_features(
                        stage=shape_hr_conditioning.stage,
                        coordinates=hr_selection.coordinates,
                        lr_projected_features=shape_hr_conditioning.projected_lr_features,
                        stage_label="shape_hr",
                    )
                    metadata["shape_hr_projection"].update(naf_projection_metadata)
                    if naf_blocker is None:
                        metadata["shape_hr_projection"]["ready"] = True
                        metadata["shape_hr_projection"]["blocker"] = None
                    else:
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
                            naf_blocker.stage,
                            naf_blocker.operation,
                            naf_blocker.reason,
                            naf_blocker.metadata,
                            metadata=metadata,
                            artifacts=(
                                projection_artifact.path,
                                sparse_structure_artifact.path,
                                shape_slat_artifact.path,
                                shape_hr_coordinates_artifact.path,
                            ),
                        )
                else:
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
                        "shape-hr-projection-conditioning",
                        shape_hr_conditioning.blocker.operation,
                        shape_hr_conditioning.blocker.reason,
                        shape_hr_conditioning.blocker.metadata,
                        metadata=metadata,
                        artifacts=(
                            projection_artifact.path,
                            sparse_structure_artifact.path,
                            shape_slat_artifact.path,
                            shape_hr_coordinates_artifact.path,
                        ),
                    )
            completed.append("projection-conditioning:shape_1024")
            timings["projection-conditioning:shape_1024"] = time.perf_counter() - started

            assert shape_hr_conditioning.global_tokens is not None
            if shape_hr_projected is None:
                assert shape_hr_conditioning.projected_features is not None
                shape_hr_projected = select_pixal3d_projected_features_at_coordinates(
                    shape_hr_conditioning.projected_features,
                    hr_selection.coordinates,
                    grid_resolution=shape_hr_conditioning.stage.grid_resolution,
                )
            metadata["shape_hr_projection"]["selected_projected_shape"] = tuple(int(dim) for dim in shape_hr_projected.shape)

            shape_hr_slat_model = _pixal3d_model_asset(config.models, "shape_slat_flow_model_1024")
            try:
                shape_hr_slat_config = read_slat_flow_config(self.root, shape_hr_slat_model.config_path)
                shape_hr_probe = probe_shape_slat_forward_boundary(
                    self.root / shape_hr_slat_model.checkpoint_path,
                    shape_hr_slat_config,
                    hr_selection.coordinates,
                    conditioning={
                        "global": shape_hr_conditioning.global_tokens,
                        "proj": shape_hr_projected,
                    },
                    steps=config.shape_slat_sampler.steps,
                    rescale_t=config.shape_slat_sampler.rescale_t,
                    guidance_strength=config.shape_slat_sampler.guidance_strength,
                    guidance_rescale=config.shape_slat_sampler.guidance_rescale,
                    guidance_interval=config.shape_slat_sampler.guidance_interval,
                    sigma_min=config.shape_slat_sampler.sigma_min,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                    "shape-hr-slat-sampling",
                    "run Pixal3D 1024 shape SLat FlowEuler cascade",
                    str(error),
                    {
                        "config_path": shape_hr_slat_model.config_path,
                        "checkpoint_path": shape_hr_slat_model.checkpoint_path,
                    },
                    metadata=metadata,
                    artifacts=(
                        projection_artifact.path,
                        sparse_structure_artifact.path,
                        shape_slat_artifact.path,
                        shape_hr_coordinates_artifact.path,
                    ),
                )

            metadata["shape_slat_hr"] = {
                "coordinate_shape": shape_hr_probe.coordinate_shape,
                "feature_shape": shape_hr_probe.feature_shape,
                "sampled_feature_shape": shape_hr_probe.sampled_feature_shape,
                "completed_blocks": shape_hr_probe.completed_blocks,
                "blocker_operation": shape_hr_probe.blocker_operation,
                "blocker_detail": shape_hr_probe.blocker_detail,
            }
            if shape_hr_probe.sampled_features is None:
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
                    "shape-hr-slat-sampling",
                    shape_hr_probe.blocker_operation,
                    shape_hr_probe.blocker_detail,
                    {
                        "config_path": shape_hr_slat_model.config_path,
                        "checkpoint_path": shape_hr_slat_model.checkpoint_path,
                    },
                    metadata=metadata,
                    artifacts=(
                        projection_artifact.path,
                        sparse_structure_artifact.path,
                        shape_slat_artifact.path,
                        shape_hr_coordinates_artifact.path,
                    ),
                )

            shape_hr_features = _apply_pixal3d_slat_normalization(
                shape_hr_probe.sampled_features,
                config.shape_slat_normalization,
                name="shape_slat_hr",
            )
            shape_hr_slat_artifact = write_pixal3d_shape_slat_npz(
                artifact_dir / "shape_slat_hr.npz",
                hr_selection.coordinates,
                shape_hr_features,
                metadata={
                    "stage": "shape_slat_hr",
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "shape_slat_model": shape_hr_slat_model.key,
                    "shape_slat_config_path": shape_hr_slat_model.config_path,
                    "shape_slat_checkpoint_path": shape_hr_slat_model.checkpoint_path,
                    "shape_slat_hr_coordinates_artifact": str(shape_hr_coordinates_artifact.path),
                    "actual_hr_resolution": hr_selection.actual_hr_resolution,
                    "actual_hr_grid_resolution": hr_selection.actual_hr_grid_resolution,
                    "blocker_next_target": "texture-projection-conditioning",
                },
            )
            completed.append("shape-slat-sampling:1024")
            completed.append("artifact:shape_slat_hr")
            metadata["artifact_paths"] = [
                projection_artifact.path,
                sparse_structure_artifact.path,
                shape_slat_artifact.path,
                shape_hr_coordinates_artifact.path,
                shape_hr_slat_artifact.path,
            ]
            metadata["shape_slat_hr_artifact"] = shape_hr_slat_artifact
            texture_stage = pixal3d_stage_with_grid_resolution(
                pixal3d_projection_stage_config("tex_1024"),
                hr_selection.actual_hr_grid_resolution,
            )
            texture_conditioning = build_pixal3d_projection_conditioning(
                projection_hidden_states,
                texture_stage,
                camera_angle_x=camera.camera_angle_x,
                distance=camera.distance,
                mesh_scale=camera.mesh_scale,
                naf_feature_map=texture_naf_feature_map,
            )
            metadata["texture_projection"] = {
                "ready": texture_conditioning.ready,
                "global_shape": tuple(int(dim) for dim in texture_conditioning.global_tokens.shape)
                if texture_conditioning.global_tokens is not None
                else None,
                "projected_shape": tuple(int(dim) for dim in texture_conditioning.projected_features.shape)
                if texture_conditioning.projected_features is not None
                else None,
                "projected_lr_shape": tuple(int(dim) for dim in texture_conditioning.projected_lr_features.shape)
                if texture_conditioning.projected_lr_features is not None
                else None,
                "blocker": texture_conditioning.blocker,
            }
            texture_projected = None
            if texture_conditioning.blocker is not None:
                if texture_naf_feature_map is None:
                    texture_projected, naf_projection_metadata, naf_blocker = runtime_naf_projected_features(
                        stage=texture_conditioning.stage,
                        coordinates=hr_selection.coordinates,
                        lr_projected_features=texture_conditioning.projected_lr_features,
                        stage_label="texture",
                    )
                    metadata["texture_projection"].update(naf_projection_metadata)
                    if naf_blocker is None:
                        metadata["texture_projection"]["ready"] = True
                        metadata["texture_projection"]["blocker"] = None
                    else:
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
                            naf_blocker.stage,
                            naf_blocker.operation,
                            naf_blocker.reason,
                            naf_blocker.metadata,
                            metadata=metadata,
                            artifacts=(
                                projection_artifact.path,
                                sparse_structure_artifact.path,
                                shape_slat_artifact.path,
                                shape_hr_coordinates_artifact.path,
                                shape_hr_slat_artifact.path,
                            ),
                        )
                else:
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
                        "texture-projection-conditioning",
                        texture_conditioning.blocker.operation,
                        texture_conditioning.blocker.reason,
                        texture_conditioning.blocker.metadata,
                        metadata=metadata,
                        artifacts=(
                            projection_artifact.path,
                            sparse_structure_artifact.path,
                            shape_slat_artifact.path,
                            shape_hr_coordinates_artifact.path,
                            shape_hr_slat_artifact.path,
                        ),
                    )
            completed.append("projection-conditioning:tex_1024")
            timings["projection-conditioning:tex_1024"] = time.perf_counter() - started

            assert texture_conditioning.global_tokens is not None
            if texture_projected is None:
                assert texture_conditioning.projected_features is not None
                texture_projected = select_pixal3d_projected_features_at_coordinates(
                    texture_conditioning.projected_features,
                    hr_selection.coordinates,
                    grid_resolution=texture_conditioning.stage.grid_resolution,
                )
            metadata["texture_projection"]["selected_projected_shape"] = tuple(int(dim) for dim in texture_projected.shape)

            texture_slat_model = _pixal3d_model_asset(config.models, "tex_slat_flow_model_1024")
            shape_hr_features_for_texture = _remove_pixal3d_slat_normalization(
                shape_hr_features,
                config.shape_slat_normalization,
                name="shape_slat_hr",
            )
            metadata["texture_slat"] = {
                "normalized_shape_feature_shape": tuple(int(dim) for dim in shape_hr_features_for_texture.shape),
            }
            try:
                texture_slat_config = read_slat_flow_config(self.root, texture_slat_model.config_path)
                texture_probe = probe_texture_slat_forward_boundary(
                    self.root / texture_slat_model.checkpoint_path,
                    texture_slat_config,
                    hr_selection.coordinates,
                    shape_hr_features_for_texture,
                    conditioning={
                        "global": texture_conditioning.global_tokens,
                        "proj": texture_projected,
                    },
                    steps=config.texture_slat_sampler.steps,
                    rescale_t=config.texture_slat_sampler.rescale_t,
                    guidance_strength=config.texture_slat_sampler.guidance_strength,
                    guidance_rescale=config.texture_slat_sampler.guidance_rescale,
                    guidance_interval=config.texture_slat_sampler.guidance_interval,
                    sigma_min=config.texture_slat_sampler.sigma_min,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                    "texture-slat-sampling",
                    "run Pixal3D 1024 texture SLat FlowEuler cascade",
                    str(error),
                    {
                        "config_path": texture_slat_model.config_path,
                        "checkpoint_path": texture_slat_model.checkpoint_path,
                    },
                    metadata=metadata,
                    artifacts=(
                        projection_artifact.path,
                        sparse_structure_artifact.path,
                        shape_slat_artifact.path,
                        shape_hr_coordinates_artifact.path,
                        shape_hr_slat_artifact.path,
                    ),
                )

            metadata["texture_slat"].update(
                {
                    "coordinate_shape": texture_probe.coordinate_shape,
                    "shape_feature_shape": texture_probe.shape_feature_shape,
                    "noise_feature_shape": texture_probe.noise_feature_shape,
                    "concat_feature_shape": texture_probe.concat_feature_shape,
                    "sampled_feature_shape": texture_probe.sampled_feature_shape,
                    "completed_blocks": texture_probe.completed_blocks,
                    "blocker_operation": texture_probe.blocker_operation,
                    "blocker_detail": texture_probe.blocker_detail,
                }
            )
            if texture_probe.sampled_features is None:
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
                    "texture-slat-sampling",
                    texture_probe.blocker_operation,
                    texture_probe.blocker_detail,
                    {
                        "config_path": texture_slat_model.config_path,
                        "checkpoint_path": texture_slat_model.checkpoint_path,
                    },
                    metadata=metadata,
                    artifacts=(
                        projection_artifact.path,
                        sparse_structure_artifact.path,
                        shape_slat_artifact.path,
                        shape_hr_coordinates_artifact.path,
                        shape_hr_slat_artifact.path,
                    ),
                )

            texture_features = _apply_pixal3d_slat_normalization(
                texture_probe.sampled_features,
                config.texture_slat_normalization,
                name="texture_slat",
            )
            texture_slat_artifact = write_pixal3d_texture_slat_npz(
                artifact_dir / "texture_slat.npz",
                hr_selection.coordinates,
                texture_features,
                metadata={
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "texture_slat_model": texture_slat_model.key,
                    "texture_slat_config_path": texture_slat_model.config_path,
                    "texture_slat_checkpoint_path": texture_slat_model.checkpoint_path,
                    "shape_slat_hr_artifact": str(shape_hr_slat_artifact.path),
                    "actual_hr_resolution": hr_selection.actual_hr_resolution,
                    "actual_hr_grid_resolution": hr_selection.actual_hr_grid_resolution,
                    "blocker_next_target": "latent-decoding",
                },
            )
            completed.append("texture-slat-sampling:1024")
            completed.append("artifact:texture_slat")
            metadata["artifact_paths"] = [
                projection_artifact.path,
                sparse_structure_artifact.path,
                shape_slat_artifact.path,
                shape_hr_coordinates_artifact.path,
                shape_hr_slat_artifact.path,
                texture_slat_artifact.path,
            ]
            metadata["texture_slat_artifact"] = texture_slat_artifact
            texture_slat_artifacts = (
                projection_artifact.path,
                sparse_structure_artifact.path,
                shape_slat_artifact.path,
                shape_hr_coordinates_artifact.path,
                shape_hr_slat_artifact.path,
                texture_slat_artifact.path,
            )

            try:
                shape_decode = run_shape_decoder_to_fields(
                    self.root / shape_decoder_model.checkpoint_path,
                    shape_decoder_config,
                    hr_selection.coordinates,
                    shape_hr_features,
                    decoder_token_limit=max_num_tokens,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
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
                    "shape-decoder",
                    "decode Pixal3D HR shape SLat into FlexiDualGrid fields",
                    str(error),
                    {
                        "config_path": shape_decoder_model.config_path,
                        "checkpoint_path": shape_decoder_model.checkpoint_path,
                        "decoder_token_limit": max_num_tokens,
                        "shape_coordinates_shape": tuple(int(dim) for dim in hr_selection.coordinates.shape),
                        "shape_features_shape": tuple(int(dim) for dim in shape_hr_features.shape),
                    },
                    metadata=metadata,
                    artifacts=texture_slat_artifacts,
                )

            metadata["shape_decoder"] = {
                "shape_decoder_model": shape_decoder_model.key,
                "shape_decoder_config_path": shape_decoder_model.config_path,
                "shape_decoder_checkpoint_path": shape_decoder_model.checkpoint_path,
                "coordinate_shape": shape_decode.probe.coordinate_shape,
                "feature_shape": shape_decode.probe.feature_shape,
                "decoder_output_coordinate_shape": shape_decode.probe.decoder_output_coordinate_shape,
                "decoder_output_shape": shape_decode.probe.decoder_output_shape,
                "completed_levels": shape_decode.probe.completed_levels,
                "subdivision_shapes": shape_decode.probe.subdivision_shapes,
                "decoder_token_limit": shape_decode.probe.reference_token_limit,
            }
            shape_decoder_artifact = write_pixal3d_shape_decoder_npz(
                artifact_dir / "shape_decoder_fields.npz",
                shape_decode.coordinates,
                shape_decode.fields,
                subdivisions=shape_decode.subdivisions,
                metadata={
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "shape_decoder_model": shape_decoder_model.key,
                    "shape_decoder_config_path": shape_decoder_model.config_path,
                    "shape_decoder_checkpoint_path": shape_decoder_model.checkpoint_path,
                    "shape_slat_hr_artifact": str(shape_hr_slat_artifact.path),
                    "texture_slat_artifact": str(texture_slat_artifact.path),
                    "actual_hr_resolution": hr_selection.actual_hr_resolution,
                    "actual_hr_grid_resolution": hr_selection.actual_hr_grid_resolution,
                    "decoder_token_limit": shape_decode.probe.reference_token_limit,
                    "blocker_next_target": "texture-decoder",
                },
            )
            completed.append("shape-decoder")
            completed.append("artifact:shape_decoder_fields")
            shape_decode_artifacts = (*texture_slat_artifacts, shape_decoder_artifact.path)
            metadata["artifact_paths"] = list(shape_decode_artifacts)
            metadata["shape_decoder_artifact"] = shape_decoder_artifact

            texture_decoder_model = None
            try:
                texture_decoder_model = _pixal3d_model_asset(config.models, "tex_slat_decoder")
                texture_decoder_config = read_structured_latent_decoder_config(
                    self.root, texture_decoder_model.config_path
                )
                texture_decode = run_texture_decoder_to_representation(
                    self.root / texture_decoder_model.checkpoint_path,
                    texture_decoder_config,
                    hr_selection.coordinates,
                    texture_features,
                    guide_subdivisions=shape_decode.subdivisions,
                    decoder_token_limit=max_num_tokens,
                    decode_resolution=hr_selection.actual_hr_resolution,
                    shape_decoder_coordinates=shape_decode.coordinates,
                )
            except (FileNotFoundError, RuntimeError, ValueError) as error:
                decoder_metadata = {
                    "decoder_token_limit": max_num_tokens,
                    "texture_coordinates_shape": tuple(int(dim) for dim in hr_selection.coordinates.shape),
                    "texture_features_shape": tuple(int(dim) for dim in texture_features.shape),
                    "guide_subdivision_shapes": tuple(
                        tuple(int(dim) for dim in subdivision.shape) for subdivision in shape_decode.subdivisions
                    ),
                }
                if texture_decoder_model is not None:
                    decoder_metadata.update(
                        {
                            "config_path": texture_decoder_model.config_path,
                            "checkpoint_path": texture_decoder_model.checkpoint_path,
                        }
                    )
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
                    "texture-decoder",
                    "decode Pixal3D texture SLat into guided PBR voxels",
                    str(error),
                    decoder_metadata,
                    metadata=metadata,
                    artifacts=shape_decode_artifacts,
                )

            metadata["texture_decoder"] = {
                "texture_decoder_model": texture_decoder_model.key,
                "texture_decoder_config_path": texture_decoder_model.config_path,
                "texture_decoder_checkpoint_path": texture_decoder_model.checkpoint_path,
                "coordinate_shape": texture_decode.probe.coordinate_shape,
                "feature_shape": texture_decode.probe.feature_shape,
                "decoder_output_coordinate_shape": texture_decode.probe.decoder_output_coordinate_shape,
                "decoder_output_shape": texture_decode.probe.decoder_output_shape,
                "completed_levels": texture_decode.probe.completed_levels,
                "subdivision_shapes": texture_decode.probe.subdivision_shapes,
                "guide_subdivision_shapes": texture_decode.guide_subdivision_shapes,
                "spatial_shape": texture_decode.spatial_shape,
                "batch_size": texture_decode.batch_size,
                "decode_resolution": texture_decode.decode_resolution,
                "voxel_size": texture_decode.voxel_size,
                "shape_decoder_coordinate_shape": texture_decode.shape_decoder_coordinate_shape,
                "decoder_token_limit": texture_decode.probe.reference_token_limit,
            }
            texture_decoder_artifact = write_pixal3d_texture_decoder_npz(
                artifact_dir / "texture_decoder_pbr.npz",
                texture_decode.coordinates,
                texture_decode.attributes,
                spatial_shape=texture_decode.spatial_shape,
                batch_size=texture_decode.batch_size,
                decode_resolution=texture_decode.decode_resolution,
                voxel_size=texture_decode.voxel_size,
                metadata={
                    "pipeline_type": pipeline_type,
                    "manual_fov": manual_fov,
                    "seed": seed,
                    "texture_decoder_model": texture_decoder_model.key,
                    "texture_decoder_config_path": texture_decoder_model.config_path,
                    "texture_decoder_checkpoint_path": texture_decoder_model.checkpoint_path,
                    "shape_decoder_artifact": str(shape_decoder_artifact.path),
                    "texture_slat_artifact": str(texture_slat_artifact.path),
                    "actual_hr_resolution": hr_selection.actual_hr_resolution,
                    "actual_hr_grid_resolution": hr_selection.actual_hr_grid_resolution,
                    "decoder_token_limit": texture_decode.probe.reference_token_limit,
                    "guide_subdivision_shapes": texture_decode.guide_subdivision_shapes,
                    "shape_decoder_coordinate_shape": texture_decode.shape_decoder_coordinate_shape,
                    "blocker_next_target": "mesh-extraction",
                },
            )
            completed.append("texture-decoder")
            completed.append("artifact:texture_decoder_pbr")
            decode_artifacts = (*shape_decode_artifacts, texture_decoder_artifact.path)
            metadata["artifact_paths"] = list(decode_artifacts)
            metadata["texture_decoder_artifact"] = texture_decoder_artifact
            clear_mlx_cache()

            try:
                mesh = flexi_dual_grid_fields_to_mesh(
                    shape_decode.coordinates,
                    shape_decode.fields,
                    grid_size=hr_selection.actual_hr_resolution,
                )
                postprocess_result = postprocess_trellis2_mesh_for_glb(mesh, target_faces=glb_target_faces)
                baked_texture = bake_trellis2_texture_fields_mac_native(
                    postprocess_result.mesh,
                    texture_decode.coordinates,
                    texture_decode.attributes,
                    decode_resolution=hr_selection.actual_hr_resolution,
                    texture_size=texture_size,
                    xatlas_face_guard=xatlas_face_guard,
                    xatlas_parallel_chunks=xatlas_parallel_chunks,
                    texture_bake_backend=texture_bake_backend,
                    projection_source_mesh=getattr(postprocess_result, "source_mesh", None),
                )
            except (ImportError, RuntimeError, ValueError) as error:
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
                    "mesh-export",
                    "extract Pixal3D FlexiDualGrid mesh and bake decoded PBR voxels",
                    str(error),
                    {
                        "shape_decoder_coordinates_shape": tuple(int(dim) for dim in shape_decode.coordinates.shape),
                        "shape_decoder_fields_shape": tuple(int(dim) for dim in shape_decode.fields.shape),
                        "texture_decoder_coordinates_shape": tuple(int(dim) for dim in texture_decode.coordinates.shape),
                        "texture_decoder_attributes_shape": tuple(int(dim) for dim in texture_decode.attributes.shape),
                        "shape_decoder_artifact_path": str(shape_decoder_artifact.path),
                        "texture_decoder_artifact_path": str(texture_decoder_artifact.path),
                        "actual_hr_resolution": hr_selection.actual_hr_resolution,
                        "texture_size": texture_size,
                        "glb_target_faces": glb_target_faces,
                        "xatlas_face_guard": xatlas_face_guard,
                        "xatlas_parallel_chunks": xatlas_parallel_chunks,
                        "texture_bake_backend": texture_bake_backend,
                    },
                    metadata=metadata,
                    artifacts=decode_artifacts,
                )

            metadata["mesh_export"] = {
                "source_mesh_vertices": int(mesh.vertices.shape[0]),
                "source_mesh_faces": int(mesh.faces.shape[0]),
                "postprocess_stats": postprocess_result.stats,
                "baked_vertices_shape": tuple(int(dim) for dim in baked_texture.vertices.shape),
                "baked_faces_shape": tuple(int(dim) for dim in baked_texture.faces.shape),
                "baked_uv_shape": tuple(int(dim) for dim in baked_texture.uvs.shape),
                "texture_size": int(baked_texture.texture_size),
                "voxel_count": int(baked_texture.voxel_count),
                "coverage_ratio": float(baked_texture.coverage_ratio),
                "raw_coverage_ratio": float(baked_texture.raw_coverage_ratio),
                "bake_backend": baked_texture.backend,
                "unwrap_backend": baked_texture.unwrap_backend,
                "unwrap_chunks": baked_texture.unwrap_chunks,
                "unwrap_chart_count": baked_texture.unwrap_chart_count,
                "unwrap_utilization": float(baked_texture.unwrap_utilization),
                "xatlas_face_guard": baked_texture.xatlas_face_guard,
                "xatlas_face_guard_mode": baked_texture.xatlas_face_guard_mode,
                "sampled_texel_count": baked_texture.sampled_texel_count,
                "missing_texel_count": baked_texture.missing_texel_count,
                "out_of_grid_texel_count": baked_texture.out_of_grid_texel_count,
                "source_projection_used": baked_texture.source_projection_used,
                "source_projection_detail": baked_texture.source_projection_detail,
            }
            completed.append("mesh-export")

            try:
                glb_artifact = write_pixal3d_textured_glb(
                    baked_texture,
                    output_path,
                    metadata={
                        "pipeline_type": pipeline_type,
                        "manual_fov": manual_fov,
                        "seed": seed,
                        "texture_size": int(baked_texture.texture_size),
                        "coverage_ratio": float(baked_texture.coverage_ratio),
                        "raw_coverage_ratio": float(baked_texture.raw_coverage_ratio),
                        "bake_backend": baked_texture.backend,
                        "unwrap_backend": baked_texture.unwrap_backend,
                        "shape_decoder_artifact": str(shape_decoder_artifact.path),
                        "texture_decoder_artifact": str(texture_decoder_artifact.path),
                    },
                )
            except (OSError, ValueError) as error:
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
                    "glb-export",
                    "write Pixal3D textured GLB",
                    str(error),
                    {
                        "output_path": str(output_path),
                        "shape_decoder_artifact_path": str(shape_decoder_artifact.path),
                        "texture_decoder_artifact_path": str(texture_decoder_artifact.path),
                    },
                    metadata=metadata,
                    artifacts=decode_artifacts,
                )

            completed.append("artifact:textured_glb")
            final_artifacts = (*decode_artifacts, glb_artifact.path)
            metadata["artifact_paths"] = list(final_artifacts)
            metadata["textured_glb_artifact"] = glb_artifact
            metadata["memory_after"] = mlx_memory_snapshot().as_dict()
            metadata["timings_sec"] = timings
            return Pixal3DGenerationResult(
                trace=Pixal3DInferenceTrace(
                    root=self.root,
                    image_path=image_path,
                    completed_stages=tuple(completed),
                    pipeline_type=pipeline_type,
                    manual_fov=manual_fov,
                    seed=seed,
                    max_num_tokens=max_num_tokens,
                    output_path=output_path,
                    blocker=None,
                    metadata=metadata,
                ),
                artifacts=final_artifacts,
            )
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
            "sparse-structure-decoding",
            decoder_probe.blocker_operation,
            decoder_probe.blocker_detail,
            {
                "config_path": sparse_decoder_model.config_path,
                "checkpoint_path": sparse_decoder_model.checkpoint_path,
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


def _apply_pixal3d_slat_normalization(features: mx.array, normalization: object, *, name: str) -> mx.array:
    feature_width = int(features.shape[-1])
    mean_values = tuple(float(value) for value in getattr(normalization, "mean"))
    std_values = tuple(float(value) for value in getattr(normalization, "std"))
    if len(mean_values) != feature_width or len(std_values) != feature_width:
        raise ValueError(
            f"{name} normalization width mismatch: expected {feature_width}, "
            f"got mean={len(mean_values)} std={len(std_values)}"
        )
    mean = mx.array(mean_values, dtype=mx.float32)[None, :]
    std = mx.array(std_values, dtype=mx.float32)[None, :]
    return features.astype(mx.float32) * std + mean


def _validate_pixal3d_export_guards(
    *,
    texture_size: int,
    glb_target_faces: int,
    xatlas_parallel_chunks: int,
    texture_bake_backend: str,
) -> str | None:
    if texture_size <= 0:
        return f"texture_size must be positive, got {texture_size}"
    if glb_target_faces <= 0:
        return f"glb_target_faces must be positive, got {glb_target_faces}"
    if xatlas_parallel_chunks < 0:
        return f"xatlas_parallel_chunks must be non-negative, got {xatlas_parallel_chunks}"
    if texture_bake_backend not in TRELLIS2_TEXTURE_BAKE_BACKENDS:
        return f"texture_bake_backend must be one of {TRELLIS2_TEXTURE_BAKE_BACKENDS}, got {texture_bake_backend}"
    return None


def _pixal3d_patch_feature_map_bchw(hidden_states: mx.array) -> mx.array:
    if hidden_states.ndim != 3:
        raise ValueError(f"projection_hidden_states must have shape [B,N,C], got {hidden_states.shape}")
    batch, token_count, channels = (int(dim) for dim in hidden_states.shape)
    global_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS
    patch_count = token_count - global_count
    side = int(patch_count**0.5)
    if patch_count <= 0 or side * side != patch_count:
        raise ValueError("projection_hidden_states patch token count must be square for NAF")
    patch_tokens = hidden_states[:, global_count:, :].reshape(batch, side, side, channels)
    return mx.transpose(patch_tokens, (0, 3, 1, 2))


def _pixal3d_project_sparse_coordinates(
    coordinates: mx.array,
    *,
    stage,
    camera_angle_x: float | mx.array,
    distance: float | mx.array,
    mesh_scale: float | mx.array,
) -> mx.array:
    coords = np.array(coordinates, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"sparse coordinates must have shape [N,4], got {coordinates.shape}")
    if coords.shape[0] == 0:
        return mx.zeros((1, 0, 2), dtype=mx.float32)
    if np.any(coords[:, 0] != 0):
        raise ValueError("NAF coordinate projection currently supports batch index 0 only")
    spatial = coords[:, 1:]
    grid_resolution = int(stage.grid_resolution)
    if np.any(spatial < 0) or np.any(spatial >= grid_resolution):
        raise ValueError(f"sparse coordinates out of bounds for grid_resolution={grid_resolution}")
    flat_index = spatial[:, 0] * grid_resolution * grid_resolution + spatial[:, 1] * grid_resolution + spatial[:, 2]
    grid_points = pixal3d_projection_grid_points(grid_resolution)
    selected_points = grid_points[mx.array(flat_index.astype(np.int32))]
    return project_pixal3d_points_to_image(
        selected_points,
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
        image_resolution=stage.image_size,
    ).points_2d


def _remove_pixal3d_slat_normalization(features: mx.array, normalization: object, *, name: str) -> mx.array:
    feature_width = int(features.shape[-1])
    mean_values = tuple(float(value) for value in getattr(normalization, "mean"))
    std_values = tuple(float(value) for value in getattr(normalization, "std"))
    if len(mean_values) != feature_width or len(std_values) != feature_width:
        raise ValueError(
            f"{name} normalization width mismatch: expected {feature_width}, "
            f"got mean={len(mean_values)} std={len(std_values)}"
        )
    mean = mx.array(mean_values, dtype=mx.float32)[None, :]
    std = mx.array(std_values, dtype=mx.float32)[None, :]
    return (features.astype(mx.float32) - mean) / std


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


def _dino_blocker_reason(reason: str, root: Path) -> str:
    if str(root) in reason and DINOv3_VITL16_REPO_ID in reason:
        return reason
    return f"{DINOv3_VITL16_REPO_ID} assets at {root}: {reason}"


def _dino_blocker_metadata(blocker: object, root: Path) -> dict[str, object]:
    return {
        "dino_root": str(root),
        "repo_id": DINOv3_VITL16_REPO_ID,
        "access_note": DINOv3_ACCESS_NOTE,
        "download_command": " ".join(dinov3_download_command(root)),
        "reference": getattr(blocker, "reference", None),
        "next_slice": getattr(blocker, "next_slice", None),
    }


def _pixal3d_model_asset(models: tuple[object, ...], key: str) -> object:
    for model in models:
        if getattr(model, "key", None) == key:
            return model
    raise ValueError(f"Pixal3D pipeline config is missing model key {key!r}")
