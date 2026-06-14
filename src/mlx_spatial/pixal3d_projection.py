"""Pixal3D pixel-aligned projection conditioning."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import mlx.core as mx
import numpy as np


PIXAL3D_DINOV3_EMBED_DIM = 1024
PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS = 4
PIXAL3D_DINOV3_PATCH_SIZE = 16


@dataclass(frozen=True)
class Pixal3DProjectionStageConfig:
    """Pixal3D projection-conditioning settings for one generation stage."""

    name: str
    image_size: int
    grid_resolution: int
    use_naf_upsample: bool = False
    naf_target_size: int | None = None
    patch_size: int = PIXAL3D_DINOV3_PATCH_SIZE

    @property
    def projected_token_count(self) -> int:
        return self.grid_resolution**3

    @property
    def expected_patch_grid(self) -> tuple[int, int]:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"image_size={self.image_size} is not divisible by patch_size={self.patch_size}")
        side = self.image_size // self.patch_size
        return (side, side)

    def expected_projected_channels(self, embed_dim: int = PIXAL3D_DINOV3_EMBED_DIM) -> int:
        return embed_dim * (2 if self.use_naf_upsample else 1)


@dataclass(frozen=True)
class Pixal3DProjectionBlocker:
    """Structured blocker for projection-conditioning boundaries."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class Pixal3DProjectedPoints:
    """Projected grid points and visibility."""

    points_2d: mx.array
    depth: mx.array
    valid_mask: mx.array


@dataclass(frozen=True)
class Pixal3DProjectionConditioning:
    """Global and projected image features consumed by Pixal3D flow blocks."""

    stage: Pixal3DProjectionStageConfig
    global_tokens: mx.array | None
    projected_features: mx.array | None
    projected_lr_features: mx.array | None = None
    blocker: Pixal3DProjectionBlocker | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.global_tokens is not None and self.projected_features is not None and self.blocker is None


PIXAL3D_PROJECTION_STAGE_CONFIGS = {
    "ss": Pixal3DProjectionStageConfig(name="ss", image_size=512, grid_resolution=16),
    "shape_512": Pixal3DProjectionStageConfig(
        name="shape_512",
        image_size=512,
        grid_resolution=32,
        use_naf_upsample=True,
        naf_target_size=512,
    ),
    "shape_1024": Pixal3DProjectionStageConfig(
        name="shape_1024",
        image_size=1024,
        grid_resolution=64,
        use_naf_upsample=True,
        naf_target_size=512,
    ),
    "tex_1024": Pixal3DProjectionStageConfig(
        name="tex_1024",
        image_size=1024,
        grid_resolution=64,
        use_naf_upsample=True,
        naf_target_size=1024,
    ),
}


def pixal3d_projection_stage_config(name: str) -> Pixal3DProjectionStageConfig:
    """Return a named Pixal3D projection stage config."""

    try:
        return PIXAL3D_PROJECTION_STAGE_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"unknown Pixal3D projection stage: {name!r}") from error


def pixal3d_projection_grid_points(grid_resolution: int) -> mx.array:
    """Create upstream-compatible rotated 3D projection grid points."""

    values = mx.linspace(-1.0, 1.0, grid_resolution)
    x, y, z = mx.meshgrid(values, values, values, indexing="ij")
    grid = mx.stack((x, y, z), axis=-1)
    rotation = mx.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=grid.dtype,
    )
    return (grid @ rotation.T).reshape(-1, 3)


def pixal3d_front_view_transform(distance: float | mx.array) -> mx.array:
    """Return the Pixal3D front-view transform with camera distance applied."""

    value = _array_scalar(distance)
    return mx.stack(
        [
            mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32),
            mx.stack(
                [
                    mx.array(0.0, dtype=mx.float32),
                    mx.array(0.0, dtype=mx.float32),
                    mx.array(-1.0, dtype=mx.float32),
                    -value.astype(mx.float32),
                ]
            ),
            mx.array([0.0, 1.0, 0.0, 0.0], dtype=mx.float32),
            mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32),
        ],
        axis=0,
    )


def project_pixal3d_points_to_image(
    points_3d: mx.array,
    *,
    camera_angle_x: float | mx.array,
    distance: float | mx.array,
    mesh_scale: float | mx.array = 1.0,
    image_resolution: int,
    transform_matrix: mx.array | None = None,
) -> Pixal3DProjectedPoints:
    """Project 3D points to Pixal3D image coordinates."""

    points = _ensure_batched_points(points_3d)
    batch = int(points.shape[0])
    cam_angle = _batch_vector(camera_angle_x, batch)
    scale = _batch_vector(mesh_scale, batch)
    points = points / scale[:, None, None] / 2.0

    if transform_matrix is None:
        transform = pixal3d_front_view_transform(distance)
        transform_matrix = mx.broadcast_to(transform[None, :, :], (batch, 4, 4))
    elif transform_matrix.ndim == 2:
        transform_matrix = mx.broadcast_to(transform_matrix[None, :, :], (batch, 4, 4))
    if transform_matrix.ndim != 3 or tuple(int(dim) for dim in transform_matrix.shape[-2:]) != (4, 4):
        raise ValueError(f"transform_matrix must have shape [B,4,4] or [4,4], got {transform_matrix.shape}")

    ones = mx.ones((*points.shape[:2], 1), dtype=points.dtype)
    points_h = mx.concatenate([points, ones], axis=-1)
    world_to_camera = _invert_rigid_transform_4x4(transform_matrix).astype(points.dtype)
    points_camera = mx.matmul(points_h, mx.swapaxes(world_to_camera, -1, -2))[..., :3]

    x_cam = points_camera[..., 0]
    y_cam = points_camera[..., 1]
    z_cam = points_camera[..., 2]
    depth = -z_cam
    focal = 16.0 / mx.tan(cam_angle / 2.0)
    focal_pixels = (focal * image_resolution / 32.0)[:, None]

    z_safe = -z_cam + 1e-8
    x_pixel = focal_pixels * x_cam / z_safe + image_resolution / 2.0
    y_pixel = -(focal_pixels * y_cam / z_safe) + image_resolution / 2.0
    points_2d = mx.stack([x_pixel, y_pixel], axis=-1)
    valid = (
        (x_pixel >= 0)
        & (x_pixel < image_resolution)
        & (y_pixel >= 0)
        & (y_pixel < image_resolution)
        & (depth > 0)
    )
    return Pixal3DProjectedPoints(points_2d=points_2d, depth=depth, valid_mask=valid)


def _invert_rigid_transform_4x4(transform_matrix: mx.array) -> mx.array:
    """Invert a batched rigid camera transform without GPU-unsupported linalg ops."""

    matrix = transform_matrix.astype(mx.float32)
    rotation = matrix[:, :3, :3]
    translation = matrix[:, :3, 3]
    rotation_inv = mx.swapaxes(rotation, -1, -2)
    translation_inv = -mx.matmul(rotation_inv, translation[..., None])[..., 0]
    top = mx.concatenate([rotation_inv, translation_inv[..., None]], axis=-1)
    bottom = mx.broadcast_to(mx.array([0.0, 0.0, 0.0, 1.0], dtype=mx.float32)[None, None, :], (int(matrix.shape[0]), 1, 4))
    return mx.concatenate([top, bottom], axis=1)


def sample_pixal3d_feature_map(
    feature_map: mx.array,
    image_points: mx.array,
    *,
    image_resolution: int,
    layout: Literal["BHWC", "BCHW"] = "BHWC",
) -> mx.array:
    """Bilinearly sample a feature map at image pixel coordinates with border padding."""

    if layout == "BCHW":
        feature_map = mx.transpose(feature_map, (0, 2, 3, 1))
        layout = "BHWC"
    elif layout != "BHWC":
        raise ValueError("layout must be BHWC or BCHW")
    if feature_map.ndim != 4 or image_points.ndim != 3 or image_points.shape[-1] != 2:
        raise ValueError("expected feature_map [B,H,W,C] and image_points [B,K,2]")

    batch, height, width, channels = (int(dim) for dim in feature_map.shape)
    if int(image_points.shape[0]) != batch:
        raise ValueError("feature_map and image_points batch dimensions must match")

    source_x = (image_points[..., 0] + 0.5) * width / image_resolution - 0.5
    source_y = (image_points[..., 1] + 0.5) * height / image_resolution - 0.5
    source_x = mx.clip(source_x, 0.0, float(width - 1))
    source_y = mx.clip(source_y, 0.0, float(height - 1))

    x0 = mx.floor(source_x).astype(mx.int32)
    y0 = mx.floor(source_y).astype(mx.int32)
    x1 = mx.clip(x0 + 1, 0, width - 1)
    y1 = mx.clip(y0 + 1, 0, height - 1)
    wx = (source_x - x0.astype(source_x.dtype))[..., None]
    wy = (source_y - y0.astype(source_y.dtype))[..., None]

    flat = feature_map.reshape(batch, height * width, channels)
    v00 = _gather_flat_pixels(flat, y0 * width + x0)
    v01 = _gather_flat_pixels(flat, y0 * width + x1)
    v10 = _gather_flat_pixels(flat, y1 * width + x0)
    v11 = _gather_flat_pixels(flat, y1 * width + x1)
    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    return top * (1.0 - wy) + bottom * wy


def build_pixal3d_projection_conditioning(
    hidden_states: mx.array,
    stage: str | Pixal3DProjectionStageConfig,
    *,
    camera_angle_x: float | mx.array,
    distance: float | mx.array,
    mesh_scale: float | mx.array = 1.0,
    patch_grid: tuple[int, int] | None = None,
    num_register_tokens: int = PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS,
    naf_feature_map: mx.array | None = None,
    naf_layout: Literal["BHWC", "BCHW"] = "BHWC",
) -> Pixal3DProjectionConditioning:
    """Build Pixal3D global/projection conditioning from DINOv3 hidden states."""

    config = pixal3d_projection_stage_config(stage) if isinstance(stage, str) else stage
    if hidden_states.ndim != 3:
        return _projection_blocked(config, "hidden-state-validation", "expected DINOv3 hidden states [B,N,C]", hidden_states.shape)

    batch, token_count, channels = (int(dim) for dim in hidden_states.shape)
    global_count = 1 + int(num_register_tokens)
    if token_count <= global_count:
        return _projection_blocked(config, "hidden-state-validation", "not enough DINOv3 tokens for cls/register plus patches", hidden_states.shape)

    patch_count = token_count - global_count
    if patch_grid is None:
        side = int(patch_count**0.5)
        if side * side != patch_count:
            return _projection_blocked(config, "patch-grid-validation", "patch token count is not square; pass patch_grid", hidden_states.shape)
        patch_grid = (side, side)
    if patch_grid[0] * patch_grid[1] != patch_count:
        return _projection_blocked(config, "patch-grid-validation", "patch_grid does not match patch token count", patch_grid)
    try:
        expected_patch_grid = config.expected_patch_grid
    except ValueError as error:
        return _projection_blocked(config, "stage-patch-grid-validation", str(error), patch_grid)
    if tuple(int(value) for value in patch_grid) != expected_patch_grid:
        return Pixal3DProjectionConditioning(
            stage=config,
            global_tokens=None,
            projected_features=None,
            blocker=Pixal3DProjectionBlocker(
                stage="projection-conditioning",
                operation="stage-patch-grid-validation",
                reason=(
                    f"{config.name} expects DINO patch grid {expected_patch_grid} "
                    f"from image_size={config.image_size}, got {tuple(int(value) for value in patch_grid)}"
                ),
                metadata={
                    "stage": config.name,
                    "image_size": config.image_size,
                    "patch_size": config.patch_size,
                    "expected_patch_grid": expected_patch_grid,
                    "actual_patch_grid": tuple(int(value) for value in patch_grid),
                    "hidden_state_shape": tuple(int(dim) for dim in hidden_states.shape),
                },
            ),
            metadata={
                "stage": config.name,
                "image_size": config.image_size,
                "patch_size": config.patch_size,
                "expected_patch_grid": expected_patch_grid,
                "actual_patch_grid": tuple(int(value) for value in patch_grid),
            },
        )

    conditioning_metadata = {
        "stage": config.name,
        "image_size": config.image_size,
        "grid_resolution": config.grid_resolution,
        "patch_size": config.patch_size,
        "patch_grid": tuple(int(value) for value in patch_grid),
        "patch_token_count": int(patch_count),
        "hidden_state_shape": tuple(int(dim) for dim in hidden_states.shape),
        "use_naf_upsample": config.use_naf_upsample,
        "naf_target_size": config.naf_target_size,
        "naf_source": "supplied" if naf_feature_map is not None else None,
    }

    global_tokens = hidden_states[:, :global_count, :]
    patch_tokens = hidden_states[:, global_count:, :].reshape(batch, patch_grid[0], patch_grid[1], channels)

    grid_points = pixal3d_projection_grid_points(config.grid_resolution).astype(hidden_states.dtype)
    projected = project_pixal3d_points_to_image(
        grid_points,
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
        image_resolution=config.image_size,
    )
    lr_features = sample_pixal3d_feature_map(
        patch_tokens,
        projected.points_2d,
        image_resolution=config.image_size,
        layout="BHWC",
    )
    if not config.use_naf_upsample:
        return Pixal3DProjectionConditioning(
            stage=config,
            global_tokens=global_tokens,
            projected_features=lr_features,
            projected_lr_features=lr_features,
            metadata={
                **conditioning_metadata,
                "global_shape": tuple(int(dim) for dim in global_tokens.shape),
                "projected_shape": tuple(int(dim) for dim in lr_features.shape),
                "projected_lr_shape": tuple(int(dim) for dim in lr_features.shape),
            },
        )

    if naf_feature_map is None:
        return Pixal3DProjectionConditioning(
            stage=config,
            global_tokens=global_tokens,
            projected_features=None,
            projected_lr_features=lr_features,
            blocker=Pixal3DProjectionBlocker(
                stage="naf-upsample",
                operation="build Pixal3D high-resolution projected features",
                reason="Pixal3D stage requires NAF-upsampled DINOv3 features; provide naf_feature_map or use the inference pipeline NAF bridge",
                metadata={
                    "stage": config.name,
                    "expected_projected_channels": config.expected_projected_channels(channels),
                    "available_lr_channels": channels,
                    "image_size": config.image_size,
                    "patch_grid": tuple(int(value) for value in patch_grid),
                    "naf_target_size": config.naf_target_size,
                },
            ),
            metadata={
                **conditioning_metadata,
                "naf_source": "runtime-bridge-required",
                "global_shape": tuple(int(dim) for dim in global_tokens.shape),
                "projected_lr_shape": tuple(int(dim) for dim in lr_features.shape),
            },
        )

    hr_features = sample_pixal3d_feature_map(
        naf_feature_map,
        projected.points_2d,
        image_resolution=config.image_size,
        layout=naf_layout,
    )
    return Pixal3DProjectionConditioning(
        stage=config,
        global_tokens=global_tokens,
        projected_features=mx.concatenate([lr_features, hr_features], axis=-1),
        projected_lr_features=lr_features,
        metadata={
            **conditioning_metadata,
            "naf_shape": tuple(int(dim) for dim in naf_feature_map.shape),
            "global_shape": tuple(int(dim) for dim in global_tokens.shape),
            "projected_shape": (int(batch), config.projected_token_count, int(channels) * 2),
            "projected_lr_shape": tuple(int(dim) for dim in lr_features.shape),
            "projected_hr_shape": tuple(int(dim) for dim in hr_features.shape),
        },
    )


def select_pixal3d_projected_features_at_coordinates(
    projected_features: mx.array,
    sparse_coordinates: mx.array,
    *,
    grid_resolution: int,
) -> mx.array:
    """Select projected grid features at sparse `(batch, z, y, x)` coordinates."""

    if grid_resolution <= 0:
        raise ValueError("grid_resolution must be positive")
    if projected_features.ndim != 3:
        raise ValueError(f"projected_features must have shape (batch, grid^3, channels), got {projected_features.shape}")
    if sparse_coordinates.ndim != 2 or int(sparse_coordinates.shape[1]) != 4:
        raise ValueError(f"sparse_coordinates must have shape (num_tokens, 4), got {sparse_coordinates.shape}")

    batch_size, token_count, channels = (int(dim) for dim in projected_features.shape)
    expected_tokens = grid_resolution**3
    if token_count != expected_tokens:
        raise ValueError(f"projected feature token count mismatch: expected {expected_tokens}, got {token_count}")

    coords = np.array(sparse_coordinates, dtype=np.int64)
    if coords.shape[0] == 0:
        return mx.zeros((0, channels), dtype=projected_features.dtype)
    batch = coords[:, 0]
    spatial = coords[:, 1:]
    if np.any(batch < 0) or np.any(batch >= batch_size):
        raise ValueError(f"sparse coordinate batch index out of bounds for batch size {batch_size}")
    if np.any(spatial < 0) or np.any(spatial >= grid_resolution):
        raise ValueError(f"sparse coordinate spatial index out of bounds for grid_resolution={grid_resolution}")

    z = spatial[:, 0]
    y = spatial[:, 1]
    x = spatial[:, 2]
    flat_index = batch * expected_tokens + z * grid_resolution * grid_resolution + y * grid_resolution + x
    flat_features = mx.reshape(projected_features, (batch_size * expected_tokens, channels))
    return flat_features[mx.array(flat_index.astype(np.int32))]


def pixal3d_stage_with_grid_resolution(
    stage: Pixal3DProjectionStageConfig,
    grid_resolution: int,
) -> Pixal3DProjectionStageConfig:
    """Return a copy of a stage config with a different projection grid resolution."""

    return replace(stage, grid_resolution=int(grid_resolution))


def _ensure_batched_points(points_3d: mx.array) -> mx.array:
    if points_3d.ndim == 2:
        return points_3d[None, :, :]
    if points_3d.ndim == 3:
        return points_3d
    raise ValueError("points_3d must have shape [N,3] or [B,N,3]")


def _batch_vector(value: float | mx.array, batch: int) -> mx.array:
    array = value if _is_mx_array(value) else mx.array(value, dtype=mx.float32)
    array = array.reshape(-1)
    if int(array.shape[0]) == 1:
        return mx.broadcast_to(array, (batch,))
    if int(array.shape[0]) != batch:
        raise ValueError(f"expected scalar or batch vector of length {batch}, got {array.shape}")
    return array


def _is_mx_array(value: object) -> bool:
    return value.__class__.__module__.startswith("mlx.") and hasattr(value, "shape")


def _array_scalar(value: float | mx.array) -> mx.array:
    array = value if _is_mx_array(value) else mx.array(value, dtype=mx.float32)
    return array.reshape(-1)[0]


def _gather_flat_pixels(flat: mx.array, indices: mx.array) -> mx.array:
    return mx.take_along_axis(flat, indices.astype(mx.int32)[..., None], axis=1)


def _projection_blocked(
    stage: Pixal3DProjectionStageConfig,
    operation: str,
    reason: str,
    shape: object,
) -> Pixal3DProjectionConditioning:
    return Pixal3DProjectionConditioning(
        stage=stage,
        global_tokens=None,
        projected_features=None,
        blocker=Pixal3DProjectionBlocker(
            stage="projection-conditioning",
            operation=operation,
            reason=reason,
            metadata={"shape": tuple(shape) if isinstance(shape, tuple | list) else str(shape)},
        ),
        metadata={
            "stage": stage.name,
            "image_size": stage.image_size,
            "grid_resolution": stage.grid_resolution,
            "patch_size": stage.patch_size,
        },
    )
