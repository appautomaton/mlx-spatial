"""Pixal3D camera and cascade planning helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


PIXAL3D_WILD_MESH_SCALE = 1.0
PIXAL3D_WILD_EXTEND_PIXEL = 0
PIXAL3D_WILD_IMAGE_RESOLUTION = 512
PIXAL3D_CASCADE_LR_RESOLUTION = 512
PIXAL3D_HR_RESOLUTION_STEP = 128
PIXAL3D_MIN_HR_RESOLUTION = 1024


@dataclass(frozen=True)
class Pixal3DCameraParams:
    """Camera params consumed by Pixal3D projection conditioning."""

    camera_angle_x: float
    distance: float
    mesh_scale: float = PIXAL3D_WILD_MESH_SCALE
    image_resolution: int = PIXAL3D_WILD_IMAGE_RESOLUTION
    extend_pixel: int = PIXAL3D_WILD_EXTEND_PIXEL


@dataclass(frozen=True)
class Pixal3DStagePlan:
    """Cascade stage settings selected before model execution."""

    pipeline_type: str
    sparse_structure_grid_resolution: int
    shape_lr_resolution: int
    requested_hr_resolution: int
    actual_hr_resolution: int
    actual_hr_grid_resolution: int
    texture_grid_resolution: int
    max_num_tokens: int
    hr_token_count: int | None = None


def pixal3d_compute_f_pixels(camera_angle_x: float, image_resolution: int = PIXAL3D_WILD_IMAGE_RESOLUTION) -> float:
    """Match upstream Pixal3D focal-pixel conversion from horizontal FOV."""

    if camera_angle_x <= 0:
        raise ValueError("camera_angle_x must be positive radians")
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    return float(focal_length * image_resolution / 32.0)


def pixal3d_distance_from_fov(
    camera_angle_x: float,
    *,
    mesh_scale: float = PIXAL3D_WILD_MESH_SCALE,
    image_resolution: int = PIXAL3D_WILD_IMAGE_RESOLUTION,
    extend_pixel: int = PIXAL3D_WILD_EXTEND_PIXEL,
) -> dict[str, float]:
    """Compute the upstream Pixal3D manual-FOV camera distance."""

    if mesh_scale <= 0:
        raise ValueError("mesh_scale must be positive")
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    grid_point = np.array([-1.0, 0.0, 0.0], dtype=np.float32) @ rotation.T
    grid_point = grid_point / float(mesh_scale) / 2.0
    xw, yw = float(grid_point[0]), float(grid_point[1])
    xt = float(0 - extend_pixel)
    f_pixels = pixal3d_compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    if abs(x_ndc) < 1e-8:
        raise ValueError("target x is too close to the image center to infer distance")
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def pixal3d_manual_camera_params(
    manual_fov: float,
    *,
    mesh_scale: float = PIXAL3D_WILD_MESH_SCALE,
    image_resolution: int = PIXAL3D_WILD_IMAGE_RESOLUTION,
    extend_pixel: int = PIXAL3D_WILD_EXTEND_PIXEL,
) -> Pixal3DCameraParams:
    """Create Pixal3D camera params from manual FOV in radians."""

    if manual_fov <= 0:
        raise ValueError("manual_fov must be positive radians")
    distance = pixal3d_distance_from_fov(
        manual_fov,
        mesh_scale=mesh_scale,
        image_resolution=image_resolution,
        extend_pixel=extend_pixel,
    )["distance_from_x"]
    return Pixal3DCameraParams(
        camera_angle_x=float(manual_fov),
        distance=distance,
        mesh_scale=float(mesh_scale),
        image_resolution=int(image_resolution),
        extend_pixel=int(extend_pixel),
    )


def pixal3d_requested_hr_resolution(pipeline_type: str) -> int:
    """Return the upstream HR resolution for a Pixal3D pipeline type."""

    if pipeline_type == "1024_cascade":
        return 1024
    if pipeline_type == "1536_cascade":
        return 1536
    raise ValueError(f"unsupported Pixal3D pipeline type: {pipeline_type!r}")


def pixal3d_select_hr_resolution(
    coordinates: mx.array | np.ndarray,
    *,
    requested_hr_resolution: int,
    max_num_tokens: int,
    lr_resolution: int = PIXAL3D_CASCADE_LR_RESOLUTION,
) -> tuple[int, int]:
    """Apply Pixal3D's HR-token guard by reducing HR resolution in 128px steps."""

    if max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be positive")
    if requested_hr_resolution < PIXAL3D_MIN_HR_RESOLUTION:
        raise ValueError(f"requested_hr_resolution must be at least {PIXAL3D_MIN_HR_RESOLUTION}")
    coords = _coordinates_np(coordinates)
    actual = int(requested_hr_resolution)
    while True:
        token_count = _quantized_unique_token_count(coords, actual_hr_resolution=actual, lr_resolution=lr_resolution)
        if token_count < max_num_tokens or actual == PIXAL3D_MIN_HR_RESOLUTION:
            return actual, token_count
        actual -= PIXAL3D_HR_RESOLUTION_STEP


def pixal3d_stage_plan(
    pipeline_type: str,
    *,
    max_num_tokens: int,
    hr_coordinates: mx.array | np.ndarray | None = None,
) -> Pixal3DStagePlan:
    """Build the Pixal3D cascade plan before running expensive stages."""

    requested_hr = pixal3d_requested_hr_resolution(pipeline_type)
    token_count = None
    actual_hr = requested_hr
    if hr_coordinates is not None:
        actual_hr, token_count = pixal3d_select_hr_resolution(
            hr_coordinates,
            requested_hr_resolution=requested_hr,
            max_num_tokens=max_num_tokens,
        )
    actual_grid = actual_hr // 16
    return Pixal3DStagePlan(
        pipeline_type=pipeline_type,
        sparse_structure_grid_resolution=16,
        shape_lr_resolution=PIXAL3D_CASCADE_LR_RESOLUTION,
        requested_hr_resolution=requested_hr,
        actual_hr_resolution=actual_hr,
        actual_hr_grid_resolution=actual_grid,
        texture_grid_resolution=actual_grid,
        max_num_tokens=max_num_tokens,
        hr_token_count=token_count,
    )


def _coordinates_np(coordinates: mx.array | np.ndarray) -> np.ndarray:
    coords = np.array(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Pixal3D coordinates must have shape (tokens, 4), got {coords.shape}")
    return coords.astype(np.int64, copy=False)


def _quantized_unique_token_count(
    coordinates: np.ndarray,
    *,
    actual_hr_resolution: int,
    lr_resolution: int,
) -> int:
    if coordinates.shape[0] == 0:
        return 0
    grid_res = actual_hr_resolution // 16
    quantized_xyz = np.rint(((coordinates[:, 1:] + 0.5) / lr_resolution) * (grid_res - 1)).astype(np.int64)
    quantized = np.concatenate([coordinates[:, :1], quantized_xyz], axis=1)
    return int(np.unique(quantized, axis=0).shape[0])
