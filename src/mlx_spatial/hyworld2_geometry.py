"""Depth/geometry utilities for HY-World 2.0 inference.

Gap IDs: HW-13, HW-17. Matches
``vendors/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/utils/geometry.py``
and ``vendors/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/utils/geometry.py``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def depth_to_camera_coords(
    depth: mx.array,
    intrinsics: mx.array,
) -> tuple[mx.array, mx.array]:
    """Unproject a depth map to 3D camera-space coordinates.

    Args:
        depth: ``(B, H, W)`` depth values.
        intrinsics: ``(B, 3, 3)`` camera intrinsic matrices
            ``[[fv, 0, cy], [0, fu, cx], [0, 0, 1]]``.

    Returns:
        ``(xyz, valid)`` where xyz is ``(B, H, W, 3)`` camera-space
        coordinates and valid is ``(B, H, W)`` boolean mask.
    """
    B, H, W = depth.shape
    fy = intrinsics[:, 0, 0:1]
    fx = intrinsics[:, 1, 1:2]
    cy = intrinsics[:, 0, 2:3]
    cx = intrinsics[:, 1, 2:3]

    y_grid = mx.broadcast_to(mx.arange(H, dtype=mx.float32).reshape(1, H, 1), (B, H, W))
    x_grid = mx.broadcast_to(mx.arange(W, dtype=mx.float32).reshape(1, 1, W), (B, H, W))

    valid = depth > 0
    z = mx.where(valid, depth, mx.zeros_like(depth))

    x_cam = (x_grid - cx) * z / fx
    y_cam = (y_grid - cy) * z / fy
    z_cam = z

    xyz = mx.stack([x_cam, y_cam, z_cam], axis=-1)
    return xyz, valid


def depth_to_world_coords(
    depth: mx.array,
    extrinsic: mx.array,
    intrinsic: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Unproject depth to world coordinates using extrinsic and intrinsic matrices.

    Args:
        depth: ``(B, H, W)`` depth values.
        extrinsic: ``(B, 4, 4)`` camera-to-world transformation.
        intrinsic: ``(B, 3, 3)`` camera intrinsics.

    Returns:
        ``(world, cam, mask)`` with world ``(B, H, W, 3)``,
        cam ``(B, H, W, 3)``, mask ``(B, H, W)``.
    """
    cam_xyz, valid = depth_to_camera_coords(depth, intrinsic)
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3:]

    cam_flat = mx.reshape(cam_xyz, (*cam_xyz.shape[:1], -1, 3))
    R_t = mx.transpose(R, (0, 2, 1))
    world_flat = mx.matmul(cam_flat, R_t) + mx.broadcast_to(t, cam_flat.shape).squeeze(-2)
    world = mx.reshape(world_flat, cam_xyz.shape)

    return world, cam_xyz, valid


def closed_form_inverse_se3(se3: mx.array) -> mx.array:
    R = se3[..., :3, :3]
    t = se3[..., :3, 3:4]
    R_inv = mx.transpose(R, (*range(R.ndim - 2), R.ndim - 1, R.ndim - 2))
    t_inv = -mx.matmul(R_inv, t)
    top = mx.concatenate([R_inv, t_inv], axis=-1)
    bottom_row = mx.zeros((*se3.shape[:-2], 1, 3), dtype=se3.dtype)
    bottom_val = mx.ones((*se3.shape[:-2], 1, 1), dtype=se3.dtype)
    bottom = mx.concatenate([bottom_row, bottom_val], axis=-1)
    return mx.concatenate([top, bottom], axis=-2)


def colmap_to_opencv_intrinsics(K: mx.array) -> mx.array:
    cx = K[..., 0, 2] - 0.5
    cy = K[..., 1, 2] - 0.5
    ones = mx.ones_like(cx)
    zeros = mx.zeros_like(cx)
    row0 = mx.stack([K[..., 0, 0], zeros, cx - 0.5 + 0.5], axis=-1)
    row1 = mx.stack([zeros, K[..., 1, 1], cy - 0.5 + 0.5], axis=-1)
    row2 = mx.stack([zeros, zeros, ones], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def opencv_to_colmap_intrinsics(K: mx.array) -> mx.array:
    cx = K[..., 0, 2] + 0.5
    cy = K[..., 1, 2] + 0.5
    ones = mx.ones_like(cx)
    zeros = mx.zeros_like(cx)
    row0 = mx.stack([K[..., 0, 0], zeros, cx], axis=-1)
    row1 = mx.stack([zeros, K[..., 1, 1], cy], axis=-1)
    row2 = mx.stack([zeros, zeros, ones], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def points_to_normals(
    points: np.ndarray,
    mask: np.ndarray | None = None,
    edge_threshold: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Estimate surface normals from a point map using cross products of 4-connectivity neighbors.

    Gap ID: HW-17. Matches the official implementation using numpy.

    Args:
        points: ``(H, W, 3)`` point map.
        mask: ``(H, W)`` boolean valid mask, or None.
        edge_threshold: Maximum edge length to consider valid, or None.

    Returns:
        ``(H, W, 3)`` normal map, or ``(normal, normal_mask)`` if mask is provided.
    """
    H, W, _ = points.shape
    normals = np.zeros_like(points)

    left = np.roll(points, 1, axis=1)
    right = np.roll(points, -1, axis=1)
    up = np.roll(points, 1, axis=0)
    down = np.roll(points, -1, axis=0)

    n1 = np.cross(right - left, down - up)
    norm = np.linalg.norm(n1, axis=-1, keepdims=True)
    valid = norm > 1e-8
    normals = np.where(valid, n1 / np.maximum(norm, 1e-8), 0.0)

    if mask is not None:
        normals = np.where(mask[..., None], normals, 0.0)
        return normals, mask & (norm[..., 0] != 0)

    if edge_threshold is not None:
        edge_len = np.linalg.norm(right - left, axis=-1)
        valid_edge = edge_len < edge_threshold
        normals = np.where(valid_edge[..., None], normals, 0.0)

    return normals


def normalize_poses(
    extrinsics: np.ndarray,
    padding: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Normalize camera extrinsics to a unit cube using percentile-based bounds.

    Gap ID: HW-16. Matches ``models/utils/priors.py::normalize_poses``.

    Args:
        extrinsics: ``(B, S, 3, 4)`` camera extrinsic matrices.
        padding: Padding around the unit cube.

    Returns:
        ``(normalized_extrinsics, stats)`` where stats contains
        ``scale_factors`` and ``translation_vectors``.
    """
    original_shape = extrinsics.shape
    B, S = original_shape[0], original_shape[1]
    positions = extrinsics[..., :3, 3].reshape(-1, 3)

    p5 = np.percentile(positions, 5, axis=0)
    p95 = np.percentile(positions, 95, axis=0)
    center = (p5 + p95) / 2.0
    extent = p95 - p5
    scale = np.max(np.abs(extent))
    scale = np.maximum(scale, 1e-8)

    scale_factor = (1.0 - 2 * padding) / scale
    translation = -center * scale_factor

    normalized = extrinsics.copy()
    normalized[..., :3, 3] = extrinsics[..., :3, 3] * scale_factor + translation

    stats = {
        "scale_factors": scale_factor,
        "translation_vectors": translation,
    }
    return normalized, stats


def normalize_depth(
    depth: np.ndarray,
    eps: float = 1e-6,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
) -> np.ndarray:
    """Per-image percentile-based depth normalization to [0, 1].

    Gap ID: HW-16. Matches ``models/utils/priors.py::normalize_depth``.

    Args:
        depth: ``(B, S, H, W)`` or any leading-dim + H,W depth map.
        eps: Small value for numerical stability.
        min_percentile: Lower percentile bound.
        max_percentile: Upper percentile bound.

    Returns:
        Normalized depth with values in [0, 1].
    """
    original_shape = depth.shape
    flat = depth.reshape(-1, original_shape[-2], original_shape[-1])

    results = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        d = flat[i]
        valid = d > 0
        if not np.any(valid):
            results[i] = d
            continue
        d_valid = d[valid]
        pmin = np.percentile(d_valid, min_percentile)
        pmax = np.percentile(d_valid, max_percentile)
        rng = max(pmax - pmin, eps)
        results[i] = np.clip((d - pmin) / rng, 0.0, 1.0)

    return results.reshape(original_shape)