"""Camera parameter encoding/decoding for HY-World 2.0.

Gap ID: HW-11. Matches
``vendors/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/utils/camera_utils.py``.

Converts between 9-dim camera parameter vectors and 3×4 extrinsic + 3×3
intrinsic matrices, following the official convention:
  vector = [tx, ty, tz, qx, qy, qz, qw, fov_v, fov_u]
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def camera_params_to_matrices(
    cam_vec: mx.array,
    image_hw: tuple[int, int] | None = None,
) -> tuple[mx.array, mx.array]:
    """Convert a 9-dim camera parameter vector to extrinsic and intrinsic matrices.

    Args:
        cam_vec: ``(B, ..., 9)`` — [tx, ty, tz, qx, qy, qz, qw, fov_v, fov_u].
        image_hw: ``(H, W)`` image size for focal length computation.
            Required when fov values are present.

    Returns:
        ``(extrinsic, intrinsic)`` where extrinsic is ``(B, ..., 3, 4)``
        and intrinsic is ``(B, ..., 3, 3)``.
    """
    translation = cam_vec[..., :3]
    quaternion = cam_vec[..., 3:7]
    fov_v = cam_vec[..., 7:8]
    fov_u = cam_vec[..., 8:9]

    rotation = quat_to_rotmat(quaternion)
    extrinsic = mx.concatenate((rotation, translation[..., None]), axis=-1)

    if image_hw is None:
        ones = mx.ones_like(fov_v)
        intrinsic = mx.stack(
            [ones, mx.zeros_like(fov_v), mx.zeros_like(fov_v),
             mx.zeros_like(fov_v), ones, mx.zeros_like(fov_v),
             mx.zeros_like(fov_v), mx.zeros_like(fov_v), ones],
            axis=-1,
        ).reshape((*cam_vec.shape[:-1], 3, 3))
    else:
        h, w = image_hw
        focal_v = (float(h) * 0.5) / mx.tan(fov_v * 0.5)
        focal_u = float(w) * 0.5 / mx.tan(fov_u * 0.5)
        cy = mx.ones_like(focal_v) * (float(h) * 0.5)
        cx = mx.ones_like(focal_u) * (float(w) * 0.5)
        zeros = mx.zeros_like(focal_v)
        ones = mx.ones_like(focal_v)
        row0 = mx.concatenate([focal_v, zeros, cy], axis=-1)
        row1 = mx.concatenate([zeros, focal_u, cx], axis=-1)
        row2 = mx.concatenate([zeros, zeros, ones], axis=-1)
        intrinsic = mx.stack([row0, row1, row2], axis=-2)

    return extrinsic, intrinsic


def quat_to_rotmat(quaternions: mx.array) -> mx.array:
    """Convert XYZW quaternions to 3×3 rotation matrices.

    Args:
        quaternions: ``(..., 4)`` — XYZW (scalar-last) convention.

    Returns:
        ``(..., 3, 3)`` rotation matrices.
    """
    qx = quaternions[..., 0]
    qy = quaternions[..., 1]
    qz = quaternions[..., 2]
    qw = quaternions[..., 3]

    norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
    norm_sq = mx.maximum(norm_sq, mx.array(1e-12, dtype=norm_sq.dtype))
    s = 2.0 / norm_sq

    r00 = 1.0 - s * (qy * qy + qz * qz)
    r01 = s * (qx * qy - qz * qw)
    r02 = s * (qx * qz + qy * qw)
    r10 = s * (qx * qy + qz * qw)
    r11 = 1.0 - s * (qx * qx + qz * qz)
    r12 = s * (qy * qz - qx * qw)
    r20 = s * (qx * qz - qy * qw)
    r21 = s * (qy * qz + qx * qw)
    r22 = 1.0 - s * (qx * qx + qy * qy)

    row0 = mx.stack([r00, r01, r02], axis=-1)
    row1 = mx.stack([r10, r11, r12], axis=-1)
    row2 = mx.stack([r20, r21, r22], axis=-1)
    return mx.stack([row0, row1, row2], axis=-2)


def rotmat_to_quat(rotation: mx.array) -> mx.array:
    """Convert 3×3 rotation matrices to XYZW quaternions.

    Uses Shepherd's method with max-diagonal selection for numerical stability.

    Args:
        rotation: ``(..., 3, 3)`` rotation matrices.

    Returns:
        ``(..., 4)`` quaternions in XYZW (scalar-last) convention.
    """
    r00 = rotation[..., 0, 0]
    r11 = rotation[..., 1, 1]
    r22 = rotation[..., 2, 2]
    trace = r00 + r11 + r22

    def _sqrt_positive_part(x: mx.array) -> mx.array:
        return mx.sqrt(mx.maximum(x, mx.zeros_like(x)))

    s = _sqrt_positive_part(1.0 + trace) * 2.0
    qw = s * 0.25
    qx = (rotation[..., 2, 1] - rotation[..., 1, 2]) / (s + 1e-12)
    qy = (rotation[..., 0, 2] - rotation[..., 2, 0]) / (s + 1e-12)
    qz = (rotation[..., 1, 0] - rotation[..., 0, 1]) / (s + 1e-12)

    result = mx.stack([qx, qy, qz, qw], axis=-1)
    sign = mx.where(result[..., 3:4] < 0, mx.array(-1.0), mx.array(1.0))
    return result * sign


def extrinsics_to_vector(extrinsics: mx.array) -> mx.array:
    """Pack a 3×4 extrinsic matrix into a 7-dim vector [tx,ty,tz,qx,qy,qz,qw].

    Args:
        extrinsics: ``(..., 3, 4)`` — [R|t] camera-to-world matrices.

    Returns:
        ``(..., 7)`` — translation + quaternion.
    """
    translation = extrinsics[..., :3, 3]
    rotation = extrinsics[..., :3, :3]
    quaternion = rotmat_to_quat(rotation)
    return mx.concatenate([translation, quaternion], axis=-1)


def vector_to_extrinsics(cam_vec: mx.array) -> mx.array:
    """Unpack a 7-dim vector [tx,ty,tz,qx,qy,qz,qw] into a 3×4 extrinsic matrix.

    Args:
        cam_vec: ``(..., 7)`` — translation + quaternion.

    Returns:
        ``(..., 3, 4)`` — [R|t] extrinsic matrices.
    """
    translation = cam_vec[..., :3]
    quaternion = cam_vec[..., 3:7]
    rotation = quat_to_rotmat(quaternion)
    return mx.concatenate([rotation, translation[..., None]], axis=-1)