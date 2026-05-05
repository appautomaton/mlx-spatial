"""SAM 3D Objects pose decoder utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SAM3D_ROTATION_6D_MEAN = np.array(
    [
        -0.06366084883674913,
        0.008438224692279752,
        0.00017084786438302483,
        0.0007126610473540038,
        -0.0030916726538816417,
        0.5166093753457688,
    ],
    dtype=np.float32,
)
SAM3D_ROTATION_6D_STD = np.array(
    [
        0.6656971967514863,
        0.6787012271867754,
        0.30345010594844524,
        0.4394504420678794,
        0.39817973931717104,
        0.6176286868761914,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Sam3dPoseDecodeResult:
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    metadata: dict[str, object]


def decode_sam3d_scale_shift_invariant_pose(
    model_output: dict[str, np.ndarray],
    *,
    scene_scale: np.ndarray | None = None,
    scene_shift: np.ndarray | None = None,
) -> Sam3dPoseDecodeResult:
    """Port the active official ScaleShiftInvariant pose decoder without PyTorch."""

    rotation_6d = _as_pose_array(model_output["6drotation_normalized"], 6)
    rotation_6d = rotation_6d * SAM3D_ROTATION_6D_STD + SAM3D_ROTATION_6D_MEAN
    rotation = rotation_6d_to_quaternion(rotation_6d)

    translation = _as_pose_array(model_output["translation"], 3)
    scale = np.exp(_as_pose_array(model_output["scale"], 3)).astype(np.float32)
    translation_scale = np.exp(_as_pose_array(model_output.get("translation_scale", np.zeros((1, 1), dtype=np.float32)), 1))

    scene_scale_np = _scene_vector(scene_scale, default=1.0)
    scene_shift_np = _scene_vector(scene_shift, default=0.0)
    metric_translation = translation * scene_scale_np[None, :] + scene_shift_np[None, :]
    metric_scale = scale * scene_scale_np[None, :]
    mean_scale = metric_scale.mean(axis=-1, keepdims=True)
    metric_scale = np.repeat(mean_scale, 3, axis=-1)

    return Sam3dPoseDecodeResult(
        translation=metric_translation.astype(np.float32, copy=False),
        rotation=rotation.astype(np.float32, copy=False),
        scale=metric_scale.astype(np.float32, copy=False),
        metadata={
            "pose_target_convention": "ScaleShiftInvariant",
            "translation_shape": tuple(int(value) for value in metric_translation.shape),
            "rotation_shape": tuple(int(value) for value in rotation.shape),
            "scale_shape": tuple(int(value) for value in metric_scale.shape),
            "translation_scale_mean": float(translation_scale.mean()),
        },
    )


def rotation_6d_to_quaternion(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert official 6D rotation representation to PyTorch3D-order quaternion."""

    rot = np.asarray(rotation_6d, dtype=np.float32)
    if rot.ndim == 1:
        rot = rot[None, :]
    if rot.ndim != 2 or rot.shape[1] != 6:
        raise ValueError(f"SAM3D 6D rotation must have shape (N, 6), got {rot.shape}")
    a1 = rot[:, 0:3]
    a2 = rot[:, 3:6]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2)
    matrix = np.stack((b1, b2, b3), axis=-1)
    return matrix_to_quaternion(matrix)


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to `(w, x, y, z)` quaternions."""

    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim == 2:
        m = m[None, ...]
    if m.ndim != 3 or m.shape[1:] != (3, 3):
        raise ValueError(f"rotation matrix must have shape (N, 3, 3), got {m.shape}")
    quats = np.empty((m.shape[0], 4), dtype=np.float32)
    for index, value in enumerate(m):
        trace = float(np.trace(value))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            quats[index, 0] = 0.25 * s
            quats[index, 1] = (value[2, 1] - value[1, 2]) / s
            quats[index, 2] = (value[0, 2] - value[2, 0]) / s
            quats[index, 3] = (value[1, 0] - value[0, 1]) / s
        else:
            diag = np.diag(value)
            axis = int(np.argmax(diag))
            if axis == 0:
                s = np.sqrt(1.0 + value[0, 0] - value[1, 1] - value[2, 2]) * 2.0
                quats[index] = [(value[2, 1] - value[1, 2]) / s, 0.25 * s, (value[0, 1] + value[1, 0]) / s, (value[0, 2] + value[2, 0]) / s]
            elif axis == 1:
                s = np.sqrt(1.0 + value[1, 1] - value[0, 0] - value[2, 2]) * 2.0
                quats[index] = [(value[0, 2] - value[2, 0]) / s, (value[0, 1] + value[1, 0]) / s, 0.25 * s, (value[1, 2] + value[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + value[2, 2] - value[0, 0] - value[1, 1]) * 2.0
                quats[index] = [(value[1, 0] - value[0, 1]) / s, (value[0, 2] + value[2, 0]) / s, (value[1, 2] + value[2, 1]) / s, 0.25 * s]
    return _normalize(quats)


def _as_pose_array(value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim == 3 and array.shape[1] == 1:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"SAM3D pose value must have shape (N, {width}), got {array.shape}")
    return array


def _scene_vector(value: np.ndarray | None, *, default: float) -> np.ndarray:
    if value is None:
        return np.full((3,), default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        return np.repeat(array, 3).astype(np.float32, copy=False)
    if array.size != 3:
        raise ValueError(f"SAM3D scene scale/shift must have 1 or 3 values, got {array.shape}")
    return array.astype(np.float32, copy=False)


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, eps)
