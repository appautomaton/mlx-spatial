import numpy as np

from mlx_spatial.sam3d_pose import (
    SAM3D_ROTATION_6D_MEAN,
    SAM3D_ROTATION_6D_STD,
    decode_sam3d_scale_shift_invariant_pose,
    rotation_6d_to_quaternion,
)


def test_rotation_6d_to_quaternion_returns_identity_for_basis_vectors():
    quat = rotation_6d_to_quaternion(np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32))

    assert np.allclose(quat, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_decode_scale_shift_invariant_pose_applies_scene_scale_and_shift():
    identity_6d = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    normalized = (identity_6d - SAM3D_ROTATION_6D_MEAN[None, :]) / SAM3D_ROTATION_6D_STD[None, :]
    result = decode_sam3d_scale_shift_invariant_pose(
        {
            "6drotation_normalized": normalized,
            "translation": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            "scale": np.zeros((1, 3), dtype=np.float32),
            "translation_scale": np.zeros((1, 1), dtype=np.float32),
        },
        scene_scale=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        scene_shift=np.array([0.5, 1.0, 1.5], dtype=np.float32),
    )

    assert np.allclose(result.translation, np.array([[2.5, 5.0, 7.5]], dtype=np.float32))
    assert np.allclose(result.rotation, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), atol=1e-6)
    assert np.allclose(result.scale, np.array([[2.0, 2.0, 2.0]], dtype=np.float32))
    assert result.metadata["pose_target_convention"] == "ScaleShiftInvariant"


def test_decode_scale_shift_invariant_pose_rejects_bad_rotation_shape():
    try:
        decode_sam3d_scale_shift_invariant_pose(
            {
                "6drotation_normalized": np.zeros((1, 5), dtype=np.float32),
                "translation": np.zeros((1, 3), dtype=np.float32),
                "scale": np.zeros((1, 3), dtype=np.float32),
            }
        )
    except ValueError as error:
        assert "shape (N, 6)" in str(error)
    else:
        raise AssertionError("expected bad 6D rotation shape to fail")
