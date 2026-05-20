"""Tests for HY-World 2.0 utility modules (Slice 3).

Covers: hyworld2_camera, hyworld2_geometry, hyworld2_sh
Gap IDs: HW-11, HW-13, HW-15, HW-16, HW-17
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.hyworld2_camera import (
    camera_params_to_matrices,
    extrinsics_to_vector,
    quat_to_rotmat,
    rotmat_to_quat,
    vector_to_extrinsics,
)
from mlx_spatial.hyworld2_geometry import (
    closed_form_inverse_se3,
    colmap_to_opencv_intrinsics,
    depth_to_camera_coords,
    depth_to_world_coords,
    normalize_depth,
    normalize_poses,
    opencv_to_colmap_intrinsics,
    points_to_normals,
)
from mlx_spatial.hyworld2_sh import eval_sh, eval_sh_numpy, rgb_to_sh, sh_to_rgb


class TestQuatRotmat:
    def test_identity_quat_gives_identity_matrix(self):
        q = mx.array([[0.0, 0.0, 0.0, 1.0]])
        R = quat_to_rotmat(q)
        expected = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)
        assert mx.allclose(R, expected, atol=1e-5).item()

    def test_180_degree_rotation_x(self):
        q = mx.array([[1.0, 0.0, 0.0, 0.0]])
        R = quat_to_rotmat(q)
        expected = mx.array([[[1, 0, 0], [0, -1, 0], [0, 0, -1]]], dtype=mx.float32)
        assert mx.allclose(R, expected, atol=1e-5).item()

    def test_90_degree_rotation_z(self):
        angle = math.pi / 2
        q = mx.array([[0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]])
        R = quat_to_rotmat(q)
        expected = mx.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=mx.float32)
        assert mx.allclose(R, expected, atol=1e-5).item()

    def test_roundtrip_quat_rotmat(self):
        q_orig = mx.array([[0.182, 0.546, 0.728, 0.364]])
        q_orig = q_orig / mx.linalg.norm(q_orig, axis=-1, keepdims=True)
        R = quat_to_rotmat(q_orig)
        q_back = rotmat_to_quat(R)
        q_back = q_back / mx.linalg.norm(q_back, axis=-1, keepdims=True)
        sign = mx.where(q_orig[..., 3:4] < 0, mx.array(-1.0), mx.array(1.0))
        sign_back = mx.where(q_back[..., 3:4] < 0, mx.array(-1.0), mx.array(1.0))
        flip = sign * sign_back
        q_aligned = q_back * flip
        assert mx.allclose(q_orig, q_aligned, atol=1e-4).item()

    def test_rotation_determinant_is_one(self):
        q = mx.array([[0.3, 0.5, 0.7, 0.4]])
        q = q / mx.linalg.norm(q, axis=-1, keepdims=True)
        R = quat_to_rotmat(q)
        R_np = np.array(R[0])
        det = np.linalg.det(R_np)
        assert abs(det - 1.0) < 1e-4

    def test_batch_operation(self):
        qs = mx.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        Rs = quat_to_rotmat(qs)
        assert Rs.shape == (2, 3, 3)


class TestCameraParams:
    def test_camera_params_identity_rotation(self):
        cam = mx.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.047, 1.047]])
        extr, intr = camera_params_to_matrices(cam, image_hw=(256, 256))
        assert extr.shape == (1, 3, 4)
        assert intr.shape == (1, 3, 3)
        assert abs(extr[0, 2, 3].item() - 5.0) < 1e-4

    def test_camera_params_without_image_hw(self):
        cam = mx.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5]])
        extr, intr = camera_params_to_matrices(cam)
        assert intr[..., 0, 0].item() == 1.0

    def test_extrinsic_vector_roundtrip(self):
        cam = mx.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.7071, 0.7071]])
        ext = vector_to_extrinsics(cam)
        back = extrinsics_to_vector(ext)
        assert mx.allclose(cam, back, atol=1e-3).item()


class TestDepthGeometry:
    def test_depth_to_camera_coords_center_pixel(self):
        depth = mx.array([[[5.0]]]).reshape(1, 1, 1)
        intr = mx.array([[[100.0, 0.0, 0.5], [0.0, 100.0, 0.5], [0.0, 0.0, 1.0]]])
        xyz, valid = depth_to_camera_coords(depth, intr)
        assert valid.item()
        assert abs(xyz[0, 0, 0, 2].item() - 5.0) < 1e-4

    def test_depth_to_camera_coords_zeros_invalid(self):
        depth = mx.zeros((1, 4, 4))
        intr = mx.eye(3, dtype=mx.float32).reshape(1, 3, 3)
        xyz, valid = depth_to_camera_coords(depth, intr)
        assert not valid.any().item()

    def test_closed_form_inverse_se3_identity(self):
        I = mx.eye(4, dtype=mx.float32).reshape(1, 4, 4)
        inv = closed_form_inverse_se3(I)
        assert mx.allclose(inv, I, atol=1e-5).item()

    def test_closed_form_inverse_se3_composition(self):
        t = mx.array([[[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, -1.0, 3.0], [0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 0.0, 1.0]]])
        inv = closed_form_inverse_se3(t)
        result = mx.matmul(t, inv)
        expected = mx.eye(4, dtype=mx.float32).reshape(1, 4, 4)
        assert mx.allclose(result, expected, atol=1e-4).item()

    def test_colmap_opencv_intrinsics_roundtrip(self):
        K = mx.array([[[500.0, 0.0, 320.5], [0.0, 500.0, 240.5], [0.0, 0.0, 1.0]]])
        K_cv = colmap_to_opencv_intrinsics(K)
        K_back = opencv_to_colmap_intrinsics(K_cv)
        assert mx.allclose(K, K_back, atol=1e-5).item()

    def test_colmap_shifts_principal_point(self):
        K = mx.array([[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]])
        K_cv = colmap_to_opencv_intrinsics(K)
        assert abs(K_cv[0, 0, 2].item() - 319.5) < 1e-5
        assert abs(K_cv[0, 1, 2].item() - 239.5) < 1e-5


class TestPointsToNormals:
    def test_flat_plane_normals(self):
        H, W = 10, 10
        z = np.ones((H, W), dtype=np.float32) * 5.0
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        points = np.stack([xx, yy, z], axis=-1)
        normals = points_to_normals(points)
        assert normals.shape == (H, W, 3)
        center_normal = normals[5, 5]
        norm = np.linalg.norm(center_normal)
        if norm > 1e-6:
            center_normal = center_normal / norm
        assert abs(center_normal[2]) > 0.9

    def test_points_with_mask(self):
        H, W = 5, 5
        z = np.ones((H, W), dtype=np.float32) * 3.0
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        points = np.stack([xx, yy, z], axis=-1)
        mask = np.ones((H, W), dtype=bool)
        mask[0, :] = False
        normals, normal_mask = points_to_normals(points, mask=mask)
        assert not normal_mask[0, :].any()


class TestNormalizePoses:
    def test_normalize_poses_preserves_shape(self):
        B, S = 2, 3
        extrinsics = np.random.randn(B, S, 3, 4).astype(np.float32)
        extrinsics[..., :3, :3] = np.eye(3)
        normed, stats = normalize_poses(extrinsics)
        assert normed.shape == (B, S, 3, 4)

    def test_normalize_depth_range(self):
        depth = np.random.rand(2, 3, 64, 64).astype(np.float32) * 10 + 0.1
        result = normalize_depth(depth)
        assert result.min() >= 0.0
        assert result.max() <= 1.01

    def test_normalize_depth_zeros(self):
        depth = np.zeros((1, 1, 4, 4), dtype=np.float32)
        result = normalize_depth(depth)
        assert np.allclose(result, 0.0)


class TestSphericalHarmonics:
    def test_sh_degree0_constant(self):
        sh = mx.ones((1, 3, 1))
        dirs = mx.array([[0.0, 0.0, 1.0]])
        result = eval_sh(0, sh, dirs)
        expected = 0.28209479177387814
        assert mx.allclose(result, mx.ones_like(result) * expected, atol=1e-5).item()

    def test_rgb_sh_roundtrip(self):
        rgb = mx.array([[0.5, 0.3, 0.8]])
        sh = rgb_to_sh(rgb)
        rgb_back = sh_to_rgb(sh)
        assert mx.allclose(rgb, rgb_back, atol=1e-5).item()

    def test_sh_numpy_matches_mlx(self):
        sh_np = np.array([[[0.5], [0.3], [0.8]]])
        dirs_np = np.array([[0.0, 0.0, 1.0]])
        result_np = eval_sh_numpy(0, sh_np, dirs_np)
        sh_mx = mx.array(sh_np)
        dirs_mx = mx.array(dirs_np)
        result_mx = eval_sh(0, sh_mx, dirs_mx)
        assert np.allclose(np.array(result_mx), result_np, atol=1e-5)

    def test_sh_degree1_directional(self):
        sh = mx.zeros((1, 1, 4))
        vals = [[0.0, 0.0, 1.0, 0.0]]
        sh = mx.array(vals).reshape(1, 1, 4)
        dirs = mx.array([[0.0, 0.0, 1.0]])
        result = eval_sh(1, sh, dirs)
        assert result.shape == (1, 1, 1)