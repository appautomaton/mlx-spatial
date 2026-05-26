import numpy as np

from mlx_spatial.mapanything_geometry import (
    mapanything_camera_poses_from_trans_quats,
    mapanything_convert_ray_dirs_depth_pose_to_pointmap,
    mapanything_depthmap_to_camera_frame,
    mapanything_depthmap_to_world_frame,
    mapanything_quaternion_to_rotation_matrix,
    mapanything_recover_pinhole_intrinsics_from_ray_directions,
)


def test_mapanything_quaternion_to_rotation_matrix_uses_xyzw_order():
    identity = mapanything_quaternion_to_rotation_matrix(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(identity, np.eye(3, dtype=np.float32), atol=1e-6)

    half = np.sqrt(0.5, dtype=np.float32)
    z_turn = mapanything_quaternion_to_rotation_matrix(np.array([0.0, 0.0, half, half], dtype=np.float32))
    np.testing.assert_allclose(
        z_turn,
        np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_mapanything_camera_pose_and_pointmap_conversion():
    ray_directions = np.array([[[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]], dtype=np.float32)
    depth = np.array([[[[2.0], [3.0]]]], dtype=np.float32)
    trans = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    quats = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    pose = mapanything_camera_poses_from_trans_quats(trans, quats)
    points = mapanything_convert_ray_dirs_depth_pose_to_pointmap(ray_directions, depth, trans, quats)

    np.testing.assert_allclose(pose[0, :3, 3], trans[0], atol=1e-6)
    np.testing.assert_allclose(points[0, 0, 0], [10.0, 20.0, 32.0], atol=1e-6)
    np.testing.assert_allclose(points[0, 0, 1], [13.0, 20.0, 30.0], atol=1e-6)


def test_mapanything_depthmap_camera_and_world_frame():
    depth = np.array([[2.0, 4.0]], dtype=np.float32)
    intrinsics = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]

    camera_points, valid = mapanything_depthmap_to_camera_frame(depth, intrinsics)
    world_points, world_valid = mapanything_depthmap_to_world_frame(depth, intrinsics, pose)

    np.testing.assert_array_equal(valid, [[True, True]])
    np.testing.assert_array_equal(world_valid, valid)
    np.testing.assert_allclose(camera_points[0, 0], [0.0, 0.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(camera_points[0, 1], [2.0, 0.0, 4.0], atol=1e-6)
    np.testing.assert_allclose(world_points[0, 1], [3.0, 2.0, 7.0], atol=1e-6)


def test_mapanything_recover_intrinsics_from_synthetic_rays():
    height, width = 6, 8
    fx, fy, cx, cy = 11.0, 13.0, 3.0, 2.0
    x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    rays = np.stack(((x_grid - cx) / fx, (y_grid - cy) / fy, np.ones_like(x_grid)), axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    intrinsics = mapanything_recover_pinhole_intrinsics_from_ray_directions(rays)

    np.testing.assert_allclose(intrinsics[0, 0], fx, atol=1e-5)
    np.testing.assert_allclose(intrinsics[1, 1], fy, atol=1e-5)
    np.testing.assert_allclose(intrinsics[0, 2], cx, atol=1e-5)
    np.testing.assert_allclose(intrinsics[1, 2], cy, atol=1e-5)
    np.testing.assert_allclose(intrinsics[2, 2], 1.0, atol=1e-6)
