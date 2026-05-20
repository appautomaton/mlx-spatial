import numpy as np

from mlx_spatial.sam3d_export import SAM3D_SH_C0, postprocess_sam3d_mesh_for_glb
from mlx_spatial.sam3d_render import (
    optimize_sam3d_layout_alignment,
    render_sam3d_gaussian_multiview,
    sam3d_gaussian_fields_to_raster_inputs,
    sam3d_orbit_cameras,
)


def _fixture_gaussians():
    rgb = np.array([[1.0, 0.25, 0.0]], dtype=np.float32)
    return {
        "xyz": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        "features_dc": ((rgb - 0.5) / SAM3D_SH_C0).reshape(1, 1, 3).astype(np.float32),
        "opacity": np.array([[8.0]], dtype=np.float32),
        "scale": np.full((1, 3), np.log(0.18), dtype=np.float32),
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }


def test_render_orbit_camera_setup_keeps_target_centered_and_in_front():
    cameras = sam3d_orbit_cameras(view_count=4, image_size=(16, 20), radius=3.0)
    target_h = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    assert len(cameras) == 4
    for camera in cameras:
        camera_target = camera.view_matrix @ target_h
        projected = camera.intrinsics @ camera_target[:3]
        projected = projected[:2] / projected[2]
        np.testing.assert_allclose(camera_target[2], 3.0, atol=1e-6)
        np.testing.assert_allclose(projected, np.array([10.0, 8.0], dtype=np.float32), atol=1e-5)


def test_render_sam3d_gaussian_multiview_produces_consistent_renders():
    result = render_sam3d_gaussian_multiview(
        **_fixture_gaussians(),
        image_size=(16, 16),
        view_count=4,
        camera_radius=2.5,
        use_metal=False,
    )

    assert result.rgba.shape == (4, 16, 16, 4)
    assert result.depth.shape == (4, 16, 16)
    assert result.metadata["view_count"] == 4
    assert result.metadata["rotation_source_convention"] == "SAM3D WXYZ"
    assert all(count > 0 for count in result.pixel_counts)
    alpha_sums = result.rgba[..., 3].sum(axis=(1, 2))
    np.testing.assert_allclose(alpha_sums, np.full((4,), alpha_sums[0]), rtol=1e-5, atol=1e-5)


def test_render_adapter_converts_sam3d_fields_to_raster_inputs():
    means, quats, scales, alphas, sh_dc = sam3d_gaussian_fields_to_raster_inputs(**_fixture_gaussians())

    np.testing.assert_allclose(means, np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(quats, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(scales, np.full((1, 3), 0.18, dtype=np.float32), rtol=1e-6)
    assert alphas[0] > 0.99
    assert sh_dc.shape == (1, 1, 3)


def test_layout_post_optimization_improves_scene_alignment():
    target = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    angle = np.deg2rad(20.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    source = (target @ rotation.T) + np.array([0.5, -0.25, 0.2], dtype=np.float32)

    result = optimize_sam3d_layout_alignment(source, target, iterations=4)

    assert result.improved
    assert result.optimized_rmse < result.initial_rmse * 0.1
    np.testing.assert_allclose(result.aligned_points, target, atol=1e-5)


def test_holes_postprocess_fills_clean_mesh_holes():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.001],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64)

    result = postprocess_sam3d_mesh_for_glb(vertices, faces, target_faces=0, simplify=False, min_component_faces=1)

    assert result.stats.hole_fill.filled_loops == 1
    assert result.stats.hole_fill.faces_added == 3
    assert result.faces.shape[0] > faces.shape[0]
