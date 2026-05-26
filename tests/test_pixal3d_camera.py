import numpy as np
import mlx.core as mx

from mlx_spatial import (
    PIXAL3D_DEFAULT_MAX_NUM_TOKENS,
    Pixal3DCameraParams,
    pixal3d_compute_f_pixels,
    pixal3d_distance_from_fov,
    pixal3d_manual_camera_params,
    pixal3d_requested_hr_resolution,
    pixal3d_select_hr_resolution,
    pixal3d_stage_plan,
)


def test_pixal3d_manual_camera_params_match_upstream_formula():
    camera = pixal3d_manual_camera_params(0.2)
    distance = pixal3d_distance_from_fov(0.2)

    assert isinstance(camera, Pixal3DCameraParams)
    assert camera.camera_angle_x == 0.2
    assert camera.mesh_scale == 1.0
    assert camera.image_resolution == 512
    assert camera.distance == distance["distance_from_x"]
    np.testing.assert_allclose(distance["f_pixels"], pixal3d_compute_f_pixels(0.2), atol=1e-6)


def test_pixal3d_stage_plan_maps_supported_cascade_types():
    plan_1024 = pixal3d_stage_plan("1024_cascade", max_num_tokens=PIXAL3D_DEFAULT_MAX_NUM_TOKENS)
    plan_1536 = pixal3d_stage_plan("1536_cascade", max_num_tokens=PIXAL3D_DEFAULT_MAX_NUM_TOKENS)

    assert pixal3d_requested_hr_resolution("1024_cascade") == 1024
    assert plan_1024.requested_hr_resolution == 1024
    assert plan_1024.actual_hr_grid_resolution == 64
    assert plan_1536.requested_hr_resolution == 1536
    assert plan_1536.actual_hr_grid_resolution == 96
    assert plan_1536.texture_grid_resolution == 96


def test_pixal3d_hr_resolution_reduces_until_token_guard_passes():
    coords = mx.array([[0, i, i * 2, i * 3] for i in range(80)], dtype=mx.int32)

    actual, token_count = pixal3d_select_hr_resolution(
        coords,
        requested_hr_resolution=1536,
        max_num_tokens=8,
    )

    assert actual == 1024
    assert token_count >= 8


def test_pixal3d_stage_plan_records_quantized_hr_token_count():
    coords = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 511, 511, 511]], dtype=np.int32)

    plan = pixal3d_stage_plan("1536_cascade", max_num_tokens=49_152, hr_coordinates=coords)

    assert plan.actual_hr_resolution == 1536
    assert plan.hr_token_count == 2
