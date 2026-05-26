import numpy as np
import mlx.core as mx

from mlx_spatial import (
    PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS,
    PIXAL3D_DINOV3_EMBED_DIM,
    Pixal3DProjectionStageConfig,
    build_pixal3d_projection_conditioning,
    pixal3d_front_view_transform,
    pixal3d_projection_grid_points,
    pixal3d_projection_stage_config,
    pixal3d_stage_with_grid_resolution,
    project_pixal3d_points_to_image,
    sample_pixal3d_feature_map,
)


def test_pixal3d_projection_stage_configs_match_upstream_shapes():
    assert PIXAL3D_DINOV3_EMBED_DIM == 1024
    assert PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS == 4

    ss = pixal3d_projection_stage_config("ss")
    shape = pixal3d_projection_stage_config("shape_512")
    texture = pixal3d_projection_stage_config("tex_1024")

    assert ss.image_size == 512
    assert ss.grid_resolution == 16
    assert ss.projected_token_count == 16**3
    assert ss.expected_projected_channels() == 1024
    assert shape.use_naf_upsample is True
    assert shape.expected_projected_channels() == 2048
    assert texture.image_size == 1024
    assert texture.naf_target_size == 1024


def test_pixal3d_projection_grid_matches_rotated_coordinate_convention():
    grid = pixal3d_projection_grid_points(2)

    assert grid.shape == (8, 3)
    np.testing.assert_allclose(np.array(grid[0]), [-1.0, 1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(np.array(grid[-1]), [1.0, -1.0, 1.0], atol=1e-6)


def test_pixal3d_front_view_transform_sets_camera_distance():
    transform = pixal3d_front_view_transform(mx.array([3.5], dtype=mx.float32))

    np.testing.assert_allclose(
        np.array(transform),
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -3.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        atol=1e-6,
    )


def test_pixal3d_projection_maps_origin_to_image_center():
    projection = project_pixal3d_points_to_image(
        mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32),
        camera_angle_x=0.8,
        distance=2.0,
        mesh_scale=1.0,
        image_resolution=512,
    )

    np.testing.assert_allclose(np.array(projection.points_2d[0, 0]), [256.0, 256.0], atol=1e-4)
    np.testing.assert_allclose(np.array(projection.depth[0, 0]), 2.0, atol=1e-6)
    assert bool(np.array(projection.valid_mask[0, 0]))


def test_pixal3d_feature_sampling_matches_grid_sample_pixel_centers():
    fmap_bhwc = mx.array(
        [[[[0.0], [1.0]], [[2.0], [3.0]]]],
        dtype=mx.float32,
    )
    points = mx.array([[[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]], dtype=mx.float32)

    sampled = sample_pixal3d_feature_map(fmap_bhwc, points, image_resolution=2)
    sampled_bchw = sample_pixal3d_feature_map(mx.transpose(fmap_bhwc, (0, 3, 1, 2)), points, image_resolution=2, layout="BCHW")

    np.testing.assert_allclose(np.array(sampled[:, :, 0]), [[0.0, 1.0, 1.5]], atol=1e-6)
    np.testing.assert_allclose(np.array(sampled_bchw), np.array(sampled), atol=1e-6)


def test_pixal3d_projection_conditioning_builds_sparse_stage_shapes():
    hidden = _hidden_states(batch=1, patch_grid=(2, 2), channels=3)
    stage = pixal3d_stage_with_grid_resolution(pixal3d_projection_stage_config("ss"), 2)

    conditioning = build_pixal3d_projection_conditioning(
        hidden,
        stage,
        camera_angle_x=0.8,
        distance=2.0,
        mesh_scale=1.0,
    )

    assert conditioning.ready
    assert conditioning.blocker is None
    assert conditioning.global_tokens is not None
    assert conditioning.projected_features is not None
    assert conditioning.global_tokens.shape == (1, 5, 3)
    assert conditioning.projected_features.shape == (1, 2**3, 3)
    assert conditioning.projected_lr_features is conditioning.projected_features


def test_pixal3d_projection_conditioning_blocks_shape_stage_without_naf_features():
    hidden = _hidden_states(batch=1, patch_grid=(2, 2), channels=3)
    stage = pixal3d_stage_with_grid_resolution(pixal3d_projection_stage_config("shape_512"), 2)

    conditioning = build_pixal3d_projection_conditioning(
        hidden,
        stage,
        camera_angle_x=0.8,
        distance=2.0,
        mesh_scale=1.0,
    )

    assert not conditioning.ready
    assert conditioning.projected_features is None
    assert conditioning.projected_lr_features is not None
    assert conditioning.projected_lr_features.shape == (1, 2**3, 3)
    assert conditioning.blocker is not None
    assert conditioning.blocker.stage == "naf-upsample"
    assert conditioning.blocker.metadata["expected_projected_channels"] == 6


def test_pixal3d_projection_conditioning_uses_supplied_naf_features():
    hidden = _hidden_states(batch=1, patch_grid=(2, 2), channels=3)
    stage = pixal3d_stage_with_grid_resolution(pixal3d_projection_stage_config("shape_512"), 2)
    naf = mx.array(np.arange(1 * 3 * 4 * 4, dtype=np.float32).reshape(1, 3, 4, 4))

    conditioning = build_pixal3d_projection_conditioning(
        hidden,
        stage,
        camera_angle_x=0.8,
        distance=2.0,
        mesh_scale=1.0,
        naf_feature_map=naf,
        naf_layout="BCHW",
    )

    assert conditioning.ready
    assert conditioning.projected_features is not None
    assert conditioning.projected_features.shape == (1, 2**3, 6)
    assert conditioning.projected_lr_features is not None
    assert conditioning.projected_lr_features.shape == (1, 2**3, 3)


def test_pixal3d_projection_conditioning_reports_invalid_patch_grid():
    hidden = mx.zeros((1, 8, 3), dtype=mx.float32)

    conditioning = build_pixal3d_projection_conditioning(
        hidden,
        pixal3d_projection_stage_config("ss"),
        camera_angle_x=0.8,
        distance=2.0,
        mesh_scale=1.0,
    )

    assert not conditioning.ready
    assert conditioning.blocker is not None
    assert conditioning.blocker.operation == "patch-grid-validation"


def test_pixal3d_projection_stage_grid_override_keeps_stage_metadata():
    stage = pixal3d_stage_with_grid_resolution(Pixal3DProjectionStageConfig("tiny", 64, 16), 3)

    assert stage.name == "tiny"
    assert stage.image_size == 64
    assert stage.grid_resolution == 3
    assert stage.projected_token_count == 27


def _hidden_states(*, batch: int, patch_grid: tuple[int, int], channels: int) -> mx.array:
    token_count = 1 + PIXAL3D_DEFAULT_NUM_REGISTER_TOKENS + patch_grid[0] * patch_grid[1]
    values = np.arange(batch * token_count * channels, dtype=np.float32).reshape(batch, token_count, channels)
    return mx.array(values)
