from pathlib import Path

import mlx.core as mx
import numpy as np

import mlx_spatial
from mlx_spatial.mapanything_geometry import (
    MapAnythingPostprocessConfig,
    mapanything_heads_output_to_raw_views,
    mapanything_postprocess_outputs_for_parity,
    postprocess_mapanything_heads_output,
)
from mlx_spatial.mapanything_heads import MapAnythingDenseHeadOutput, MapAnythingHeadsOutput
from mlx_spatial.mapanything_preprocess import MapAnythingPreprocessedInput, MapAnythingPreprocessedView


def test_mapanything_heads_output_to_raw_views_applies_scale_and_splits_views():
    heads_output = _tiny_heads_output(mask_zero=False)

    raw_views = mapanything_heads_output_to_raw_views(heads_output, view_count=2)

    assert len(raw_views) == 2
    np.testing.assert_allclose(raw_views[0]["depth_along_ray"][0, 0, 0, 0], 4.0, atol=1e-6)
    np.testing.assert_allclose(
        raw_views[0]["pts3d_cam"][0, 0, 0],
        raw_views[0]["ray_directions"][0, 0, 0] * 4.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(raw_views[1]["cam_trans"][0], [2.0, 0.0, 0.0], atol=1e-6)
    assert raw_views[1]["pts3d"][0, 0, 0, 0] > 1.0
    assert raw_views[0]["non_ambiguous_mask"].dtype == np.bool_


def test_mapanything_postprocess_builds_final_outputs_and_scene_payload():
    heads_output = _tiny_heads_output(mask_zero=True)
    preprocessed = _tiny_preprocessed_input()

    result = postprocess_mapanything_heads_output(
        heads_output,
        preprocessed,
        config=MapAnythingPostprocessConfig(mask_edges=False),
    )

    assert result.trace["runtime_depends_on_torch"] is False
    assert tuple(result.views[0]["camera_poses"].shape) == (1, 4, 4)
    assert tuple(result.views[0]["intrinsics"].shape) == (1, 3, 3)
    assert result.views[0]["mask"][0, 0, 1, 0] == 0
    np.testing.assert_allclose(result.views[0]["depth_z"][0, 0, 1, 0], 0.0, atol=1e-6)
    assert result.views[0]["depth_z"][0, 0, 0, 0] > 0.0

    scene = result.scene_payload
    assert set(scene) == {
        "images",
        "depth",
        "confidence",
        "masks",
        "intrinsics",
        "camera_poses",
        "extrinsics",
        "world_points",
    }
    assert tuple(scene["depth"].shape) == (2, 2, 3)
    assert tuple(scene["world_points"].shape) == (2, 2, 3, 3)
    assert scene["masks"][0, 0, 1] == 0
    np.testing.assert_allclose(scene["camera_poses"][1, :3, 3], [2.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(scene["extrinsics"][1] @ scene["camera_poses"][1], np.eye(4), atol=1e-6)

    parity = mapanything_postprocess_outputs_for_parity(result)
    assert "final.pts3d.0" in parity
    assert "final.mask.1" in parity
    assert "scene.world_points" in parity


def test_mapanything_postprocess_public_exports():
    assert mlx_spatial.MapAnythingPostprocessConfig is MapAnythingPostprocessConfig
    assert mlx_spatial.postprocess_mapanything_heads_output is postprocess_mapanything_heads_output
    assert mlx_spatial.mapanything_heads_output_to_raw_views is mapanything_heads_output_to_raw_views


def _tiny_heads_output(*, mask_zero: bool) -> MapAnythingHeadsOutput:
    height, width = 2, 3
    dense_value = np.zeros((2, 4, height, width), dtype=np.float32)
    x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")
    rays = np.stack(((x_grid - 1.0) / 4.0, (y_grid - 0.5) / 4.0, np.ones_like(x_grid)), axis=0)
    rays = rays / np.linalg.norm(rays, axis=0, keepdims=True)
    dense_value[:, :3, :, :] = rays
    dense_value[:, 3, :, :] = 2.0
    confidence = np.ones((2, 1, height, width), dtype=np.float32) * 3.0
    mask = np.ones((2, 1, height, width), dtype=np.float32)
    if mask_zero:
        mask[0, 0, 0, 1] = 0.25
    logits = np.where(mask > 0.5, 5.0, -5.0).astype(np.float32)
    pose_value = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale_value = np.array([[2.0]], dtype=np.float32)
    dense = MapAnythingDenseHeadOutput(
        value=mx.array(dense_value),
        confidence=mx.array(confidence),
        mask=mx.array(mask),
        logits=mx.array(logits),
        decoded_channels=mx.zeros((2, 6, height, width), dtype=mx.float32),
    )
    return MapAnythingHeadsOutput(
        dense=dense,
        pose_value=mx.array(pose_value),
        scale_value=mx.array(scale_value),
        trace={"stage": "tiny-test"},
    )


def _tiny_preprocessed_input() -> MapAnythingPreprocessedInput:
    views = []
    for index in range(2):
        views.append(
            MapAnythingPreprocessedView(
                image_path=Path(f"view-{index}.jpg"),
                img=mx.zeros((1, 3, 2, 3), dtype=mx.float32),
                true_shape=(2, 3),
                idx=index,
                instance=str(index),
                data_norm_type="dinov2",
                original_size=(3, 2),
                processed_size=(2, 3),
                target_size=(3, 2),
            )
        )
    return MapAnythingPreprocessedInput(
        views=tuple(views),
        target_size=(3, 2),
        average_aspect_ratio=1.5,
        resize_mode="fixed_mapping",
        patch_size=1,
        resolution_set=518,
    )
