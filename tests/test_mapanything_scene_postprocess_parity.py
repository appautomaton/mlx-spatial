import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.mapanything_geometry import (
    MapAnythingPostprocessConfig,
    mapanything_postprocess_outputs_for_parity,
    postprocess_mapanything_heads_output,
)
from mlx_spatial.mapanything_heads import MapAnythingDenseHeadOutput, MapAnythingHeadsOutput
from mlx_spatial.mapanything_parity import (
    MAPANYTHING_TORCH_PARITY_ENV,
    compare_mapanything_parity_tensors,
    load_mapanything_parity_bundle,
    mapanything_parity_report_to_dict,
)
from mlx_spatial.mapanything_preprocess import MapAnythingPreprocessedInput, MapAnythingPreprocessedView


DEFAULT_REFERENCE = Path("/tmp/mapanything-desk-scene-reference.npz")

pytestmark = pytest.mark.skipif(
    os.environ.get(MAPANYTHING_TORCH_PARITY_ENV) != "1",
    reason="opt-in MapAnything Torch reference parity",
)


def test_mapanything_scene_postprocess_matches_desk_reference():
    reference_path = Path(os.environ.get("MAPANYTHING_SCENE_REFERENCE", str(DEFAULT_REFERENCE)))
    if not reference_path.is_file():
        pytest.fail(
            f"missing scene reference bundle: {reference_path}; run "
            "tools/mapanything_dump_torch_scene_reference.py first"
        )

    reference = load_mapanything_parity_bundle(reference_path)
    heads_output = _heads_output_from_reference(reference.tensors)
    preprocessed = _preprocessed_input_from_reference(reference.tensors)

    result = postprocess_mapanything_heads_output(
        heads_output,
        preprocessed,
        config=MapAnythingPostprocessConfig(apply_mask=True, mask_edges=True),
    )

    stable_names = (
        "final.ray_directions.0",
        "final.ray_directions.1",
        "final.cam_trans.0",
        "final.cam_trans.1",
        "final.cam_quats.0",
        "final.cam_quats.1",
        "final.camera_poses.0",
        "final.camera_poses.1",
        "final.metric_scaling_factor.0",
        "final.metric_scaling_factor.1",
        "final.intrinsics.0",
        "final.intrinsics.1",
        "final.conf.0",
        "final.conf.1",
        "final.non_ambiguous_mask.0",
        "final.non_ambiguous_mask.1",
        "final.non_ambiguous_mask_logits.0",
        "final.non_ambiguous_mask_logits.1",
        "final.img_no_norm.0",
        "final.img_no_norm.1",
        "scene.intrinsics",
        "scene.camera_poses",
        "scene.conf",
        "scene.images",
    )
    actual = mapanything_postprocess_outputs_for_parity(result)
    report = compare_mapanything_parity_tensors(
        actual,
        reference,
        names=stable_names,
        atol=5e-3,
        rtol=2e-4,
    )

    assert result.trace["runtime_depends_on_torch"] is False
    assert "edge-mask logic" in result.trace["edge_mask_numeric_boundary"]
    assert report.passed, mapanything_parity_report_to_dict(report)

    # The edge-mask path is intentionally NumPy runtime code, while the saved
    # reference mask came from Torch pointmaps. The only tolerated mismatch is a
    # tiny set of threshold pixels along normal/depth edges.
    for view_index in range(2):
        actual_mask = actual[f"final.mask.{view_index}"].astype(bool)
        expected_mask = reference.tensors[f"final.mask.{view_index}"].astype(bool)
        mismatch_count = int((actual_mask != expected_mask).sum())
        assert mismatch_count <= 8

        common_mask = actual_mask & expected_mask
        for key in ("pts3d", "pts3d_cam", "depth_along_ray", "depth_z"):
            actual_tensor = actual[f"final.{key}.{view_index}"]
            expected_tensor = reference.tensors[f"final.{key}.{view_index}"]
            common_values = common_mask.repeat(actual_tensor.shape[-1], axis=-1)
            actual_values = actual_tensor[common_values]
            expected_values = expected_tensor[common_values]
            assert actual_values.size > 0
            np.testing.assert_allclose(actual_values, expected_values, atol=5e-3, rtol=2e-4)

    actual_scene_mask = actual["scene.final_masks"].astype(bool)
    expected_scene_mask = reference.tensors["scene.final_masks"].astype(bool)
    assert int((actual_scene_mask != expected_scene_mask).sum()) <= 9


def _heads_output_from_reference(tensors: dict[str, object]) -> MapAnythingHeadsOutput:
    dense_value = mx.array(tensors["head.dense.value"])
    dense = MapAnythingDenseHeadOutput(
        value=dense_value,
        confidence=mx.array(tensors["head.dense.confidence"]),
        mask=mx.array(tensors["head.dense.mask"]),
        logits=mx.array(tensors["head.dense.logits"]),
        decoded_channels=mx.zeros(
            (
                int(dense_value.shape[0]),
                6,
                int(dense_value.shape[2]),
                int(dense_value.shape[3]),
            ),
            dtype=mx.float32,
        ),
    )
    return MapAnythingHeadsOutput(
        dense=dense,
        pose_value=mx.array(tensors["head.pose.value"]),
        scale_value=mx.array(tensors["head.scale.value"]),
        trace={"source": "desk-reference"},
    )


def _preprocessed_input_from_reference(tensors: dict[str, object]) -> MapAnythingPreprocessedInput:
    views = []
    for index in range(2):
        image = tensors[f"input.img.{index}"]
        height, width = int(image.shape[2]), int(image.shape[3])
        views.append(
            MapAnythingPreprocessedView(
                image_path=Path(f"reference-view-{index}.jpg"),
                img=mx.array(image),
                true_shape=(height, width),
                idx=index,
                instance=str(index),
                data_norm_type="dinov2",
                original_size=(width, height),
                processed_size=(height, width),
                target_size=(width, height),
            )
        )
    return MapAnythingPreprocessedInput(
        views=tuple(views),
        target_size=(518, 392),
        average_aspect_ratio=518 / 392,
        resize_mode="fixed_mapping",
        patch_size=14,
        resolution_set=518,
    )
