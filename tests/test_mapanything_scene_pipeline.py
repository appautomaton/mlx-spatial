from pathlib import Path

import numpy as np
import pytest

from mlx_spatial.mapanything_scene import (
    MAPANYTHING_SCENE_OUTPUT_KEYS,
    MapAnythingScenePipeline,
    write_mapanything_scene_npz,
)


ROOT = Path(__file__).resolve().parents[1]


def test_mapanything_scene_pipeline_generates_local_desk_npz(tmp_path):
    model_root = ROOT / "weights/map-anything"
    image_root = ROOT / "inputs/map-anything/desk"
    if not (model_root / "model.safetensors").is_file() or not image_root.is_dir():
        pytest.skip("local MapAnything weights or Desk inputs are absent")

    result = MapAnythingScenePipeline(model_root).generate(image_root)

    assert result.ready, result.trace.blocker
    assert result.predictions is not None
    assert result.trace.completed_stages == (
        "asset-config-validation",
        "image-preprocessing",
        "model-config",
        "checkpoint-loading:encoder",
        "full-encoder",
        "checkpoint-loading:heads",
        "fusion-norm",
        "checkpoint-loading:info-sharing",
        "info-sharing",
        "prediction-heads",
        "scene-postprocess",
    )
    assert result.trace.frame_count == 2
    assert result.trace.target_size == (518, 392)
    assert result.trace.metadata["runtime_depends_on_torch"] is False
    assert result.trace.metadata["implemented_boundary"] == "scene-generation"
    assert result.predictions.images.shape == (2, 392, 518, 3)
    assert result.predictions.depth.shape == (2, 392, 518)
    assert result.predictions.confidence.shape == (2, 392, 518)
    assert result.predictions.masks.shape == (2, 392, 518)
    assert result.predictions.intrinsics.shape == (2, 3, 3)
    assert result.predictions.camera_poses.shape == (2, 4, 4)
    assert result.predictions.extrinsics.shape == (2, 4, 4)
    assert result.predictions.world_points.shape == (2, 392, 518, 3)
    assert np.isfinite(result.predictions.world_points).all()

    output_path = write_mapanything_scene_npz(tmp_path / "desk-scene.npz", result.predictions)

    with np.load(output_path, allow_pickle=False) as data:
        assert set(MAPANYTHING_SCENE_OUTPUT_KEYS).issubset(data.files)
        assert data["world_points"].shape == (2, 392, 518, 3)
        assert "scene-generation" in str(data["__metadata_json__"])
