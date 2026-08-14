import json
from pathlib import Path

import numpy as np
import pytest

import mlx_spatial.mapanything_scene as mapanything_scene
from mlx_spatial.mapanything_scene import (
    MAPANYTHING_SCENE_OUTPUT_KEYS,
    MapAnythingScenePipeline,
    MapAnythingSceneResult,
    write_mapanything_scene_npz,
)
from tests.golden_assertions import assert_golden_close, summarize_array
from tests.mapanything_scene_fixture import (
    build_mapanything_miniature_scene_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_MANIFEST = ROOT / "tests/data/mapanything_miniature_scene_golden.json"


@pytest.mark.integration
def test_mapanything_miniature_scene_pipeline_matches_golden(
    tmp_path,
    monkeypatch,
):
    fixture = build_mapanything_miniature_scene_fixture(tmp_path)
    monkeypatch.setattr(
        mapanything_scene,
        "mapanything_heads_config_from_model_config",
        lambda _: fixture.heads_config,
    )

    result = MapAnythingScenePipeline(fixture.model_root).generate(
        fixture.image_root,
        resize_mode="fixed_size",
        size=(4, 4),
    )

    assert result.ready, result.trace.blocker
    assert result.predictions is not None
    output_path = write_mapanything_scene_npz(
        tmp_path / "outputs/mapanything/miniature-scene.npz",
        result.predictions,
        metadata={"completed_stages": list(result.trace.completed_stages)},
    )
    with np.load(output_path, allow_pickle=False) as payload:
        assert set(MAPANYTHING_SCENE_OUTPUT_KEYS).issubset(payload.files)
        assert "scene-generation" in str(payload["__metadata_json__"])

    expected = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    assert_golden_close(_summarize_miniature_scene(result), expected)


def _summarize_miniature_scene(
    result: MapAnythingSceneResult,
) -> dict[str, object]:
    assert result.predictions is not None
    stable_prediction_keys = tuple(
        key
        for key in MAPANYTHING_SCENE_OUTPUT_KEYS
        if key not in {"intrinsics", "world_points"}
    )
    return {
        "schema_version": 2,
        "fixture": {
            "kind": "generated-miniature-checkpoint",
            "source": "deterministic synthetic tensors",
            "quantization": "none",
            "random_seeds": {"encoder": 42, "heads": 123},
            "views": 2,
            "image_size": [4, 4],
            "patch_size": 2,
            "encoder_layers": 2,
            "info_sharing_layers": 2,
            "covered": [
                "asset inspection",
                "safetensors loading",
                "image preprocessing",
                "full encoder",
                "fusion norm",
                "multi-view info sharing",
                "dense pose and scale heads",
                "scene geometry postprocess",
                "NPZ artifact writing",
            ],
            "not_covered": [
                "numeric parity with official MapAnything weights",
                "production image resolution",
                "performance or memory benchmarking",
            ],
        },
        "trace": {
            "completed_stages": list(result.trace.completed_stages),
            "frame_count": result.trace.frame_count,
            "target_size": list(result.trace.target_size or ()),
            "patch_grid": result.trace.metadata["patch_grid"],
            "implemented_boundary": result.trace.metadata["implemented_boundary"],
        },
        "predictions": {
            key: summarize_array(getattr(result.predictions, key))
            for key in stable_prediction_keys
        },
        "geometry_invariants": _summarize_geometry_invariants(result),
        "artifact": {
            "keys": sorted(
                (*MAPANYTHING_SCENE_OUTPUT_KEYS, "__metadata_json__")
            ),
            "format": "npz",
        },
    }


def _summarize_geometry_invariants(
    result: MapAnythingSceneResult,
) -> dict[str, object]:
    assert result.predictions is not None
    intrinsics = np.asarray(result.predictions.intrinsics)
    world_points = np.asarray(result.predictions.world_points)
    focal_lengths = np.stack((intrinsics[:, 0, 0], intrinsics[:, 1, 1]), axis=1)
    return {
        "intrinsics": {
            "shape": list(intrinsics.shape),
            "dtype": str(intrinsics.dtype),
            "finite": bool(np.isfinite(intrinsics).all()),
            "positive_focal_lengths": bool((focal_lengths > 0.0).all()),
            "homogeneous_bottom_row": bool(
                np.allclose(
                    intrinsics[:, 2, :],
                    np.array([0.0, 0.0, 1.0], dtype=np.float32),
                    atol=1e-6,
                    rtol=0.0,
                )
            ),
        },
        "world_points": {
            "shape": list(world_points.shape),
            "dtype": str(world_points.dtype),
            "finite": bool(np.isfinite(world_points).all()),
            "nonzero": bool(np.linalg.norm(world_points) > 1e-3),
            "bounded": bool(np.max(np.abs(world_points)) < 10.0),
        },
    }


@pytest.mark.integration
@pytest.mark.real_assets
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
