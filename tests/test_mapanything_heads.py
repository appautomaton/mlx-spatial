from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from tests.safetensors_test_utils import save_file

import mlx_spatial
from mlx_spatial.mapanything_heads import (
    MAPANYTHING_DPT_INPUT_PROCESS_REQUIRED_KEYS,
    MAPANYTHING_DPT_REFINENET_REQUIRED_KEYS,
    MAPANYTHING_DPT_REGRESSOR_REQUIRED_KEYS,
    MAPANYTHING_HEAD_BASE_REQUIRED_KEYS,
    MAPANYTHING_POSE_HEAD_REQUIRED_KEYS,
    MAPANYTHING_SCALE_HEAD_REQUIRED_KEYS,
    MapAnythingHeads,
    MapAnythingHeadsConfig,
    apply_mapanything_fusion_norm,
    load_mapanything_heads_weights,
    mapanything_heads_config_from_model_config,
    mapanything_heads_outputs_for_parity,
    mapanything_heads_required_keys,
    run_mapanything_heads,
    validate_mapanything_heads_weights,
)
from tests.mapanything_scene_fixture import (
    tiny_heads_config as _tiny_heads_config,
    tiny_heads_weights as _tiny_heads_weights,
    tiny_model_config_json,
)


def test_mapanything_heads_required_keys_are_explicit_and_unique():
    keys = mapanything_heads_required_keys(_tiny_heads_config())

    assert keys[:2] == MAPANYTHING_HEAD_BASE_REQUIRED_KEYS
    assert "fusion_norm_layer.weight" in keys
    assert "dense_head.0.input_process.0.0.1.weight" in keys
    assert "dense_head.0.scratch.refinenet4.resConfUnit1.conv1.weight" not in keys
    assert "pose_head.fc_rot.weight" in keys
    assert "scale_head.output_proj.weight" in keys
    assert len(keys) == len(
        (
            *MAPANYTHING_HEAD_BASE_REQUIRED_KEYS,
            *MAPANYTHING_DPT_INPUT_PROCESS_REQUIRED_KEYS,
            *MAPANYTHING_DPT_REFINENET_REQUIRED_KEYS,
            *MAPANYTHING_DPT_REGRESSOR_REQUIRED_KEYS,
            *MAPANYTHING_POSE_HEAD_REQUIRED_KEYS,
            *MAPANYTHING_SCALE_HEAD_REQUIRED_KEYS,
        )
    )
    assert len(keys) == len(set(keys))


def test_mapanything_heads_validate_missing_and_shape_errors():
    config = _tiny_heads_config()
    weights = _tiny_heads_weights(config)
    weights.pop("scale_head.output_proj.weight")

    with pytest.raises(ValueError, match="missing MapAnything head tensors"):
        validate_mapanything_heads_weights(weights, config)

    weights = _tiny_heads_weights(config)
    weights["pose_head.fc_t.weight"] = mx.zeros((4, config.pose_hidden_dim), dtype=mx.float32)
    with pytest.raises(ValueError, match="pose_head.fc_t.weight has shape"):
        validate_mapanything_heads_weights(weights, config)


def test_mapanything_fusion_norm_matches_channel_layer_norm():
    config = _tiny_heads_config()
    weights = _tiny_heads_weights(config)
    feature = mx.arange(1 * config.input_feature_dim * 2 * 3, dtype=mx.float32).reshape(
        (1, config.input_feature_dim, 2, 3)
    )

    fused = apply_mapanything_fusion_norm((feature,), weights, config=config)[0]
    mx.eval(fused)

    expected = _numpy_nchw_layer_norm(np.asarray(feature))
    np.testing.assert_allclose(np.asarray(fused), expected, atol=2e-5, rtol=2e-5)


def test_mapanything_heads_run_tiny_shapes_and_trace():
    config = _tiny_heads_config()
    weights = _tiny_heads_weights(config)
    dense_features = tuple(
        mx.array(np.random.default_rng(index).normal(size=(1, config.input_feature_dim, 2, 3)).astype(np.float32))
        for index in range(4)
    )
    scale_tokens = mx.ones((1, config.input_feature_dim, 1), dtype=mx.float32) * 0.1

    output = MapAnythingHeads(weights, config)(dense_features, scale_tokens, image_shape=(4, 6))
    mx.eval(output.dense.value, output.dense.confidence, output.dense.mask, output.pose_value, output.scale_value)

    assert tuple(output.dense.value.shape) == (1, 4, 4, 6)
    assert tuple(output.dense.confidence.shape) == (1, 1, 4, 6)
    assert tuple(output.dense.mask.shape) == (1, 1, 4, 6)
    assert tuple(output.pose_value.shape) == (1, 7)
    assert tuple(output.scale_value.shape) == (1, 1)
    assert output.trace["runtime_depends_on_torch"] is False
    assert output.trace["dense_output_type"] == "raydirs+depth+confidence+mask"
    assert set(mapanything_heads_outputs_for_parity(output)) >= {
        "head.dense.value",
        "head.dense.confidence",
        "head.dense.mask",
        "head.pose.value",
        "head.scale.value",
    }


def test_load_mapanything_heads_weights_maps_configured_layers(tmp_path: Path):
    config = _tiny_heads_config()
    model_root = tmp_path / "weights"
    model_root.mkdir()
    (model_root / "config.json").write_text(_tiny_config_json(), encoding="utf-8")
    save_file(_tiny_heads_weights(config), model_root / "model.safetensors")

    loaded = load_mapanything_heads_weights(model_root, config=config)

    assert set(loaded) == set(mapanything_heads_required_keys(config))
    assert tuple(loaded["fusion_norm_layer.weight"].shape) == (config.input_feature_dim,)
    assert tuple(loaded["dense_head.1.conv2.2.weight"].shape) == (
        config.dense_output_dim,
        config.feature_dim // 2,
        1,
        1,
    )


def test_mapanything_heads_public_exports():
    assert mlx_spatial.MapAnythingHeads is MapAnythingHeads
    assert mlx_spatial.MapAnythingHeadsConfig is MapAnythingHeadsConfig
    assert mlx_spatial.load_mapanything_heads_weights is load_mapanything_heads_weights
    assert mlx_spatial.mapanything_heads_config_from_model_config is mapanything_heads_config_from_model_config


def _numpy_nchw_layer_norm(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    nhwc = np.transpose(values, (0, 2, 3, 1))
    mean = nhwc.mean(axis=-1, keepdims=True)
    centered = nhwc - mean
    normalized = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    return np.transpose(normalized, (0, 3, 1, 2)).astype(np.float32)


def _tiny_config_json() -> str:
    return tiny_model_config_json(info_dim=4)
