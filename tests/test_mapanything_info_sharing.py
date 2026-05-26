from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.mapanything_model import (
    MAPANYTHING_INFO_SHARING_BASE_REQUIRED_KEYS,
    MAPANYTHING_INFO_SHARING_BLOCK_SUFFIXES,
    MapAnythingInfoSharing,
    MapAnythingInfoSharingConfig,
    load_mapanything_info_sharing_weights,
    mapanything_info_sharing_config_from_model_config,
    mapanything_info_sharing_outputs_for_parity,
    mapanything_info_sharing_required_keys,
    run_mapanything_info_sharing,
    validate_mapanything_info_sharing_weights,
)


def test_mapanything_info_sharing_required_keys_cover_configured_layers():
    config = _tiny_info_config(depth=3, indices=(0, 2))

    keys = mapanything_info_sharing_required_keys(config)

    assert keys[:4] == MAPANYTHING_INFO_SHARING_BASE_REQUIRED_KEYS
    assert "scale_token" in keys
    assert "info_sharing.self_attention_blocks.0.attn.qkv.weight" in keys
    assert "info_sharing.self_attention_blocks.2.mlp.w3.bias" in keys
    assert "info_sharing.self_attention_blocks.3.norm1.weight" not in keys
    assert len(keys) == len(MAPANYTHING_INFO_SHARING_BASE_REQUIRED_KEYS) + (
        config.depth * len(MAPANYTHING_INFO_SHARING_BLOCK_SUFFIXES)
    )
    assert len(keys) == len(set(keys))


def test_mapanything_info_sharing_validates_missing_shape_and_unsupported_configs():
    config = _tiny_info_config()
    weights = _tiny_info_weights(config)
    weights.pop("info_sharing.self_attention_blocks.1.attn.qkv.weight")

    with pytest.raises(ValueError, match="missing MapAnything info-sharing tensors"):
        validate_mapanything_info_sharing_weights(weights, config)

    weights = _tiny_info_weights(config)
    weights["info_sharing.norm.weight"] = mx.zeros((config.dim + 1,), dtype=mx.float32)
    with pytest.raises(ValueError, match="info_sharing.norm.weight has shape"):
        validate_mapanything_info_sharing_weights(weights, config)

    with pytest.raises(ValueError, match="identity proj_embed only"):
        validate_mapanything_info_sharing_weights(
            _tiny_info_weights(config),
            MapAnythingInfoSharingConfig(input_embed_dim=4, dim=8, depth=2, num_heads=2),
        )


def test_mapanything_info_sharing_runs_tiny_alternating_attention_shapes_and_trace():
    config = _tiny_info_config(indices=(0, 1))
    weights = _tiny_info_weights(config)
    features, registers = _tiny_features_and_registers(config)

    output = MapAnythingInfoSharing(weights, config)(
        features,
        additional_tokens_per_view=registers,
    )
    mx.eval(
        output.final.features[0],
        output.final.features[1],
        output.final.additional_token_features,
        output.final.additional_token_features_per_view[0],
    )

    assert len(output.final.features) == 2
    assert tuple(output.final.features[0].shape) == (1, config.dim, 2, 2)
    assert tuple(output.final.features[1].shape) == (1, config.dim, 2, 2)
    assert tuple(output.final.additional_token_features.shape) == (1, config.dim, 1)
    assert tuple(output.final.additional_token_features_per_view[0].shape) == (1, config.dim, 1)
    assert len(output.intermediates) == 2
    assert output.trace["runtime_depends_on_torch"] is False
    assert output.trace["attention_schedule"] == "even-global/odd-frame"
    assert output.trace["additional_tokens_per_view"] == 1
    assert set(mapanything_info_sharing_outputs_for_parity(output)) >= {
        "info.final.features.0",
        "info.final.features.1",
        "info.final.additional_token_features",
        "info.final.additional_token_features_per_view.0",
        "info.intermediate.0.features.0",
        "info.intermediate.1.features.1",
    }


def test_mapanything_info_sharing_odd_frame_attention_holds_global_scale_token_out():
    config = _tiny_info_config(indices=(1,))
    weights = _tiny_info_weights(config, identity_odd_attention=True)
    features = tuple(mx.zeros((1, config.dim, 2, 2), dtype=mx.float32) for _ in range(2))

    output = run_mapanything_info_sharing(features, weights, config=config)
    mx.eval(output.final.additional_token_features)

    scale_token = np.asarray(weights["scale_token"]).reshape(1, config.dim, 1)
    expected = _numpy_channel_layer_norm(scale_token)
    np.testing.assert_allclose(
        np.asarray(output.final.additional_token_features),
        expected,
        atol=2e-5,
        rtol=2e-5,
    )


def test_load_mapanything_info_sharing_weights_maps_configured_layers(tmp_path: Path):
    config = _tiny_info_config()
    model_root = tmp_path / "weights"
    model_root.mkdir()
    (model_root / "config.json").write_text(_tiny_config_json(), encoding="utf-8")
    save_file(_tiny_info_weights(config), model_root / "model.safetensors")

    loaded = load_mapanything_info_sharing_weights(model_root, config=config)

    assert set(loaded) == set(mapanything_info_sharing_required_keys(config))
    assert tuple(loaded["scale_token"].shape) == (config.dim,)
    assert tuple(loaded["info_sharing.self_attention_blocks.1.attn.qkv.weight"].shape) == (
        3 * config.dim,
        config.dim,
    )


def test_mapanything_info_sharing_public_exports():
    assert mlx_spatial.MapAnythingInfoSharing is MapAnythingInfoSharing
    assert mlx_spatial.MapAnythingInfoSharingConfig is MapAnythingInfoSharingConfig
    assert mlx_spatial.load_mapanything_info_sharing_weights is load_mapanything_info_sharing_weights
    assert (
        mlx_spatial.mapanything_info_sharing_config_from_model_config
        is mapanything_info_sharing_config_from_model_config
    )


def _tiny_info_config(depth: int = 2, indices: tuple[int, ...] = (0, 1)) -> MapAnythingInfoSharingConfig:
    return MapAnythingInfoSharingConfig(
        input_embed_dim=8,
        dim=8,
        depth=depth,
        num_heads=2,
        indices=indices,
        norm_intermediate=True,
    )


def _tiny_info_weights(
    config: MapAnythingInfoSharingConfig,
    *,
    identity_odd_attention: bool = False,
) -> dict[str, mx.array]:
    hidden = config.swiglu_hidden_features
    weights: dict[str, mx.array] = {
        "scale_token": mx.array(np.linspace(-0.5, 0.6, config.dim, dtype=np.float32)),
        "info_sharing.norm.weight": mx.ones((config.dim,), dtype=mx.float32),
        "info_sharing.norm.bias": mx.zeros((config.dim,), dtype=mx.float32),
        "info_sharing.view_pos_table": mx.zeros((1, config.dim), dtype=mx.float32),
    }
    for block_index in range(config.depth):
        prefix = f"info_sharing.self_attention_blocks.{block_index}"
        weights[f"{prefix}.norm1.weight"] = mx.ones((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.norm1.bias"] = mx.zeros((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.attn.qkv.weight"] = mx.zeros((3 * config.dim, config.dim), dtype=mx.float32)
        weights[f"{prefix}.attn.qkv.bias"] = mx.zeros((3 * config.dim,), dtype=mx.float32)
        weights[f"{prefix}.attn.proj.weight"] = mx.zeros((config.dim, config.dim), dtype=mx.float32)
        weights[f"{prefix}.attn.proj.bias"] = mx.zeros((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.ls1.gamma"] = mx.ones((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.norm2.weight"] = mx.ones((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.norm2.bias"] = mx.zeros((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.mlp.w12.weight"] = mx.zeros((2 * hidden, config.dim), dtype=mx.float32)
        weights[f"{prefix}.mlp.w12.bias"] = mx.zeros((2 * hidden,), dtype=mx.float32)
        weights[f"{prefix}.mlp.w3.weight"] = mx.zeros((config.dim, hidden), dtype=mx.float32)
        weights[f"{prefix}.mlp.w3.bias"] = mx.zeros((config.dim,), dtype=mx.float32)
        weights[f"{prefix}.ls2.gamma"] = mx.ones((config.dim,), dtype=mx.float32)

    if identity_odd_attention and config.depth > 1:
        prefix = "info_sharing.self_attention_blocks.1"
        identity = np.eye(config.dim, dtype=np.float32)
        weights[f"{prefix}.attn.qkv.weight"] = mx.array(np.concatenate((identity, identity, identity), axis=0))
        weights[f"{prefix}.attn.proj.weight"] = mx.array(identity)
    return weights


def _tiny_features_and_registers(
    config: MapAnythingInfoSharingConfig,
) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
    values = mx.arange(2 * config.dim * 2 * 2, dtype=mx.float32).reshape((2, config.dim, 2, 2)) / 100
    features = (values[0:1], values[1:2])
    registers = (
        mx.ones((1, config.dim, 1), dtype=mx.float32) * 0.1,
        mx.ones((1, config.dim, 1), dtype=mx.float32) * -0.1,
    )
    return features, registers


def _numpy_channel_layer_norm(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    transposed = np.transpose(values, (0, 2, 1))
    mean = transposed.mean(axis=-1, keepdims=True)
    centered = transposed - mean
    normalized = centered / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    return np.transpose(normalized, (0, 2, 1)).astype(np.float32)


def _tiny_config_json() -> str:
    return """{
  "encoder_config": {
    "data_norm_type": "dinov2",
    "name": "tiny-test",
    "size": "giant",
    "keep_first_n_layers": 1,
    "uses_torch_hub": false
  },
  "info_sharing_config": {
    "model_type": "alternating_attention",
    "model_return_type": "intermediate_features",
    "module_args": {
      "depth": 2,
      "dim": 8,
      "num_heads": 2,
      "indices": [0, 1]
    }
  },
  "pred_head_config": {
    "type": "dpt+pose",
    "adaptor_type": "raydirs+depth+pose+confidence+mask",
    "feature_head": {"patch_size": 2},
    "adaptor_config": {
      "dense_pred_init_dict": {"name": "raydirs+depth+pose+confidence+mask+scale"}
    }
  },
  "use_register_tokens_from_encoder": true
}"""
