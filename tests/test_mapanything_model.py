from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.mapanything_model import (
    MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS,
    MapAnythingEncoder,
    MapAnythingEncoderPrefix,
    MapAnythingEncoderPrefixConfig,
    interpolate_mapanything_dinov2_pos_embed,
    load_mapanything_full_encoder_weights,
    load_mapanything_encoder_prefix_weights,
    mapanything_full_encoder_outputs_for_parity,
    mapanything_full_encoder_required_keys,
    mapanything_encoder_prefix_config_from_model_config,
    mapanything_encoder_prefix_outputs_for_parity,
    mapanything_encoder_prefix_required_keys,
    run_mapanything_full_encoder,
    run_mapanything_encoder_prefix,
    validate_mapanything_full_encoder_weights,
    validate_mapanything_encoder_prefix_weights,
)
from mlx_spatial.mapanything_parity import (
    compare_mapanything_parity_tensors,
    load_mapanything_parity_bundle,
    mapanything_parity_report_to_dict,
)


ROOT = Path(__file__).resolve().parents[1]
TINY_REFERENCE = ROOT / "tests/fixtures/mapanything/encoder_prefix_tiny_reference.npz"


def test_mapanything_encoder_prefix_required_keys_are_explicit():
    keys = mapanything_encoder_prefix_required_keys()

    assert keys == MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS
    assert "encoder.model.patch_embed.proj.weight" in keys
    assert "encoder.model.blocks.0.attn.qkv.weight" in keys
    assert "encoder.model.blocks.0.mlp.w12.weight" in keys
    assert len(keys) == len(set(keys))


def test_mapanything_full_encoder_required_keys_cover_configured_layers():
    config = MapAnythingEncoderPrefixConfig(
        embed_dim=8,
        num_heads=2,
        patch_size=2,
        keep_first_n_layers=3,
    )

    keys = mapanything_full_encoder_required_keys(config)

    assert "encoder.model.patch_embed.proj.weight" in keys
    assert "encoder.model.blocks.0.attn.qkv.weight" in keys
    assert "encoder.model.blocks.2.mlp.w3.bias" in keys
    assert "encoder.model.blocks.3.norm1.weight" not in keys
    assert len(keys) == len(set(keys))


def test_mapanything_encoder_prefix_validates_missing_and_shape_errors():
    weights = _tiny_encoder_prefix_weights()
    weights.pop("encoder.model.blocks.0.attn.qkv.weight")

    with pytest.raises(ValueError, match="missing MapAnything encoder-prefix tensors"):
        validate_mapanything_encoder_prefix_weights(weights, _tiny_config())

    weights = _tiny_encoder_prefix_weights()
    weights["encoder.model.pos_embed"] = mx.zeros((1, 4, _tiny_config().embed_dim), dtype=mx.float32)
    with pytest.raises(ValueError, match="square patch-position grid"):
        validate_mapanything_encoder_prefix_weights(weights, _tiny_config())


def test_mapanything_full_encoder_validates_missing_later_block():
    config = _tiny_full_config()
    weights = _tiny_full_encoder_weights()
    weights.pop("encoder.model.blocks.1.attn.qkv.weight")

    with pytest.raises(ValueError, match="missing MapAnything full-encoder tensors"):
        validate_mapanything_full_encoder_weights(weights, config)


def test_mapanything_encoder_prefix_runs_tiny_boundary_shapes_and_trace():
    config = _tiny_config()
    images = mx.arange(1 * 3 * 4 * 4, dtype=mx.float32).reshape((1, 3, 4, 4)) / 100

    output = MapAnythingEncoderPrefix(_tiny_encoder_prefix_weights(), config)(images)
    mx.eval(output.patch_embeddings, output.tokens_with_position, output.block0)

    assert tuple(output.patch_embeddings.shape) == (1, 4, config.embed_dim)
    assert tuple(output.tokens_with_position.shape) == (1, 5, config.embed_dim)
    assert tuple(output.block0.shape) == (1, 5, config.embed_dim)
    assert output.patch_grid == (2, 2)
    assert str(output.block0.dtype).endswith("float32")
    assert output.trace["runtime_depends_on_torch"] is False
    assert output.trace["implemented_layers"] == ("patch_embed", "block0")


def test_mapanything_full_encoder_runs_tiny_two_layer_shapes_and_trace():
    config = _tiny_full_config()
    images = mx.arange(1 * 3 * 4 * 4, dtype=mx.float32).reshape((1, 3, 4, 4)) / 100

    output = MapAnythingEncoder(_tiny_full_encoder_weights(), config)(images)
    mx.eval(output.features, output.registers, output.block0, output.final_tokens)

    assert tuple(output.patch_embeddings.shape) == (1, 4, config.embed_dim)
    assert tuple(output.tokens_with_position.shape) == (1, 5, config.embed_dim)
    assert tuple(output.block0.shape) == (1, 5, config.embed_dim)
    assert tuple(output.final_tokens.shape) == (1, 5, config.embed_dim)
    assert tuple(output.features.shape) == (1, config.embed_dim, 2, 2)
    assert tuple(output.registers.shape) == (1, config.embed_dim, 1)
    assert output.patch_grid == (2, 2)
    assert output.trace["runtime_depends_on_torch"] is False
    assert output.trace["implemented_layers"] == (0, 1)
    assert set(mapanything_full_encoder_outputs_for_parity(output)) >= {
        "encoder.patch_embed",
        "encoder.block0",
        "encoder.features.0",
        "encoder.registers.0",
    }


def test_mapanything_encoder_prefix_matches_tiny_pytorch_reference_bundle():
    assert TINY_REFERENCE.is_file(), f"missing test fixture: {TINY_REFERENCE}"
    reference = load_mapanything_parity_bundle(TINY_REFERENCE)
    config = _tiny_config()
    images = mx.array(reference.tensors["input.img.0"], dtype=mx.float32)

    output = run_mapanything_encoder_prefix(images, _tiny_encoder_prefix_weights(), config=config)
    mx.eval(output.patch_embeddings, output.block0)
    report = compare_mapanything_parity_tensors(
        mapanything_encoder_prefix_outputs_for_parity(output),
        reference,
        names=("encoder.patch_embed", "encoder.block0"),
        atol=2e-5,
        rtol=2e-5,
    )

    assert reference.metadata["case"] == "tiny-pytorch-prefix"
    assert reference.metadata["runtime_depends_on_torch"] is False
    assert report.passed, mapanything_parity_report_to_dict(report)


def test_mapanything_dinov2_position_interpolation_matches_pytorch_recording():
    config = MapAnythingEncoderPrefixConfig(embed_dim=2, num_heads=1, patch_size=2)
    pos_embed = mx.array(np.arange(10, dtype=np.float32).reshape(1, 5, 2) / 10)

    interpolated = interpolate_mapanything_dinov2_pos_embed(
        pos_embed,
        token_count=9,
        image_height=4,
        image_width=8,
        config=config,
    )
    mx.eval(interpolated)

    expected = np.array(
        [
            [
                [0.0, 0.10000000149011612],
                [0.17193478345870972, 0.27193471789360046],
                [0.2347584217786789, 0.33475837111473083],
                [0.3415256440639496, 0.44152557849884033],
                [0.4127330183982849, 0.5127329230308533],
                [0.5558551549911499, 0.655855119228363],
                [0.6186788082122803, 0.7186787128448486],
                [0.7254461646080017, 0.8254461288452148],
                [0.7966534495353699, 0.8966533541679382],
            ]
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(interpolated), expected, atol=2e-5, rtol=2e-5)


def test_load_mapanything_encoder_prefix_weights_maps_official_keys(tmp_path):
    config = _tiny_config()
    model_root = tmp_path / "weights"
    model_root.mkdir()
    (model_root / "config.json").write_text(_tiny_config_json(), encoding="utf-8")
    save_file(_tiny_encoder_prefix_weights(), model_root / "model.safetensors")

    loaded = load_mapanything_encoder_prefix_weights(model_root, config=config)

    assert set(loaded) == set(MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS)
    assert tuple(loaded["encoder.model.patch_embed.proj.weight"].shape) == (8, 3, 2, 2)

    missing = dict(_tiny_encoder_prefix_weights())
    missing.pop("encoder.model.blocks.0.ls2.gamma")
    save_file(missing, model_root / "model.safetensors")
    with pytest.raises(ValueError, match="missing requested tensors"):
        load_mapanything_encoder_prefix_weights(model_root, config=config)


def test_load_mapanything_full_encoder_weights_maps_configured_layers(tmp_path):
    config = _tiny_full_config()
    model_root = tmp_path / "weights"
    model_root.mkdir()
    (model_root / "config.json").write_text(_tiny_config_json(), encoding="utf-8")
    save_file(_tiny_full_encoder_weights(), model_root / "model.safetensors")

    loaded = load_mapanything_full_encoder_weights(model_root, config=config)

    assert set(loaded) == set(mapanything_full_encoder_required_keys(config))
    assert tuple(loaded["encoder.model.blocks.1.attn.qkv.weight"].shape) == (
        3 * config.embed_dim,
        config.embed_dim,
    )


def test_mapanything_encoder_prefix_public_exports():
    assert mlx_spatial.MapAnythingEncoderPrefix is MapAnythingEncoderPrefix
    assert mlx_spatial.MapAnythingEncoder is MapAnythingEncoder
    assert mlx_spatial.load_mapanything_encoder_prefix_weights is load_mapanything_encoder_prefix_weights
    assert mlx_spatial.load_mapanything_full_encoder_weights is load_mapanything_full_encoder_weights
    assert (
        mlx_spatial.mapanything_encoder_prefix_config_from_model_config
        is mapanything_encoder_prefix_config_from_model_config
    )


def _tiny_config() -> MapAnythingEncoderPrefixConfig:
    return MapAnythingEncoderPrefixConfig(
        embed_dim=8,
        num_heads=2,
        patch_size=2,
        data_norm_type="dinov2",
        encoder_size="giant",
        keep_first_n_layers=1,
    )


def _tiny_full_config() -> MapAnythingEncoderPrefixConfig:
    return MapAnythingEncoderPrefixConfig(
        embed_dim=8,
        num_heads=2,
        patch_size=2,
        data_norm_type="dinov2",
        encoder_size="giant",
        keep_first_n_layers=2,
    )


def _tiny_encoder_prefix_weights() -> dict[str, mx.array]:
    config = _tiny_config()
    hidden = config.swiglu_hidden_features
    rng = np.random.default_rng(42)

    def randn(shape: tuple[int, ...], scale: float = 0.02, offset: float = 0.0) -> mx.array:
        return mx.array(rng.normal(loc=offset, scale=scale, size=shape).astype(np.float32))

    weights = {
        "encoder.model.cls_token": randn((1, 1, config.embed_dim), 0.03),
        "encoder.model.pos_embed": randn((1, 5, config.embed_dim), 0.02),
        "encoder.model.patch_embed.proj.weight": randn(
            (config.embed_dim, 3, config.patch_size, config.patch_size),
            0.04,
        ),
        "encoder.model.patch_embed.proj.bias": randn((config.embed_dim,), 0.01),
        "encoder.model.blocks.0.norm1.weight": randn((config.embed_dim,), 0.01, 1.0),
        "encoder.model.blocks.0.norm1.bias": randn((config.embed_dim,), 0.01),
        "encoder.model.blocks.0.attn.qkv.weight": randn((3 * config.embed_dim, config.embed_dim), 0.03),
        "encoder.model.blocks.0.attn.qkv.bias": randn((3 * config.embed_dim,), 0.01),
        "encoder.model.blocks.0.attn.proj.weight": randn((config.embed_dim, config.embed_dim), 0.03),
        "encoder.model.blocks.0.attn.proj.bias": randn((config.embed_dim,), 0.01),
        "encoder.model.blocks.0.ls1.gamma": randn((config.embed_dim,), 0.01, 0.1),
        "encoder.model.blocks.0.norm2.weight": randn((config.embed_dim,), 0.01, 1.0),
        "encoder.model.blocks.0.norm2.bias": randn((config.embed_dim,), 0.01),
        "encoder.model.blocks.0.mlp.w12.weight": randn((2 * hidden, config.embed_dim), 0.02),
        "encoder.model.blocks.0.mlp.w12.bias": randn((2 * hidden,), 0.01),
        "encoder.model.blocks.0.mlp.w3.weight": randn((config.embed_dim, hidden), 0.02),
        "encoder.model.blocks.0.mlp.w3.bias": randn((config.embed_dim,), 0.01),
        "encoder.model.blocks.0.ls2.gamma": randn((config.embed_dim,), 0.01, 0.1),
    }
    return weights


def _tiny_full_encoder_weights() -> dict[str, mx.array]:
    prefix = _tiny_encoder_prefix_weights()
    full = dict(prefix)
    for key, value in prefix.items():
        block1_key = key.replace("encoder.model.blocks.0.", "encoder.model.blocks.1.")
        if block1_key != key:
            full[block1_key] = value
    return full


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
      "depth": 1,
      "dim": 8,
      "num_heads": 2,
      "indices": [0]
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
  "use_register_tokens_from_encoder": false
}"""
