import json

import mlx.core as mx
import pytest
from safetensors.mlx import save_file

from mlx_spatial.trellis2_slat import (
    SLAT_BLOCK0_INSPECTION_NAMES,
    SLAT_FULL_SELF_ATTN_TOKEN_LIMIT,
    SLAT_INPUT_TENSOR_NAMES,
    SLAT_WINDOWED_SELF_ATTN_THRESHOLD,
    SLatFlowConfig,
    TextureSLatRoute,
    _attention,
    _slat_full_self_attention_chunked,
    _slat_self_attention_kernel,
    _slat_window_groups,
    probe_shape_slat_forward_boundary,
    probe_texture_slat_forward_boundary,
    read_slat_flow_config,
    select_shape_slat_route,
    select_texture_slat_route,
)


def _small_slat_config(*, in_channels=32):
    return SLatFlowConfig(
        name="SLatFlowModel",
        resolution=32,
        in_channels=in_channels,
        out_channels=32,
        model_channels=6,
        cond_channels=1024,
        num_blocks=1,
        num_heads=1,
        mlp_ratio=2.0,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="bfloat16",
    )


def _write_slat_config(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "SLatFlowModel",
                "args": {
                    "resolution": 32,
                    "in_channels": 32,
                    "out_channels": 32,
                    "model_channels": 6,
                    "cond_channels": 1024,
                    "num_blocks": 1,
                    "num_heads": 1,
                    "mlp_ratio": 2.0,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "dtype": "bfloat16",
                },
            }
        )
    )


def _write_slat_checkpoint(path, config=None, *, omit=(), bad_input_shape=False):
    config = config or _small_slat_config()
    tensors = {
        "input_layer.weight": mx.ones(
            (config.model_channels, config.in_channels)
            if not bad_input_shape
            else (config.model_channels + 1, config.in_channels),
            dtype=mx.float32,
        ),
        "input_layer.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "out_layer.weight": mx.zeros((config.out_channels, config.model_channels), dtype=mx.float32),
        "out_layer.bias": mx.zeros((config.out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": mx.zeros((config.model_channels, 256), dtype=mx.float32),
        "t_embedder.mlp.0.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "t_embedder.mlp.2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": mx.zeros((config.model_channels * 6, config.model_channels), dtype=mx.float32),
        "adaLN_modulation.1.bias": mx.zeros((config.model_channels * 6,), dtype=mx.float32),
        "blocks.0.modulation": mx.zeros((config.model_channels * 6,), dtype=mx.float32),
        "blocks.0.norm2.weight": mx.ones((config.model_channels,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.weight": mx.zeros((config.model_channels * 3, config.model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.bias": mx.zeros((config.model_channels * 3,), dtype=mx.float32),
        "blocks.0.self_attn.q_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.self_attn.k_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.self_attn.to_out.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_out.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.weight": mx.zeros((config.model_channels * 2, config.cond_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.bias": mx.zeros((config.model_channels * 2,), dtype=mx.float32),
        "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.cross_attn.to_out.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.weight": mx.zeros((int(config.model_channels * config.mlp_ratio), config.model_channels), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.bias": mx.zeros((int(config.model_channels * config.mlp_ratio),), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.weight": mx.zeros((config.model_channels, int(config.model_channels * config.mlp_ratio)), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
    }
    for name in omit:
        tensors.pop(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def test_read_slat_flow_config_maps_shape_flow_fields(tmp_path):
    path = tmp_path / "slat.json"
    _write_slat_config(path)

    config = read_slat_flow_config(tmp_path, "slat.json")

    assert config.name == "SLatFlowModel"
    assert config.resolution == 32
    assert config.in_channels == 32
    assert config.out_channels == 32
    assert config.model_channels == 6
    assert config.cond_channels == 1024
    assert config.pe_mode == "rope"
    assert config.share_mod is True


@pytest.mark.parametrize(
    ("pipeline_type", "keys", "resolution", "cascade"),
    [
        ("512", ("shape_slat_flow_model_512",), 512, False),
        ("1024", ("shape_slat_flow_model_1024",), 1024, False),
        ("1024_cascade", ("shape_slat_flow_model_512", "shape_slat_flow_model_1024"), 1024, True),
        ("1536_cascade", ("shape_slat_flow_model_512", "shape_slat_flow_model_1024"), 1536, True),
    ],
)
def test_select_shape_slat_route_maps_pipeline_types(pipeline_type, keys, resolution, cascade):
    route = select_shape_slat_route(pipeline_type)

    assert route.pipeline_type == pipeline_type
    assert route.model_keys == keys
    assert route.output_resolution == resolution
    assert route.cascade is cascade


def test_select_shape_slat_route_rejects_unknown_pipeline_type():
    with pytest.raises(ValueError, match="unsupported shape SLat pipeline type"):
        select_shape_slat_route("2048")


@pytest.mark.parametrize(
    ("pipeline_type", "key", "resolution"),
    [
        ("512", "tex_slat_flow_model_512", 512),
        ("1024", "tex_slat_flow_model_1024", 1024),
        ("1024_cascade", "tex_slat_flow_model_1024", 1024),
        ("1536_cascade", "tex_slat_flow_model_1024", 1024),
    ],
)
def test_select_texture_slat_route_maps_pipeline_types(pipeline_type, key, resolution):
    route = select_texture_slat_route(pipeline_type)

    assert isinstance(route, TextureSLatRoute)
    assert route.pipeline_type == pipeline_type
    assert route.model_key == key
    assert route.output_resolution == resolution


def test_select_texture_slat_route_rejects_unknown_pipeline_type():
    with pytest.raises(ValueError, match="unsupported texture SLat pipeline type"):
        select_texture_slat_route("2048")


def test_shape_slat_probe_runs_sparse_feature_projection_and_blocks_at_transformer(tmp_path):
    config = _small_slat_config()
    checkpoint = tmp_path / "slat.safetensors"
    _write_slat_checkpoint(checkpoint, config)
    coordinates = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)

    probe = probe_shape_slat_forward_boundary(checkpoint, config, coordinates)

    assert probe.coordinate_shape == (2, 4)
    assert probe.feature_shape == (2, 32)
    assert probe.input_projection_shape == (2, 6)
    assert probe.block0_output_shape == (2, 6)
    assert probe.output_projection_shape == (2, 32)
    assert probe.sampled_feature_shape == (2, 32)
    assert set(SLAT_INPUT_TENSOR_NAMES).issubset(set(probe.loaded_tensor_names))
    assert set(SLAT_BLOCK0_INSPECTION_NAMES).issubset(set(probe.inspected_tensor_names))
    assert probe.blocker_operation == "shape SLat texture handoff"
    assert "FlowEuler sampler executed" in probe.blocker_detail


def test_shape_slat_sampling_respects_mlx_seed(tmp_path):
    config = _small_slat_config()
    checkpoint = tmp_path / "slat.safetensors"
    _write_slat_checkpoint(checkpoint, config)
    coordinates = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)

    mx.random.seed(7)
    first = probe_shape_slat_forward_boundary(checkpoint, config, coordinates).sampled_features
    mx.random.seed(7)
    second = probe_shape_slat_forward_boundary(checkpoint, config, coordinates).sampled_features

    assert first is not None
    assert second is not None
    assert mx.allclose(first, second)


def test_shape_slat_probe_reports_bad_sparse_coordinate_shape(tmp_path):
    config = _small_slat_config()
    checkpoint = tmp_path / "slat.safetensors"
    _write_slat_checkpoint(checkpoint, config)
    coordinates = mx.array([[0, 0, 0]], dtype=mx.int32)

    with pytest.raises(ValueError, match="sparse coordinates must have shape"):
        probe_shape_slat_forward_boundary(checkpoint, config, coordinates)


def test_shape_slat_probe_reports_missing_checkpoint_tensor(tmp_path):
    config = _small_slat_config()
    checkpoint = tmp_path / "slat.safetensors"
    _write_slat_checkpoint(checkpoint, config, omit=("out_layer.weight",))
    coordinates = mx.array([[0, 0, 0, 0]], dtype=mx.int32)

    with pytest.raises(ValueError, match="out_layer.weight"):
        probe_shape_slat_forward_boundary(checkpoint, config, coordinates)


def test_shape_slat_probe_reports_input_projection_shape_mismatch(tmp_path):
    config = _small_slat_config()
    checkpoint = tmp_path / "slat.safetensors"
    _write_slat_checkpoint(checkpoint, config, bad_input_shape=True)
    coordinates = mx.array([[0, 0, 0, 0]], dtype=mx.int32)

    with pytest.raises(ValueError, match="input_layer.weight shape mismatch"):
        probe_shape_slat_forward_boundary(checkpoint, config, coordinates)


def test_texture_slat_probe_runs_concat_feature_projection_and_blocks_at_transformer(tmp_path):
    config = _small_slat_config(in_channels=64)
    checkpoint = tmp_path / "tex.safetensors"
    _write_slat_checkpoint(checkpoint, config)
    coordinates = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    features = mx.zeros((2, 32), dtype=mx.float32)

    probe = probe_texture_slat_forward_boundary(checkpoint, config, coordinates, features)

    assert probe.coordinate_shape == (2, 4)
    assert probe.shape_feature_shape == (2, 32)
    assert probe.noise_feature_shape == (2, 32)
    assert probe.concat_feature_shape == (2, 64)
    assert probe.input_projection_shape == (2, 6)
    assert probe.block0_output_shape == (2, 6)
    assert probe.output_projection_shape == (2, 32)
    assert probe.sampled_feature_shape == (2, 32)
    assert set(SLAT_INPUT_TENSOR_NAMES).issubset(set(probe.loaded_tensor_names))
    assert set(SLAT_BLOCK0_INSPECTION_NAMES).issubset(set(probe.inspected_tensor_names))
    assert probe.blocker_operation == "texture SLat decode handoff"
    assert "FlowEuler sampler executed" in probe.blocker_detail


def test_texture_slat_probe_reports_bad_shape_feature_width(tmp_path):
    config = _small_slat_config(in_channels=64)
    checkpoint = tmp_path / "tex.safetensors"
    _write_slat_checkpoint(checkpoint, config)
    coordinates = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    features = mx.zeros((1, 64), dtype=mx.float32)

    with pytest.raises(ValueError, match="shape SLat feature width"):
        probe_texture_slat_forward_boundary(checkpoint, config, coordinates, features)


def test_slat_window_groups_partition_large_sparse_coordinates():
    coordinates = mx.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 8, 0, 0],
            [0, 9, 1, 1],
            [0, 0, 8, 0],
        ],
        dtype=mx.int32,
    )

    groups = _slat_window_groups(coordinates, window_size=8)

    assert groups == ((0, 1), (4,), (2, 3))
    assert SLAT_WINDOWED_SELF_ATTN_THRESHOLD == 4096


def test_chunked_full_slat_attention_matches_dense_attention():
    query = mx.array(
        [[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]]],
        dtype=mx.float32,
    )
    key = query + 0.1
    value = query - 0.1

    dense = _attention(query, key, value, head_dim=2)
    chunked = _slat_full_self_attention_chunked(query, key, value, head_dim=2, query_chunk_size=1)

    assert mx.allclose(chunked, dense, atol=1e-3)


def test_large_slat_attention_uses_exact_chunked_path(monkeypatch):
    calls = {"chunked": 0}

    def fake_chunked(query, key, value, *, head_dim, query_chunk_size=512):
        calls["chunked"] += 1
        return mx.zeros_like(query)

    monkeypatch.setattr("mlx_spatial.trellis2_slat._slat_full_self_attention_chunked", fake_chunked)
    token_count = SLAT_WINDOWED_SELF_ATTN_THRESHOLD + 1
    query = mx.zeros((1, token_count, 1, 2), dtype=mx.float32)
    coordinates = mx.concatenate(
        [
            mx.zeros((token_count, 1), dtype=mx.int32),
            mx.arange(token_count, dtype=mx.int32)[:, None],
            mx.zeros((token_count, 2), dtype=mx.int32),
        ],
        axis=1,
    )

    output = _slat_self_attention_kernel(query, query, query, coordinates, head_dim=2)

    assert output.shape == query.shape
    assert calls["chunked"] == 1


def test_slat_attention_blocks_above_exact_token_guard():
    token_count = SLAT_FULL_SELF_ATTN_TOKEN_LIMIT + 1
    query = mx.zeros((1, token_count, 1, 2), dtype=mx.float32)
    coordinates = mx.zeros((token_count, 4), dtype=mx.int32)

    with pytest.raises(ValueError, match="exact full SLat self-attention"):
        _slat_self_attention_kernel(query, query, query, coordinates, head_dim=2)
