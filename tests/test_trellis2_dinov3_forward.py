import mlx.core as mx
import numpy as np
from safetensors.mlx import save_file

from mlx_spatial.trellis2_dinov3 import DinoV3ModelConfig
from mlx_spatial.trellis2_dinov3_forward import (
    assemble_dinov3_tokens,
    dinov3_full_forward_required_keys,
    dinov3_forward_required_keys,
    inspect_dinov3_forward_key_map,
    load_dinov3_forward_tensors,
    probe_dinov3_rope,
    run_dinov3_layer_stack,
    run_dinov3_transformer_block,
)


def _config(*, rope_theta=100.0, num_register_tokens=1, num_hidden_layers=1):
    return DinoV3ModelConfig(
        model_type="dinov3_vit",
        image_size=2,
        patch_size=1,
        hidden_size=8,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        intermediate_size=16,
        layer_norm_eps=1e-5,
        use_swiglu_ffn=False,
        num_register_tokens=num_register_tokens,
        expected_feature_width=8,
        rope_theta=rope_theta,
    )


def _checkpoint_tensors(config):
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    tensors = {
        "embeddings.cls_token": mx.zeros((1, 1, hidden), dtype=mx.float32),
        "embeddings.patch_embeddings.bias": mx.zeros((hidden,), dtype=mx.float32),
        "embeddings.patch_embeddings.weight": mx.ones((hidden, 3, 1, 1), dtype=mx.float32),
        "norm.bias": mx.zeros((hidden,), dtype=mx.float32),
        "norm.weight": mx.ones((hidden,), dtype=mx.float32),
    }
    for layer_index in range(config.num_hidden_layers):
        layer = f"layer.{layer_index}"
        tensors.update(
            {
                f"{layer}.attention.k_proj.weight": mx.ones((hidden, hidden), dtype=mx.float32),
                f"{layer}.attention.o_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{layer}.attention.o_proj.weight": mx.ones((hidden, hidden), dtype=mx.float32),
                f"{layer}.attention.q_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{layer}.attention.q_proj.weight": mx.ones((hidden, hidden), dtype=mx.float32),
                f"{layer}.attention.v_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{layer}.attention.v_proj.weight": mx.ones((hidden, hidden), dtype=mx.float32),
                f"{layer}.layer_scale1.lambda1": mx.ones((hidden,), dtype=mx.float32),
                f"{layer}.layer_scale2.lambda1": mx.ones((hidden,), dtype=mx.float32),
                f"{layer}.mlp.down_proj.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{layer}.mlp.down_proj.weight": mx.ones((hidden, intermediate), dtype=mx.float32),
                f"{layer}.mlp.up_proj.bias": mx.zeros((intermediate,), dtype=mx.float32),
                f"{layer}.mlp.up_proj.weight": mx.ones((intermediate, hidden), dtype=mx.float32),
                f"{layer}.norm1.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{layer}.norm1.weight": mx.ones((hidden,), dtype=mx.float32),
                f"{layer}.norm2.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{layer}.norm2.weight": mx.ones((hidden,), dtype=mx.float32),
            }
        )
    if config.num_register_tokens:
        tensors["embeddings.register_tokens"] = mx.ones(
            (1, config.num_register_tokens, hidden),
            dtype=mx.float32,
        )
    return tensors


def _write_checkpoint(path, config, *, omit=()):
    tensors = _checkpoint_tensors(config)
    for key in omit:
        tensors.pop(key)
    save_file(tensors, path)


def test_forward_key_map_uses_real_dinov3_checkpoint_names(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)

    result = inspect_dinov3_forward_key_map(checkpoint_path, config)

    assert result.blocker is None
    assert result.key_map is not None
    assert "layer.0.attention.q_proj.weight" in result.key_map.required_keys
    assert "layer.0.attention.query.weight" not in result.key_map.required_keys
    assert result.key_map.required_keys == dinov3_forward_required_keys(config)


def test_forward_key_map_reports_first_missing_required_key(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config, omit=("layer.0.attention.q_proj.weight",))

    result = inspect_dinov3_forward_key_map(checkpoint_path, config)

    assert result.key_map is None
    assert result.blocker is not None
    assert result.blocker.operation == "DINOv3 forward checkpoint key validation"
    assert "layer.0.attention.q_proj.weight" in result.blocker.reason


def test_forward_tensor_loader_loads_only_selected_probe_keys(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)

    result = load_dinov3_forward_tensors(checkpoint_path, config)

    assert result.ready
    assert result.tensors is not None
    assert tuple(result.tensors) == tuple(sorted(dinov3_forward_required_keys(config)))


def test_full_forward_tensor_loader_loads_all_configured_layers(tmp_path):
    config = _config(num_hidden_layers=2)
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)

    result = load_dinov3_forward_tensors(checkpoint_path, config, all_layers=True)

    assert result.ready
    assert result.tensors is not None
    assert tuple(result.tensors) == tuple(sorted(dinov3_full_forward_required_keys(config)))
    assert "layer.1.attention.q_proj.weight" in result.tensors


def test_patch_embedding_assembles_cls_register_and_patch_tokens(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)
    loaded = load_dinov3_forward_tensors(checkpoint_path, config)
    image_tensor = mx.ones((1, 3, 2, 2), dtype=mx.float32)

    result = assemble_dinov3_tokens(image_tensor, config, loaded.tensors)

    assert result.ready
    assert result.patch_grid == (2, 2)
    assert result.tokens is not None
    assert tuple(result.tokens.shape) == (1, 6, 8)
    np.testing.assert_allclose(np.array(result.tokens)[0, 0], np.zeros((8,)))
    np.testing.assert_allclose(np.array(result.tokens)[0, 1], np.ones((8,)))
    np.testing.assert_allclose(np.array(result.tokens)[0, 2], np.full((8,), 3.0))


def test_patch_embedding_blocks_on_non_bchw_image_tensor(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)
    loaded = load_dinov3_forward_tensors(checkpoint_path, config)

    result = assemble_dinov3_tokens(mx.ones((1, 2, 2, 3), dtype=mx.float32), config, loaded.tensors)

    assert result.tokens is None
    assert result.blocker is not None
    assert result.blocker.operation == "DINOv3 image tensor validation"
    assert "RGB channel count 3" in result.blocker.reason


def test_rope_probe_reports_geometry_from_runtime_image_size():
    result = probe_dinov3_rope(_config(), runtime_image_size=4)

    assert result.ready
    assert result.patch_grid == (4, 4)
    assert result.patch_token_count == 16
    assert result.head_dim == 4
    assert result.theta == 100.0


def test_rope_probe_blocks_when_theta_is_missing():
    result = probe_dinov3_rope(_config(rope_theta=None))

    assert result.blocker is not None
    assert result.blocker.operation == "DINOv3 RoPE parameter validation"
    assert "rope_theta" in result.blocker.reason


def test_single_transformer_block_returns_hidden_states_with_same_shape(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)
    loaded = load_dinov3_forward_tensors(checkpoint_path, config)
    tokens = assemble_dinov3_tokens(mx.ones((1, 3, 2, 2), dtype=mx.float32), config, loaded.tensors)

    result = run_dinov3_transformer_block(
        tokens.tokens,
        config,
        loaded.tensors,
        patch_grid=tokens.patch_grid,
    )

    assert result.ready
    assert result.hidden_states is not None
    assert tuple(result.hidden_states.shape) == (1, 6, 8)
    assert str(result.hidden_states.dtype).removeprefix("mlx.core.") == "float32"


def test_single_transformer_block_reports_missing_block_tensor(tmp_path):
    config = _config()
    checkpoint_path = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint_path, config)
    loaded = load_dinov3_forward_tensors(checkpoint_path, config)
    tokens = assemble_dinov3_tokens(mx.ones((1, 3, 2, 2), dtype=mx.float32), config, loaded.tensors)
    tensors = dict(loaded.tensors)
    tensors.pop("layer.0.mlp.down_proj.weight")

    result = run_dinov3_transformer_block(
        tokens.tokens,
        config,
        tensors,
        patch_grid=tokens.patch_grid,
    )

    assert result.hidden_states is None
    assert result.blocker is not None
    assert result.blocker.operation == "DINOv3 transformer block tensor lookup"
    assert "layer.0.mlp.down_proj.weight" in result.blocker.reason


def test_layer_stack_runs_all_layers_and_vendor_final_layer_norm(tmp_path):
    config = _config(num_hidden_layers=2)
    checkpoint_path = tmp_path / "model.safetensors"
    tensors = _checkpoint_tensors(config)
    tensors["norm.bias"] = mx.arange(config.hidden_size, dtype=mx.float32)
    tensors["norm.weight"] = mx.full((config.hidden_size,), 3.0, dtype=mx.float32)
    save_file(tensors, checkpoint_path)
    loaded = load_dinov3_forward_tensors(checkpoint_path, config, all_layers=True)
    tokens = assemble_dinov3_tokens(mx.ones((1, 3, 2, 2), dtype=mx.float32), config, loaded.tensors)

    result = run_dinov3_layer_stack(
        tokens.tokens,
        config,
        loaded.tensors,
        patch_grid=tokens.patch_grid,
    )

    assert result.ready
    assert result.completed_layers == 2
    assert result.hidden_states is not None
    assert tuple(result.hidden_states.shape) == (1, 6, 8)
    assert "norm.bias" not in loaded.tensors
    assert "norm.weight" not in loaded.tensors
    np.testing.assert_allclose(np.array(result.hidden_states).mean(axis=-1), 0.0, atol=1e-5)
    assert not np.allclose(np.array(result.hidden_states)[0, 0], np.arange(config.hidden_size, dtype=np.float32))
