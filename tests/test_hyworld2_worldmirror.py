import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.hyworld2_worldmirror import (
    VisualGeometryTransformerConfig,
    assemble_worldmirror_tokens,
    default_visual_geometry_tensors,
    exact_full_attention,
    run_visual_geometry_transformer,
)


def _config(**overrides):
    values = {
        "img_size": 2,
        "patch_size": 1,
        "embed_dim": 8,
        "depth": 3,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "num_register_tokens": 2,
        "max_tokens": 64,
        "max_attention_bytes": 1_000_000,
        "intermediate_layers": (0, 2),
    }
    values.update(overrides)
    return VisualGeometryTransformerConfig(**values)


def test_token_assembly_models_camera_register_and_patch_contract():
    config = _config(depth=1)
    image = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)

    result = assemble_worldmirror_tokens(image, config)

    assert result.ready
    assert config.enable_cond is True
    assert result.patch_start_idx == 1 + config.num_register_tokens + 2
    assert result.patch_grid == (2, 2)
    assert result.frame_token_count == 9
    assert result.tokens is not None
    assert result.patch_tokens is not None
    assert result.rope_positions is not None
    assert tuple(result.tokens.shape) == (1, 18, 8)
    assert tuple(result.patch_tokens.shape) == (1, 2, 4, 8)
    assert tuple(result.rope_positions.shape) == (2, 9, 2)
    np.testing.assert_allclose(np.array(result.tokens)[0, 0], np.zeros((8,)))
    np.testing.assert_allclose(np.array(result.tokens)[0, 1], np.zeros((8,)))
    np.testing.assert_allclose(np.array(result.tokens)[0, 3], np.zeros((8,)))
    np.testing.assert_allclose(np.array(result.tokens)[0, 5], np.full((8,), 3.0))
    np.testing.assert_allclose(np.array(result.rope_positions)[0, :5], np.zeros((5, 2)))
    np.testing.assert_allclose(
        np.array(result.rope_positions)[0, 5:],
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )


def test_official_scale_token_assembly_blocks_fixture_creation_before_allocation(monkeypatch):
    def fail_fixture(_config):
        raise AssertionError("default fixture tensors should not be allocated")

    monkeypatch.setattr(
        "mlx_spatial.hyworld2_worldmirror.default_visual_geometry_tensors",
        fail_fixture,
    )
    image = mx.ones((1, 1, 3, 518, 518), dtype=mx.float32)

    result = assemble_worldmirror_tokens(image, VisualGeometryTransformerConfig())

    assert result.tokens is None
    assert result.blocker is not None
    assert result.blocker.stage == "model-construction"
    assert result.blocker.operation == "HY-World deterministic fixture tensor allocation"
    assert (
        result.blocker.metadata["estimated_fixture_bytes"]
        > result.blocker.metadata["max_fixture_bytes"]
    )


def test_official_scale_transformer_blocks_fixture_creation_before_allocation(monkeypatch):
    def fail_fixture(_config):
        raise AssertionError("default fixture tensors should not be allocated")

    monkeypatch.setattr(
        "mlx_spatial.hyworld2_worldmirror.default_visual_geometry_tensors",
        fail_fixture,
    )
    image = mx.ones((1, 1, 3, 518, 518), dtype=mx.float32)

    result = run_visual_geometry_transformer(image, VisualGeometryTransformerConfig())

    assert result.tokens is None
    assert result.blocker is not None
    assert result.blocker.stage == "model-construction"
    assert result.blocker.operation == "HY-World deterministic fixture tensor allocation"
    assert (
        result.blocker.metadata["estimated_fixture_bytes"]
        > result.blocker.metadata["max_fixture_bytes"]
    )


def test_official_scale_default_fixture_builder_rejects_direct_oversized_request():
    with pytest.raises(ValueError, match="deterministic fixture tensors would exceed"):
        default_visual_geometry_tensors(VisualGeometryTransformerConfig())


def test_token_assembly_can_disable_condition_slots():
    config = _config(depth=1, enable_cond=False)
    image = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)

    result = assemble_worldmirror_tokens(image, config)

    assert result.ready
    assert result.patch_start_idx == 1 + config.num_register_tokens
    assert result.frame_token_count == 7
    assert result.tokens is not None
    assert tuple(result.tokens.shape) == (1, 14, 8)
    np.testing.assert_allclose(np.array(result.tokens)[0, 3], np.full((8,), 3.0))


def test_token_assembly_accepts_official_dino_patch_embed_keys():
    config = _config(depth=0, enable_cond=False)
    image = mx.ones((1, 1, 3, 2, 2), dtype=mx.float32)
    tensors = {
        "patch_embed.patch_embed.proj.weight": mx.ones((8, 3, 1, 1), dtype=mx.float32),
        "patch_embed.patch_embed.proj.bias": mx.zeros((8,), dtype=mx.float32),
        "patch_embed.cls_token": mx.zeros((1, 1, 8), dtype=mx.float32),
        "patch_embed.pos_embed": mx.zeros((1, 5, 8), dtype=mx.float32),
        "patch_embed.register_tokens": mx.zeros((1, 2, 8), dtype=mx.float32),
        "patch_embed.norm.weight": mx.ones((8,), dtype=mx.float32),
        "patch_embed.norm.bias": mx.zeros((8,), dtype=mx.float32),
        "cam_token": mx.zeros((1, 2, 1, 8), dtype=mx.float32),
        "reg_token": mx.zeros((1, 2, 2, 8), dtype=mx.float32),
    }

    result = assemble_worldmirror_tokens(image, config, tensors)

    assert result.ready
    assert result.patch_start_idx == 3
    assert result.frame_token_count == 7
    assert result.patch_tokens is not None
    assert tuple(result.patch_tokens.shape) == (1, 1, 4, 8)


def test_official_dino_positional_interpolation_handles_non_checkpoint_grid():
    config = _config(depth=0, enable_cond=False)
    image = mx.ones((1, 1, 3, 2, 2), dtype=mx.float32)
    tensors = {
        "patch_embed.patch_embed.proj.weight": mx.ones((8, 3, 1, 1), dtype=mx.float32),
        "patch_embed.patch_embed.proj.bias": mx.zeros((8,), dtype=mx.float32),
        "patch_embed.cls_token": mx.zeros((1, 1, 8), dtype=mx.float32),
        "patch_embed.pos_embed": mx.zeros((1, 10, 8), dtype=mx.float32),
        "patch_embed.register_tokens": mx.zeros((1, 2, 8), dtype=mx.float32),
        "patch_embed.norm.weight": mx.ones((8,), dtype=mx.float32),
        "patch_embed.norm.bias": mx.zeros((8,), dtype=mx.float32),
        "cam_token": mx.zeros((1, 2, 1, 8), dtype=mx.float32),
        "reg_token": mx.zeros((1, 2, 2, 8), dtype=mx.float32),
    }

    result = assemble_worldmirror_tokens(image, config, tensors)

    assert result.ready
    assert result.blocker is None
    assert result.patch_tokens is not None
    assert tuple(result.patch_tokens.shape) == (1, 1, 4, 8)


def test_first_and_remaining_frames_use_distinct_special_token_slots():
    config = _config(depth=1)
    image = mx.ones((1, 3, 3, 2, 2), dtype=mx.float32)
    tensors = default_visual_geometry_tensors(config)
    tensors["cam_token"] = mx.array(
        np.array([[[[10.0] * 8], [[20.0] * 8]]], dtype=np.float32)
    )
    tensors["reg_token"] = mx.array(
        np.array(
            [
                [
                    [[30.0] * 8, [31.0] * 8],
                    [[40.0] * 8, [41.0] * 8],
                ]
            ],
            dtype=np.float32,
        )
    )

    result = assemble_worldmirror_tokens(image, config, tensors)

    assert result.ready
    assert result.tokens is not None
    assert result.frame_token_count == 9
    tokens = np.array(result.tokens)[0]
    np.testing.assert_allclose(tokens[0], np.full((8,), 10.0))
    np.testing.assert_allclose(tokens[1], np.full((8,), 30.0))
    np.testing.assert_allclose(tokens[2], np.full((8,), 31.0))
    np.testing.assert_allclose(tokens[9], np.full((8,), 20.0))
    np.testing.assert_allclose(tokens[10], np.full((8,), 40.0))
    np.testing.assert_allclose(tokens[11], np.full((8,), 41.0))
    np.testing.assert_allclose(tokens[18], np.full((8,), 20.0))
    np.testing.assert_allclose(tokens[19], np.full((8,), 40.0))
    np.testing.assert_allclose(tokens[20], np.full((8,), 41.0))


def test_default_tensors_use_official_frame_and_global_qkv_namespace():
    config = _config(depth=1)

    tensors = default_visual_geometry_tensors(config)

    assert "frame_blocks.0.attn.qkv.weight" in tensors
    assert "global_blocks.0.attn.qkv.weight" in tensors
    assert tuple(tensors["frame_blocks.0.attn.qkv.weight"].shape) == (24, 8)
    assert tuple(tensors["global_blocks.0.attn.qkv.bias"].shape) == (24,)
    assert "layers.0.attn.q.weight" not in tensors


def test_visual_geometry_transformer_captures_intermediate_patch_tokens():
    config = _config()
    image = mx.arange(1 * 2 * 3 * 2 * 2, dtype=mx.float32)
    image = mx.reshape(image, (1, 2, 3, 2, 2))

    result = run_visual_geometry_transformer(image, config)

    assert result.ready
    assert result.patch_start_idx == 5
    assert result.patch_grid == (2, 2)
    assert result.frame_token_count == 9
    assert result.attention_modes == ("frame", "global", "frame", "global", "frame", "global")
    assert result.tokens is not None
    assert tuple(result.tokens.shape) == (1, 18, 8)
    assert len(result.intermediate_tokens) == 2
    assert tuple(result.intermediate_tokens[0].shape) == (1, 2, 4, 16)
    assert tuple(result.intermediate_tokens[1].shape) == (1, 2, 4, 16)


def test_visual_geometry_transformer_captures_full_intermediate_tokens_for_camera_head():
    config = _config()
    image = mx.arange(1 * 2 * 3 * 2 * 2, dtype=mx.float32)
    image = mx.reshape(image, (1, 2, 3, 2, 2))

    result = run_visual_geometry_transformer(image, config)

    assert result.ready
    assert result.frame_token_count == 9
    assert len(result.intermediate_full_tokens) == 2
    assert tuple(result.intermediate_full_tokens[0].shape) == (1, 2, 9, 16)
    assert tuple(result.intermediate_full_tokens[1].shape) == (1, 2, 9, 16)
    np.testing.assert_allclose(
        np.array(result.intermediate_tokens[0]),
        np.array(result.intermediate_full_tokens[0])[:, :, result.patch_start_idx :, :],
    )


def test_rope_changes_attention_output_against_no_position_attention():
    image = mx.arange(1 * 2 * 3 * 2 * 2, dtype=mx.float32)
    image = mx.reshape(image, (1, 2, 3, 2, 2))
    tensors = default_visual_geometry_tensors(_config(depth=1))
    tensors["patch_embed.weight"] = mx.reshape(
        mx.arange(8 * 3, dtype=mx.float32) / 10.0,
        (8, 3, 1, 1),
    )

    with_rope = run_visual_geometry_transformer(image, _config(depth=1, enable_rope=True), tensors)
    without_rope = run_visual_geometry_transformer(
        image,
        _config(depth=1, enable_rope=False),
        tensors,
    )

    assert with_rope.ready
    assert without_rope.ready
    assert with_rope.tokens is not None
    assert without_rope.tokens is not None
    assert not np.allclose(np.array(with_rope.tokens), np.array(without_rope.tokens))


def test_rope_blocks_when_head_dim_cannot_split_into_2d_pairs():
    config = _config(depth=1, embed_dim=6, num_heads=2)
    image = mx.ones((1, 1, 3, 2, 2), dtype=mx.float32)

    result = run_visual_geometry_transformer(image, config)

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World 2D RoPE head dimension validation"
    assert result.blocker.metadata["head_dim"] == 3


def test_query_chunked_full_attention_matches_dense_attention():
    query = mx.reshape(mx.arange(1 * 2 * 5 * 4, dtype=mx.float32), (1, 2, 5, 4)) / 10.0
    key = mx.reshape(mx.arange(1 * 2 * 7 * 4, dtype=mx.float32), (1, 2, 7, 4)) / 11.0
    value = mx.reshape(mx.arange(1 * 2 * 7 * 4, dtype=mx.float32), (1, 2, 7, 4)) / 13.0

    dense = exact_full_attention(query, key, value, max_attention_bytes=1_000_000)
    chunked = exact_full_attention(
        query,
        key,
        value,
        max_attention_bytes=1_000_000,
        query_chunk_size=2,
    )

    assert dense.ready
    assert chunked.ready
    assert dense.hidden_states is not None
    assert chunked.hidden_states is not None
    np.testing.assert_allclose(
        np.array(chunked.hidden_states),
        np.array(dense.hidden_states),
        rtol=1e-3,
        atol=5e-3,
    )


def test_token_guard_blocks_before_patch_embedding_allocation():
    config = _config(max_tokens=13)
    image = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)

    result = assemble_worldmirror_tokens(image, config)

    assert result.tokens is None
    assert result.blocker is not None
    assert result.blocker.stage == "visual-transformer"
    assert result.blocker.operation == "assemble HY-World camera/register/patch tokens"
    assert result.blocker.metadata["token_count"] == 18
    assert result.blocker.metadata["max_tokens"] == 13


def test_activation_guard_blocks_before_unsafe_global_attention_allocation():
    config = _config(depth=2, max_attention_bytes=2_000)
    image = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)

    result = run_visual_geometry_transformer(image, config, default_visual_geometry_tensors(config))

    assert result.tokens is None
    assert result.blocker is not None
    assert result.blocker.stage == "visual-transformer"
    assert result.blocker.operation == "HY-World exact global full attention"
    assert result.blocker.metadata["estimated_attention_bytes"] == 1 * 2 * 18 * 18 * 4
    assert result.blocker.metadata["exact_attention"] is True
    assert result.attention_modes == ("frame", "global")


def test_attention_guard_respects_query_chunked_exact_full_attention():
    config = _config(depth=2, max_attention_bytes=144, query_chunk_size=1)
    image = mx.ones((1, 2, 3, 2, 2), dtype=mx.float32)

    result = run_visual_geometry_transformer(image, config)

    assert result.ready
    assert result.attention_modes == ("frame", "global", "frame", "global")
    assert result.tokens is not None
    assert tuple(result.tokens.shape) == (1, 18, 8)


def test_worldmirror_blocks_on_non_official_image_tensor_shape():
    result = assemble_worldmirror_tokens(mx.ones((1, 3, 2, 2), dtype=mx.float32), _config())

    assert result.blocker is not None
    assert result.blocker.operation == "HY-World image tensor validation"
    assert "[B,S,3,H,W]" in result.blocker.reason
