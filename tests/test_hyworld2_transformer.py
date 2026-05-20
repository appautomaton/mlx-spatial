"""Tests for HY-World 2.0 transformer and ViT modules (Slice 2).

Gap IDs: HW-03, HW-05, HW-09.
Covers: hyworld2_transformer (run_vgt_block, run_dino_block)
         hyworld2_vit (run_dino_vit, interpolate_dino_pos_embed)
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.hyworld2_transformer import (
    Block,
    DistBlock,
    NestedTensorBlock,
    run_dino_block,
    run_vgt_block,
)
from mlx_spatial.hyworld2_vit import (
    DinoVisionTransformer,
    interpolate_dino_pos_embed,
    run_dino_vit,
)
from mlx_spatial.hyworld2_worldmirror import (
    _official_dino_patch_tokens,
    _run_dino_transformer_block,
    _run_transformer_block,
)


def _simple_config(**overrides):
    """Minimal config-like object for testing."""
    defaults = dict(
        img_size=56,
        patch_size=14,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        num_register_tokens=1,
        enable_cond=True,
        rope_base=100.0,
        normalized_rope=True,
        enable_rope=True,
        layer_norm_eps=1e-5,
        max_tokens=8192,
        max_attention_bytes=4_000_000_000,
        query_chunk_size=None,
    )
    defaults.update(overrides)

    class _Config:
        pass

    cfg = _Config()
    for k, v in defaults.items():
        setattr(cfg, k, v)
    cfg.head_dim = cfg.embed_dim // cfg.num_heads
    return cfg


def _make_block_tensors(prefix: str, embed_dim: int, intermediate: int):
    """Create minimal required tensors for a transformer block."""
    eye = mx.eye(embed_dim, dtype=mx.float32)
    qkv_weight = mx.concatenate((eye, eye, eye), axis=0)
    qkv_bias = mx.zeros((embed_dim * 3,), dtype=mx.float32)
    return {
        f"{prefix}.norm1.weight": mx.ones((embed_dim,), dtype=mx.float32),
        f"{prefix}.norm1.bias": mx.zeros((embed_dim,), dtype=mx.float32),
        f"{prefix}.norm2.weight": mx.ones((embed_dim,), dtype=mx.float32),
        f"{prefix}.norm2.bias": mx.zeros((embed_dim,), dtype=mx.float32),
        f"{prefix}.attn.qkv.weight": qkv_weight,
        f"{prefix}.attn.qkv.bias": qkv_bias,
        f"{prefix}.attn.out.weight": eye,
        f"{prefix}.attn.out.bias": mx.zeros((embed_dim,), dtype=mx.float32),
        f"{prefix}.mlp.up.weight": mx.zeros((intermediate, embed_dim), dtype=mx.float32),
        f"{prefix}.mlp.up.bias": mx.zeros((intermediate,), dtype=mx.float32),
        f"{prefix}.mlp.down.weight": mx.zeros((embed_dim, intermediate), dtype=mx.float32),
        f"{prefix}.mlp.down.bias": mx.zeros((embed_dim,), dtype=mx.float32),
    }


def _assert_allclose_mlx(actual, expected, *, rtol=1e-3, atol=5e-3):
    np.testing.assert_allclose(np.array(actual), np.array(expected), rtol=rtol, atol=atol)


class TestVGTBlock:
    def test_vgt_block_missing_tensors(self):
        cfg = _simple_config()
        hidden = mx.zeros((2, 10, cfg.embed_dim))
        result, blocker = run_vgt_block(
            hidden, cfg, {},
            layer_index=0, mode="frame", rope_positions=None,
        )
        assert blocker is not None

    def test_vgt_block_runs_with_fixture_tensors(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        result, blocker = run_vgt_block(
            hidden, cfg, tensors,
            layer_index=0, mode="frame", rope_positions=None,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_vgt_block_with_rope(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        rope_pos = mx.zeros((2, 10, 2), dtype=mx.float32)
        result, blocker = run_vgt_block(
            hidden, cfg, tensors,
            layer_index=0, mode="frame", rope_positions=rope_pos,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_vgt_block_global_mode(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("global_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        result, blocker = run_vgt_block(
            hidden, cfg, tensors,
            layer_index=0, mode="global", rope_positions=None,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_vgt_block_embed_dim_not_divisible(self):
        cfg = _simple_config(embed_dim=63, num_heads=4)
        hidden = mx.random.normal((2, 10, 63))
        result, blocker = run_vgt_block(
            hidden, cfg, {},
            layer_index=0, mode="frame", rope_positions=None,
        )
        assert blocker is not None

    def test_vgt_block_with_proj_prefix(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        tensors["frame_blocks.0.attn.proj.weight"] = tensors.pop("frame_blocks.0.attn.out.weight")
        tensors["frame_blocks.0.attn.proj.bias"] = tensors.pop("frame_blocks.0.attn.out.bias")
        result, blocker = run_vgt_block(
            hidden, cfg, tensors,
            layer_index=0, mode="frame", rope_positions=None,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_vgt_block_with_fc_mlp_prefix(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        tensors["frame_blocks.0.mlp.fc1.weight"] = tensors.pop("frame_blocks.0.mlp.up.weight")
        tensors["frame_blocks.0.mlp.fc1.bias"] = tensors.pop("frame_blocks.0.mlp.up.bias")
        tensors["frame_blocks.0.mlp.fc2.weight"] = tensors.pop("frame_blocks.0.mlp.down.weight")
        tensors["frame_blocks.0.mlp.fc2.bias"] = tensors.pop("frame_blocks.0.mlp.down.bias")
        result, blocker = run_vgt_block(
            hidden, cfg, tensors,
            layer_index=0, mode="frame", rope_positions=None,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_public_block_facades_match_functional_runner(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))

        expected, expected_blocker = run_vgt_block(
            hidden, cfg, tensors, layer_index=0, mode="frame"
        )
        for block_cls in (Block, DistBlock, NestedTensorBlock):
            result, blocker = block_cls(cfg, tensors, layer_index=0, mode="frame")(hidden)
            assert blocker is None
            assert expected_blocker is None
            _assert_allclose_mlx(result, expected)

    def test_vgt_block_matches_worldmirror_reference_block(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        rope_pos = mx.zeros((2, 10, 2), dtype=mx.float32)

        result, blocker = run_vgt_block(
            hidden, cfg, tensors, layer_index=0, mode="frame", rope_positions=rope_pos
        )
        reference = _run_transformer_block(
            hidden, cfg, tensors, layer_index=0, mode="frame", rope_positions=rope_pos
        )

        assert blocker is None
        assert reference.blocker is None
        assert reference.hidden_states is not None
        _assert_allclose_mlx(result, reference.hidden_states)

    def test_vgt_block_supports_vendor_swiglu_keys(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("frame_blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        tensors["frame_blocks.0.mlp.w12.weight"] = mx.zeros(
            (int(cfg.embed_dim * cfg.mlp_ratio) * 2, cfg.embed_dim), dtype=mx.float32
        )
        tensors["frame_blocks.0.mlp.w12.bias"] = mx.zeros(
            (int(cfg.embed_dim * cfg.mlp_ratio) * 2,), dtype=mx.float32
        )
        tensors["frame_blocks.0.mlp.w3.weight"] = mx.zeros(
            (cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio)), dtype=mx.float32
        )
        tensors["frame_blocks.0.mlp.w3.bias"] = mx.zeros((cfg.embed_dim,), dtype=mx.float32)
        for key in (
            "frame_blocks.0.mlp.up.weight",
            "frame_blocks.0.mlp.up.bias",
            "frame_blocks.0.mlp.down.weight",
            "frame_blocks.0.mlp.down.bias",
        ):
            tensors.pop(key)

        result, blocker = run_vgt_block(
            hidden, cfg, tensors, layer_index=0, mode="frame", rope_positions=None
        )

        assert blocker is None
        assert result.shape == hidden.shape


class TestDINOBlock:
    def test_dino_block_missing_tensors(self):
        cfg = _simple_config()
        hidden = mx.zeros((2, 10, cfg.embed_dim))
        result, blocker = run_dino_block(
            hidden, cfg, {},
            block_index=0,
        )
        assert blocker is not None

    def test_dino_block_runs(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("patch_embed.blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        result, blocker = run_dino_block(
            hidden, cfg, tensors,
            block_index=0,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_dino_block_with_layer_scale(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("patch_embed.blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))
        tensors["patch_embed.blocks.0.ls1.gamma"] = mx.ones((cfg.embed_dim,), dtype=mx.float32) * 0.1
        tensors["patch_embed.blocks.0.ls2.gamma"] = mx.ones((cfg.embed_dim,), dtype=mx.float32) * 0.1
        result, blocker = run_dino_block(
            hidden, cfg, tensors,
            block_index=0,
        )
        assert blocker is None
        assert result.shape == hidden.shape

    def test_dino_block_matches_worldmirror_reference_block(self):
        cfg = _simple_config()
        hidden = mx.random.normal((2, 10, cfg.embed_dim))
        tensors = _make_block_tensors("patch_embed.blocks.0", cfg.embed_dim, int(cfg.embed_dim * cfg.mlp_ratio))

        result, blocker = run_dino_block(hidden, cfg, tensors, block_index=0)
        reference = _run_dino_transformer_block(hidden, cfg, tensors, block_index=0)

        assert blocker is None
        assert reference.blocker is None
        assert reference.hidden_states is not None
        _assert_allclose_mlx(result, reference.hidden_states)


class TestDINOViT:
    def _make_dino_tensors(self, cfg):
        embed_dim = cfg.embed_dim
        depth = cfg.depth
        patch_size = cfg.patch_size
        num_reg = cfg.num_register_tokens
        patch_h = cfg.img_size // patch_size
        patch_w = cfg.img_size // patch_size
        num_patches = patch_h * patch_w
        tensors = {
            "patch_embed.patch_embed.proj.weight": mx.random.normal(
                (embed_dim, 3, patch_size, patch_size), dtype=mx.float32
            ) * 0.01,
            "patch_embed.patch_embed.proj.bias": mx.zeros((embed_dim,), dtype=mx.float32),
            "patch_embed.cls_token": mx.random.normal((1, 1, embed_dim), dtype=mx.float32) * 0.01,
            "patch_embed.pos_embed": mx.random.normal(
                (1, 1 + num_patches, embed_dim), dtype=mx.float32
            ) * 0.01,
            "patch_embed.register_tokens": mx.random.normal(
                (1, num_reg, embed_dim), dtype=mx.float32
            ) * 0.01,
            "patch_embed.norm.weight": mx.ones((embed_dim,), dtype=mx.float32),
            "patch_embed.norm.bias": mx.zeros((embed_dim,), dtype=mx.float32),
        }
        intermediate = int(embed_dim * cfg.mlp_ratio)
        for i in range(depth):
            tensors.update(
                _make_block_tensors(
                    f"patch_embed.blocks.{i}", embed_dim, intermediate
                )
            )
        return tensors

    def test_dino_vit_runs(self):
        cfg = _simple_config()
        tensors = self._make_dino_tensors(cfg)
        image = mx.random.normal((1, 1, 3, cfg.img_size, cfg.img_size), dtype=mx.float32)
        patch_grid = (cfg.img_size // cfg.patch_size, cfg.img_size // cfg.patch_size)
        result, blocker = run_dino_vit(image, cfg, tensors, patch_grid=patch_grid)
        assert blocker is None
        assert result is not None
        assert result.shape[0] == 1
        assert result.shape[1] == 1
        num_patches = patch_grid[0] * patch_grid[1]
        assert result.shape[2] == num_patches
        assert result.shape[3] == cfg.embed_dim

    def test_public_dino_vit_facade_extracts_patch_tokens(self):
        cfg = _simple_config()
        tensors = self._make_dino_tensors(cfg)
        image = mx.random.normal((1, 1, 3, cfg.img_size, cfg.img_size), dtype=mx.float32)
        model = DinoVisionTransformer(cfg, tensors)

        result, blocker = model(image)
        features, feature_blocker = model.forward_features(image)

        assert blocker is None
        assert feature_blocker is None
        assert result is not None
        assert features is not None
        _assert_allclose_mlx(features["x_norm_patchtokens"], result)

    def test_dino_vit_matches_worldmirror_reference_patch_tokens(self):
        cfg = _simple_config()
        tensors = self._make_dino_tensors(cfg)
        image = mx.random.normal((1, 1, 3, cfg.img_size, cfg.img_size), dtype=mx.float32)
        patch_grid = (cfg.img_size // cfg.patch_size, cfg.img_size // cfg.patch_size)

        result, blocker = run_dino_vit(image, cfg, tensors, patch_grid=patch_grid)
        reference = _official_dino_patch_tokens(image, cfg, tensors, patch_grid=patch_grid)

        assert blocker is None
        assert reference.blocker is None
        assert result is not None
        assert reference.patch_tokens is not None
        _assert_allclose_mlx(result, reference.patch_tokens)

    def test_dino_vit_missing_tensor(self):
        cfg = _simple_config()
        image = mx.random.normal((1, 1, 3, cfg.img_size, cfg.img_size))
        result, blocker = run_dino_vit(image, cfg, {}, patch_grid=(4, 4))
        assert blocker is not None
        assert result is None


class TestInterpolateDinoPosEmbed:
    def test_exact_match_no_interpolation(self):
        embed_dim = 8
        num_patches = 16
        pos_embed = mx.random.normal((1, 1 + num_patches, embed_dim), dtype=mx.float32)
        result, blocker = interpolate_dino_pos_embed(pos_embed, (4, 4))
        assert blocker is None
        assert result is not None
        assert mx.allclose(result, pos_embed).item()

    def test_upsample_requires_scipy(self):
        embed_dim = 8
        pos_embed = mx.random.normal((1, 1 + 9, embed_dim), dtype=mx.float32)
        result, blocker = interpolate_dino_pos_embed(pos_embed, (6, 6))
        if blocker is not None:
            assert "scipy" in blocker.reason or "interpolation" in blocker.reason
