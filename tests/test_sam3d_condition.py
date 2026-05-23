import mlx.core as mx
import numpy as np

from mlx_spatial.sam3d_condition import (
    Sam3dDinoConfig,
    Sam3dPointPatchConfig,
    _resize_nearest_bchw,
    fuse_sam3d_condition_tokens,
    run_sam3d_dino_vitl14_reg,
    run_sam3d_point_patch_embed,
    run_sam3d_projection_net,
    run_sam3d_ss_condition_stack,
)


def _point_patch_tensors(config: Sam3dPointPatchConfig):
    dim = config.embed_dim
    hidden = int(dim * config.mlp_ratio)
    return {
        "point_proj.weight": mx.ones((dim, 3), dtype=mx.float32) * 0.01,
        "point_proj.bias": mx.zeros((dim,), dtype=mx.float32),
        "invalid_xyz_token": mx.ones((dim,), dtype=mx.float32) * 0.5,
        "cls_token": mx.zeros((1, 1, dim), dtype=mx.float32),
        "pos_embed": mx.zeros((1, dim, config.input_size // config.patch_size, config.input_size // config.patch_size), dtype=mx.float32),
        "pos_embed_window": mx.zeros((1, 1 + config.patch_size * config.patch_size, dim), dtype=mx.float32),
        "blocks.0.norm1.weight": mx.ones((dim,), dtype=mx.float32),
        "blocks.0.norm1.bias": mx.zeros((dim,), dtype=mx.float32),
        "blocks.0.norm2.weight": mx.ones((dim,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.zeros((dim,), dtype=mx.float32),
        "blocks.0.attn.qkv.weight": mx.zeros((dim * 3, dim), dtype=mx.float32),
        "blocks.0.attn.qkv.bias": mx.zeros((dim * 3,), dtype=mx.float32),
        "blocks.0.attn.proj.weight": mx.zeros((dim, dim), dtype=mx.float32),
        "blocks.0.attn.proj.bias": mx.zeros((dim,), dtype=mx.float32),
        "blocks.0.mlp.fc1.weight": mx.zeros((hidden, dim), dtype=mx.float32),
        "blocks.0.mlp.fc1.bias": mx.zeros((hidden,), dtype=mx.float32),
        "blocks.0.mlp.fc2.weight": mx.zeros((dim, hidden), dtype=mx.float32),
        "blocks.0.mlp.fc2.bias": mx.zeros((dim,), dtype=mx.float32),
    }


def _dino_tensors(config: Sam3dDinoConfig, *, prefix: str = "module_list.0.backbone."):
    dim = config.embed_dim
    hidden = dim * 2
    tensors = {
        f"{prefix}patch_embed.proj.weight": mx.zeros((dim, 3, config.patch_size, config.patch_size), dtype=mx.float32),
        f"{prefix}patch_embed.proj.bias": mx.zeros((dim,), dtype=mx.float32),
        f"{prefix}cls_token": mx.zeros((1, 1, dim), dtype=mx.float32),
        f"{prefix}register_tokens": mx.zeros((1, config.register_count, dim), dtype=mx.float32),
        f"{prefix}pos_embed": mx.zeros((1, 1 + (config.input_size // config.patch_size) ** 2, dim), dtype=mx.float32),
        f"{prefix}norm.weight": mx.ones((dim,), dtype=mx.float32),
        f"{prefix}norm.bias": mx.zeros((dim,), dtype=mx.float32),
    }
    for index in range(config.num_blocks):
        block = f"{prefix}blocks.{index}"
        tensors.update(
            {
                f"{block}.norm1.weight": mx.ones((dim,), dtype=mx.float32),
                f"{block}.norm1.bias": mx.zeros((dim,), dtype=mx.float32),
                f"{block}.norm2.weight": mx.ones((dim,), dtype=mx.float32),
                f"{block}.norm2.bias": mx.zeros((dim,), dtype=mx.float32),
                f"{block}.attn.qkv.weight": mx.zeros((dim * 3, dim), dtype=mx.float32),
                f"{block}.attn.qkv.bias": mx.zeros((dim * 3,), dtype=mx.float32),
                f"{block}.attn.proj.weight": mx.zeros((dim, dim), dtype=mx.float32),
                f"{block}.attn.proj.bias": mx.zeros((dim,), dtype=mx.float32),
                f"{block}.ls1.gamma": mx.ones((dim,), dtype=mx.float32),
                f"{block}.mlp.fc1.weight": mx.zeros((hidden, dim), dtype=mx.float32),
                f"{block}.mlp.fc1.bias": mx.zeros((hidden,), dtype=mx.float32),
                f"{block}.mlp.fc2.weight": mx.zeros((dim, hidden), dtype=mx.float32),
                f"{block}.mlp.fc2.bias": mx.zeros((dim,), dtype=mx.float32),
                f"{block}.ls2.gamma": mx.ones((dim,), dtype=mx.float32),
            }
        )
    return tensors


def _projection_tensors(prefix: str, *, in_dim: int, out_dim: int, hidden_dim: int):
    return {
        f"{prefix}0.weight": mx.ones((in_dim,), dtype=mx.float32),
        f"{prefix}0.bias": mx.zeros((in_dim,), dtype=mx.float32),
        f"{prefix}1.w1.weight": mx.zeros((hidden_dim, in_dim), dtype=mx.float32),
        f"{prefix}1.w2.weight": mx.zeros((out_dim, hidden_dim), dtype=mx.float32),
        f"{prefix}1.w3.weight": mx.zeros((hidden_dim, in_dim), dtype=mx.float32),
    }


def test_run_sam3d_point_patch_embed_returns_one_token_per_window():
    config = Sam3dPointPatchConfig(input_size=16, patch_size=4, embed_dim=32, num_heads=4)
    pointmap = np.zeros((3, 8, 8), dtype=np.float32)

    tokens = run_sam3d_point_patch_embed(pointmap, _point_patch_tensors(config), config)

    assert tuple(tokens.shape) == (1, 16, 32)


def test_run_sam3d_point_patch_embed_accepts_hwc_and_invalid_points():
    config = Sam3dPointPatchConfig(input_size=8, patch_size=4, embed_dim=16, num_heads=4)
    pointmap = np.zeros((8, 8, 3), dtype=np.float32)
    pointmap[0, 0, :] = np.inf

    tokens = run_sam3d_point_patch_embed(pointmap, _point_patch_tensors(config), config)

    assert tuple(tokens.shape) == (1, 4, 16)
    assert np.isfinite(np.array(tokens)).all()


def test_resize_nearest_bchw_matches_torch_floor_indices():
    values = mx.array(np.arange(3, dtype=np.float32).reshape(1, 1, 3, 1))

    resized = _resize_nearest_bchw(values, 5)

    assert np.array_equal(np.array(resized[0, 0, :, 0]), np.array([0, 0, 1, 1, 2], dtype=np.float32))


def test_run_sam3d_dino_vitl14_reg_omits_registers_after_final_norm():
    config = Sam3dDinoConfig(
        input_size=4,
        patch_size=2,
        embed_dim=8,
        num_heads=2,
        num_blocks=1,
        register_count=1,
        normalize_images=False,
    )
    image = np.zeros((3, 4, 4), dtype=np.float32)

    tokens = run_sam3d_dino_vitl14_reg(image, _dino_tensors(config), module_index=0, config=config)

    assert tuple(tokens.shape) == (1, 5, 8)


def test_run_sam3d_dino_vitl14_reg_prenorm_keeps_register_tokens():
    config = Sam3dDinoConfig(
        input_size=4,
        patch_size=2,
        embed_dim=8,
        num_heads=2,
        num_blocks=1,
        register_count=1,
        prenorm_features=True,
        normalize_images=False,
    )

    tokens = run_sam3d_dino_vitl14_reg(
        np.zeros((1, 4, 4), dtype=np.float32),
        _dino_tensors(config),
        module_index=0,
        config=config,
    )

    assert tuple(tokens.shape) == (1, 6, 8)


def test_projection_net_uses_llama_feed_forward_shapes():
    tokens = mx.ones((1, 2, 4), dtype=mx.float32)
    tensors = {
        "proj.0.weight": mx.ones((4,), dtype=mx.float32),
        "proj.0.bias": mx.zeros((4,), dtype=mx.float32),
        "proj.1.w1.weight": mx.ones((8, 4), dtype=mx.float32),
        "proj.1.w2.weight": mx.ones((6, 8), dtype=mx.float32),
        "proj.1.w3.weight": mx.ones((8, 4), dtype=mx.float32),
    }

    projected = run_sam3d_projection_net(tokens, tensors, prefix="proj.")

    assert tuple(projected.shape) == (1, 2, 6)


def test_fuse_sam3d_condition_tokens_adds_learned_position_embeddings():
    first = mx.zeros((1, 2, 4), dtype=mx.float32)
    second = mx.ones((1, 3, 4), dtype=mx.float32)
    idx_emb = mx.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=mx.float32)

    fused = fuse_sam3d_condition_tokens((first, second), idx_emb, positional_indices=(0, 1))

    assert tuple(fused.shape) == (1, 5, 4)
    assert np.allclose(np.array(fused[0, 0]), np.ones((4,)))
    assert np.allclose(np.array(fused[0, -1]), np.ones((4,)) * 3)


def test_run_sam3d_ss_condition_stack_fuses_active_modalities():
    class Official:
        image = np.zeros((3, 4, 4), dtype=np.float32)
        rgb_image = np.zeros((3, 4, 4), dtype=np.float32)
        mask = np.zeros((1, 4, 4), dtype=np.float32)
        rgb_image_mask = np.zeros((1, 4, 4), dtype=np.float32)
        pointmap = np.zeros((3, 4, 4), dtype=np.float32)
        rgb_pointmap = np.zeros((3, 4, 4), dtype=np.float32)

    dino_config = Sam3dDinoConfig(
        input_size=4,
        patch_size=2,
        embed_dim=8,
        num_heads=2,
        num_blocks=1,
        register_count=1,
        normalize_images=False,
    )
    point_config = Sam3dPointPatchConfig(input_size=4, patch_size=2, embed_dim=4, num_heads=2)
    tensors = {
        **_dino_tensors(dino_config, prefix="module_list.0.backbone."),
        **_dino_tensors(dino_config, prefix="module_list.1.backbone."),
        **{f"module_list.2.{key}": value for key, value in _point_patch_tensors(point_config).items()},
        **_projection_tensors("projection_nets.0.", in_dim=8, out_dim=8, hidden_dim=12),
        **_projection_tensors("projection_nets.1.", in_dim=8, out_dim=8, hidden_dim=12),
        **_projection_tensors("projection_nets.2.", in_dim=4, out_dim=8, hidden_dim=12),
        "idx_emb": mx.zeros((3, 8), dtype=mx.float32),
    }

    output = run_sam3d_ss_condition_stack(Official(), tensors, dino_config=dino_config, point_config=point_config)

    assert tuple(output.tokens.shape) == (1, 28, 8)
    assert output.metadata["kind"] == "ss"
