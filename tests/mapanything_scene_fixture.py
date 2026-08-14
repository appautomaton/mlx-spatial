"""Deterministic miniature assets for MapAnything scene-pipeline tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_spatial.mapanything_heads import MapAnythingHeadsConfig
from mlx_spatial.mapanything_model import (
    MapAnythingEncoderPrefixConfig,
    MapAnythingInfoSharingConfig,
)
from tests.safetensors_test_utils import save_file


@dataclass(frozen=True)
class MapAnythingMiniatureSceneFixture:
    """Paths and test-only config for a generated scene fixture."""

    model_root: Path
    image_root: Path
    heads_config: MapAnythingHeadsConfig


def build_mapanything_miniature_scene_fixture(
    root: Path,
) -> MapAnythingMiniatureSceneFixture:
    """Generate a tiny, fully executable MapAnything checkpoint and two views."""

    model_root = root / "model"
    image_root = root / "images"
    model_root.mkdir(parents=True)
    image_root.mkdir(parents=True)

    encoder_config = tiny_encoder_config(layers=2)
    info_config = tiny_info_config()
    heads_config = tiny_heads_config(input_feature_dim=8)
    weights = {
        **tiny_encoder_weights(encoder_config),
        **tiny_info_weights(info_config),
        **tiny_heads_weights(heads_config),
    }
    (model_root / "config.json").write_text(
        tiny_model_config_json(encoder_layers=2),
        encoding="utf-8",
    )
    save_file(weights, model_root / "model.safetensors")

    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    Image.fromarray(pixels, mode="RGB").save(image_root / "view-0.png")
    Image.fromarray(255 - pixels, mode="RGB").save(image_root / "view-1.png")
    return MapAnythingMiniatureSceneFixture(
        model_root=model_root,
        image_root=image_root,
        heads_config=heads_config,
    )


def tiny_encoder_config(*, layers: int = 1) -> MapAnythingEncoderPrefixConfig:
    return MapAnythingEncoderPrefixConfig(
        embed_dim=8,
        num_heads=2,
        patch_size=2,
        data_norm_type="dinov2",
        encoder_size="giant",
        keep_first_n_layers=layers,
    )


def tiny_info_config(
    *,
    depth: int = 2,
    indices: tuple[int, ...] = (0, 1),
    dim: int = 8,
) -> MapAnythingInfoSharingConfig:
    return MapAnythingInfoSharingConfig(
        input_embed_dim=dim,
        dim=dim,
        depth=depth,
        num_heads=2,
        indices=indices,
        norm_intermediate=True,
    )


def tiny_heads_config(*, input_feature_dim: int = 4) -> MapAnythingHeadsConfig:
    return MapAnythingHeadsConfig(
        input_feature_dim=input_feature_dim,
        patch_size=2,
        layer_dims=(2, 2, 2, 2),
        feature_dim=2,
        dense_output_dim=6,
        pose_resconv_blocks=2,
        pose_rot_dim=4,
        scale_hidden_dim=3,
        scale_output_dim=1,
    )


def tiny_encoder_weights(
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> dict[str, mx.array]:
    config = config or tiny_encoder_config()
    hidden = config.swiglu_hidden_features
    randn = _random_tensor_factory(seed=42)
    weights: dict[str, mx.array] = {
        "encoder.model.cls_token": randn((1, 1, config.embed_dim), 0.03),
        "encoder.model.pos_embed": randn((1, 5, config.embed_dim), 0.02),
        "encoder.model.patch_embed.proj.weight": randn(
            (config.embed_dim, 3, config.patch_size, config.patch_size), 0.04
        ),
        "encoder.model.patch_embed.proj.bias": randn((config.embed_dim,), 0.01),
    }
    for block_index in range(config.keep_first_n_layers):
        prefix = f"encoder.model.blocks.{block_index}"
        weights.update(
            {
                f"{prefix}.norm1.weight": randn((config.embed_dim,), 0.01, 1.0),
                f"{prefix}.norm1.bias": randn((config.embed_dim,), 0.01),
                f"{prefix}.attn.qkv.weight": randn(
                    (3 * config.embed_dim, config.embed_dim), 0.03
                ),
                f"{prefix}.attn.qkv.bias": randn((3 * config.embed_dim,), 0.01),
                f"{prefix}.attn.proj.weight": randn(
                    (config.embed_dim, config.embed_dim), 0.03
                ),
                f"{prefix}.attn.proj.bias": randn((config.embed_dim,), 0.01),
                f"{prefix}.ls1.gamma": randn((config.embed_dim,), 0.01, 0.1),
                f"{prefix}.norm2.weight": randn((config.embed_dim,), 0.01, 1.0),
                f"{prefix}.norm2.bias": randn((config.embed_dim,), 0.01),
                f"{prefix}.mlp.w12.weight": randn(
                    (2 * hidden, config.embed_dim), 0.02
                ),
                f"{prefix}.mlp.w12.bias": randn((2 * hidden,), 0.01),
                f"{prefix}.mlp.w3.weight": randn(
                    (config.embed_dim, hidden), 0.02
                ),
                f"{prefix}.mlp.w3.bias": randn((config.embed_dim,), 0.01),
                f"{prefix}.ls2.gamma": randn((config.embed_dim,), 0.01, 0.1),
            }
        )
    return weights


def tiny_info_weights(
    config: MapAnythingInfoSharingConfig,
    *,
    identity_odd_attention: bool = False,
) -> dict[str, mx.array]:
    hidden = config.swiglu_hidden_features
    weights: dict[str, mx.array] = {
        "scale_token": mx.array(
            np.linspace(-0.5, 0.6, config.dim, dtype=np.float32)
        ),
        "info_sharing.norm.weight": mx.ones((config.dim,), dtype=mx.float32),
        "info_sharing.norm.bias": mx.zeros((config.dim,), dtype=mx.float32),
        "info_sharing.view_pos_table": mx.zeros((1, config.dim), dtype=mx.float32),
    }
    for block_index in range(config.depth):
        prefix = f"info_sharing.self_attention_blocks.{block_index}"
        weights.update(
            {
                f"{prefix}.norm1.weight": mx.ones((config.dim,), dtype=mx.float32),
                f"{prefix}.norm1.bias": mx.zeros((config.dim,), dtype=mx.float32),
                f"{prefix}.attn.qkv.weight": mx.zeros(
                    (3 * config.dim, config.dim), dtype=mx.float32
                ),
                f"{prefix}.attn.qkv.bias": mx.zeros(
                    (3 * config.dim,), dtype=mx.float32
                ),
                f"{prefix}.attn.proj.weight": mx.zeros(
                    (config.dim, config.dim), dtype=mx.float32
                ),
                f"{prefix}.attn.proj.bias": mx.zeros(
                    (config.dim,), dtype=mx.float32
                ),
                f"{prefix}.ls1.gamma": mx.ones((config.dim,), dtype=mx.float32),
                f"{prefix}.norm2.weight": mx.ones((config.dim,), dtype=mx.float32),
                f"{prefix}.norm2.bias": mx.zeros((config.dim,), dtype=mx.float32),
                f"{prefix}.mlp.w12.weight": mx.zeros(
                    (2 * hidden, config.dim), dtype=mx.float32
                ),
                f"{prefix}.mlp.w12.bias": mx.zeros(
                    (2 * hidden,), dtype=mx.float32
                ),
                f"{prefix}.mlp.w3.weight": mx.zeros(
                    (config.dim, hidden), dtype=mx.float32
                ),
                f"{prefix}.mlp.w3.bias": mx.zeros((config.dim,), dtype=mx.float32),
                f"{prefix}.ls2.gamma": mx.ones((config.dim,), dtype=mx.float32),
            }
        )
    if identity_odd_attention and config.depth > 1:
        prefix = "info_sharing.self_attention_blocks.1"
        identity = np.eye(config.dim, dtype=np.float32)
        weights[f"{prefix}.attn.qkv.weight"] = mx.array(
            np.concatenate((identity, identity, identity), axis=0)
        )
        weights[f"{prefix}.attn.proj.weight"] = mx.array(identity)
    return weights


def tiny_heads_weights(config: MapAnythingHeadsConfig) -> dict[str, mx.array]:
    randn = _random_tensor_factory(seed=123)
    weights: dict[str, mx.array] = {
        "fusion_norm_layer.weight": mx.ones(
            (config.input_feature_dim,), dtype=mx.float32
        ),
        "fusion_norm_layer.bias": mx.zeros(
            (config.input_feature_dim,), dtype=mx.float32
        ),
    }
    for index, channels in enumerate(config.layer_dims):
        prefix = f"dense_head.0.input_process.{index}"
        weights[f"{prefix}.0.0.weight"] = randn(
            (channels, config.input_feature_dim, 1, 1)
        )
        weights[f"{prefix}.0.0.bias"] = randn((channels,))
        weights[f"{prefix}.1.weight"] = randn(
            (config.feature_dim, channels, 3, 3)
        )
    for index, kernel_size in ((0, 4), (1, 2), (3, 3)):
        channels = config.layer_dims[index]
        prefix = f"dense_head.0.input_process.{index}.0.1"
        weights[f"{prefix}.weight"] = randn(
            (channels, channels, kernel_size, kernel_size)
        )
        weights[f"{prefix}.bias"] = randn((channels,))

    for block in ("refinenet1", "refinenet2", "refinenet3"):
        for unit in ("resConfUnit1", "resConfUnit2"):
            for conv in ("conv1", "conv2"):
                prefix = f"dense_head.0.scratch.{block}.{unit}.{conv}"
                weights[f"{prefix}.weight"] = randn(
                    (config.feature_dim, config.feature_dim, 3, 3)
                )
                weights[f"{prefix}.bias"] = randn((config.feature_dim,))
        prefix = f"dense_head.0.scratch.{block}.out_conv"
        weights[f"{prefix}.weight"] = randn(
            (config.feature_dim, config.feature_dim, 1, 1)
        )
        weights[f"{prefix}.bias"] = randn((config.feature_dim,))
    for conv in ("conv1", "conv2"):
        prefix = f"dense_head.0.scratch.refinenet4.resConfUnit2.{conv}"
        weights[f"{prefix}.weight"] = randn(
            (config.feature_dim, config.feature_dim, 3, 3)
        )
        weights[f"{prefix}.bias"] = randn((config.feature_dim,))
    weights["dense_head.0.scratch.refinenet4.out_conv.weight"] = randn(
        (config.feature_dim, config.feature_dim, 1, 1)
    )
    weights["dense_head.0.scratch.refinenet4.out_conv.bias"] = randn(
        (config.feature_dim,)
    )

    final_channels = config.feature_dim // 2
    weights["dense_head.1.conv1.weight"] = randn(
        (final_channels, config.feature_dim, 3, 3)
    )
    weights["dense_head.1.conv1.bias"] = randn((final_channels,))
    weights["dense_head.1.conv2.0.weight"] = randn(
        (final_channels, final_channels, 3, 3)
    )
    weights["dense_head.1.conv2.0.bias"] = randn((final_channels,))
    weights["dense_head.1.conv2.2.weight"] = randn(
        (config.dense_output_dim, final_channels, 1, 1)
    )
    weights["dense_head.1.conv2.2.bias"] = mx.array(
        [0.1, -0.1, 1.0, 0.0, 0.0, 2.0],
        dtype=mx.float32,
    )

    pose_hidden = config.pose_hidden_dim
    weights["pose_head.proj.weight"] = randn(
        (pose_hidden, config.input_feature_dim, 1, 1)
    )
    weights["pose_head.proj.bias"] = randn((pose_hidden,))
    for block_index in range(config.pose_resconv_blocks):
        for conv in ("res_conv1", "res_conv2", "res_conv3"):
            prefix = f"pose_head.res_conv.{block_index}.{conv}"
            weights[f"{prefix}.weight"] = randn(
                (pose_hidden, pose_hidden, 1, 1)
            )
            weights[f"{prefix}.bias"] = randn((pose_hidden,))
    for layer in (0, 2):
        weights[f"pose_head.more_mlps.{layer}.weight"] = randn(
            (pose_hidden, pose_hidden)
        )
        weights[f"pose_head.more_mlps.{layer}.bias"] = randn((pose_hidden,))
    weights["pose_head.fc_t.weight"] = randn((3, pose_hidden))
    weights["pose_head.fc_t.bias"] = randn((3,))
    weights["pose_head.fc_rot.weight"] = randn(
        (config.pose_rot_dim, pose_hidden)
    )
    weights["pose_head.fc_rot.bias"] = randn((config.pose_rot_dim,))

    weights["scale_head.proj.weight"] = randn(
        (config.scale_hidden_dim, config.input_feature_dim)
    )
    weights["scale_head.proj.bias"] = randn((config.scale_hidden_dim,))
    for layer in (0, 1):
        weights[f"scale_head.mlp.{layer}.0.weight"] = randn(
            (config.scale_hidden_dim, config.scale_hidden_dim)
        )
        weights[f"scale_head.mlp.{layer}.0.bias"] = randn(
            (config.scale_hidden_dim,)
        )
    weights["scale_head.output_proj.weight"] = randn(
        (config.scale_output_dim, config.scale_hidden_dim)
    )
    weights["scale_head.output_proj.bias"] = randn((config.scale_output_dim,))
    return weights


def _random_tensor_factory(seed: int):
    rng = np.random.default_rng(seed)

    def randn(
        shape: tuple[int, ...],
        scale: float = 0.02,
        offset: float = 0.0,
    ) -> mx.array:
        values = rng.normal(loc=offset, scale=scale, size=shape).astype(np.float32)
        return mx.array(values)

    return randn


def tiny_features_and_registers(
    config: MapAnythingInfoSharingConfig,
) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
    values = mx.arange(2 * config.dim * 2 * 2, dtype=mx.float32).reshape(
        (2, config.dim, 2, 2)
    ) / 100
    features = (values[0:1], values[1:2])
    registers = (
        mx.ones((1, config.dim, 1), dtype=mx.float32) * 0.1,
        mx.ones((1, config.dim, 1), dtype=mx.float32) * -0.1,
    )
    return features, registers


def tiny_prefix_asset_weights() -> dict[str, mx.array]:
    config = tiny_encoder_config()
    weights = tiny_encoder_weights(config)
    weights.update(
        {
            "info_sharing.dummy": mx.zeros((1,), dtype=mx.float32),
            "dense_head.dummy": mx.zeros((1,), dtype=mx.float32),
            "pose_head.dummy": mx.zeros((1,), dtype=mx.float32),
            "scale_head.dummy": mx.zeros((1,), dtype=mx.float32),
            "fusion_norm_layer.weight": mx.ones(
                (config.embed_dim,), dtype=mx.float32
            ),
            "scale_token": mx.zeros((config.embed_dim,), dtype=mx.float32),
        }
    )
    return weights


def write_tiny_mapanything_prefix_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        tiny_model_config_json(
            info_depth=1,
            info_indices=(0,),
            use_register_tokens=True,
        ),
        encoding="utf-8",
    )
    save_file(tiny_prefix_asset_weights(), root / "model.safetensors")
    return root


def tiny_model_config_json(
    *,
    encoder_layers: int = 1,
    info_depth: int = 2,
    info_dim: int = 8,
    info_indices: tuple[int, ...] = (0, 1),
    use_register_tokens: bool = True,
) -> str:
    payload = _tiny_model_config(
        encoder_layers=encoder_layers,
        info_depth=info_depth,
        info_dim=info_dim,
        info_indices=info_indices,
        use_register_tokens=use_register_tokens,
    )
    return json.dumps(payload, indent=2) + "\n"


def _tiny_model_config(
    *,
    encoder_layers: int,
    info_depth: int,
    info_dim: int,
    info_indices: tuple[int, ...],
    use_register_tokens: bool,
) -> dict[str, object]:
    return {
        "encoder_config": {
            "data_norm_type": "dinov2",
            "name": "tiny-generated-fixture",
            "size": "giant",
            "keep_first_n_layers": encoder_layers,
            "uses_torch_hub": False,
        },
        "info_sharing_config": {
            "model_type": "alternating_attention",
            "model_return_type": "intermediate_features",
            "module_args": {
                "depth": info_depth,
                "dim": info_dim,
                "num_heads": 2,
                "indices": list(info_indices),
            },
        },
        "pred_head_config": {
            "type": "dpt+pose",
            "adaptor_type": "raydirs+depth+pose+confidence+mask",
            "feature_head": {"patch_size": 2},
            "adaptor_config": {
                "dense_pred_init_dict": {
                    "name": "raydirs+depth+pose+confidence+mask+scale"
                }
            },
        },
        "use_register_tokens_from_encoder": use_register_tokens,
    }


__all__ = [
    "MapAnythingMiniatureSceneFixture",
    "build_mapanything_miniature_scene_fixture",
    "tiny_encoder_config",
    "tiny_encoder_weights",
    "tiny_features_and_registers",
    "tiny_heads_config",
    "tiny_heads_weights",
    "tiny_info_config",
    "tiny_info_weights",
    "tiny_model_config_json",
    "tiny_prefix_asset_weights",
    "write_tiny_mapanything_prefix_fixture",
]
