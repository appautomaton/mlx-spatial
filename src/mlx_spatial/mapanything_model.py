"""MapAnything MLX model components.

This module implements inference-time MapAnything model stages without importing
Torch, TorchVision, UniCeption, or vendored MapAnything runtime code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import mlx.core as mx

from .checkpoint import load_checkpoint_tensors
from .mapanything_assets import (
    MAPANYTHING_DEFAULT_ROOT,
    MapAnythingModelConfig,
    read_mapanything_model_config,
)


MAPANYTHING_ENCODER_PREFIX_KEY_MAP = {
    "cls_token": "encoder.model.cls_token",
    "pos_embed": "encoder.model.pos_embed",
    "patch_embed.proj.weight": "encoder.model.patch_embed.proj.weight",
    "patch_embed.proj.bias": "encoder.model.patch_embed.proj.bias",
    "blocks.0.norm1.weight": "encoder.model.blocks.0.norm1.weight",
    "blocks.0.norm1.bias": "encoder.model.blocks.0.norm1.bias",
    "blocks.0.attn.qkv.weight": "encoder.model.blocks.0.attn.qkv.weight",
    "blocks.0.attn.qkv.bias": "encoder.model.blocks.0.attn.qkv.bias",
    "blocks.0.attn.proj.weight": "encoder.model.blocks.0.attn.proj.weight",
    "blocks.0.attn.proj.bias": "encoder.model.blocks.0.attn.proj.bias",
    "blocks.0.ls1.gamma": "encoder.model.blocks.0.ls1.gamma",
    "blocks.0.norm2.weight": "encoder.model.blocks.0.norm2.weight",
    "blocks.0.norm2.bias": "encoder.model.blocks.0.norm2.bias",
    "blocks.0.mlp.w12.weight": "encoder.model.blocks.0.mlp.w12.weight",
    "blocks.0.mlp.w12.bias": "encoder.model.blocks.0.mlp.w12.bias",
    "blocks.0.mlp.w3.weight": "encoder.model.blocks.0.mlp.w3.weight",
    "blocks.0.mlp.w3.bias": "encoder.model.blocks.0.mlp.w3.bias",
    "blocks.0.ls2.gamma": "encoder.model.blocks.0.ls2.gamma",
}
MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS = tuple(MAPANYTHING_ENCODER_PREFIX_KEY_MAP.values())
MAPANYTHING_ENCODER_BASE_REQUIRED_KEYS = (
    "encoder.model.cls_token",
    "encoder.model.pos_embed",
    "encoder.model.patch_embed.proj.weight",
    "encoder.model.patch_embed.proj.bias",
)
MAPANYTHING_ENCODER_BLOCK_SUFFIXES = (
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "ls1.gamma",
    "norm2.weight",
    "norm2.bias",
    "mlp.w12.weight",
    "mlp.w12.bias",
    "mlp.w3.weight",
    "mlp.w3.bias",
    "ls2.gamma",
)
MAPANYTHING_DINOV2_GIANT_EMBED_DIM = 1536
MAPANYTHING_DINOV2_GIANT_NUM_HEADS = 24
MAPANYTHING_DINOV2_LAYER_NORM_EPS = 1e-6
MAPANYTHING_DINOV2_POS_INTERPOLATE_OFFSET = 0.1
MAPANYTHING_INFO_SHARING_BASE_REQUIRED_KEYS = (
    "scale_token",
    "info_sharing.norm.weight",
    "info_sharing.norm.bias",
    "info_sharing.view_pos_table",
)
MAPANYTHING_INFO_SHARING_BLOCK_SUFFIXES = (
    "norm1.weight",
    "norm1.bias",
    "attn.qkv.weight",
    "attn.qkv.bias",
    "attn.proj.weight",
    "attn.proj.bias",
    "ls1.gamma",
    "norm2.weight",
    "norm2.bias",
    "mlp.w12.weight",
    "mlp.w12.bias",
    "mlp.w3.weight",
    "mlp.w3.bias",
    "ls2.gamma",
)


@dataclass(frozen=True)
class MapAnythingEncoderPrefixConfig:
    """Static DINOv2 encoder-prefix shape/config values used by MapAnything."""

    embed_dim: int = MAPANYTHING_DINOV2_GIANT_EMBED_DIM
    num_heads: int = MAPANYTHING_DINOV2_GIANT_NUM_HEADS
    patch_size: int = 14
    mlp_ratio: float = 4.0
    layer_norm_eps: float = MAPANYTHING_DINOV2_LAYER_NORM_EPS
    pos_interpolate_offset: float = MAPANYTHING_DINOV2_POS_INTERPOLATE_OFFSET
    data_norm_type: str = "dinov2"
    encoder_size: str = "giant"
    keep_first_n_layers: int = 24
    with_registers: bool = False

    @property
    def head_dim(self) -> int:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {self.embed_dim} and {self.num_heads}"
            )
        return self.embed_dim // self.num_heads

    @property
    def swiglu_hidden_features(self) -> int:
        hidden = int(self.embed_dim * self.mlp_ratio)
        return (int(hidden * 2 / 3) + 7) // 8 * 8


@dataclass(frozen=True)
class MapAnythingInfoSharingConfig:
    """Static config for MapAnything's alternating multi-view transformer."""

    input_embed_dim: int = MAPANYTHING_DINOV2_GIANT_EMBED_DIM
    dim: int = MAPANYTHING_DINOV2_GIANT_EMBED_DIM
    depth: int = 16
    num_heads: int = MAPANYTHING_DINOV2_GIANT_NUM_HEADS
    mlp_ratio: float = 4.0
    layer_norm_eps: float = MAPANYTHING_DINOV2_LAYER_NORM_EPS
    indices: tuple[int, ...] = (7, 11)
    norm_intermediate: bool = True
    distinguish_ref_and_non_ref_views: bool = True
    use_pe_for_non_reference_views: bool = False
    use_register_tokens_from_encoder: bool = True

    @property
    def head_dim(self) -> int:
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim must be divisible by num_heads, got {self.dim} and {self.num_heads}")
        return self.dim // self.num_heads

    @property
    def swiglu_hidden_features(self) -> int:
        hidden = int(self.dim * self.mlp_ratio)
        return (int(hidden * 2 / 3) + 7) // 8 * 8


@dataclass(frozen=True)
class MapAnythingEncoderPrefixOutput:
    """Outputs from the first MapAnything MLX model-stage boundary."""

    patch_embeddings: mx.array
    tokens_with_position: mx.array
    block0: mx.array
    patch_grid: tuple[int, int]
    trace: dict[str, object]

    @property
    def parity_tensors(self) -> dict[str, mx.array]:
        return {
            "encoder.patch_embed": self.patch_embeddings,
            "encoder.tokens": self.tokens_with_position,
            "encoder.block0": self.block0,
        }


@dataclass(frozen=True)
class MapAnythingEncoderOutput:
    """Outputs from the full MapAnything DINOv2 image encoder."""

    patch_embeddings: mx.array
    tokens_with_position: mx.array
    block0: mx.array
    final_tokens: mx.array
    features: mx.array
    registers: mx.array
    patch_grid: tuple[int, int]
    trace: dict[str, object]

    @property
    def parity_tensors(self) -> dict[str, mx.array]:
        tensors = {
            "encoder.patch_embed": self.patch_embeddings,
            "encoder.tokens": self.tokens_with_position,
            "encoder.block0": self.block0,
        }
        batch = int(self.features.shape[0])
        for index in range(batch):
            tensors[f"encoder.features.{index}"] = self.features[index : index + 1]
            tensors[f"encoder.registers.{index}"] = self.registers[index : index + 1]
        return tensors


@dataclass(frozen=True)
class MapAnythingInfoSharingOutput:
    """One unpacked output from the multi-view information-sharing transformer."""

    features: tuple[mx.array, ...]
    additional_token_features: mx.array | None = None
    additional_token_features_per_view: tuple[mx.array, ...] | None = None


@dataclass(frozen=True)
class MapAnythingInfoSharingResult:
    """Final and intermediate MapAnything info-sharing outputs."""

    final: MapAnythingInfoSharingOutput
    intermediates: tuple[MapAnythingInfoSharingOutput, ...]
    trace: dict[str, object]

    @property
    def parity_tensors(self) -> dict[str, mx.array]:
        tensors: dict[str, mx.array] = {}
        _add_info_sharing_parity_tensors(tensors, "info.final", self.final)
        for index, intermediate in enumerate(self.intermediates):
            _add_info_sharing_parity_tensors(tensors, f"info.intermediate.{index}", intermediate)
        return tensors


class MapAnythingEncoderPrefix:
    """MLX implementation of DINOv2 patch embedding plus encoder block 0."""

    def __init__(
        self,
        weights: Mapping[str, mx.array],
        config: MapAnythingEncoderPrefixConfig | None = None,
    ) -> None:
        self.config = config or MapAnythingEncoderPrefixConfig()
        self.weights = dict(weights)
        validate_mapanything_encoder_prefix_weights(self.weights, self.config)

    def __call__(self, images: mx.array) -> MapAnythingEncoderPrefixOutput:
        return run_mapanything_encoder_prefix(images, self.weights, config=self.config)


class MapAnythingEncoder:
    """MLX implementation of the full MapAnything DINOv2 image encoder."""

    def __init__(
        self,
        weights: Mapping[str, mx.array],
        config: MapAnythingEncoderPrefixConfig | None = None,
    ) -> None:
        self.config = config or MapAnythingEncoderPrefixConfig()
        self.weights = dict(weights)
        validate_mapanything_full_encoder_weights(self.weights, self.config)

    def __call__(self, images: mx.array) -> MapAnythingEncoderOutput:
        return run_mapanything_full_encoder(images, self.weights, config=self.config)


class MapAnythingInfoSharing:
    """MLX implementation of MapAnything's alternating multi-view transformer."""

    def __init__(
        self,
        weights: Mapping[str, mx.array],
        config: MapAnythingInfoSharingConfig | None = None,
    ) -> None:
        self.config = config or MapAnythingInfoSharingConfig()
        self.weights = dict(weights)
        validate_mapanything_info_sharing_weights(self.weights, self.config)

    def __call__(
        self,
        features: Sequence[mx.array],
        *,
        additional_tokens_per_view: Sequence[mx.array] | None = None,
        additional_tokens: mx.array | None = None,
        use_checkpoint_scale_token: bool = True,
    ) -> MapAnythingInfoSharingResult:
        return run_mapanything_info_sharing(
            features,
            self.weights,
            additional_tokens_per_view=additional_tokens_per_view,
            additional_tokens=additional_tokens,
            use_checkpoint_scale_token=use_checkpoint_scale_token,
            config=self.config,
        )


def mapanything_encoder_prefix_config_from_model_config(
    model_config: MapAnythingModelConfig,
) -> MapAnythingEncoderPrefixConfig:
    """Build the MLX prefix config from the parsed official MapAnything config."""

    if model_config.encoder_size != "giant":
        raise ValueError(
            f"only the local MapAnything DINOv2 giant encoder is supported, got {model_config.encoder_size!r}"
        )
    if model_config.encoder_with_registers:
        raise ValueError("MapAnything encoder-prefix port does not support DINO register-token models yet")
    return MapAnythingEncoderPrefixConfig(
        embed_dim=model_config.info_sharing_dim,
        num_heads=model_config.info_sharing_num_heads,
        patch_size=model_config.patch_size,
        data_norm_type=model_config.data_norm_type,
        encoder_size=model_config.encoder_size,
        keep_first_n_layers=model_config.encoder_keep_first_n_layers,
        with_registers=model_config.encoder_with_registers,
    )


def mapanything_info_sharing_config_from_model_config(
    model_config: MapAnythingModelConfig,
) -> MapAnythingInfoSharingConfig:
    """Build the MLX info-sharing config from the official MapAnything config."""

    if model_config.info_sharing_model_type != "alternating_attention":
        raise ValueError(
            "only MapAnything alternating_attention info sharing is supported, "
            f"got {model_config.info_sharing_model_type!r}"
        )
    if model_config.info_sharing_return_type != "intermediate_features":
        raise ValueError(
            "MapAnything scene generation expects intermediate info-sharing features, "
            f"got {model_config.info_sharing_return_type!r}"
        )
    return MapAnythingInfoSharingConfig(
        input_embed_dim=model_config.info_sharing_dim,
        dim=model_config.info_sharing_dim,
        depth=model_config.info_sharing_depth,
        num_heads=model_config.info_sharing_num_heads,
        indices=model_config.info_sharing_indices,
        use_register_tokens_from_encoder=model_config.use_register_tokens_from_encoder,
    )


def load_mapanything_encoder_prefix_weights(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
    *,
    config: MapAnythingEncoderPrefixConfig | None = None,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    """Load the explicit MapAnything encoder-prefix tensors from local safetensors."""

    root_path = Path(root)
    if config is None:
        model_config = read_mapanything_model_config(root_path / "config.json")
        config = mapanything_encoder_prefix_config_from_model_config(model_config)
    tensors = load_checkpoint_tensors(
        root_path / "model.safetensors",
        names=MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS,
    )
    loaded = {name: tensors[name].astype(dtype) for name in MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS}
    validate_mapanything_encoder_prefix_weights(loaded, config)
    return loaded


def mapanything_full_encoder_required_keys(
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> tuple[str, ...]:
    """Return checkpoint keys required for the configured full DINOv2 encoder."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    keys = list(MAPANYTHING_ENCODER_BASE_REQUIRED_KEYS)
    for block_index in range(cfg.keep_first_n_layers):
        keys.extend(_block_key(block_index, suffix) for suffix in MAPANYTHING_ENCODER_BLOCK_SUFFIXES)
    return tuple(keys)


def load_mapanything_full_encoder_weights(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
    *,
    config: MapAnythingEncoderPrefixConfig | None = None,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    """Load full MapAnything DINOv2 encoder tensors from local safetensors."""

    root_path = Path(root)
    if config is None:
        model_config = read_mapanything_model_config(root_path / "config.json")
        config = mapanything_encoder_prefix_config_from_model_config(model_config)
    required = mapanything_full_encoder_required_keys(config)
    tensors = load_checkpoint_tensors(root_path / "model.safetensors", names=required)
    loaded = {name: tensors[name].astype(dtype) for name in required}
    validate_mapanything_full_encoder_weights(loaded, config)
    return loaded


def mapanything_info_sharing_required_keys(
    config: MapAnythingInfoSharingConfig | None = None,
) -> tuple[str, ...]:
    """Return checkpoint keys required for the configured info-sharing stage."""

    cfg = config or MapAnythingInfoSharingConfig()
    keys = list(MAPANYTHING_INFO_SHARING_BASE_REQUIRED_KEYS)
    for block_index in range(cfg.depth):
        keys.extend(_info_block_key(block_index, suffix) for suffix in MAPANYTHING_INFO_SHARING_BLOCK_SUFFIXES)
    return tuple(keys)


def load_mapanything_info_sharing_weights(
    root: str | Path = MAPANYTHING_DEFAULT_ROOT,
    *,
    config: MapAnythingInfoSharingConfig | None = None,
    dtype: mx.Dtype = mx.float32,
) -> dict[str, mx.array]:
    """Load MapAnything info-sharing tensors from local safetensors."""

    root_path = Path(root)
    if config is None:
        model_config = read_mapanything_model_config(root_path / "config.json")
        config = mapanything_info_sharing_config_from_model_config(model_config)
    required = mapanything_info_sharing_required_keys(config)
    tensors = load_checkpoint_tensors(root_path / "model.safetensors", names=required)
    loaded = {name: tensors[name].astype(dtype) for name in required}
    validate_mapanything_info_sharing_weights(loaded, config)
    return loaded


def validate_mapanything_encoder_prefix_weights(
    weights: Mapping[str, mx.array],
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> None:
    """Validate required key presence and shapes for the encoder-prefix boundary."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    missing = tuple(name for name in MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS if name not in weights)
    if missing:
        raise ValueError(f"missing MapAnything encoder-prefix tensors: {missing}")

    hidden = cfg.swiglu_hidden_features
    expected_shapes: dict[str, tuple[int | None, ...]] = {
        "encoder.model.cls_token": (1, 1, cfg.embed_dim),
        "encoder.model.pos_embed": (1, None, cfg.embed_dim),
        "encoder.model.patch_embed.proj.weight": (
            cfg.embed_dim,
            3,
            cfg.patch_size,
            cfg.patch_size,
        ),
        "encoder.model.patch_embed.proj.bias": (cfg.embed_dim,),
        "encoder.model.blocks.0.norm1.weight": (cfg.embed_dim,),
        "encoder.model.blocks.0.norm1.bias": (cfg.embed_dim,),
        "encoder.model.blocks.0.attn.qkv.weight": (3 * cfg.embed_dim, cfg.embed_dim),
        "encoder.model.blocks.0.attn.qkv.bias": (3 * cfg.embed_dim,),
        "encoder.model.blocks.0.attn.proj.weight": (cfg.embed_dim, cfg.embed_dim),
        "encoder.model.blocks.0.attn.proj.bias": (cfg.embed_dim,),
        "encoder.model.blocks.0.ls1.gamma": (cfg.embed_dim,),
        "encoder.model.blocks.0.norm2.weight": (cfg.embed_dim,),
        "encoder.model.blocks.0.norm2.bias": (cfg.embed_dim,),
        "encoder.model.blocks.0.mlp.w12.weight": (2 * hidden, cfg.embed_dim),
        "encoder.model.blocks.0.mlp.w12.bias": (2 * hidden,),
        "encoder.model.blocks.0.mlp.w3.weight": (cfg.embed_dim, hidden),
        "encoder.model.blocks.0.mlp.w3.bias": (cfg.embed_dim,),
        "encoder.model.blocks.0.ls2.gamma": (cfg.embed_dim,),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(int(dim) for dim in weights[name].shape)
        if len(actual) != len(expected) or any(
            expected_dim is not None and expected_dim != actual_dim
            for actual_dim, expected_dim in zip(actual, expected)
        ):
            raise ValueError(f"{name} has shape {actual}, expected {expected}")

    pos_tokens = int(weights["encoder.model.pos_embed"].shape[1]) - 1
    if pos_tokens <= 0 or int(math.isqrt(pos_tokens)) ** 2 != pos_tokens:
        raise ValueError(
            "encoder.model.pos_embed must contain one cls token plus a square patch-position grid"
        )


def validate_mapanything_full_encoder_weights(
    weights: Mapping[str, mx.array],
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> None:
    """Validate key presence and shapes for the configured full DINOv2 encoder."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    missing = tuple(name for name in mapanything_full_encoder_required_keys(cfg) if name not in weights)
    if missing:
        raise ValueError(f"missing MapAnything full-encoder tensors: {missing}")
    validate_mapanything_encoder_prefix_weights(weights, cfg)

    hidden = cfg.swiglu_hidden_features
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for block_index in range(cfg.keep_first_n_layers):
        expected_shapes.update(
            {
                _block_key(block_index, "norm1.weight"): (cfg.embed_dim,),
                _block_key(block_index, "norm1.bias"): (cfg.embed_dim,),
                _block_key(block_index, "attn.qkv.weight"): (3 * cfg.embed_dim, cfg.embed_dim),
                _block_key(block_index, "attn.qkv.bias"): (3 * cfg.embed_dim,),
                _block_key(block_index, "attn.proj.weight"): (cfg.embed_dim, cfg.embed_dim),
                _block_key(block_index, "attn.proj.bias"): (cfg.embed_dim,),
                _block_key(block_index, "ls1.gamma"): (cfg.embed_dim,),
                _block_key(block_index, "norm2.weight"): (cfg.embed_dim,),
                _block_key(block_index, "norm2.bias"): (cfg.embed_dim,),
                _block_key(block_index, "mlp.w12.weight"): (2 * hidden, cfg.embed_dim),
                _block_key(block_index, "mlp.w12.bias"): (2 * hidden,),
                _block_key(block_index, "mlp.w3.weight"): (cfg.embed_dim, hidden),
                _block_key(block_index, "mlp.w3.bias"): (cfg.embed_dim,),
                _block_key(block_index, "ls2.gamma"): (cfg.embed_dim,),
            }
        )
    for name, expected in expected_shapes.items():
        actual = tuple(int(dim) for dim in weights[name].shape)
        if actual != expected:
            raise ValueError(f"{name} has shape {actual}, expected {expected}")


def validate_mapanything_info_sharing_weights(
    weights: Mapping[str, mx.array],
    config: MapAnythingInfoSharingConfig | None = None,
) -> None:
    """Validate key presence and shapes for the configured info-sharing stage."""

    cfg = config or MapAnythingInfoSharingConfig()
    if cfg.input_embed_dim != cfg.dim:
        raise ValueError("MapAnything info-sharing MLX port currently supports identity proj_embed only")
    if cfg.use_pe_for_non_reference_views:
        raise ValueError("MapAnything info-sharing MLX port currently supports reference-view PE only")
    missing = tuple(name for name in mapanything_info_sharing_required_keys(cfg) if name not in weights)
    if missing:
        raise ValueError(f"missing MapAnything info-sharing tensors: {missing}")

    hidden = cfg.swiglu_hidden_features
    expected_shapes: dict[str, tuple[int, ...]] = {
        "scale_token": (cfg.input_embed_dim,),
        "info_sharing.norm.weight": (cfg.dim,),
        "info_sharing.norm.bias": (cfg.dim,),
        "info_sharing.view_pos_table": (1, cfg.dim),
    }
    for block_index in range(cfg.depth):
        expected_shapes.update(
            {
                _info_block_key(block_index, "norm1.weight"): (cfg.dim,),
                _info_block_key(block_index, "norm1.bias"): (cfg.dim,),
                _info_block_key(block_index, "attn.qkv.weight"): (3 * cfg.dim, cfg.dim),
                _info_block_key(block_index, "attn.qkv.bias"): (3 * cfg.dim,),
                _info_block_key(block_index, "attn.proj.weight"): (cfg.dim, cfg.dim),
                _info_block_key(block_index, "attn.proj.bias"): (cfg.dim,),
                _info_block_key(block_index, "ls1.gamma"): (cfg.dim,),
                _info_block_key(block_index, "norm2.weight"): (cfg.dim,),
                _info_block_key(block_index, "norm2.bias"): (cfg.dim,),
                _info_block_key(block_index, "mlp.w12.weight"): (2 * hidden, cfg.dim),
                _info_block_key(block_index, "mlp.w12.bias"): (2 * hidden,),
                _info_block_key(block_index, "mlp.w3.weight"): (cfg.dim, hidden),
                _info_block_key(block_index, "mlp.w3.bias"): (cfg.dim,),
                _info_block_key(block_index, "ls2.gamma"): (cfg.dim,),
            }
        )
    for name, expected in expected_shapes.items():
        actual = tuple(int(dim) for dim in weights[name].shape)
        if actual != expected:
            raise ValueError(f"{name} has shape {actual}, expected {expected}")
    _feature_take_indices(cfg.depth, cfg.indices)


def run_mapanything_encoder_prefix(
    images: mx.array,
    weights: Mapping[str, mx.array],
    *,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> MapAnythingEncoderPrefixOutput:
    """Run the MLX DINOv2 patch embedding and encoder block 0."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    validate_mapanything_encoder_prefix_weights(weights, cfg)
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
    batch, channels, height, width = tuple(int(dim) for dim in images.shape)
    if channels != 3:
        raise ValueError(f"MapAnything images must have 3 channels, got {channels}")
    if height % cfg.patch_size != 0 or width % cfg.patch_size != 0:
        raise ValueError(
            f"image height/width must be divisible by patch_size={cfg.patch_size}, got {(height, width)}"
        )

    values = images.astype(mx.float32)
    patch_embeddings = mapanything_patch_embed(values, weights, config=cfg)
    tokens = add_mapanything_dinov2_cls_and_position(
        patch_embeddings,
        weights,
        image_height=height,
        image_width=width,
        config=cfg,
    )
    block0 = mapanything_encoder_block0(tokens, weights, config=cfg)
    patch_grid = (height // cfg.patch_size, width // cfg.patch_size)
    return MapAnythingEncoderPrefixOutput(
        patch_embeddings=patch_embeddings,
        tokens_with_position=tokens,
        block0=block0,
        patch_grid=patch_grid,
        trace={
            "stage": "encoder-prefix",
            "runtime_depends_on_torch": False,
            "batch": batch,
            "patch_grid": patch_grid,
            "patch_size": cfg.patch_size,
            "embed_dim": cfg.embed_dim,
            "num_heads": cfg.num_heads,
            "data_norm_type": cfg.data_norm_type,
            "implemented_layers": ("patch_embed", "block0"),
        },
    )


def run_mapanything_full_encoder(
    images: mx.array,
    weights: Mapping[str, mx.array],
    *,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> MapAnythingEncoderOutput:
    """Run the full MLX DINOv2 image encoder used by MapAnything."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    validate_mapanything_full_encoder_weights(weights, cfg)
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
    batch, channels, height, width = tuple(int(dim) for dim in images.shape)
    if channels != 3:
        raise ValueError(f"MapAnything images must have 3 channels, got {channels}")
    if height % cfg.patch_size != 0 or width % cfg.patch_size != 0:
        raise ValueError(
            f"image height/width must be divisible by patch_size={cfg.patch_size}, got {(height, width)}"
        )

    values = images.astype(mx.float32)
    patch_embeddings = mapanything_patch_embed(values, weights, config=cfg)
    tokens_with_position = add_mapanything_dinov2_cls_and_position(
        patch_embeddings,
        weights,
        image_height=height,
        image_width=width,
        config=cfg,
    )
    hidden = tokens_with_position
    block0 = None
    for block_index in range(cfg.keep_first_n_layers):
        hidden = mapanything_encoder_block(hidden, weights, block_index=block_index, config=cfg)
        if block_index == 0:
            block0 = hidden
    if block0 is None:
        raise ValueError("keep_first_n_layers must be at least 1")

    patch_grid = (height // cfg.patch_size, width // cfg.patch_size)
    patch_tokens = hidden[:, 1:, :]
    features = mx.reshape(patch_tokens, (batch, patch_grid[0], patch_grid[1], cfg.embed_dim))
    features = mx.transpose(features, (0, 3, 1, 2))
    registers = mx.transpose(hidden[:, :1, :], (0, 2, 1))
    return MapAnythingEncoderOutput(
        patch_embeddings=patch_embeddings,
        tokens_with_position=tokens_with_position,
        block0=block0,
        final_tokens=hidden,
        features=features,
        registers=registers,
        patch_grid=patch_grid,
        trace={
            "stage": "full-encoder",
            "runtime_depends_on_torch": False,
            "batch": batch,
            "patch_grid": patch_grid,
            "patch_size": cfg.patch_size,
            "embed_dim": cfg.embed_dim,
            "num_heads": cfg.num_heads,
            "data_norm_type": cfg.data_norm_type,
            "implemented_layers": tuple(range(cfg.keep_first_n_layers)),
        },
    )


def run_mapanything_info_sharing(
    features: Sequence[mx.array],
    weights: Mapping[str, mx.array],
    *,
    additional_tokens_per_view: Sequence[mx.array] | None = None,
    additional_tokens: mx.array | None = None,
    use_checkpoint_scale_token: bool = True,
    config: MapAnythingInfoSharingConfig | None = None,
) -> MapAnythingInfoSharingResult:
    """Run the MLX multi-view alternating attention transformer used by MapAnything."""

    cfg = config or MapAnythingInfoSharingConfig()
    validate_mapanything_info_sharing_weights(weights, cfg)
    feature_tuple = tuple(features)
    if not feature_tuple:
        raise ValueError("features must contain at least one view")

    batch, channels, height, width = _validate_info_sharing_features(feature_tuple, cfg)
    per_view_tokens = None
    per_view_token_count = 0
    if additional_tokens_per_view is not None:
        per_view_tokens = tuple(additional_tokens_per_view)
        per_view_token_count = _validate_info_sharing_per_view_tokens(
            per_view_tokens,
            batch=batch,
            view_count=len(feature_tuple),
            config=cfg,
        )

    if additional_tokens is None and use_checkpoint_scale_token:
        scale_token = mx.reshape(weights["scale_token"].astype(mx.float32), (1, cfg.input_embed_dim, 1))
        additional_tokens = mx.broadcast_to(scale_token, (batch, cfg.input_embed_dim, 1))
    if additional_tokens is not None:
        if additional_tokens.ndim != 3:
            raise ValueError(f"additional_tokens must have shape [B, C, T], got {tuple(additional_tokens.shape)}")
        token_shape = tuple(int(dim) for dim in additional_tokens.shape)
        if token_shape[0] != batch or token_shape[1] != cfg.input_embed_dim:
            raise ValueError(
                "additional_tokens must match feature batch/channel dimensions, "
                f"got {token_shape}, expected batch={batch}, channels={cfg.input_embed_dim}"
            )

    view_count = len(feature_tuple)
    spatial_token_count = height * width
    tokens_per_view = spatial_token_count + per_view_token_count
    hidden = _pack_info_sharing_tokens(
        feature_tuple,
        per_view_tokens,
        additional_tokens,
        batch=batch,
        height=height,
        width=width,
        config=cfg,
    )
    hidden = _add_info_sharing_view_positional_encoding(
        hidden,
        weights,
        batch=batch,
        view_count=view_count,
        tokens_per_view=tokens_per_view,
        config=cfg,
    )

    take_indices = set(_feature_take_indices(cfg.depth, cfg.indices))
    intermediate_outputs: list[MapAnythingInfoSharingOutput] = []
    has_global_tokens = additional_tokens is not None
    view_token_count = view_count * tokens_per_view
    for block_index in range(cfg.depth):
        if block_index % 2 == 0:
            hidden = _info_sharing_block(hidden, weights, block_index=block_index, config=cfg)
        else:
            global_tokens = None
            view_hidden = hidden
            if has_global_tokens:
                global_tokens = hidden[:, view_token_count:, :]
                view_hidden = hidden[:, :view_token_count, :]
            view_hidden = mx.reshape(view_hidden, (batch * view_count, tokens_per_view, cfg.dim))
            view_hidden = _info_sharing_block(view_hidden, weights, block_index=block_index, config=cfg)
            view_hidden = mx.reshape(view_hidden, (batch, view_token_count, cfg.dim))
            hidden = (
                mx.concatenate((view_hidden, global_tokens), axis=1)
                if global_tokens is not None
                else view_hidden
            )

        if block_index in take_indices:
            selected = (
                _layer_norm(
                    hidden,
                    weights["info_sharing.norm.weight"],
                    weights["info_sharing.norm.bias"],
                    eps=cfg.layer_norm_eps,
                )
                if cfg.norm_intermediate
                else hidden
            )
            intermediate_outputs.append(
                _unpack_info_sharing_tokens(
                    selected,
                    batch=batch,
                    view_count=view_count,
                    height=height,
                    width=width,
                    tokens_per_view=tokens_per_view,
                    per_view_token_count=per_view_token_count,
                    has_global_tokens=has_global_tokens,
                    config=cfg,
                )
            )

    final_hidden = _layer_norm(
        hidden,
        weights["info_sharing.norm.weight"],
        weights["info_sharing.norm.bias"],
        eps=cfg.layer_norm_eps,
    )
    final = _unpack_info_sharing_tokens(
        final_hidden,
        batch=batch,
        view_count=view_count,
        height=height,
        width=width,
        tokens_per_view=tokens_per_view,
        per_view_token_count=per_view_token_count,
        has_global_tokens=has_global_tokens,
        config=cfg,
    )
    return MapAnythingInfoSharingResult(
        final=final,
        intermediates=tuple(intermediate_outputs),
        trace={
            "stage": "info-sharing",
            "runtime_depends_on_torch": False,
            "batch": batch,
            "view_count": view_count,
            "patch_grid": (height, width),
            "spatial_tokens_per_view": spatial_token_count,
            "additional_tokens_per_view": per_view_token_count,
            "global_additional_tokens": int(additional_tokens.shape[2]) if additional_tokens is not None else 0,
            "depth": cfg.depth,
            "indices": cfg.indices,
            "attention_schedule": "even-global/odd-frame",
        },
    )


def mapanything_patch_embed(
    images: mx.array,
    weights: Mapping[str, mx.array],
    *,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> mx.array:
    """Apply the DINOv2 patch embedding as ``Conv2d -> flatten -> transpose``."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    x_nhwc = mx.transpose(images, (0, 2, 3, 1))
    weight = weights["encoder.model.patch_embed.proj.weight"].astype(x_nhwc.dtype)
    bias = weights["encoder.model.patch_embed.proj.bias"].astype(x_nhwc.dtype)
    weight_ohwi = mx.transpose(weight, (0, 2, 3, 1))
    embedded = mx.conv2d(x_nhwc, weight_ohwi, stride=cfg.patch_size)
    embedded = embedded + bias[None, None, None, :]
    batch, patch_h, patch_w, embed_dim = tuple(int(dim) for dim in embedded.shape)
    return mx.reshape(embedded, (batch, patch_h * patch_w, embed_dim))


def add_mapanything_dinov2_cls_and_position(
    patch_embeddings: mx.array,
    weights: Mapping[str, mx.array],
    *,
    image_height: int,
    image_width: int,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> mx.array:
    """Add DINOv2 cls token and positional embeddings, including rectangular interpolation."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    batch = int(patch_embeddings.shape[0])
    cls_token = weights["encoder.model.cls_token"].astype(patch_embeddings.dtype)
    cls_tokens = mx.broadcast_to(cls_token, (batch, 1, cfg.embed_dim))
    tokens = mx.concatenate((cls_tokens, patch_embeddings), axis=1)
    pos_embed = interpolate_mapanything_dinov2_pos_embed(
        weights["encoder.model.pos_embed"].astype(patch_embeddings.dtype),
        token_count=int(tokens.shape[1]),
        image_height=image_height,
        image_width=image_width,
        config=cfg,
    )
    return tokens + pos_embed.astype(tokens.dtype)


def interpolate_mapanything_dinov2_pos_embed(
    pos_embed: mx.array,
    *,
    token_count: int,
    image_height: int,
    image_width: int,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> mx.array:
    """Match DINOv2 bicubic positional embedding interpolation for rectangular inputs."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    patch_count = token_count - 1
    source_patch_count = int(pos_embed.shape[1]) - 1
    grid_h = image_height // cfg.patch_size
    grid_w = image_width // cfg.patch_size
    if patch_count != grid_h * grid_w:
        raise ValueError(
            f"token_count={token_count} is inconsistent with patch grid {(grid_h, grid_w)}"
        )
    if patch_count == source_patch_count and image_height == image_width:
        return pos_embed

    source_grid = int(math.isqrt(source_patch_count))
    if source_grid * source_grid != source_patch_count:
        raise ValueError("pos_embed patch tokens must form a square source grid")

    class_pos_embed = pos_embed[:, :1, :]
    patch_pos_embed = mx.reshape(
        pos_embed[:, 1:, :],
        (1, source_grid, source_grid, int(pos_embed.shape[-1])),
    )
    patch_pos_embed = _bicubic_interpolate_nhwc(
        patch_pos_embed.astype(mx.float32),
        output_size=(grid_h, grid_w),
        scale_factor=(
            (grid_h + cfg.pos_interpolate_offset) / source_grid,
            (grid_w + cfg.pos_interpolate_offset) / source_grid,
        ),
    ).astype(pos_embed.dtype)
    patch_pos_embed = mx.reshape(patch_pos_embed, (1, grid_h * grid_w, int(pos_embed.shape[-1])))
    return mx.concatenate((class_pos_embed, patch_pos_embed), axis=1)


def mapanything_encoder_block0(
    tokens: mx.array,
    weights: Mapping[str, mx.array],
    *,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> mx.array:
    """Run DINOv2 encoder block 0 in inference mode."""

    return mapanything_encoder_block(tokens, weights, block_index=0, config=config)


def mapanything_encoder_block(
    tokens: mx.array,
    weights: Mapping[str, mx.array],
    *,
    block_index: int,
    config: MapAnythingEncoderPrefixConfig | None = None,
) -> mx.array:
    """Run one DINOv2 encoder block in inference mode."""

    cfg = config or MapAnythingEncoderPrefixConfig()
    attn_input = _layer_norm(
        tokens,
        weights[_block_key(block_index, "norm1.weight")],
        weights[_block_key(block_index, "norm1.bias")],
        eps=cfg.layer_norm_eps,
    )
    hidden = tokens + _apply_layer_scale(
        _self_attention(attn_input, weights, block_index=block_index, config=cfg),
        weights[_block_key(block_index, "ls1.gamma")],
    )
    ffn_input = _layer_norm(
        hidden,
        weights[_block_key(block_index, "norm2.weight")],
        weights[_block_key(block_index, "norm2.bias")],
        eps=cfg.layer_norm_eps,
    )
    hidden = hidden + _apply_layer_scale(
        _swiglu_ffn(ffn_input, weights, block_index=block_index),
        weights[_block_key(block_index, "ls2.gamma")],
    )
    return hidden


def mapanything_encoder_prefix_required_keys() -> tuple[str, ...]:
    """Return the official checkpoint keys required for the first MLX boundary."""

    return MAPANYTHING_ENCODER_PREFIX_REQUIRED_KEYS


def _self_attention(
    hidden_states: mx.array,
    weights: Mapping[str, mx.array],
    *,
    block_index: int,
    config: MapAnythingEncoderPrefixConfig,
) -> mx.array:
    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
    head_dim = config.head_dim
    qkv = _linear(
        hidden_states,
        weights[_block_key(block_index, "attn.qkv.weight")],
        weights[_block_key(block_index, "attn.qkv.bias")],
    )
    qkv = mx.reshape(qkv, (batch, token_count, 3, config.num_heads, head_dim))
    qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
    query = qkv[0]
    key = qkv[1]
    value = qkv[2]
    attended = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=head_dim**-0.5,
    )
    attended = mx.reshape(
        mx.transpose(attended, (0, 2, 1, 3)),
        (batch, token_count, config.embed_dim),
    )
    return _linear(
        attended,
        weights[_block_key(block_index, "attn.proj.weight")],
        weights[_block_key(block_index, "attn.proj.bias")],
    )


def _swiglu_ffn(
    hidden_states: mx.array,
    weights: Mapping[str, mx.array],
    *,
    block_index: int,
) -> mx.array:
    x12 = _linear(
        hidden_states,
        weights[_block_key(block_index, "mlp.w12.weight")],
        weights[_block_key(block_index, "mlp.w12.bias")],
    )
    x1, x2 = mx.split(x12, 2, axis=-1)
    gated = (x1 * mx.sigmoid(x1)) * x2
    return _linear(
        gated,
        weights[_block_key(block_index, "mlp.w3.weight")],
        weights[_block_key(block_index, "mlp.w3.bias")],
    )


def _block_key(block_index: int, suffix: str) -> str:
    return f"encoder.model.blocks.{block_index}.{suffix}"


def _info_block_key(block_index: int, suffix: str) -> str:
    return f"info_sharing.self_attention_blocks.{block_index}.{suffix}"


def _validate_info_sharing_features(
    features: tuple[mx.array, ...],
    config: MapAnythingInfoSharingConfig,
) -> tuple[int, int, int, int]:
    first = features[0]
    if first.ndim != 4:
        raise ValueError(f"features must have shape [B, C, H, W], got {tuple(first.shape)}")
    batch, channels, height, width = tuple(int(dim) for dim in first.shape)
    if channels != config.input_embed_dim:
        raise ValueError(f"features channel dimension must be {config.input_embed_dim}, got {channels}")
    for index, feature in enumerate(features):
        shape = tuple(int(dim) for dim in feature.shape)
        if feature.ndim != 4 or shape != (batch, channels, height, width):
            raise ValueError(
                "all info-sharing features must have matching [B, C, H, W] shape, "
                f"view 0 has {(batch, channels, height, width)}, view {index} has {shape}"
            )
    return batch, channels, height, width


def _validate_info_sharing_per_view_tokens(
    tokens: tuple[mx.array, ...],
    *,
    batch: int,
    view_count: int,
    config: MapAnythingInfoSharingConfig,
) -> int:
    if len(tokens) != view_count:
        raise ValueError(f"expected {view_count} per-view token arrays, got {len(tokens)}")
    first = tokens[0]
    if first.ndim != 3:
        raise ValueError(f"per-view tokens must have shape [B, C, T], got {tuple(first.shape)}")
    token_batch, channels, token_count = tuple(int(dim) for dim in first.shape)
    if token_batch != batch or channels != config.input_embed_dim:
        raise ValueError(
            "per-view tokens must match feature batch/channel dimensions, "
            f"got {(token_batch, channels, token_count)}, expected batch={batch}, channels={config.input_embed_dim}"
        )
    for index, value in enumerate(tokens):
        shape = tuple(int(dim) for dim in value.shape)
        if value.ndim != 3 or shape != (batch, channels, token_count):
            raise ValueError(
                "all per-view token arrays must have matching [B, C, T] shape, "
                f"view 0 has {(batch, channels, token_count)}, view {index} has {shape}"
            )
    return token_count


def _pack_info_sharing_tokens(
    features: tuple[mx.array, ...],
    per_view_tokens: tuple[mx.array, ...] | None,
    additional_tokens: mx.array | None,
    *,
    batch: int,
    height: int,
    width: int,
    config: MapAnythingInfoSharingConfig,
) -> mx.array:
    view_count = len(features)
    spatial_tokens = height * width
    if per_view_tokens is not None:
        view_sequences = []
        for feature, tokens in zip(features, per_view_tokens):
            feature_flat = mx.reshape(feature.astype(mx.float32), (batch, config.input_embed_dim, spatial_tokens))
            view_sequences.append(mx.concatenate((feature_flat, tokens.astype(mx.float32)), axis=2))
        hidden = mx.stack(view_sequences, axis=1)
        hidden = mx.transpose(hidden, (0, 1, 3, 2))
        hidden = mx.reshape(hidden, (batch, view_count * int(hidden.shape[2]), config.input_embed_dim))
    else:
        hidden = mx.stack([feature.astype(mx.float32) for feature in features], axis=1)
        hidden = mx.transpose(hidden, (0, 1, 3, 4, 2))
        hidden = mx.reshape(hidden, (batch, view_count * spatial_tokens, config.input_embed_dim))

    if additional_tokens is not None:
        hidden = mx.concatenate((hidden, mx.transpose(additional_tokens.astype(mx.float32), (0, 2, 1))), axis=1)
    return hidden


def _add_info_sharing_view_positional_encoding(
    hidden: mx.array,
    weights: Mapping[str, mx.array],
    *,
    batch: int,
    view_count: int,
    tokens_per_view: int,
    config: MapAnythingInfoSharingConfig,
) -> mx.array:
    if not config.distinguish_ref_and_non_ref_views:
        return hidden
    ref_pe = mx.reshape(weights["info_sharing.view_pos_table"][0].astype(hidden.dtype), (1, 1, config.dim))
    ref_pe = mx.broadcast_to(ref_pe, (batch, tokens_per_view, config.dim))
    ref_features = hidden[:, :tokens_per_view, :] + ref_pe
    non_ref_features = hidden[:, tokens_per_view : view_count * tokens_per_view, :]
    if hidden.shape[1] > view_count * tokens_per_view:
        additional_features = hidden[:, view_count * tokens_per_view :, :]
        return mx.concatenate((ref_features, non_ref_features, additional_features), axis=1)
    return mx.concatenate((ref_features, non_ref_features), axis=1)


def _info_sharing_block(
    tokens: mx.array,
    weights: Mapping[str, mx.array],
    *,
    block_index: int,
    config: MapAnythingInfoSharingConfig,
) -> mx.array:
    prefix = f"info_sharing.self_attention_blocks.{block_index}"
    attn_input = _layer_norm(
        tokens,
        weights[f"{prefix}.norm1.weight"],
        weights[f"{prefix}.norm1.bias"],
        eps=config.layer_norm_eps,
    )
    hidden = tokens + _apply_layer_scale(
        _self_attention_with_prefix(
            attn_input,
            weights,
            prefix=prefix,
            dim=config.dim,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
        ),
        weights[f"{prefix}.ls1.gamma"],
    )
    ffn_input = _layer_norm(
        hidden,
        weights[f"{prefix}.norm2.weight"],
        weights[f"{prefix}.norm2.bias"],
        eps=config.layer_norm_eps,
    )
    hidden = hidden + _apply_layer_scale(
        _swiglu_ffn_with_prefix(ffn_input, weights, prefix=prefix),
        weights[f"{prefix}.ls2.gamma"],
    )
    return hidden


def _self_attention_with_prefix(
    hidden_states: mx.array,
    weights: Mapping[str, mx.array],
    *,
    prefix: str,
    dim: int,
    num_heads: int,
    head_dim: int,
) -> mx.array:
    batch, token_count, _ = tuple(int(dim_value) for dim_value in hidden_states.shape)
    qkv = _linear(
        hidden_states,
        weights[f"{prefix}.attn.qkv.weight"],
        weights[f"{prefix}.attn.qkv.bias"],
    )
    qkv = mx.reshape(qkv, (batch, token_count, 3, num_heads, head_dim))
    qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
    query = qkv[0]
    key = qkv[1]
    value = qkv[2]
    attended = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=head_dim**-0.5,
    )
    attended = mx.reshape(
        mx.transpose(attended, (0, 2, 1, 3)),
        (batch, token_count, dim),
    )
    return _linear(
        attended,
        weights[f"{prefix}.attn.proj.weight"],
        weights[f"{prefix}.attn.proj.bias"],
    )


def _swiglu_ffn_with_prefix(
    hidden_states: mx.array,
    weights: Mapping[str, mx.array],
    *,
    prefix: str,
) -> mx.array:
    x12 = _linear(
        hidden_states,
        weights[f"{prefix}.mlp.w12.weight"],
        weights[f"{prefix}.mlp.w12.bias"],
    )
    x1, x2 = mx.split(x12, 2, axis=-1)
    gated = (x1 * mx.sigmoid(x1)) * x2
    return _linear(
        gated,
        weights[f"{prefix}.mlp.w3.weight"],
        weights[f"{prefix}.mlp.w3.bias"],
    )


def _unpack_info_sharing_tokens(
    hidden: mx.array,
    *,
    batch: int,
    view_count: int,
    height: int,
    width: int,
    tokens_per_view: int,
    per_view_token_count: int,
    has_global_tokens: bool,
    config: MapAnythingInfoSharingConfig,
) -> MapAnythingInfoSharingOutput:
    view_token_count = view_count * tokens_per_view
    view_flat = hidden[:, :view_token_count, :]
    per_view_outputs = None
    if per_view_token_count:
        view_with_tokens = mx.reshape(view_flat, (batch, view_count, tokens_per_view, config.dim))
        spatial = view_with_tokens[:, :, : height * width, :]
        per_view = view_with_tokens[:, :, height * width :, :]
        view_features = mx.reshape(spatial, (batch, view_count, height, width, config.dim))
        view_features = mx.transpose(view_features, (0, 1, 4, 2, 3))
        per_view_outputs = tuple(
            mx.transpose(per_view[:, view_index, :, :], (0, 2, 1)) for view_index in range(view_count)
        )
    else:
        view_features = mx.reshape(view_flat, (batch, view_count, height, width, config.dim))
        view_features = mx.transpose(view_features, (0, 1, 4, 2, 3))
    features = tuple(view_features[:, view_index, :, :, :] for view_index in range(view_count))

    global_output = None
    if has_global_tokens:
        global_output = mx.transpose(hidden[:, view_token_count:, :], (0, 2, 1))
    return MapAnythingInfoSharingOutput(
        features=features,
        additional_token_features=global_output,
        additional_token_features_per_view=per_view_outputs,
    )


def _feature_take_indices(num_features: int, indices: tuple[int, ...] | int | None) -> tuple[int, ...]:
    if indices is None:
        return tuple(range(num_features))
    if isinstance(indices, int):
        if indices <= 0 or indices > num_features:
            raise ValueError(f"last-n ({indices}) is out of range (1 to {num_features})")
        return tuple(num_features - indices + offset for offset in range(indices))
    result = []
    for index in indices:
        resolved = num_features + index if index < 0 else index
        if resolved < 0 or resolved >= num_features:
            raise ValueError(f"feature index {resolved} is out of range (0 to {num_features - 1})")
        result.append(resolved)
    return tuple(result)


def _add_info_sharing_parity_tensors(
    tensors: dict[str, mx.array],
    prefix: str,
    output: MapAnythingInfoSharingOutput,
) -> None:
    for index, feature in enumerate(output.features):
        tensors[f"{prefix}.features.{index}"] = feature
    if output.additional_token_features is not None:
        tensors[f"{prefix}.additional_token_features"] = output.additional_token_features
    if output.additional_token_features_per_view is not None:
        for index, tokens in enumerate(output.additional_token_features_per_view):
            tensors[f"{prefix}.additional_token_features_per_view.{index}"] = tokens


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight.astype(values.dtype))
    if bias is not None:
        output = output + bias.astype(output.dtype)
    return output


def _layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    values = values.astype(mx.float32)
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight.astype(values.dtype) + bias.astype(values.dtype)


def _apply_layer_scale(values: mx.array, gamma: mx.array) -> mx.array:
    return values * gamma.astype(values.dtype)


def _bicubic_interpolate_nhwc(
    values: mx.array,
    *,
    output_size: tuple[int, int],
    scale_factor: tuple[float, float],
) -> mx.array:
    """PyTorch-compatible bicubic interpolation for DINOv2 pos embeddings."""

    _, input_h, input_w, _ = tuple(int(dim) for dim in values.shape)
    output_h, output_w = output_size
    row_indices, row_weights = _bicubic_indices_and_weights(
        input_h,
        output_h,
        scale_factor[0],
        dtype=values.dtype,
    )
    col_indices, col_weights = _bicubic_indices_and_weights(
        input_w,
        output_w,
        scale_factor[1],
        dtype=values.dtype,
    )
    rows = mx.take(values, row_indices, axis=1)
    rows = mx.sum(rows * row_weights[None, :, :, None, None], axis=2)
    cols = mx.take(rows, col_indices, axis=2)
    return mx.sum(cols * col_weights[None, None, :, :, None], axis=3)


def _bicubic_indices_and_weights(
    input_size: int,
    output_size: int,
    scale_factor: float,
    *,
    dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    source = (mx.arange(output_size, dtype=mx.float32) + 0.5) / float(scale_factor) - 0.5
    base = mx.floor(source).astype(mx.int32)
    offsets = mx.array([-1, 0, 1, 2], dtype=mx.int32)
    indices = base[:, None] + offsets[None, :]
    weights = _cubic_convolution1(source[:, None] - indices.astype(mx.float32)).astype(dtype)
    return mx.clip(indices, 0, input_size - 1), weights


def _cubic_convolution1(values: mx.array) -> mx.array:
    a = -0.75
    abs_values = mx.abs(values)
    squared = abs_values * abs_values
    cubed = squared * abs_values
    inside_one = (a + 2.0) * cubed - (a + 3.0) * squared + 1.0
    inside_two = a * cubed - 5.0 * a * squared + 8.0 * a * abs_values - 4.0 * a
    return mx.where(abs_values <= 1.0, inside_one, mx.where(abs_values < 2.0, inside_two, 0.0))


def mapanything_encoder_prefix_outputs_for_parity(
    output: MapAnythingEncoderPrefixOutput,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, mx.array]:
    """Return named tensors for comparison with a MapAnything parity bundle."""

    tensors = output.parity_tensors
    if names is None:
        return tensors
    return {name: tensors[name] for name in names if name in tensors}


def mapanything_full_encoder_outputs_for_parity(
    output: MapAnythingEncoderOutput,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, mx.array]:
    """Return named full-encoder tensors for comparison with a reference bundle."""

    tensors = output.parity_tensors
    if names is None:
        return tensors
    return {name: tensors[name] for name in names if name in tensors}


def mapanything_info_sharing_outputs_for_parity(
    output: MapAnythingInfoSharingResult,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, mx.array]:
    """Return named info-sharing tensors for comparison with a reference bundle."""

    tensors = output.parity_tensors
    if names is None:
        return tensors
    return {name: tensors[name] for name in names if name in tensors}
