"""Runtime boundary for checkpoint-backed LiTo generation.

This module intentionally has no dependency on Torch, upstream ``lito``, CUDA,
or vendored runtime code. The real path is a direct safetensors-to-MLX port; the
vendored project is source reference only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


_DIT_CHECKPOINT = Path("image_to_3d/lito_dit_rgba.safetensors")
_TOKENIZER_CHECKPOINT = Path("tokenizer/lito_new.safetensors")
_DIT_EMA_PREFIX = "velocity_estimator_ema.module."
_DIT_SOURCE_PREFIX = "velocity_estimator."
_PATCH_ENCODER_PREFIX = "patch_encoder."
_GS_PREFIX = "gs_decoder."
_VOXEL_PREFIX = "voxel_decoder."
_TRELLIS_SS_DECODER_CONFIG = Path("ckpts/ss_dec_conv3d_16l8_fp16.json")
_TRELLIS_SS_DECODER_CHECKPOINT = Path("ckpts/ss_dec_conv3d_16l8_fp16.safetensors")
_LITO_DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
_LITO_DINO_IMAGE_STD = (0.229, 0.224, 0.225)
_LITO_REAL_MAX_INIT_COORDS_BY_PROFILE = {
    "safe": 512,
    "balanced": 2048,
    "large": 8192,
}
LITO_INIT_COORD_CAP_PROFILE = "profile"
LITO_PLY_STORAGES = ("binary_little_endian", "ascii")
LITO_DEFAULT_PLY_STORAGE = "binary_little_endian"


@dataclass(frozen=True)
class LitoRealBackendConfig:
    """Configuration for a checkpoint-backed LiTo backend."""

    weights_root: Path
    asset_summary: Any
    memory_profile: str
    max_init_coords_per_batch: int | str | None = LITO_INIT_COORD_CAP_PROFILE
    raw_weights_root: Path | None = None
    allow_cuda: bool = False
    mlx_compute_dtype: str = "float16"


@dataclass(frozen=True)
class LitoRealGenerateRequest:
    """Inputs for one checkpoint-backed LiTo generation request."""

    cond_rgba: np.ndarray
    num_steps: int
    cfg_scale: float
    seed: int | None


@dataclass(frozen=True)
class LitoRealArchitectureInventory:
    """Header-only architecture facts inferred from converted LiTo weights."""

    weights_root: Path
    checkpoint_key_counts: dict[str, int]
    prefix_counts: dict[str, dict[str, int]]
    dit: dict[str, Any]
    gaussian_decoder: dict[str, Any]
    voxel_decoder: dict[str, Any]


@dataclass(frozen=True)
class LitoGaussianDecoderProfile:
    """Fixed LiTo Gaussian decode constants from the tokenizer config."""

    expansion_ratio: int = 64
    shape_dim: int = 10
    color_dim: int = 49
    rgb_sh_degree: int = 3
    region_scaling: float = 0.05
    scaling_logit_bias: float = 0.0
    scaling_scalar: float = 0.01
    min_scaling: float = 0.001
    opacity_logit_bias: float = 0.1
    opacity_logit_scale: float = 1.0
    min_opacity: float = 0.0
    max_opacity: float = 1.0


@dataclass(frozen=True)
class LitoPatchEncoderConfig:
    """Local MLX execution parameters for LiTo's DINO/RGBA patch encoder."""

    input_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    num_heads: int = 16
    num_blocks: int = 24
    register_count: int = 4
    normalize_concat_tokens: bool = True
    attention_chunk_size: int | None = 192
    max_attention_bytes: int = 1_600_000_000


class LitoBackendUnavailable(RuntimeError):
    """Raised when the checkpoint-backed LiTo backend cannot run locally."""


class DirectMlxLitoBackend:
    """Boundary for the local no-CUDA checkpoint-backed backend implementation."""

    def __init__(self, config: LitoRealBackendConfig, *, architecture: LitoRealArchitectureInventory) -> None:
        self.config = config
        self.architecture = architecture

    def load_dit_weight_arrays(
        self,
        *,
        names: Iterable[str] | None = None,
        dtype: Any = np.float32,
    ) -> dict[str, np.ndarray]:
        """Load remapped real DiT arrays for local MLX model construction."""

        return load_lito_dit_weight_arrays(self.config.weights_root, names=names, dtype=dtype)

    def load_patch_encoder_weight_arrays(
        self,
        *,
        names: Iterable[str] | None = None,
        dtype: Any = np.float32,
    ) -> dict[str, np.ndarray]:
        """Load remapped real DINO/RGBA patch encoder arrays for local MLX construction."""

        return load_lito_patch_encoder_weight_arrays(self.config.weights_root, names=names, dtype=dtype)

    def condition_rgba(
        self,
        cond_rgba: Any,
        *,
        weights: dict[str, Any] | None = None,
        block_indices: Iterable[int] | None = None,
        config: LitoPatchEncoderConfig = LitoPatchEncoderConfig(),
    ) -> mx.array:
        """Run LiTo's real DINO/RGBA image conditioner to DiT condition tokens."""

        active_weights = weights if weights is not None else self.load_patch_encoder_weight_arrays()
        return run_lito_patch_encoder_condition_tokens(
            cond_rgba,
            active_weights,
            block_indices=block_indices,
            config=config,
        )

    def run_dit_velocity(
        self,
        latent_tokens: Any,
        timestep: Any,
        condition_tokens: Any,
        *,
        weights: dict[str, Any] | None = None,
        block_indices: Iterable[int] | None = None,
        num_heads: int = 16,
        cond_drop_ids: Any | None = None,
    ) -> mx.array:
        """Run the real LiTo DiT velocity model for caller-supplied condition tokens."""

        active_weights = weights if weights is not None else self.load_dit_weight_arrays()
        return run_lito_dit_velocity(
            latent_tokens,
            timestep,
            condition_tokens,
            active_weights,
            block_indices=block_indices,
            num_heads=num_heads,
            cond_drop_ids=cond_drop_ids,
        )

    def sample_dit_latents(
        self,
        condition_tokens: Any,
        *,
        weights: dict[str, Any] | None = None,
        initial_latent: Any | None = None,
        seed: int | None = None,
        num_steps: int = 20,
        cfg_scale: float = 1.0,
        method: str = "heun",
        block_indices: Iterable[int] | None = None,
        num_heads: int = 16,
        latent_mean: float = 0.0661,
        latent_std: float = 1.6464,
        t_eps: float = 1e-4,
    ) -> mx.array:
        """Run LiTo DiT ODE sampling from caller-supplied condition tokens."""

        active_weights = weights if weights is not None else self.load_dit_weight_arrays()
        return sample_lito_dit_latents(
            condition_tokens,
            active_weights,
            initial_latent=initial_latent,
            seed=seed,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            method=method,
            block_indices=block_indices,
            num_heads=num_heads,
            latent_mean=latent_mean,
            latent_std=latent_std,
            t_eps=t_eps,
        )

    def load_gaussian_decoder_weight_arrays(
        self,
        *,
        names: Iterable[str] | None = None,
        dtype: Any = np.float32,
    ) -> dict[str, np.ndarray]:
        """Load remapped real Gaussian decoder arrays for local MLX model construction."""

        return load_lito_gaussian_decoder_weight_arrays(self.config.weights_root, names=names, dtype=dtype)

    def load_voxel_decoder_weight_arrays(
        self,
        *,
        names: Iterable[str] | None = None,
        dtype: Any = np.float32,
    ) -> dict[str, np.ndarray]:
        """Load remapped real LiTo voxel decoder arrays for local MLX construction."""

        return load_lito_voxel_decoder_weight_arrays(self.config.weights_root, names=names, dtype=dtype)

    def decode_gaussian_outputs(
        self,
        shape_out: Any,
        color_out: Any,
        init_coord: Any,
    ) -> dict[str, np.ndarray]:
        """Decode raw Gaussian decoder outputs with the fixed LiTo profile."""

        return decode_lito_gaussian_outputs(shape_out, color_out, init_coord)

    def decode_gaussian_query_latents(
        self,
        query_latent: Any,
        init_coord: Any,
        *,
        weights: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run real LiTo Gaussian output heads for caller-supplied query latents."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return decode_lito_gaussian_query_latents(query_latent, init_coord, active_weights)

    def encode_gaussian_query_points(
        self,
        init_coord: Any,
        *,
        weights: dict[str, Any] | None = None,
    ) -> mx.array:
        """Run LiTo's real point/Fourier query stem for explicit init coordinates."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return encode_lito_gaussian_query_points(init_coord, active_weights)

    def decode_gaussian_query_points(
        self,
        init_coord: Any,
        *,
        weights: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run point query stem plus output heads for explicit init coordinates."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return decode_lito_gaussian_query_points(init_coord, active_weights)

    def run_gaussian_perceiver_block0_cross_only(
        self,
        query_latent: Any,
        latent_tokens: Any,
        *,
        weights: dict[str, Any] | None = None,
        q_seq_lens: Iterable[int] | None = None,
        kv_seq_lens: Iterable[int] | None = None,
        num_heads: int = 8,
    ) -> mx.array:
        """Run the real block-0 cross-attention/CA-MLP subpath only."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return run_lito_gaussian_perceiver_block0_cross_only(
            query_latent,
            latent_tokens,
            active_weights,
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            num_heads=num_heads,
        )

    def run_gaussian_perceiver_block0_with_local_voxel_self_attention(
        self,
        query_latent: Any,
        latent_tokens: Any,
        init_coord: Any,
        *,
        weights: dict[str, Any] | None = None,
        q_seq_lens: Iterable[int] | None = None,
        kv_seq_lens: Iterable[int] | None = None,
        num_heads: int = 8,
        self_cell_width: float = 0.25,
    ) -> mx.array:
        """Run real block-0 cross-attention and localized-voxel self-attention."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention(
            query_latent,
            latent_tokens,
            init_coord,
            active_weights,
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            num_heads=num_heads,
            self_cell_width=self_cell_width,
        )

    def run_gaussian_perceiver_all_blocks_with_local_voxel_self_attention(
        self,
        query_latent: Any,
        latent_tokens: Any,
        init_coord: Any,
        *,
        weights: dict[str, Any] | None = None,
        q_seq_lens: Iterable[int] | None = None,
        kv_seq_lens: Iterable[int] | None = None,
        block_indices: Iterable[int] | None = None,
        num_heads: int = 8,
        self_cell_width: float = 0.25,
    ) -> mx.array:
        """Run the full real Gaussian Perceiver stack for explicit query coordinates."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention(
            query_latent,
            latent_tokens,
            init_coord,
            active_weights,
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            block_indices=block_indices,
            num_heads=num_heads,
            self_cell_width=self_cell_width,
        )

    def decode_gaussian_perceiver_all_blocks(
        self,
        latent_tokens: Any,
        init_coord: Any,
        *,
        weights: dict[str, Any] | None = None,
        q_seq_lens: Iterable[int] | None = None,
        kv_seq_lens: Iterable[int] | None = None,
        block_indices: Iterable[int] | None = None,
        num_heads: int = 8,
        self_cell_width: float = 0.25,
    ) -> dict[str, np.ndarray]:
        """Run query stem, full Gaussian Perceiver, output heads, and Gaussian decode."""

        active_weights = weights if weights is not None else self.load_gaussian_decoder_weight_arrays()
        return decode_lito_gaussian_perceiver_all_blocks(
            latent_tokens,
            init_coord,
            active_weights,
            q_seq_lens=q_seq_lens,
            kv_seq_lens=kv_seq_lens,
            block_indices=block_indices,
            num_heads=num_heads,
            self_cell_width=self_cell_width,
        )

    def run_voxel_decoder_lowres_latent(
        self,
        latent_tokens: Any,
        *,
        weights: dict[str, Any] | None = None,
        block_indices: Iterable[int] | None = None,
        num_heads: int = 8,
    ) -> mx.array:
        """Run LiTo's real voxel decoder to low-res sparse-structure latents."""

        active_weights = weights if weights is not None else self.load_voxel_decoder_weight_arrays()
        return run_lito_voxel_decoder_lowres_latent(
            latent_tokens,
            active_weights,
            block_indices=block_indices,
            num_heads=num_heads,
        )

    def decode_trellis_sparse_structure_logits(
        self,
        ss_latent: Any,
        *,
        trellis_root: str | Path,
        config_path: str | Path = _TRELLIS_SS_DECODER_CONFIG,
        checkpoint_path: str | Path = _TRELLIS_SS_DECODER_CHECKPOINT,
    ) -> mx.array:
        """Run TRELLIS sparse-structure decoder logits from LiTo voxel ``ss_latent``."""

        return decode_lito_trellis_sparse_structure_logits(
            ss_latent,
            trellis_root=trellis_root,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

    def decode_init_coords_from_latents(
        self,
        latent_tokens: Any,
        *,
        trellis_root: str | Path,
        voxel_weights: dict[str, Any] | None = None,
        block_indices: Iterable[int] | None = None,
        num_heads: int = 8,
        config_path: str | Path = _TRELLIS_SS_DECODER_CONFIG,
        checkpoint_path: str | Path = _TRELLIS_SS_DECODER_CHECKPOINT,
        max_cells_per_batch: int | None = None,
    ) -> dict[str, Any]:
        """Run LiTo voxel + TRELLIS occupancy decode into packed Gaussian init coordinates."""

        active_weights = voxel_weights if voxel_weights is not None else self.load_voxel_decoder_weight_arrays()
        return decode_lito_init_coords_from_latents(
            latent_tokens,
            active_weights,
            trellis_root=trellis_root,
            block_indices=block_indices,
            num_heads=num_heads,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            max_cells_per_batch=max_cells_per_batch,
        )

    def decode_sampled_latents_to_gaussians(self, latent_tokens: Any) -> dict[str, np.ndarray]:
        """Decode sampled LiTo latent tokens through voxel/TRELLIS and Gaussian Perceiver."""

        trellis_root = _resolve_lito_trellis_root(self.config)
        max_cells = resolve_lito_init_coord_cap(
            self.config.memory_profile,
            self.config.max_init_coords_per_batch,
        )
        init = self.decode_init_coords_from_latents(
            latent_tokens,
            trellis_root=trellis_root,
            max_cells_per_batch=max_cells,
        )
        init_coord = init["init_coord"]
        q_seq_lens = init["q_seq_lens"]
        if int(init_coord.shape[0]) == 0:
            raise LitoBackendUnavailable(
                "checkpoint-backed LiTo generation produced no occupied TRELLIS cells at the upstream "
                "occupancy threshold; no real Gaussian PLY was written"
            )
        latent_array = mx.array(latent_tokens, dtype=mx.float32)
        batch_size, latent_count, latent_dim = (int(value) for value in latent_array.shape)
        latent_packed = mx.reshape(latent_array, (batch_size * latent_count, latent_dim))
        return self.decode_gaussian_perceiver_all_blocks(
            latent_packed,
            init_coord,
            q_seq_lens=q_seq_lens,
            kv_seq_lens=[latent_count for _ in range(batch_size)],
        )

    def generate_gaussians(self, request: LitoRealGenerateRequest) -> dict[str, np.ndarray]:
        _validate_request(request)
        condition = self.condition_rgba(request.cond_rgba)
        latent = self.sample_dit_latents(
            condition,
            seed=request.seed,
            num_steps=request.num_steps,
            cfg_scale=request.cfg_scale,
            method="heun",
        )
        return self.decode_sampled_latents_to_gaussians(latent)


def create_lito_real_backend(config: LitoRealBackendConfig) -> DirectMlxLitoBackend:
    """Create the runtime backend or raise a precise availability error."""

    if config.allow_cuda:
        raise LitoBackendUnavailable("CUDA is not allowed for LiTo generation; refusing CUDA backend request")
    architecture = inspect_lito_real_architecture(config.weights_root)
    return DirectMlxLitoBackend(config, architecture=architecture)


def inspect_lito_real_architecture(weights_root: str | Path) -> LitoRealArchitectureInventory:
    """Infer real LiTo module architecture from safetensors headers only."""

    root = Path(weights_root)
    dit_headers = _read_safetensor_headers(root / _DIT_CHECKPOINT)
    tokenizer_headers = _read_safetensor_headers(root / _TOKENIZER_CHECKPOINT)
    return LitoRealArchitectureInventory(
        weights_root=root,
        checkpoint_key_counts={
            _DIT_CHECKPOINT.as_posix(): len(dit_headers),
            _TOKENIZER_CHECKPOINT.as_posix(): len(tokenizer_headers),
        },
        prefix_counts={
            _DIT_CHECKPOINT.as_posix(): _top_level_prefix_counts(dit_headers),
            _TOKENIZER_CHECKPOINT.as_posix(): _top_level_prefix_counts(tokenizer_headers),
        },
        dit=_infer_dit_architecture(dit_headers),
        gaussian_decoder=_infer_gaussian_decoder_architecture(tokenizer_headers),
        voxel_decoder=_infer_voxel_decoder_architecture(tokenizer_headers),
    )


def load_lito_dit_weight_arrays(
    weights_root: str | Path,
    *,
    use_ema: bool = True,
    names: Iterable[str] | None = None,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Load selected DiT weights from converted safetensors using local MLX names."""

    root = Path(weights_root)
    path = root / _DIT_CHECKPOINT
    source_prefix = _DIT_EMA_PREFIX if use_ema else _DIT_SOURCE_PREFIX
    requested = _normalize_name_filter(names)
    arrays: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as handle:
        for source_key in handle.keys():
            if not source_key.startswith(source_prefix):
                continue
            local_name = _remap_lito_dit_key(source_key[len(source_prefix) :])
            if requested is not None and local_name not in requested:
                continue
            arrays[local_name] = handle.get_tensor(source_key).astype(dtype, copy=False)
    _raise_missing_requested_names("DiT", requested, arrays)
    return arrays


def load_lito_patch_encoder_weight_arrays(
    weights_root: str | Path,
    *,
    names: Iterable[str] | None = None,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Load selected DINO/RGBA patch encoder weights from converted safetensors."""

    root = Path(weights_root)
    path = root / _DIT_CHECKPOINT
    requested = _normalize_name_filter(names)
    arrays: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as handle:
        for source_key in handle.keys():
            if not source_key.startswith(_PATCH_ENCODER_PREFIX):
                continue
            local_name = source_key[len(_PATCH_ENCODER_PREFIX) :]
            if requested is not None and local_name not in requested:
                continue
            arrays[local_name] = handle.get_tensor(source_key).astype(dtype, copy=False)
    _raise_missing_requested_names("patch encoder", requested, arrays)
    return arrays


def load_lito_gaussian_decoder_weight_arrays(
    weights_root: str | Path,
    *,
    names: Iterable[str] | None = None,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Load selected Gaussian decoder weights from tokenizer safetensors."""

    root = Path(weights_root)
    path = root / _TOKENIZER_CHECKPOINT
    requested = _normalize_name_filter(names)
    arrays: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as handle:
        for source_key in handle.keys():
            if not source_key.startswith(_GS_PREFIX):
                continue
            stripped = source_key[len(_GS_PREFIX) :]
            tensor = handle.get_tensor(source_key)
            for local_name, array in _remap_lito_gaussian_decoder_tensor(stripped, tensor):
                if requested is not None and local_name not in requested:
                    continue
                arrays[local_name] = array.astype(dtype, copy=False)
    _raise_missing_requested_names("Gaussian decoder", requested, arrays)
    return arrays


def load_lito_voxel_decoder_weight_arrays(
    weights_root: str | Path,
    *,
    names: Iterable[str] | None = None,
    dtype: Any = np.float32,
) -> dict[str, np.ndarray]:
    """Load selected LiTo voxel decoder weights from tokenizer safetensors."""

    root = Path(weights_root)
    path = root / _TOKENIZER_CHECKPOINT
    requested = _normalize_name_filter(names)
    arrays: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as handle:
        for source_key in handle.keys():
            if not source_key.startswith(_VOXEL_PREFIX):
                continue
            local_name = source_key[len(_VOXEL_PREFIX) :]
            if requested is not None and local_name not in requested:
                continue
            arrays[local_name] = handle.get_tensor(source_key).astype(dtype, copy=False)
    _raise_missing_requested_names("Voxel decoder", requested, arrays)
    return arrays


def run_lito_patch_encoder_condition_tokens(
    cond_rgba: Any,
    weights: dict[str, Any],
    *,
    block_indices: Iterable[int] | None = None,
    config: LitoPatchEncoderConfig = LitoPatchEncoderConfig(),
) -> mx.array:
    """Run LiTo's real DINO/RGBA patch encoder from straight RGBA image input."""

    rgba = _lito_rgba_to_bqhwc(cond_rgba)
    straight_rgb = rgba[..., :3]
    alpha = rgba[..., 3:4]
    premultiplied_rgb = straight_rgb * alpha
    batch, views, height, width, _ = (int(value) for value in rgba.shape)
    premultiplied_bchw = mx.transpose(mx.reshape(premultiplied_rgb, (batch * views, height, width, 3)), (0, 3, 1, 2))
    alpha_bchw = mx.transpose(mx.reshape(alpha, (batch * views, height, width, 1)), (0, 3, 1, 2))
    if height != config.input_size or width != config.input_size:
        premultiplied_bchw = _resize_linear_bchw(premultiplied_bchw, (config.input_size, config.input_size))
        alpha_bchw = _resize_linear_bchw(alpha_bchw, (config.input_size, config.input_size))

    dino_tokens, patch_grid = _run_lito_dino_vitl14_reg(
        premultiplied_bchw,
        weights,
        block_indices=block_indices,
        config=config,
    )
    learnable_tokens = _run_lito_rgba_learnable_branch(
        premultiplied_bchw,
        alpha_bchw,
        weights,
        patch_grid=patch_grid,
        config=config,
    )
    if int(dino_tokens.shape[1]) != int(learnable_tokens.shape[1]):
        raise ValueError(f"DINO token count {dino_tokens.shape[1]} does not match RGBA token count {learnable_tokens.shape[1]}")
    tokens = mx.concatenate((dino_tokens, learnable_tokens), axis=-1)
    return mx.reshape(tokens, (batch, views * int(tokens.shape[1]), int(tokens.shape[2])))


def run_lito_dit_velocity(
    latent_tokens: Any,
    timestep: Any,
    condition_tokens: Any,
    weights: dict[str, Any],
    *,
    block_indices: Iterable[int] | None = None,
    num_heads: int = 16,
    cond_drop_ids: Any | None = None,
) -> mx.array:
    """Run the real LiTo DiT velocity forward pass from explicit condition tokens."""

    latent = mx.array(latent_tokens, dtype=mx.float32)
    cond = mx.array(condition_tokens, dtype=mx.float32)
    if latent.ndim != 3 or int(latent.shape[-1]) != 32:
        raise ValueError(f"latent_tokens must have shape (B, N, 32), got {latent.shape}")
    if cond.ndim != 3 or int(cond.shape[-1]) != 2048:
        raise ValueError(f"condition_tokens must have shape (B, M, 2048), got {cond.shape}")
    if int(latent.shape[0]) != int(cond.shape[0]):
        raise ValueError(f"latent and condition batch sizes must match, got {latent.shape[0]} and {cond.shape[0]}")

    t_emb, t0 = _run_lito_dit_timestep_embedding(timestep, int(latent.shape[0]), weights)
    cond = _run_lito_dit_condition_embedder(cond, weights, cond_drop_ids=cond_drop_ids)
    latent = _mx_linear(latent, _mx_weight(weights, "z_proj.weight"), _mx_weight(weights, "z_proj.bias"))
    latent = _mx_layer_norm(
        latent,
        _mx_weight(weights, "z_proj_ln.weight"),
        _mx_weight(weights, "z_proj_ln.bias"),
        eps=1e-6,
    )
    pos = _mx_weight(weights, "pos_mtx")
    if int(latent.shape[1]) > int(pos.shape[0]):
        raise ValueError(f"latent token count {latent.shape[1]} exceeds DiT pos_mtx length {pos.shape[0]}")
    pos = pos[: int(latent.shape[1])]
    if "pos_proj.weight" in weights:
        pos = _mx_linear(pos, _mx_weight(weights, "pos_proj.weight"), _mx_weight(weights, "pos_proj.bias"))
    latent = latent + pos[None, :, :]
    for block_index in _normalize_dit_block_indices(block_indices, weights):
        latent = _run_lito_dit_block(latent, cond, t0, weights, block_index, num_heads=num_heads)
    return _run_lito_dit_final_layer(latent, t_emb, weights)


def sample_lito_dit_latents(
    condition_tokens: Any,
    weights: dict[str, Any],
    *,
    initial_latent: Any | None = None,
    seed: int | None = None,
    num_steps: int = 20,
    cfg_scale: float = 1.0,
    method: str = "heun",
    block_indices: Iterable[int] | None = None,
    num_heads: int = 16,
    latent_mean: float = 0.0661,
    latent_std: float = 1.6464,
    t_eps: float = 1e-4,
) -> mx.array:
    """Sample LiTo latent tokens with the real DiT ODE loop from explicit condition tokens."""

    cond = mx.array(condition_tokens, dtype=mx.float32)
    if cond.ndim != 3 or int(cond.shape[-1]) != 2048:
        raise ValueError(f"condition_tokens must have shape (B, M, 2048), got {cond.shape}")
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if cfg_scale <= 0:
        raise ValueError(f"cfg_scale must be positive, got {cfg_scale}")
    if initial_latent is None:
        if seed is not None:
            mx.random.seed(int(seed))
        latent = mx.random.normal(shape=(int(cond.shape[0]), int(_mx_weight(weights, "pos_mtx").shape[0]), 32))
    else:
        latent = mx.array(initial_latent, dtype=mx.float32)
        if latent.ndim != 3 or int(latent.shape[-1]) != 32:
            raise ValueError(f"initial_latent must have shape (B, N, 32), got {latent.shape}")
        if int(latent.shape[0]) != int(cond.shape[0]):
            raise ValueError(f"initial_latent and condition batch sizes must match, got {latent.shape[0]} and {cond.shape[0]}")

    use_cfg = float(cfg_scale) > 1.0
    cond_drop_ids = None
    if use_cfg:
        batch_size = int(cond.shape[0])
        cond = mx.concatenate((cond, cond), axis=0)
        latent = mx.concatenate((latent, latent), axis=0)
        cond_drop_ids = mx.concatenate(
            (
                mx.zeros((batch_size,), dtype=mx.bool_),
                mx.ones((batch_size,), dtype=mx.bool_),
            ),
            axis=0,
        )

    def guided_velocity(x_value: mx.array, t_value: mx.array) -> mx.array:
        velocity = run_lito_dit_velocity(
            x_value,
            t_value,
            cond,
            weights,
            block_indices=block_indices,
            num_heads=num_heads,
            cond_drop_ids=cond_drop_ids,
        ).astype(mx.float32)
        if not use_cfg:
            return velocity
        mid = int(velocity.shape[0]) // 2
        cond_velocity = velocity[:mid]
        uncond_velocity = velocity[mid:]
        guided = uncond_velocity + float(cfg_scale) * (cond_velocity - uncond_velocity)
        return mx.concatenate((guided, guided), axis=0)

    ts = mx.linspace(float(t_eps), 1.0, int(num_steps))
    x = latent.astype(mx.float32)
    if method == "euler":
        for index in range(int(ts.shape[0]) - 1):
            t = mx.broadcast_to(ts[index], (int(x.shape[0]),))
            dt = (ts[index + 1] - ts[index]).astype(mx.float32)
            dx = guided_velocity(x, t)
            x = x + dt * dx
            mx.eval(x)
    elif method == "heun":
        for index in range(int(ts.shape[0]) - 1):
            t = mx.broadcast_to(ts[index], (int(x.shape[0]),))
            t_next = mx.broadcast_to(ts[index + 1], (int(x.shape[0]),))
            dt = (ts[index + 1] - ts[index]).astype(mx.float32)
            dx = guided_velocity(x, t)
            x_pred = x + dt * dx
            dx_next = guided_velocity(x_pred, t_next)
            x = x + 0.5 * dt * (dx + dx_next)
            mx.eval(x)
    else:
        raise ValueError(f"unsupported LiTo DiT ODE method: {method}")
    if use_cfg:
        x = x[: int(x.shape[0]) // 2]
    return x * float(latent_std) + float(latent_mean)


def decode_lito_gaussian_outputs(
    shape_out: Any,
    color_out: Any,
    init_coord: Any,
    *,
    profile: LitoGaussianDecoderProfile = LitoGaussianDecoderProfile(),
) -> dict[str, np.ndarray]:
    """Decode raw LiTo Gaussian decoder outputs into export-ready fields."""

    shape = _as_numpy(shape_out, "shape_out")
    color = _as_numpy(color_out, "color_out")
    coord = _as_numpy(init_coord, "init_coord")
    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError(f"init_coord must have shape (M, 3), got {coord.shape}")
    count = int(coord.shape[0])
    shape = _reshape_decoder_output(shape, "shape_out", count, profile.expansion_ratio, profile.shape_dim)
    color = _reshape_decoder_output(color, "color_out", count, profile.expansion_ratio, profile.color_dim)

    xyz = (_sigmoid(shape[..., 0:3]) * 2.0 - 1.0) * profile.region_scaling
    xyz = xyz + coord[:, None, :]
    quaternion = _normalize_quaternion(shape[..., 3:7].reshape((-1, 4))).reshape((count, profile.expansion_ratio, 4))
    scaling_logit = shape[..., 7:10] + profile.scaling_logit_bias
    scaling = _sigmoid(scaling_logit) * profile.scaling_scalar
    if profile.min_scaling > 1e-8:
        scaling = np.sqrt(np.square(scaling) + profile.min_scaling**2)

    opacity_logit = color[..., 0:1] * profile.opacity_logit_scale + profile.opacity_logit_bias
    opacity = _sigmoid(opacity_logit)
    opacity = opacity * (profile.max_opacity - profile.min_opacity) + profile.min_opacity
    rgb_coeffs = (profile.rgb_sh_degree + 1) ** 2
    rgb_sh = color[..., 1:].reshape((count, profile.expansion_ratio, rgb_coeffs, 3))
    return {
        "xyz_w": xyz.astype(np.float32, copy=False),
        "scaling": scaling.astype(np.float32, copy=False),
        "quaternion": quaternion.astype(np.float32, copy=False),
        "opacity": opacity.astype(np.float32, copy=False),
        "rgb_sh": rgb_sh.astype(np.float32, copy=False),
    }


def run_lito_gaussian_output_heads(
    query_latent: Any,
    weights: dict[str, Any],
) -> tuple[mx.array, mx.array]:
    """Run LiTo's real shape/color output MLP heads from decoder query latents."""

    query = mx.array(query_latent, dtype=mx.float32)
    if query.ndim != 2 or int(query.shape[-1]) != 512:
        raise ValueError(f"query_latent must have shape (M, 512), got {query.shape}")
    shape_out = _run_lito_output_mlp(query, weights, "gs_output_shape_mlp")
    color_out = _run_lito_output_mlp(query, weights, "gs_output_color_mlp")
    return shape_out, color_out


def encode_lito_gaussian_query_points(
    init_coord: Any,
    weights: dict[str, Any],
) -> mx.array:
    """Run LiTo's real coordinate/Fourier point-query stem before Perceiver attention."""

    coord = mx.array(init_coord, dtype=mx.float32)
    if coord.ndim != 2 or int(coord.shape[-1]) != 3:
        raise ValueError(f"init_coord must have shape (M, 3), got {coord.shape}")
    freq_bands = _mx_weight(weights, "xyz_encoding.freq_bands")
    encoded = _fourier_encode_xyz(coord, freq_bands)
    point_features = mx.concatenate([coord, encoded], axis=-1)
    query = _mx_linear(
        point_features,
        _mx_weight(weights, "point_linear.weight"),
        _mx_weight(weights, "point_linear.bias"),
    )
    return _run_lito_output_mlp(query, weights, "point_mlp")


def decode_lito_gaussian_query_points(
    init_coord: Any,
    weights: dict[str, Any],
    *,
    profile: LitoGaussianDecoderProfile = LitoGaussianDecoderProfile(),
) -> dict[str, np.ndarray]:
    """Run explicit-coordinate query stem, output heads, and Gaussian decode."""

    query = encode_lito_gaussian_query_points(init_coord, weights)
    return decode_lito_gaussian_query_latents(query, init_coord, weights, profile=profile)


def run_lito_gaussian_perceiver_block0_cross_only(
    query_latent: Any,
    latent_tokens: Any,
    weights: dict[str, Any],
    *,
    q_seq_lens: Iterable[int] | None = None,
    kv_seq_lens: Iterable[int] | None = None,
    num_heads: int = 8,
    include_localized_self_attention: bool = False,
) -> mx.array:
    """Run LiTo Gaussian Perceiver block 0 through cross-attention and CA MLP.

    This is an intermediate checkpoint-backed subpath. It intentionally stops
    before the block's localized-voxel self-attention layers, which require
    voxel window metadata that is not ported yet.
    """

    if include_localized_self_attention:
        raise LitoBackendUnavailable(
            "localized_voxel self-attention requires init_coord; use "
            "run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention"
        )
    query = mx.array(query_latent, dtype=mx.float32)
    latent = mx.array(latent_tokens, dtype=mx.float32)
    if query.ndim != 2:
        raise ValueError(f"query_latent must have shape (M, C), got {query.shape}")
    if latent.ndim != 2:
        raise ValueError(f"latent_tokens must have shape (N, C), got {latent.shape}")

    prefix = "perceiver.blocks.0"
    q_lens = _normalize_seq_lens(q_seq_lens, int(query.shape[0]), "q_seq_lens")
    kv_lens = _normalize_seq_lens(kv_seq_lens, int(latent.shape[0]), "kv_seq_lens")
    if len(q_lens) != len(kv_lens):
        raise ValueError(
            f"q_seq_lens and kv_seq_lens must describe the same batch count, got {len(q_lens)} and {len(kv_lens)}"
        )

    latent = _mx_linear(latent, _mx_weight(weights, f"{prefix}.kv_linear.weight"), None)
    cross = _run_lito_gaussian_cross_attention_layer(
        query,
        latent,
        weights,
        f"{prefix}.ca_layer",
        q_seq_lens=q_lens,
        kv_seq_lens=kv_lens,
        num_heads=num_heads,
    )
    query = query + cross
    ca_ln = _mx_layer_norm(
        query,
        _mx_weight(weights, f"{prefix}.ca_ln.weight"),
        _mx_weight(weights, f"{prefix}.ca_ln.bias"),
        eps=1e-6,
    )
    return query + _run_lito_swiglu(ca_ln, weights, f"{prefix}.ca_mlp")


def run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention(
    query_latent: Any,
    latent_tokens: Any,
    init_coord: Any,
    weights: dict[str, Any],
    *,
    q_seq_lens: Iterable[int] | None = None,
    kv_seq_lens: Iterable[int] | None = None,
    num_heads: int = 8,
    self_cell_width: float = 0.25,
) -> mx.array:
    """Run LiTo Gaussian Perceiver block 0 through both localized self-attention layers."""

    coord = mx.array(init_coord, dtype=mx.float32)
    if coord.ndim != 2 or int(coord.shape[-1]) != 3:
        raise ValueError(f"init_coord must have shape (M, 3), got {coord.shape}")
    q_lens = _normalize_seq_lens(q_seq_lens, int(coord.shape[0]), "q_seq_lens")
    query = run_lito_gaussian_perceiver_block0_cross_only(
        query_latent,
        latent_tokens,
        weights,
        q_seq_lens=q_lens,
        kv_seq_lens=kv_seq_lens,
        num_heads=num_heads,
    )
    if int(query.shape[0]) != int(coord.shape[0]):
        raise ValueError(
            "query_latent and init_coord must have the same packed length, "
            f"got {query.shape[0]} and {coord.shape[0]}"
        )

    prefix = "perceiver.blocks.0"
    for layer_index in (0, 1):
        voxel_info = build_lito_local_voxel_info(
            coord,
            q_lens,
            cell_width=self_cell_width,
            shift=0.5 * (layer_index % 2) * self_cell_width,
        )
        normed = _mx_layer_norm(
            query,
            _mx_weight(weights, f"{prefix}.ln1_layers.{layer_index}.weight"),
            _mx_weight(weights, f"{prefix}.ln1_layers.{layer_index}.bias"),
            eps=1e-6,
        )
        query = query + _run_lito_gaussian_self_attention_layer(
            normed,
            weights,
            f"{prefix}.sa_layers.{layer_index}",
            q_seq_lens=q_lens,
            voxel_info=voxel_info,
            num_heads=num_heads,
        )
        normed = _mx_layer_norm(
            query,
            _mx_weight(weights, f"{prefix}.ln2_layers.{layer_index}.weight"),
            _mx_weight(weights, f"{prefix}.ln2_layers.{layer_index}.bias"),
            eps=1e-6,
        )
        query = query + _run_lito_swiglu(normed, weights, f"{prefix}.mlp_layers.{layer_index}")
    return query


def run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention(
    query_latent: Any,
    latent_tokens: Any,
    init_coord: Any,
    weights: dict[str, Any],
    *,
    q_seq_lens: Iterable[int] | None = None,
    kv_seq_lens: Iterable[int] | None = None,
    block_indices: Iterable[int] | None = None,
    num_heads: int = 8,
    self_cell_width: float = 0.25,
) -> mx.array:
    """Run LiTo's real Gaussian Perceiver blocks with localized-voxel self-attention."""

    query = mx.array(query_latent, dtype=mx.float32)
    latent = mx.array(latent_tokens, dtype=mx.float32)
    coord = mx.array(init_coord, dtype=mx.float32)
    if query.ndim != 2:
        raise ValueError(f"query_latent must have shape (M, C), got {query.shape}")
    if latent.ndim != 2:
        raise ValueError(f"latent_tokens must have shape (N, C), got {latent.shape}")
    if coord.ndim != 2 or int(coord.shape[-1]) != 3:
        raise ValueError(f"init_coord must have shape (M, 3), got {coord.shape}")
    if int(query.shape[0]) != int(coord.shape[0]):
        raise ValueError(
            "query_latent and init_coord must have the same packed length, "
            f"got {query.shape[0]} and {coord.shape[0]}"
        )

    q_lens = _normalize_seq_lens(q_seq_lens, int(query.shape[0]), "q_seq_lens")
    kv_lens = _normalize_seq_lens(kv_seq_lens, int(latent.shape[0]), "kv_seq_lens")
    if len(q_lens) != len(kv_lens):
        raise ValueError(
            f"q_seq_lens and kv_seq_lens must describe the same batch count, got {len(q_lens)} and {len(kv_lens)}"
        )
    block_ids = _normalize_perceiver_block_indices(block_indices, weights)
    voxel_infos = [
        build_lito_local_voxel_info(
            coord,
            q_lens,
            cell_width=self_cell_width,
            shift=0.5 * (layer_index % 2) * self_cell_width,
        )
        for layer_index in (0, 1)
    ]
    for block_index in block_ids:
        query = _run_lito_gaussian_perceiver_block_with_local_voxel_self_attention(
            query,
            latent,
            weights,
            block_index,
            q_seq_lens=q_lens,
            kv_seq_lens=kv_lens,
            voxel_infos=voxel_infos,
            num_heads=num_heads,
        )
    return query


def decode_lito_gaussian_perceiver_all_blocks(
    latent_tokens: Any,
    init_coord: Any,
    weights: dict[str, Any],
    *,
    q_seq_lens: Iterable[int] | None = None,
    kv_seq_lens: Iterable[int] | None = None,
    block_indices: Iterable[int] | None = None,
    num_heads: int = 8,
    self_cell_width: float = 0.25,
    profile: LitoGaussianDecoderProfile = LitoGaussianDecoderProfile(),
) -> dict[str, np.ndarray]:
    """Run query stem, all Gaussian Perceiver blocks, output heads, and decode."""

    query = encode_lito_gaussian_query_points(init_coord, weights)
    decoded_query = run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention(
        query,
        latent_tokens,
        init_coord,
        weights,
        q_seq_lens=q_seq_lens,
        kv_seq_lens=kv_seq_lens,
        block_indices=block_indices,
        num_heads=num_heads,
        self_cell_width=self_cell_width,
    )
    return decode_lito_gaussian_query_latents(decoded_query, init_coord, weights, profile=profile)


def run_lito_voxel_decoder_lowres_latent(
    latent_tokens: Any,
    weights: dict[str, Any],
    *,
    block_indices: Iterable[int] | None = None,
    num_heads: int = 8,
) -> mx.array:
    """Run LiTo SSLatentDecoder up to ``ss_latent`` without Trellis occupancy decode."""

    latent = mx.array(latent_tokens, dtype=mx.float32)
    if latent.ndim != 3 or int(latent.shape[-1]) != 32:
        raise ValueError(f"latent_tokens must have shape (B, N, 32), got {latent.shape}")
    latent = _mx_linear(latent, _mx_weight(weights, "input_linear.weight"), _mx_weight(weights, "input_linear.bias"))
    query = _build_lito_voxel_init_query(int(latent.shape[0]), weights)
    for block_index in _normalize_voxel_block_indices(block_indices, weights):
        query = _run_lito_voxel_perceiver_block(
            query,
            latent,
            weights,
            block_index,
            num_heads=num_heads,
        )
    init_query = _mx_weight(weights, "net.init_query")
    query = query.reshape(
        int(latent.shape[0]),
        int(init_query.shape[0]),
        int(init_query.shape[1]),
        int(init_query.shape[2]),
        -1,
    )
    out = _mx_layer_norm(
        query,
        _mx_weight(weights, "final_layer.norm_final.weight"),
        _mx_weight(weights, "final_layer.norm_final.bias"),
        eps=1e-6,
    )
    out = _mx_linear(out, _mx_weight(weights, "final_layer.linear.weight"), _mx_weight(weights, "final_layer.linear.bias"))
    return mx.transpose(out, axes=(0, 4, 1, 2, 3))


def occ_grid_to_lito_init_coord(
    occ_grid: Any,
    *,
    threshold: float = 0.5,
    min_xyz_w: float = -1.0,
    max_xyz_w: float = 1.0,
    max_cells_per_batch: int | None = None,
) -> dict[str, np.ndarray | list[int]]:
    """Convert LiTo/Trellis occupancy grid probabilities into packed Gaussian query coordinates."""

    occ = np.asarray(occ_grid)
    if occ.ndim == 4:
        occ = occ[:, None, ...]
    if occ.ndim != 5 or occ.shape[1] != 1 or occ.shape[-1] != occ.shape[-2] or occ.shape[-2] != occ.shape[-3]:
        raise ValueError(f"occ_grid must have shape (B, 1, R, R, R) or (B, R, R, R), got {occ.shape}")
    if occ.dtype == np.bool_:
        occupied = occ
    else:
        occupied = occ >= float(threshold)
    batch_size = int(occupied.shape[0])
    grid_size = int(occupied.shape[-1])
    cell_width = (float(max_xyz_w) - float(min_xyz_w)) / float(grid_size)
    occupied_bijk = np.transpose(occupied[:, 0], axes=(0, 3, 2, 1))
    bijk = np.argwhere(occupied_bijk)
    if bijk.size == 0:
        return {
            "init_coord": np.zeros((0, 3), dtype=np.float32),
            "q_seq_lens": [0 for _ in range(batch_size)],
            "occ_bool_grid": occupied.astype(np.bool_, copy=False),
        }
    if max_cells_per_batch is not None:
        max_cells = int(max_cells_per_batch)
        if max_cells <= 0:
            raise ValueError(f"max_cells_per_batch must be positive, got {max_cells_per_batch}")
        if occ.dtype == np.bool_:
            score_grid = occupied_bijk.astype(np.float32)
        else:
            score_grid = np.transpose(occ[:, 0], axes=(0, 3, 2, 1)).astype(np.float32, copy=False)
        selected = []
        for batch_index in range(batch_size):
            batch_rows = bijk[bijk[:, 0] == batch_index]
            if batch_rows.shape[0] <= max_cells:
                selected.append(batch_rows)
                continue
            scores = score_grid[batch_rows[:, 0], batch_rows[:, 1], batch_rows[:, 2], batch_rows[:, 3]]
            top = np.argpartition(-scores, max_cells - 1)[:max_cells]
            batch_rows = batch_rows[top]
            order = np.lexsort((batch_rows[:, 3], batch_rows[:, 2], batch_rows[:, 1], -scores[top]))
            selected.append(batch_rows[order])
        bijk = np.concatenate(selected, axis=0) if selected else np.zeros((0, 4), dtype=np.int64)
    order = np.argsort(bijk[:, 0], kind="stable")
    bijk = bijk[order]
    coord = (bijk[:, 1:].astype(np.float32) + 0.5) * cell_width + float(min_xyz_w)
    seq_lens = np.bincount(bijk[:, 0], minlength=batch_size).astype(np.int64).tolist()
    return {
        "init_coord": coord.astype(np.float32, copy=False),
        "q_seq_lens": [int(length) for length in seq_lens],
        "occ_bool_grid": occupied.astype(np.bool_, copy=False),
    }


def decode_lito_trellis_sparse_structure_logits(
    ss_latent: Any,
    *,
    trellis_root: str | Path,
    config_path: str | Path = _TRELLIS_SS_DECODER_CONFIG,
    checkpoint_path: str | Path = _TRELLIS_SS_DECODER_CHECKPOINT,
) -> mx.array:
    """Decode LiTo voxel ``ss_latent`` to TRELLIS occupancy logits with local MLX ops."""

    from .trellis2_sparse_structure import probe_sparse_structure_decoder_boundary, read_sparse_structure_decoder_config

    root = Path(trellis_root)
    config = read_sparse_structure_decoder_config(root, str(config_path))
    checkpoint = root / Path(checkpoint_path)
    latent = mx.array(ss_latent, dtype=mx.float32)
    probe = probe_sparse_structure_decoder_boundary(checkpoint, config, sparse_latent=latent)
    if probe.decoded_logits is None:
        raise LitoBackendUnavailable(probe.blocker_detail)
    return probe.decoded_logits


def decode_lito_init_coords_from_latents(
    latent_tokens: Any,
    voxel_weights: dict[str, Any],
    *,
    trellis_root: str | Path,
    block_indices: Iterable[int] | None = None,
    num_heads: int = 8,
    config_path: str | Path = _TRELLIS_SS_DECODER_CONFIG,
    checkpoint_path: str | Path = _TRELLIS_SS_DECODER_CHECKPOINT,
    max_cells_per_batch: int | None = None,
) -> dict[str, Any]:
    """Run real LiTo/TRELLIS init-coordinate generation from caller-supplied latent tokens."""

    ss_latent = run_lito_voxel_decoder_lowres_latent(
        latent_tokens,
        voxel_weights,
        block_indices=block_indices,
        num_heads=num_heads,
    )
    logits = decode_lito_trellis_sparse_structure_logits(
        ss_latent,
        trellis_root=trellis_root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    result = occ_grid_to_lito_init_coord(logits, threshold=0.0, max_cells_per_batch=max_cells_per_batch)
    result["ss_latent"] = ss_latent
    result["occ_logits"] = logits
    return result


def build_lito_local_voxel_info(
    init_coord: Any,
    seq_lens: Iterable[int],
    *,
    cell_width: float,
    shift: float = 0.0,
    chunk_size: int = 30000,
) -> dict[str, Any]:
    """Build LiTo localized-voxel attention metadata without Torch or vendor runtime."""

    coord = _as_numpy(init_coord, "init_coord")
    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError(f"init_coord must have shape (M, 3), got {coord.shape}")
    lens = _normalize_seq_lens(seq_lens, int(coord.shape[0]), "seq_lens")
    if cell_width <= 0:
        raise ValueError(f"cell_width must be positive, got {cell_width}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    batch_indices = np.repeat(np.arange(len(lens), dtype=np.int64), np.asarray(lens, dtype=np.int64))
    ijk = np.floor((coord + float(shift)) / float(cell_width)).astype(np.int64)
    bijk = np.concatenate([batch_indices[:, None], ijk[:, ::-1]], axis=1)
    cell_bijk, linear_idx, cell_counts = np.unique(bijk, axis=0, return_inverse=True, return_counts=True)
    forward_idxs = np.argsort(linear_idx, kind="stable").astype(np.int64)
    backward_idxs = np.empty_like(forward_idxs)
    backward_idxs[forward_idxs] = np.arange(forward_idxs.shape[0], dtype=np.int64)
    cell_bidx = cell_bijk[:, 0]
    new_seq_lens = np.bincount(cell_bidx, minlength=len(lens)).astype(np.int32)
    cell_counts = cell_counts.astype(np.int32)

    cu_seq_lens: list[mx.array] = []
    max_seq_lens: list[int] = []
    chunk_start_idxs: list[int] = []
    current_idx = 0
    for chunk_start in range(0, int(cell_counts.shape[0]), chunk_size):
        chunk_counts = cell_counts[chunk_start : chunk_start + chunk_size]
        cu = np.concatenate([np.zeros((1,), dtype=np.int32), np.cumsum(chunk_counts, dtype=np.int32)])
        cu_seq_lens.append(mx.array(cu, dtype=mx.int32))
        max_seq_lens.append(int(chunk_counts.max(initial=0)))
        chunk_start_idxs.append(int(current_idx))
        current_idx += int(cu[-1])
    chunk_start_idxs.append(int(coord.shape[0]))
    if current_idx != int(coord.shape[0]):
        raise ValueError(f"voxel metadata covers {current_idx} points, expected {coord.shape[0]}")
    return {
        "linear_idx": mx.array(linear_idx.astype(np.int64), dtype=mx.int64),
        "new_seq_lens": mx.array(new_seq_lens, dtype=mx.int32),
        "cell_counts": mx.array(cell_counts, dtype=mx.int32),
        "total_cells": int(cell_counts.shape[0]),
        "forward_idxs": mx.array(forward_idxs, dtype=mx.int64),
        "backward_idxs": mx.array(backward_idxs, dtype=mx.int64),
        "cu_seq_lens": cu_seq_lens,
        "max_seq_lens": max_seq_lens,
        "chunk_start_idxs": chunk_start_idxs,
    }


def decode_lito_gaussian_query_latents(
    query_latent: Any,
    init_coord: Any,
    weights: dict[str, Any],
    *,
    profile: LitoGaussianDecoderProfile = LitoGaussianDecoderProfile(),
) -> dict[str, np.ndarray]:
    """Run real output heads and decode Gaussians for explicit query/init coordinates."""

    shape_out, color_out = run_lito_gaussian_output_heads(query_latent, weights)
    mx.eval(shape_out, color_out)
    return decode_lito_gaussian_outputs(shape_out, color_out, init_coord, profile=profile)


def normalize_lito_gs_dict(gs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    """Normalize upstream LiTo Gaussian tensors into the local export schema."""

    required = ("xyz_w", "scaling", "quaternion", "opacity", "rgb_sh")
    missing = [key for key in required if key not in gs_dict]
    if missing:
        raise ValueError(f"LiTo gaussian dict missing required fields: {', '.join(missing)}")

    xyz = _flatten_field(_as_numpy(gs_dict["xyz_w"], "xyz_w"), "xyz_w", 3)
    count = int(xyz.shape[0])
    normalized = {
        "xyz_w": xyz,
        "scaling": _flatten_field(_as_numpy(gs_dict["scaling"], "scaling"), "scaling", 3, count=count),
        "quaternion": _normalize_quaternion(
            _flatten_field(_as_numpy(gs_dict["quaternion"], "quaternion"), "quaternion", 4, count=count)
        ),
        "opacity": _flatten_field(_as_numpy(gs_dict["opacity"], "opacity"), "opacity", 1, count=count),
        "rgb_sh": _flatten_rgb_sh(_as_numpy(gs_dict["rgb_sh"], "rgb_sh"), count=count, require_square=False),
    }
    for optional in ("normal_w", "albedo", "roughness_metallic", "lf"):
        if optional in gs_dict:
            value = _as_numpy(gs_dict[optional], optional)
            normalized[optional] = value.reshape((count, -1)).astype(np.float32, copy=False)
    return normalized


def normalize_lito_ply_storage(ply_storage: str) -> str:
    """Normalize the checkpoint-backed LiTo PLY storage mode."""

    normalized = str(ply_storage).strip().lower().replace("-", "_")
    if normalized == "binary":
        normalized = "binary_little_endian"
    if normalized not in LITO_PLY_STORAGES:
        raise ValueError(f"unsupported LiTo PLY storage: {ply_storage}")
    return normalized


def write_lito_gaussians_ply(
    path: str | Path,
    gaussians: dict[str, Any],
    *,
    ply_storage: str = LITO_DEFAULT_PLY_STORAGE,
) -> None:
    """Write checkpoint-backed LiTo Gaussians using upstream 3DGS PLY fields."""

    ply_storage = normalize_lito_ply_storage(ply_storage)
    normalized = normalize_lito_gs_dict(gaussians)
    xyz = normalized["xyz_w"]
    count = int(xyz.shape[0])
    scaling = np.log(np.clip(normalized["scaling"], 1e-12, None)).astype(np.float32)
    quaternion = normalized["quaternion"]
    opacity = _logit(np.clip(normalized["opacity"], 1e-6, 1.0 - 1e-6)).astype(np.float32)
    rgb_sh = _flatten_rgb_sh(normalized["rgb_sh"], count=count, require_square=True)
    sh_coeffs = int(rgb_sh.shape[1])
    sh_degree = _sh_degree_from_coeff_count(sh_coeffs)
    normal = np.zeros_like(xyz, dtype=np.float32)
    f_dc = rgb_sh[:, :1, :].transpose(0, 2, 1).reshape(count, -1).astype(np.float32)
    rest_coeffs = (sh_degree + 1) ** 2 - 1
    if rest_coeffs > 0:
        f_rest = rgb_sh[:, 1 : 1 + rest_coeffs, :].transpose(0, 2, 1).reshape(count, -1).astype(np.float32)
    else:
        f_rest = np.zeros((count, 0), dtype=np.float32)

    properties = ["x", "y", "z", "nx", "ny", "nz"]
    properties.extend(f"f_dc_{index}" for index in range(f_dc.shape[1]))
    properties.extend(f"f_rest_{index}" for index in range(f_rest.shape[1]))
    properties.append("opacity")
    properties.extend(f"scale_{index}" for index in range(3))
    properties.extend(f"rot_{index}" for index in range(4))
    rows = np.concatenate([xyz, normal, f_dc, f_rest, opacity, scaling, quaternion], axis=1)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", f"format {ply_storage} 1.0", "comment mlx-spatial LiTo checkpoint-backed 3DGS export"]
    header.append(f"element vertex {count}")
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")
    if ply_storage == "ascii":
        with output.open("w", encoding="ascii", newline="\n") as handle:
            handle.write("\n".join(header))
            handle.write("\n")
            for row in rows:
                handle.write(" ".join(f"{float(value):.8g}" for value in row))
                handle.write("\n")
        return

    with output.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        rows.astype("<f4", copy=False).tofile(handle)


def _read_safetensor_headers(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    if not path.is_file():
        raise FileNotFoundError(f"LiTo checkpoint not found: {path}")
    headers: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(path, framework="np") as handle:
        for key in handle.keys():
            tensor_slice = handle.get_slice(key)
            headers[key] = (tuple(int(dim) for dim in tensor_slice.get_shape()), str(tensor_slice.get_dtype()))
    return headers


def _normalize_name_filter(names: Iterable[str] | None) -> set[str] | None:
    if names is None:
        return None
    if isinstance(names, str):
        raise ValueError("names must be an iterable of strings, not a string")
    requested = {name for name in names}
    if not requested or any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("names must contain non-empty strings")
    return requested


def _raise_missing_requested_names(kind: str, requested: set[str] | None, arrays: dict[str, np.ndarray]) -> None:
    if requested is None:
        return
    missing = sorted(requested.difference(arrays))
    if missing:
        raise ValueError(f"{kind} checkpoint did not contain requested remapped weights: {', '.join(missing)}")


def _remap_lito_dit_key(key: str) -> str:
    remaps = (
        ("t_proj.0.", "t_proj_linear1."),
        ("t_proj.2.", "t_proj_linear2."),
        ("t0_proj.1.", "t0_proj_linear."),
        ("final_layer.adaLN_modulation.0.", "final_layer.adaLN_linear1."),
        ("final_layer.adaLN_modulation.2.", "final_layer.adaLN_linear2."),
    )
    for source, target in remaps:
        if key.startswith(source):
            return target + key[len(source) :]
    return key


def _remap_lito_gaussian_decoder_tensor(key: str, tensor: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if ".w12." in key:
        prefix, suffix = key.split(".w12.", 1)
        half = int(tensor.shape[0]) // 2
        return [
            (_remap_lito_gaussian_decoder_key(f"{prefix}.w1.{suffix}"), tensor[:half]),
            (_remap_lito_gaussian_decoder_key(f"{prefix}.w2.{suffix}"), tensor[half:]),
        ]
    return [(_remap_lito_gaussian_decoder_key(key), tensor)]


def _remap_lito_gaussian_decoder_key(key: str) -> str:
    sequential_prefixes = ("point_mlp", "gs_output_shape_mlp", "gs_output_color_mlp")
    for prefix in sequential_prefixes:
        dotted = f"{prefix}."
        if not key.startswith(dotted):
            continue
        rest = key[len(dotted) :]
        if rest.startswith("0."):
            return f"{prefix}.norms.0.{rest[2:]}"
        if rest.startswith("1.w"):
            return f"{prefix}.mlps.0.{rest[2:]}"
        if rest.startswith("2."):
            return f"{prefix}.final_layer.{rest[2:]}"
    if key.startswith(("perceiver.", "point_linear.", "xyz_encoding.")):
        return key
    raise ValueError(f"unmapped LiTo Gaussian decoder tensor: {key}")


def _reshape_decoder_output(
    values: np.ndarray,
    name: str,
    count: int,
    expansion_ratio: int,
    expected_dim: int,
) -> np.ndarray:
    if values.ndim == 2:
        expected_flat = expansion_ratio * expected_dim
        if values.shape != (count, expected_flat):
            raise ValueError(f"{name} must have shape ({count}, {expected_flat}), got {values.shape}")
        return values.reshape((count, expansion_ratio, expected_dim)).astype(np.float32, copy=False)
    if values.ndim == 3 and values.shape == (count, expansion_ratio, expected_dim):
        return values.astype(np.float32, copy=False)
    raise ValueError(f"{name} must have shape ({count}, {expansion_ratio}, {expected_dim}), got {values.shape}")


def _run_lito_output_mlp(query: mx.array, weights: dict[str, Any], prefix: str) -> mx.array:
    x = _mx_layer_norm(
        query,
        _mx_weight(weights, f"{prefix}.norms.0.weight"),
        _mx_weight(weights, f"{prefix}.norms.0.bias"),
        eps=1e-6,
    )
    x = _run_lito_swiglu(x, weights, f"{prefix}.mlps.0")
    x = _mx_layer_norm(
        x,
        _mx_weight(weights, f"{prefix}.final_layer.norm_final.weight"),
        _mx_weight(weights, f"{prefix}.final_layer.norm_final.bias"),
        eps=1e-6,
    )
    return _mx_linear(
        x,
        _mx_weight(weights, f"{prefix}.final_layer.linear.weight"),
        _mx_weight(weights, f"{prefix}.final_layer.linear.bias"),
    )


def _run_lito_swiglu(values: mx.array, weights: dict[str, Any], prefix: str) -> mx.array:
    w1 = _mx_weight(weights, f"{prefix}.w1.weight")
    w2 = _mx_weight(weights, f"{prefix}.w2.weight")
    w3 = _mx_weight(weights, f"{prefix}.w3.weight")
    b1 = _mx_optional_weight(weights, f"{prefix}.w1.bias")
    b2 = _mx_optional_weight(weights, f"{prefix}.w2.bias")
    b3 = _mx_optional_weight(weights, f"{prefix}.w3.bias")
    return _mx_linear(_mx_silu(_mx_linear(values, w1, b1)) * _mx_linear(values, w2, b2), w3, b3)


def _run_lito_gaussian_cross_attention_layer(
    query: mx.array,
    key_value: mx.array,
    weights: dict[str, Any],
    prefix: str,
    *,
    q_seq_lens: list[int],
    kv_seq_lens: list[int],
    num_heads: int,
) -> mx.array:
    query_dim = int(query.shape[-1])
    kv_dim = int(key_value.shape[-1])
    if int(_mx_weight(weights, f"{prefix}.layernorm_q.weight").shape[0]) != query_dim:
        raise ValueError(f"{prefix}.layernorm_q does not match query dim {query_dim}")
    if int(_mx_weight(weights, f"{prefix}.layernorm_kv.weight").shape[0]) != kv_dim:
        raise ValueError(f"{prefix}.layernorm_kv does not match latent dim {kv_dim}")

    q = _mx_layer_norm(
        query,
        _mx_weight(weights, f"{prefix}.layernorm_q.weight"),
        _mx_weight(weights, f"{prefix}.layernorm_q.bias"),
        eps=1e-5,
    )
    kv = _mx_layer_norm(
        key_value,
        _mx_weight(weights, f"{prefix}.layernorm_kv.weight"),
        _mx_weight(weights, f"{prefix}.layernorm_kv.bias"),
        eps=1e-5,
    )
    q = _mx_linear(q, _mx_weight(weights, f"{prefix}.linear_q.weight"), _mx_weight(weights, f"{prefix}.linear_q.bias"))
    kv = _mx_linear(
        kv,
        _mx_weight(weights, f"{prefix}.linear_kv.weight"),
        _mx_weight(weights, f"{prefix}.linear_kv.bias"),
    )
    k, v = mx.split(kv, 2, axis=-1)
    q = _mx_rms_norm(q, _mx_weight(weights, f"{prefix}.rmsnorm_q.scale"))
    k = _mx_rms_norm(k, _mx_weight(weights, f"{prefix}.rmsnorm_k.scale"))

    qkv_dim = int(q.shape[-1])
    if qkv_dim % int(num_heads) != 0:
        raise ValueError(f"qkv dim {qkv_dim} must be divisible by num_heads {num_heads}")
    head_dim = qkv_dim // int(num_heads)
    q = q.reshape(int(q.shape[0]), int(num_heads), head_dim)
    k = k.reshape(int(k.shape[0]), int(num_heads), head_dim)
    v = v.reshape(int(v.shape[0]), int(num_heads), head_dim)

    q_batched, _ = _packed_to_batched(q, q_seq_lens)
    k_batched, kv_mask = _packed_to_batched(k, kv_seq_lens)
    v_batched, _ = _packed_to_batched(v, kv_seq_lens)
    q_batched = mx.transpose(q_batched, axes=(0, 2, 1, 3))
    k_batched = mx.transpose(k_batched, axes=(0, 2, 1, 3))
    v_batched = mx.transpose(v_batched, axes=(0, 2, 1, 3))
    out = mx.fast.scaled_dot_product_attention(
        q_batched,
        k_batched,
        v_batched,
        scale=head_dim**-0.5,
        mask=kv_mask[:, None, None, :],
    )
    out = mx.transpose(out, axes=(0, 2, 1, 3))
    out = out.reshape(out.shape[0], out.shape[1], qkv_dim)
    out = _batched_to_packed(out, q_seq_lens)
    return _mx_linear(
        out,
        _mx_weight(weights, f"{prefix}.linear_out.weight"),
        _mx_weight(weights, f"{prefix}.linear_out.bias"),
    )


def _run_lito_gaussian_self_attention_layer(
    values: mx.array,
    weights: dict[str, Any],
    prefix: str,
    *,
    q_seq_lens: list[int],
    voxel_info: dict[str, Any],
    num_heads: int,
) -> mx.array:
    qkv = _mx_linear(
        values,
        _mx_weight(weights, f"{prefix}.linear_qkv.weight"),
        _mx_weight(weights, f"{prefix}.linear_qkv.bias"),
    )
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = _mx_rms_norm(q, _mx_weight(weights, f"{prefix}.rmsnorm_q.scale"))
    k = _mx_rms_norm(k, _mx_weight(weights, f"{prefix}.rmsnorm_k.scale"))
    qkv_dim = int(q.shape[-1])
    if qkv_dim % int(num_heads) != 0:
        raise ValueError(f"qkv dim {qkv_dim} must be divisible by num_heads {num_heads}")
    head_dim = qkv_dim // int(num_heads)
    q = q.reshape(int(q.shape[0]), int(num_heads), head_dim)
    k = k.reshape(int(k.shape[0]), int(num_heads), head_dim)
    v = v.reshape(int(v.shape[0]), int(num_heads), head_dim)
    out = _run_lito_localized_voxel_attention(q, k, v, voxel_info)
    out = out.reshape(int(out.shape[0]), qkv_dim)
    if sum(q_seq_lens) != int(out.shape[0]):
        raise ValueError(f"q_seq_lens must sum to packed length {out.shape[0]}, got {sum(q_seq_lens)}")
    return _mx_linear(
        out,
        _mx_weight(weights, f"{prefix}.linear_out.weight"),
        _mx_weight(weights, f"{prefix}.linear_out.bias"),
    )


def _run_lito_gaussian_perceiver_block_with_local_voxel_self_attention(
    query: mx.array,
    latent_tokens: mx.array,
    weights: dict[str, Any],
    block_index: int,
    *,
    q_seq_lens: list[int],
    kv_seq_lens: list[int],
    voxel_infos: list[dict[str, Any]],
    num_heads: int,
) -> mx.array:
    prefix = f"perceiver.blocks.{block_index}"
    latent = _mx_linear(latent_tokens, _mx_weight(weights, f"{prefix}.kv_linear.weight"), None)
    query = query + _run_lito_gaussian_cross_attention_layer(
        query,
        latent,
        weights,
        f"{prefix}.ca_layer",
        q_seq_lens=q_seq_lens,
        kv_seq_lens=kv_seq_lens,
        num_heads=num_heads,
    )
    ca_ln = _mx_layer_norm(
        query,
        _mx_weight(weights, f"{prefix}.ca_ln.weight"),
        _mx_weight(weights, f"{prefix}.ca_ln.bias"),
        eps=1e-6,
    )
    query = query + _run_lito_swiglu(ca_ln, weights, f"{prefix}.ca_mlp")
    for layer_index, voxel_info in enumerate(voxel_infos):
        normed = _mx_layer_norm(
            query,
            _mx_weight(weights, f"{prefix}.ln1_layers.{layer_index}.weight"),
            _mx_weight(weights, f"{prefix}.ln1_layers.{layer_index}.bias"),
            eps=1e-6,
        )
        query = query + _run_lito_gaussian_self_attention_layer(
            normed,
            weights,
            f"{prefix}.sa_layers.{layer_index}",
            q_seq_lens=q_seq_lens,
            voxel_info=voxel_info,
            num_heads=num_heads,
        )
        normed = _mx_layer_norm(
            query,
            _mx_weight(weights, f"{prefix}.ln2_layers.{layer_index}.weight"),
            _mx_weight(weights, f"{prefix}.ln2_layers.{layer_index}.bias"),
            eps=1e-6,
        )
        query = query + _run_lito_swiglu(normed, weights, f"{prefix}.mlp_layers.{layer_index}")
    return query


def _run_lito_dit_timestep_embedding(timestep: Any, batch_size: int, weights: dict[str, Any]) -> tuple[mx.array, mx.array]:
    t = mx.array(timestep, dtype=mx.float32)
    if t.ndim == 0:
        t = mx.broadcast_to(t, (int(batch_size),))
    if t.ndim == 2 and int(t.shape[1]) == 1:
        t = mx.squeeze(t, axis=1)
    if t.ndim != 1 or int(t.shape[0]) != int(batch_size):
        raise ValueError(f"timestep must be scalar or have shape ({batch_size},), got {t.shape}")
    t_encoded = _fourier_encode_xyz(t[:, None], _mx_weight(weights, "t_embedder.freq_bands"))
    t_emb = _mx_linear(t_encoded, _mx_weight(weights, "t_proj_linear1.weight"), _mx_weight(weights, "t_proj_linear1.bias"))
    t_emb = _mx_silu(t_emb)
    t_emb = _mx_linear(t_emb, _mx_weight(weights, "t_proj_linear2.weight"), _mx_weight(weights, "t_proj_linear2.bias"))
    t0 = _mx_linear(
        _mx_silu(t_emb),
        _mx_weight(weights, "t0_proj_linear.weight"),
        _mx_weight(weights, "t0_proj_linear.bias"),
    )
    return t_emb, t0


def _run_lito_dit_condition_embedder(
    condition_tokens: mx.array,
    weights: dict[str, Any],
    *,
    cond_drop_ids: Any | None,
) -> mx.array:
    cond = condition_tokens
    if cond_drop_ids is not None:
        drop_ids = mx.array(cond_drop_ids, dtype=mx.bool_)
        if drop_ids.ndim != 1 or int(drop_ids.shape[0]) != int(cond.shape[0]):
            raise ValueError(f"cond_drop_ids must have shape ({cond.shape[0]},), got {drop_ids.shape}")
        y_embedding = _mx_weight(weights, "cond_embedder.y_embedding")
        cond = mx.where(drop_ids[:, None, None], y_embedding[None, None, :], cond)
    hidden = _mx_linear(
        cond,
        _mx_weight(weights, "cond_embedder.y_proj.fc1.weight"),
        _mx_weight(weights, "cond_embedder.y_proj.fc1.bias"),
    )
    hidden = _mx_gelu_tanh(hidden)
    return _mx_linear(
        hidden,
        _mx_weight(weights, "cond_embedder.y_proj.fc2.weight"),
        _mx_weight(weights, "cond_embedder.y_proj.fc2.bias"),
    )


def _run_lito_dit_block(
    latent: mx.array,
    condition: mx.array,
    timestep_modulation: mx.array,
    weights: dict[str, Any],
    block_index: int,
    *,
    num_heads: int,
) -> mx.array:
    prefix = f"blocks.{block_index}"
    dim = int(latent.shape[-1])
    mod = _mx_weight(weights, f"{prefix}.scale_shift_table")[None, :, :] + timestep_modulation.reshape(
        int(latent.shape[0]), 6, dim
    )
    shift_msa = mod[:, 0:1, :]
    scale_msa = mod[:, 1:2, :]
    gate_msa = mod[:, 2:3, :]
    shift_mlp = mod[:, 3:4, :]
    scale_mlp = mod[:, 4:5, :]
    gate_mlp = mod[:, 5:6, :]

    normed = _mx_layer_norm_no_affine(latent, eps=1e-6)
    latent = latent + gate_msa * _run_lito_batched_self_attention_layer(
        _mx_modulate(normed, shift_msa, scale_msa),
        weights,
        f"{prefix}.attn",
        num_heads=num_heads,
    )
    latent = latent + _run_lito_batched_cross_attention_layer(
        latent,
        condition,
        weights,
        f"{prefix}.cross_attn",
        num_heads=num_heads,
    )
    latent = _mx_layer_norm_no_affine(latent, eps=1e-6)
    latent = latent + gate_mlp * _run_lito_dit_swiglu(_mx_modulate(latent, shift_mlp, scale_mlp), weights, f"{prefix}.mlp")
    return latent


def _run_lito_dit_final_layer(latent: mx.array, timestep_embedding: mx.array, weights: dict[str, Any]) -> mx.array:
    ada = _mx_linear(
        timestep_embedding,
        _mx_weight(weights, "final_layer.adaLN_linear1.weight"),
        _mx_weight(weights, "final_layer.adaLN_linear1.bias"),
    )
    ada = _mx_silu(ada)
    ada = _mx_linear(
        ada,
        _mx_weight(weights, "final_layer.adaLN_linear2.weight"),
        _mx_weight(weights, "final_layer.adaLN_linear2.bias"),
    )
    shift, scale = mx.split(ada, 2, axis=-1)
    latent = _mx_modulate(_mx_layer_norm_no_affine(latent, eps=1e-6), shift[:, None, :], scale[:, None, :])
    return _mx_linear(
        latent,
        _mx_weight(weights, "final_layer.linear.weight"),
        _mx_weight(weights, "final_layer.linear.bias"),
    )


def _run_lito_dit_swiglu(values: mx.array, weights: dict[str, Any], prefix: str) -> mx.array:
    hidden = _mx_silu(
        _mx_linear(values, _mx_weight(weights, f"{prefix}.w1.weight"), _mx_optional_weight(weights, f"{prefix}.w1.bias"))
    ) * _mx_linear(values, _mx_weight(weights, f"{prefix}.w3.weight"), _mx_optional_weight(weights, f"{prefix}.w3.bias"))
    return _mx_linear(hidden, _mx_weight(weights, f"{prefix}.w2.weight"), _mx_weight(weights, f"{prefix}.w2.bias"))


def _build_lito_voxel_init_query(batch_size: int, weights: dict[str, Any]) -> mx.array:
    init_query = _mx_weight(weights, "net.init_query")
    if init_query.ndim != 4:
        raise ValueError(f"net.init_query must have shape (Z, Y, X, C), got {init_query.shape}")
    z, y, x, dim = (int(value) for value in init_query.shape)
    zyx = _lito_regular_grid_coords(z, y, x)
    encoded = _fourier_encode_with_input(zyx, _mx_weight(weights, "net.zyx_pos_encoder.freq_bands"))
    pos = _mx_linear(
        encoded,
        _mx_weight(weights, "net.init_query_linear.weight"),
        _mx_weight(weights, "net.init_query_linear.bias"),
    ).reshape(z, y, x, dim)
    query = init_query + pos
    query = mx.broadcast_to(query[None, ...], (int(batch_size), z, y, x, dim))
    return query.reshape(int(batch_size), z * y * x, dim)


def _run_lito_voxel_perceiver_block(
    query: mx.array,
    latent_tokens: mx.array,
    weights: dict[str, Any],
    block_index: int,
    *,
    num_heads: int,
) -> mx.array:
    prefix = f"net.encoder.blocks.{block_index}"
    query = query + _run_lito_batched_cross_attention_layer(
        query,
        latent_tokens,
        weights,
        f"{prefix}.ca_layer",
        num_heads=num_heads,
    )
    ca_ln = _mx_layer_norm(
        query,
        _mx_weight(weights, f"{prefix}.ca_ln.weight"),
        _mx_weight(weights, f"{prefix}.ca_ln.bias"),
        eps=1e-6,
    )
    query = query + _run_lito_timm_mlp(ca_ln, weights, f"{prefix}.ca_mlp")
    layer_indices = _normalize_voxel_self_layer_indices(prefix, weights)
    for layer_index in layer_indices:
        normed = _mx_layer_norm(
            query,
            _mx_weight(weights, f"{prefix}.ln1_layers.{layer_index}.weight"),
            _mx_weight(weights, f"{prefix}.ln1_layers.{layer_index}.bias"),
            eps=1e-6,
        )
        query = query + _run_lito_batched_self_attention_layer(
            normed,
            weights,
            f"{prefix}.sa_layers.{layer_index}",
            num_heads=num_heads,
        )
        normed = _mx_layer_norm(
            query,
            _mx_weight(weights, f"{prefix}.ln2_layers.{layer_index}.weight"),
            _mx_weight(weights, f"{prefix}.ln2_layers.{layer_index}.bias"),
            eps=1e-6,
        )
        query = query + _run_lito_timm_mlp(normed, weights, f"{prefix}.mlp_layers.{layer_index}")
    return query


def _run_lito_batched_cross_attention_layer(
    query: mx.array,
    key_value: mx.array,
    weights: dict[str, Any],
    prefix: str,
    *,
    num_heads: int,
) -> mx.array:
    q = _mx_layer_norm(
        query,
        _mx_weight(weights, f"{prefix}.layernorm_q.weight"),
        _mx_weight(weights, f"{prefix}.layernorm_q.bias"),
        eps=1e-5,
    )
    kv = _mx_layer_norm(
        key_value,
        _mx_weight(weights, f"{prefix}.layernorm_kv.weight"),
        _mx_weight(weights, f"{prefix}.layernorm_kv.bias"),
        eps=1e-5,
    )
    q = _mx_linear(q, _mx_weight(weights, f"{prefix}.linear_q.weight"), _mx_weight(weights, f"{prefix}.linear_q.bias"))
    kv = _mx_linear(
        kv,
        _mx_weight(weights, f"{prefix}.linear_kv.weight"),
        _mx_weight(weights, f"{prefix}.linear_kv.bias"),
    )
    k, v = mx.split(kv, 2, axis=-1)
    q = _mx_rms_norm(q, _mx_weight(weights, f"{prefix}.rmsnorm_q.scale"))
    k = _mx_rms_norm(k, _mx_weight(weights, f"{prefix}.rmsnorm_k.scale"))
    out = _run_lito_batched_attention(q, k, v, num_heads=num_heads)
    return _mx_linear(
        out,
        _mx_weight(weights, f"{prefix}.linear_out.weight"),
        _mx_weight(weights, f"{prefix}.linear_out.bias"),
    )


def _run_lito_batched_self_attention_layer(
    values: mx.array,
    weights: dict[str, Any],
    prefix: str,
    *,
    num_heads: int,
) -> mx.array:
    qkv = _mx_linear(
        values,
        _mx_weight(weights, f"{prefix}.linear_qkv.weight"),
        _mx_weight(weights, f"{prefix}.linear_qkv.bias"),
    )
    q, k, v = mx.split(qkv, 3, axis=-1)
    q = _mx_rms_norm(q, _mx_weight(weights, f"{prefix}.rmsnorm_q.scale"))
    k = _mx_rms_norm(k, _mx_weight(weights, f"{prefix}.rmsnorm_k.scale"))
    out = _run_lito_batched_attention(q, k, v, num_heads=num_heads)
    return _mx_linear(
        out,
        _mx_weight(weights, f"{prefix}.linear_out.weight"),
        _mx_weight(weights, f"{prefix}.linear_out.bias"),
    )


def _run_lito_batched_attention(q: mx.array, k: mx.array, v: mx.array, *, num_heads: int) -> mx.array:
    qkv_dim = int(q.shape[-1])
    if qkv_dim % int(num_heads) != 0:
        raise ValueError(f"qkv dim {qkv_dim} must be divisible by num_heads {num_heads}")
    head_dim = qkv_dim // int(num_heads)
    batch = int(q.shape[0])
    q = q.reshape(batch, int(q.shape[1]), int(num_heads), head_dim)
    k = k.reshape(batch, int(k.shape[1]), int(num_heads), head_dim)
    v = v.reshape(batch, int(v.shape[1]), int(num_heads), head_dim)
    q = mx.transpose(q, axes=(0, 2, 1, 3))
    k = mx.transpose(k, axes=(0, 2, 1, 3))
    v = mx.transpose(v, axes=(0, 2, 1, 3))
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=head_dim**-0.5)
    out = mx.transpose(out, axes=(0, 2, 1, 3))
    return out.reshape(batch, int(out.shape[1]), qkv_dim)


def _run_lito_timm_mlp(values: mx.array, weights: dict[str, Any], prefix: str) -> mx.array:
    hidden = _mx_linear(
        values,
        _mx_weight(weights, f"{prefix}.fc1.weight"),
        _mx_weight(weights, f"{prefix}.fc1.bias"),
    )
    hidden = _mx_gelu_tanh(hidden)
    return _mx_linear(
        hidden,
        _mx_weight(weights, f"{prefix}.fc2.weight"),
        _mx_weight(weights, f"{prefix}.fc2.bias"),
    )


def _run_lito_localized_voxel_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    voxel_info: dict[str, Any],
) -> mx.array:
    forward_idxs = voxel_info["forward_idxs"]
    backward_idxs = voxel_info["backward_idxs"]
    cu_seq_lens_list = voxel_info["cu_seq_lens"]
    max_seq_lens_list = voxel_info["max_seq_lens"]
    chunk_start_idxs = voxel_info["chunk_start_idxs"]

    sorted_q = q[forward_idxs]
    sorted_k = k[forward_idxs]
    sorted_v = v[forward_idxs]
    out_chunks: list[mx.array] = []
    for chunk_index, cu in enumerate(cu_seq_lens_list):
        chunk_start = int(chunk_start_idxs[chunk_index])
        chunk_end = int(chunk_start_idxs[chunk_index + 1])
        max_len = int(max_seq_lens_list[chunk_index])
        chunk_q = sorted_q[chunk_start:chunk_end]
        chunk_k = sorted_k[chunk_start:chunk_end]
        chunk_v = sorted_v[chunk_start:chunk_end]
        cell_lens = np.diff(np.asarray(cu, dtype=np.int32)).astype(np.int32)
        cu_list = np.asarray(cu, dtype=np.int32).tolist()
        padded_q = _pad_voxel_cells(chunk_q, cell_lens, cu_list, max_len)
        padded_k = _pad_voxel_cells(chunk_k, cell_lens, cu_list, max_len)
        padded_v = _pad_voxel_cells(chunk_v, cell_lens, cu_list, max_len)
        padded_q = mx.transpose(padded_q, axes=(0, 2, 1, 3))
        padded_k = mx.transpose(padded_k, axes=(0, 2, 1, 3))
        padded_v = mx.transpose(padded_v, axes=(0, 2, 1, 3))
        mask = mx.arange(max_len)[None, :] < mx.array(cell_lens, dtype=mx.int32)[:, None]
        chunk_out = mx.fast.scaled_dot_product_attention(
            padded_q,
            padded_k,
            padded_v,
            scale=int(q.shape[-1]) ** -0.5,
            mask=mask[:, None, None, :],
        )
        chunk_out = mx.transpose(chunk_out, axes=(0, 2, 1, 3))
        out_chunks.append(_unpad_voxel_cells(chunk_out, cell_lens))
    return mx.concatenate(out_chunks, axis=0)[backward_idxs]


def _pad_voxel_cells(chunk: mx.array, cell_lens: np.ndarray, cu_list: list[int], max_len: int) -> mx.array:
    if max_len <= 0:
        raise ValueError("localized voxel attention requires non-empty cells")
    cells = []
    for cell_index, cell_len in enumerate(cell_lens.tolist()):
        start = int(cu_list[cell_index])
        part = chunk[start : start + int(cell_len)]
        pad_width = [(0, max_len - int(cell_len))] + [(0, 0)] * (chunk.ndim - 1)
        cells.append(mx.pad(part, pad_width))
    return mx.stack(cells, axis=0)


def _unpad_voxel_cells(chunk: mx.array, cell_lens: np.ndarray) -> mx.array:
    return mx.concatenate([chunk[index, : int(cell_len)] for index, cell_len in enumerate(cell_lens.tolist())], axis=0)


def _mx_weight(weights: dict[str, Any], name: str) -> mx.array:
    try:
        return mx.array(weights[name], dtype=mx.float32)
    except KeyError as error:
        raise ValueError(f"missing LiTo checkpoint weight: {name}") from error


def _mx_optional_weight(weights: dict[str, Any], name: str) -> mx.array | None:
    if name not in weights:
        return None
    return mx.array(weights[name], dtype=mx.float32)


def _mx_linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def _mx_layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values.astype(mx.float32), axis=-1, keepdims=True)
    centered = values.astype(mx.float32) - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight + bias


def _mx_layer_norm_no_affine(values: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values.astype(mx.float32), axis=-1, keepdims=True)
    centered = values.astype(mx.float32) - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps)


def _mx_rms_norm(values: mx.array, scale: mx.array, *, eps: float = 1e-8) -> mx.array:
    values_f = values.astype(mx.float32)
    return values_f * mx.rsqrt(mx.mean(values_f * values_f, axis=-1, keepdims=True) + eps) * scale


def _mx_silu(values: mx.array) -> mx.array:
    return values * mx.sigmoid(values)


def _mx_modulate(values: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return values * (1.0 + scale) + shift


def _mx_gelu_tanh(values: mx.array) -> mx.array:
    return 0.5 * values * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * values * values * values)))


def _lito_rgba_to_bqhwc(cond_rgba: Any) -> mx.array:
    values = mx.array(cond_rgba, dtype=mx.float32)
    if values.ndim == 3:
        values = values[None, None, ...]
    elif values.ndim == 4:
        values = values[:, None, ...]
    if values.ndim != 5 or int(values.shape[-1]) != 4:
        raise ValueError(f"cond_rgba must have shape (H,W,4), (B,H,W,4), or (B,Q,H,W,4), got {values.shape}")
    return values


def _run_lito_dino_vitl14_reg(
    premultiplied_rgb: mx.array,
    weights: dict[str, Any],
    *,
    block_indices: Iterable[int] | None,
    config: LitoPatchEncoderConfig,
) -> tuple[mx.array, tuple[int, int]]:
    x = premultiplied_rgb.astype(mx.float32)
    if x.ndim != 4 or int(x.shape[1]) != 3:
        raise ValueError(f"premultiplied_rgb must have shape (B,3,H,W), got {x.shape}")
    if int(x.shape[2]) != config.input_size or int(x.shape[3]) != config.input_size:
        x = _resize_linear_bchw(x, (config.input_size, config.input_size))
    if int(x.shape[2]) % config.patch_size or int(x.shape[3]) % config.patch_size:
        raise ValueError(f"LiTo DINO input size must be divisible by patch size {config.patch_size}, got {x.shape}")

    mean = mx.array(_LITO_DINO_IMAGE_MEAN, dtype=x.dtype)[None, :, None, None]
    std = mx.array(_LITO_DINO_IMAGE_STD, dtype=x.dtype)[None, :, None, None]
    x = (x - mean) / std
    patch_map = _conv2d_nchw(
        x,
        _mx_weight(weights, "dinov2_model.model.patch_embed.proj.weight"),
        _mx_weight(weights, "dinov2_model.model.patch_embed.proj.bias"),
        stride=config.patch_size,
    )
    patch_h, patch_w = int(patch_map.shape[2]), int(patch_map.shape[3])
    patch_tokens = mx.reshape(
        mx.transpose(patch_map, (0, 2, 3, 1)),
        (int(patch_map.shape[0]), patch_h * patch_w, config.embed_dim),
    )
    cls = mx.broadcast_to(
        _mx_weight(weights, "dinov2_model.model.cls_token").astype(patch_tokens.dtype),
        (int(patch_tokens.shape[0]), 1, config.embed_dim),
    )
    tokens = mx.concatenate((cls, patch_tokens), axis=1)
    tokens = tokens + _interpolate_lito_dino_pos_embed(
        _mx_weight(weights, "dinov2_model.model.pos_embed"),
        (patch_h, patch_w),
        config=config,
    ).astype(tokens.dtype)
    registers = mx.broadcast_to(
        _mx_weight(weights, "dinov2_model.model.register_tokens").astype(tokens.dtype),
        (int(tokens.shape[0]), config.register_count, config.embed_dim),
    )
    hidden = mx.concatenate((tokens[:, :1, :], registers, tokens[:, 1:, :]), axis=1)
    for block_index in _normalize_dino_block_indices(block_indices, weights):
        hidden = _run_lito_dino_block(hidden, weights, block_index, config=config)
        mx.eval(hidden)
    if config.normalize_concat_tokens:
        hidden = _mx_layer_norm_no_affine(hidden, eps=1e-5)
    return hidden, (patch_h, patch_w)


def _run_lito_rgba_learnable_branch(
    premultiplied_rgb: mx.array,
    alpha: mx.array,
    weights: dict[str, Any],
    *,
    patch_grid: tuple[int, int],
    config: LitoPatchEncoderConfig,
) -> mx.array:
    if alpha.ndim != 4 or int(alpha.shape[1]) != 1:
        raise ValueError(f"alpha must have shape (B,1,H,W), got {alpha.shape}")
    mean = mx.array(_LITO_DINO_IMAGE_MEAN, dtype=premultiplied_rgb.dtype)[None, :, None, None]
    std = mx.array(_LITO_DINO_IMAGE_STD, dtype=premultiplied_rgb.dtype)[None, :, None, None]
    rgb = (premultiplied_rgb.astype(mx.float32) - mean) / std
    rgba_features = mx.concatenate((rgb, alpha.astype(mx.float32)), axis=1)
    patch_map = _conv2d_nchw(
        rgba_features,
        _mx_weight(weights, "learnable_model.weight"),
        _mx_weight(weights, "learnable_model.bias"),
        stride=config.patch_size,
    )
    patch_h, patch_w = patch_grid
    if int(patch_map.shape[2]) != patch_h or int(patch_map.shape[3]) != patch_w:
        raise ValueError(f"RGBA branch patch grid {patch_map.shape[2:4]} does not match DINO grid {patch_grid}")
    tokens = mx.reshape(
        mx.transpose(patch_map, (0, 2, 3, 1)),
        (int(patch_map.shape[0]), patch_h * patch_w, config.embed_dim),
    )
    paddings = _mx_weight(weights, "learnable_paddings")
    expected_extra = 1 + config.register_count
    if tuple(int(value) for value in paddings.shape) != (expected_extra, config.embed_dim):
        raise ValueError(f"learnable_paddings must have shape ({expected_extra}, {config.embed_dim}), got {paddings.shape}")
    paddings = mx.broadcast_to(paddings[None, :, :].astype(tokens.dtype), (int(tokens.shape[0]), expected_extra, config.embed_dim))
    return mx.concatenate((paddings, tokens), axis=1)


def _run_lito_dino_block(
    hidden: mx.array,
    weights: dict[str, Any],
    block_index: int,
    *,
    config: LitoPatchEncoderConfig,
) -> mx.array:
    prefix = f"dinov2_model.model.blocks.{block_index}"
    residual = hidden
    normalized = _mx_layer_norm(
        hidden,
        _mx_weight(weights, f"{prefix}.norm1.weight"),
        _mx_weight(weights, f"{prefix}.norm1.bias"),
        eps=1e-6,
    )
    attended = _run_lito_dino_self_attention(
        normalized,
        weights,
        prefix=f"{prefix}.attn",
        num_heads=config.num_heads,
        attention_chunk_size=config.attention_chunk_size,
        max_attention_bytes=config.max_attention_bytes,
    )
    hidden = residual + attended * _mx_weight(weights, f"{prefix}.ls1.gamma").astype(attended.dtype)
    residual = hidden
    normalized = _mx_layer_norm(
        hidden,
        _mx_weight(weights, f"{prefix}.norm2.weight"),
        _mx_weight(weights, f"{prefix}.norm2.bias"),
        eps=1e-6,
    )
    mlp = _mx_linear(
        nn.gelu(_mx_linear(normalized, _mx_weight(weights, f"{prefix}.mlp.fc1.weight"), _mx_weight(weights, f"{prefix}.mlp.fc1.bias"))),
        _mx_weight(weights, f"{prefix}.mlp.fc2.weight"),
        _mx_weight(weights, f"{prefix}.mlp.fc2.bias"),
    )
    return residual + mlp * _mx_weight(weights, f"{prefix}.ls2.gamma").astype(mlp.dtype)


def _run_lito_dino_self_attention(
    tokens: mx.array,
    weights: dict[str, Any],
    *,
    prefix: str,
    num_heads: int,
    attention_chunk_size: int | None,
    max_attention_bytes: int,
) -> mx.array:
    batch, token_count, dim = tuple(int(value) for value in tokens.shape)
    if dim % int(num_heads) != 0:
        raise ValueError(f"DINO hidden dim {dim} must be divisible by num_heads {num_heads}")
    head_dim = dim // int(num_heads)
    qkv = _mx_linear(tokens, _mx_weight(weights, f"{prefix}.qkv.weight"), _mx_weight(weights, f"{prefix}.qkv.bias"))
    query, key, value = (
        mx.transpose(mx.reshape(part, (batch, token_count, int(num_heads), head_dim)), (0, 2, 1, 3))
        for part in mx.split(qkv, 3, axis=-1)
    )
    scale = head_dim**-0.5
    if attention_chunk_size is None:
        attention = mx.softmax((query @ mx.transpose(key, (0, 1, 3, 2))) * scale, axis=-1) @ value
    else:
        estimated = batch * int(num_heads) * min(token_count, int(attention_chunk_size)) * token_count * 4
        if estimated > int(max_attention_bytes):
            raise ValueError(f"LiTo DINO attention chunk exceeds activation guard ({estimated} > {max_attention_bytes} bytes)")
        key_t = mx.transpose(key, (0, 1, 3, 2))
        chunks = []
        for start in range(0, token_count, int(attention_chunk_size)):
            stop = min(start + int(attention_chunk_size), token_count)
            chunks.append(mx.softmax((query[:, :, start:stop, :] @ key_t) * scale, axis=-1) @ value)
        attention = mx.concatenate(chunks, axis=2)
    merged = mx.reshape(mx.transpose(attention, (0, 2, 1, 3)), (batch, token_count, dim))
    return _mx_linear(merged, _mx_weight(weights, f"{prefix}.proj.weight"), _mx_weight(weights, f"{prefix}.proj.bias"))


def _interpolate_lito_dino_pos_embed(
    pos_embed: mx.array,
    patch_grid: tuple[int, int],
    *,
    config: LitoPatchEncoderConfig,
) -> mx.array:
    patch_h, patch_w = patch_grid
    expected = patch_h * patch_w + 1
    if int(pos_embed.shape[1]) == expected:
        return pos_embed
    pos_np = np.asarray(pos_embed, dtype=np.float32)
    stored_count = int(pos_np.shape[1]) - 1
    stored_side = int(round(math.sqrt(stored_count)))
    if stored_side * stored_side != stored_count:
        raise ValueError(f"LiTo DINO positional embedding patch count is not square: {pos_np.shape}")
    cls = mx.array(pos_np[:, :1, :], dtype=mx.float32)
    patch = mx.array(pos_np[:, 1:, :].reshape(1, stored_side, stored_side, config.embed_dim), dtype=mx.float32)
    patch_bchw = mx.transpose(patch, (0, 3, 1, 2))
    resized = _resize_linear_bchw(patch_bchw, (patch_h, patch_w))
    patch_tokens = mx.reshape(mx.transpose(resized, (0, 2, 3, 1)), (1, patch_h * patch_w, config.embed_dim))
    return mx.concatenate((cls, patch_tokens), axis=1)


def _resize_linear_bchw(values: mx.array, size: tuple[int, int]) -> mx.array:
    height, width = (int(value) for value in size)
    if int(values.shape[2]) == height and int(values.shape[3]) == width:
        return values
    scale = (height / int(values.shape[2]), width / int(values.shape[3]))
    resized = nn.Upsample(scale, mode="linear", align_corners=False)(mx.transpose(values, (0, 2, 3, 1)))
    return mx.transpose(resized[:, :height, :width, :], (0, 3, 1, 2))


def _conv2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    out = mx.conv2d(
        mx.transpose(x, (0, 2, 3, 1)),
        mx.transpose(weight.astype(x.dtype), (0, 2, 3, 1)),
        stride=stride,
        padding=padding,
    )
    if bias is not None:
        out = out + bias.astype(out.dtype)[None, None, None, :]
    return mx.transpose(out, (0, 3, 1, 2))


def _normalize_seq_lens(seq_lens: Iterable[int] | None, total: int, name: str) -> list[int]:
    if seq_lens is None:
        return [int(total)]
    lens = [int(length) for length in seq_lens]
    if not lens or any(length <= 0 for length in lens):
        raise ValueError(f"{name} must contain positive sequence lengths")
    if sum(lens) != int(total):
        raise ValueError(f"{name} must sum to packed length {total}, got {sum(lens)}")
    return lens


def _normalize_perceiver_block_indices(
    block_indices: Iterable[int] | None,
    weights: dict[str, Any],
) -> list[int]:
    if block_indices is None:
        discovered = sorted(
            {
                int(match.group(1))
                for name in weights
                for match in [re.match(r"^perceiver\.blocks\.(\d+)\.", name)]
                if match is not None
            }
        )
        if not discovered:
            raise ValueError("Gaussian decoder weights do not contain perceiver block weights")
        return discovered
    normalized = [int(index) for index in block_indices]
    if not normalized:
        raise ValueError("block_indices must not be empty")
    if any(index < 0 for index in normalized):
        raise ValueError("block_indices must contain non-negative integers")
    return normalized


def _normalize_voxel_block_indices(
    block_indices: Iterable[int] | None,
    weights: dict[str, Any],
) -> list[int]:
    if block_indices is None:
        discovered = sorted(
            {
                int(match.group(1))
                for name in weights
                for match in [re.match(r"^net\.encoder\.blocks\.(\d+)\.", name)]
                if match is not None
            }
        )
        if not discovered:
            raise ValueError("Voxel decoder weights do not contain encoder block weights")
        return discovered
    normalized = [int(index) for index in block_indices]
    if not normalized:
        raise ValueError("block_indices must not be empty")
    if any(index < 0 for index in normalized):
        raise ValueError("block_indices must contain non-negative integers")
    return normalized


def _normalize_dit_block_indices(
    block_indices: Iterable[int] | None,
    weights: dict[str, Any],
) -> list[int]:
    if block_indices is None:
        discovered = sorted(
            {
                int(match.group(1))
                for name in weights
                for match in [re.match(r"^blocks\.(\d+)\.", name)]
                if match is not None
            }
        )
        if not discovered:
            raise ValueError("DiT weights do not contain transformer block weights")
        return discovered
    normalized = [int(index) for index in block_indices]
    if not normalized:
        raise ValueError("block_indices must not be empty")
    if any(index < 0 for index in normalized):
        raise ValueError("block_indices must contain non-negative integers")
    return normalized


def _normalize_dino_block_indices(
    block_indices: Iterable[int] | None,
    weights: dict[str, Any],
) -> list[int]:
    if block_indices is None:
        discovered = sorted(
            {
                int(match.group(1))
                for name in weights
                for match in [re.match(r"^dinov2_model\.model\.blocks\.(\d+)\.", name)]
                if match is not None
            }
        )
        if not discovered:
            raise ValueError("patch encoder weights do not contain DINO block weights")
        return discovered
    normalized = [int(index) for index in block_indices]
    if not normalized:
        raise ValueError("block_indices must not be empty")
    if any(index < 0 for index in normalized):
        raise ValueError("block_indices must contain non-negative integers")
    return normalized


def _normalize_voxel_self_layer_indices(prefix: str, weights: dict[str, Any]) -> list[int]:
    discovered = sorted(
        {
            int(match.group(1))
            for name in weights
            for match in [re.match(rf"^{re.escape(prefix)}\.sa_layers\.(\d+)\.", name)]
            if match is not None
        }
    )
    if not discovered:
        raise ValueError(f"Voxel decoder weights do not contain self-attention layers for {prefix}")
    return discovered


def _packed_to_batched(packed: mx.array, seq_lens: list[int]) -> tuple[mx.array, mx.array]:
    max_len = max(seq_lens)
    parts = []
    offset = 0
    for length in seq_lens:
        part = packed[offset : offset + length]
        pad_width = [(0, max_len - length)] + [(0, 0)] * (packed.ndim - 1)
        parts.append(mx.pad(part, pad_width))
        offset += length
    mask = mx.arange(max_len)[None, :] < mx.array(seq_lens, dtype=mx.int32)[:, None]
    return mx.stack(parts, axis=0), mask


def _batched_to_packed(batched: mx.array, seq_lens: list[int]) -> mx.array:
    return mx.concatenate([batched[index, :length] for index, length in enumerate(seq_lens)], axis=0)


def _fourier_encode_xyz(coord: mx.array, freq_bands: mx.array) -> mx.array:
    if freq_bands.ndim != 1:
        raise ValueError(f"xyz_encoding.freq_bands must have shape (F,), got {freq_bands.shape}")
    expanded = mx.expand_dims(coord, axis=-1) * freq_bands
    flat_dim = int(coord.shape[-1]) * int(freq_bands.shape[0])
    return mx.concatenate(
        [
            mx.sin(expanded).reshape(coord.shape[0], flat_dim),
            mx.cos(expanded).reshape(coord.shape[0], flat_dim),
        ],
        axis=-1,
    )


def _fourier_encode_with_input(coord: mx.array, freq_bands: mx.array) -> mx.array:
    if freq_bands.ndim != 1:
        raise ValueError(f"freq_bands must have shape (F,), got {freq_bands.shape}")
    encoded = _fourier_encode_xyz(coord, freq_bands)
    return mx.concatenate([coord, encoded], axis=-1)


def _lito_regular_grid_coords(z: int, y: int, x: int) -> mx.array:
    z_vals = (mx.arange(z, dtype=mx.float32) + 0.5) * (2.0 / float(z)) - 1.0
    y_vals = (mx.arange(y, dtype=mx.float32) + 0.5) * (2.0 / float(y)) - 1.0
    x_vals = (mx.arange(x, dtype=mx.float32) + 0.5) * (2.0 / float(x)) - 1.0
    zz = mx.broadcast_to(z_vals[:, None, None], (z, y, x))
    yy = mx.broadcast_to(y_vals[None, :, None], (z, y, x))
    xx = mx.broadcast_to(x_vals[None, None, :], (z, y, x))
    return mx.stack([zz, yy, xx], axis=-1).reshape(z * y * x, 3)


def _top_level_prefix_counts(headers: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in headers:
        prefix = key.split(".", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _infer_dit_architecture(headers: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, Any]:
    prefix = _DIT_EMA_PREFIX
    z_proj = _shape(headers, f"{prefix}z_proj.weight")
    pos_mtx = _shape(headers, f"{prefix}pos_mtx")
    cond_proj = _shape(headers, f"{prefix}cond_embedder.y_proj.fc1.weight")
    final = _shape(headers, f"{prefix}final_layer.linear.weight")
    t_proj = _shape(headers, f"{prefix}t_proj.0.weight")
    t_freq = _shape(headers, f"{prefix}t_embedder.freq_bands")
    pos_proj = _shape(headers, f"{prefix}pos_proj.weight")
    block_indices = _numbered_indices(headers, rf"^{re.escape(prefix)}blocks\.(\d+)\.")
    swiglu_hidden = _shape(headers, f"{prefix}blocks.0.mlp.w1.weight")[0]
    dim_hidden = pos_mtx[1]
    return {
        "weight_prefix": prefix[:-1],
        "source_prefix": _DIT_SOURCE_PREFIX[:-1],
        "num_latent": pos_mtx[0],
        "dim_latent": z_proj[1],
        "dim_hidden": dim_hidden,
        "dim_cond_token": cond_proj[1],
        "dim_output": final[0],
        "patch_size": max(1, final[0] // max(z_proj[1], 1)),
        "num_blocks": len(block_indices),
        "block_indices": tuple(block_indices),
        "num_heads": 16,
        "head_dim": dim_hidden // 16,
        "use_rmsnorm": _has_key_suffix(headers, f"{prefix}blocks.0.attn.rmsnorm_q.scale"),
        "use_swiglu": _has_key_suffix(headers, f"{prefix}blocks.0.mlp.w1.weight"),
        "swiglu_hidden_dim": swiglu_hidden,
        "mlp_ratio": swiglu_hidden / dim_hidden,
        "timestep_fourier_dim": t_proj[1],
        "timestep_num_freqs": t_freq[0],
        "has_pos_proj": True,
        "init_pos_emb_dim": pos_proj[1],
        "remaps": {
            "t_proj.0": "t_proj_linear1",
            "t_proj.2": "t_proj_linear2",
            "t0_proj.1": "t0_proj_linear",
            "final_layer.adaLN_modulation.0": "final_layer.adaLN_linear1",
            "final_layer.adaLN_modulation.2": "final_layer.adaLN_linear2",
        },
    }


def _infer_gaussian_decoder_architecture(headers: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, Any]:
    prefix = _GS_PREFIX
    point_linear = _shape(headers, f"{prefix}point_linear.weight")
    cross_kv = _shape(headers, f"{prefix}perceiver.blocks.0.ca_layer.linear_kv.weight")
    self_qkv = _shape(headers, f"{prefix}perceiver.blocks.0.sa_layers.0.linear_qkv.weight")
    shape_out = _shape(headers, f"{prefix}gs_output_shape_mlp.2.linear.weight")
    color_out = _shape(headers, f"{prefix}gs_output_color_mlp.2.linear.weight")
    block_indices = _numbered_indices(headers, rf"^{re.escape(prefix)}perceiver\.blocks\.(\d+)\.")
    self_attn_indices = _numbered_indices(headers, rf"^{re.escape(prefix)}perceiver\.blocks\.0\.sa_layers\.(\d+)\.")
    expansion_ratio = 64
    shape_dim = shape_out[0] // expansion_ratio
    color_dim = color_out[0] // expansion_ratio
    rgb_dim = color_dim - 1
    rgb_sh_degree = _sh_degree_from_coeff_count(rgb_dim // 3)
    return {
        "weight_prefix": prefix[:-1],
        "dim_latent": cross_kv[1],
        "perceiver_dim": point_linear[0],
        "point_input_dim": point_linear[1],
        "point_inputs": ("xyz", "xyz_encoded"),
        "xyz_fourier_dim": point_linear[1] - 3,
        "xyz_fourier_num_freqs": _shape(headers, f"{prefix}xyz_encoding.freq_bands")[0],
        "qkv_dim": cross_kv[0] // 2,
        "num_blocks": len(block_indices),
        "block_indices": tuple(block_indices),
        "num_self_attn": len(self_attn_indices),
        "num_heads": 8,
        "head_dim": self_qkv[0] // 3 // 8,
        "cross_attn_type": "global",
        "self_attn_type": "localized_voxel",
        "self_cell_width": 0.25,
        "use_rmsnorm": _has_key_suffix(headers, f"{prefix}perceiver.blocks.0.ca_layer.rmsnorm_q.scale"),
        "mlp_type": "swiglu",
        "mlp_hidden_dim": _shape(headers, f"{prefix}perceiver.blocks.0.ca_mlp.w12.weight")[0] // 2,
        "mlp_add_bias": False,
        "gs_expansion_ratio": expansion_ratio,
        "shape_output_dim": shape_dim,
        "color_output_dim": color_dim,
        "shape_outputs": ("xyz_w", "quaternion_prenorm", "scaling_logit"),
        "color_outputs": ("opacity_logit", "rgb_sh"),
        "rgb_sh_degree": rgb_sh_degree,
        "region_scaling": 0.05,
        "scaling_activation": "sigmoid",
        "scaling_scalar": 0.01,
        "min_scaling": 0.001,
        "opacity_logit_bias": 0.1,
        "opacity_logit_scale": 1.0,
        "remaps": {
            "sequential_mlp": "map .0 norm, split .1.w12 into w1/w2, map .2 to final_layer",
            "perceiver_swiglu": "split ca_mlp.w12 and mlp_layers.*.w12 on axis 0",
        },
    }


def _infer_voxel_decoder_architecture(headers: dict[str, tuple[tuple[int, ...], str]]) -> dict[str, Any]:
    prefix = _VOXEL_PREFIX
    input_linear = _shape(headers, f"{prefix}input_linear.weight")
    init_query = _shape(headers, f"{prefix}net.init_query")
    init_query_linear = _shape(headers, f"{prefix}net.init_query_linear.weight")
    self_qkv = _shape(headers, f"{prefix}net.encoder.blocks.0.sa_layers.0.linear_qkv.weight")
    final = _shape(headers, f"{prefix}final_layer.linear.weight")
    block_indices = _numbered_indices(headers, rf"^{re.escape(prefix)}net\.encoder\.blocks\.(\d+)\.")
    self_attn_indices = _numbered_indices(headers, rf"^{re.escape(prefix)}net\.encoder\.blocks\.0\.sa_layers\.(\d+)\.")
    return {
        "weight_prefix": prefix[:-1],
        "dim_latent": input_linear[1],
        "perceiver_dim": input_linear[0],
        "qkv_dim": self_qkv[0] // 3,
        "num_blocks": len(block_indices),
        "block_indices": tuple(block_indices),
        "num_self_attn": len(self_attn_indices),
        "num_heads": 8,
        "head_dim": self_qkv[0] // 3 // 8,
        "init_query_shape": init_query,
        "init_query_input_dim": init_query_linear[1],
        "zyx_fourier_num_freqs": _shape(headers, f"{prefix}net.zyx_pos_encoder.freq_bands")[0],
        "final_output_dim": final[0],
        "mlp_type": "timm",
    }


def _shape(headers: dict[str, tuple[tuple[int, ...], str]], key: str) -> tuple[int, ...]:
    try:
        return headers[key][0]
    except KeyError as error:
        raise ValueError(f"LiTo checkpoint missing architecture tensor: {key}") from error


def _has_key_suffix(headers: dict[str, tuple[tuple[int, ...], str]], key: str) -> bool:
    return key in headers


def _numbered_indices(headers: dict[str, tuple[tuple[int, ...], str]], pattern: str) -> list[int]:
    regex = re.compile(pattern)
    return sorted({int(match.group(1)) for key in headers for match in [regex.match(key)] if match is not None})


def _validate_request(request: LitoRealGenerateRequest) -> None:
    rgba = np.asarray(request.cond_rgba)
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(f"cond_rgba must have shape (H, W, 4), got {rgba.shape}")
    if request.num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {request.num_steps}")


def _resolve_lito_trellis_root(config: LitoRealBackendConfig) -> Path:
    candidates = [
        config.weights_root.parent / "trellis2" / "microsoft" / "TRELLIS-image-large",
        Path("weights/trellis2/microsoft/TRELLIS-image-large"),
    ]
    for candidate in candidates:
        if (candidate / _TRELLIS_SS_DECODER_CHECKPOINT).is_file() and (candidate / _TRELLIS_SS_DECODER_CONFIG).is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise LitoBackendUnavailable(f"TRELLIS sparse-structure decoder weights are required for LiTo init coords; searched {searched}")


def _max_init_coords_for_memory_profile(memory_profile: str) -> int:
    try:
        return _LITO_REAL_MAX_INIT_COORDS_BY_PROFILE[memory_profile]
    except KeyError as error:
        allowed = ", ".join(sorted(_LITO_REAL_MAX_INIT_COORDS_BY_PROFILE))
        raise ValueError(f"unsupported LiTo memory profile {memory_profile!r}; expected one of {allowed}") from error


def resolve_lito_init_coord_cap(
    memory_profile: str,
    cap: int | str | None = LITO_INIT_COORD_CAP_PROFILE,
) -> int | None:
    """Resolve profile, no-cap, or explicit integer mode for LiTo init-coordinate generation."""

    if cap == LITO_INIT_COORD_CAP_PROFILE:
        return _max_init_coords_for_memory_profile(memory_profile)
    if cap is None:
        return None
    if isinstance(cap, str):
        if cap.lower() == "none":
            return None
        if cap.isdecimal():
            cap = int(cap)
        else:
            raise ValueError(
                "max_init_coords_per_batch must be 'profile', 'none', or a positive integer, "
                f"got {cap!r}"
            )
    max_cells = int(cap)
    if max_cells <= 0:
        raise ValueError(f"max_init_coords_per_batch must be positive, got {cap!r}")
    return max_cells


def _as_numpy(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _flatten_field(values: np.ndarray, name: str, trailing: int, *, count: int | None = None) -> np.ndarray:
    if values.shape[-1] != trailing:
        raise ValueError(f"{name} trailing dimension must be {trailing}, got {values.shape}")
    flattened = values.reshape((-1, trailing)).astype(np.float32, copy=False)
    if count is not None and flattened.shape[0] != count:
        raise ValueError(f"{name} gaussian count must be {count}, got {flattened.shape[0]}")
    return flattened


def _flatten_rgb_sh(values: np.ndarray, *, count: int, require_square: bool) -> np.ndarray:
    if values.shape[-1] != 3:
        raise ValueError(f"rgb_sh trailing dimension must be 3, got {values.shape}")
    if values.ndim == 2:
        values = values[:, None, :]
    flattened = values.reshape((count, -1, 3)).astype(np.float32, copy=False)
    coeffs = flattened.shape[1]
    if require_square and _sh_degree_from_coeff_count(coeffs) < 0:
        raise ValueError(f"rgb_sh coefficient count must be a square SH count, got {coeffs}")
    return flattened


def _normalize_quaternion(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    fallback = np.zeros_like(values, dtype=np.float32)
    fallback[:, 3] = 1.0
    return np.where(norm > 1e-8, values / np.maximum(norm, 1e-8), fallback).astype(np.float32)


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _sh_degree_from_coeff_count(coeffs: int) -> int:
    degree = int(round(math.sqrt(coeffs) - 1))
    return degree if (degree + 1) ** 2 == coeffs else -1


__all__ = [
    "DirectMlxLitoBackend",
    "LITO_DEFAULT_PLY_STORAGE",
    "LITO_INIT_COORD_CAP_PROFILE",
    "LITO_PLY_STORAGES",
    "LitoBackendUnavailable",
    "LitoGaussianDecoderProfile",
    "LitoPatchEncoderConfig",
    "LitoRealArchitectureInventory",
    "LitoRealBackendConfig",
    "LitoRealGenerateRequest",
    "build_lito_local_voxel_info",
    "create_lito_real_backend",
    "decode_lito_gaussian_perceiver_all_blocks",
    "decode_lito_gaussian_query_points",
    "decode_lito_gaussian_query_latents",
    "decode_lito_gaussian_outputs",
    "decode_lito_init_coords_from_latents",
    "decode_lito_trellis_sparse_structure_logits",
    "encode_lito_gaussian_query_points",
    "inspect_lito_real_architecture",
    "load_lito_dit_weight_arrays",
    "load_lito_gaussian_decoder_weight_arrays",
    "load_lito_patch_encoder_weight_arrays",
    "load_lito_voxel_decoder_weight_arrays",
    "normalize_lito_gs_dict",
    "normalize_lito_ply_storage",
    "occ_grid_to_lito_init_coord",
    "resolve_lito_init_coord_cap",
    "run_lito_patch_encoder_condition_tokens",
    "run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention",
    "run_lito_gaussian_perceiver_block0_cross_only",
    "run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention",
    "run_lito_gaussian_output_heads",
    "run_lito_dit_velocity",
    "sample_lito_dit_latents",
    "run_lito_voxel_decoder_lowres_latent",
    "write_lito_gaussians_ply",
]
