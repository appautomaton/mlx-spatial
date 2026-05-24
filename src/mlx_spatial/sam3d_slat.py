"""SAM 3D Objects Stage-2 structured-latent sparse flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .checkpoint import load_checkpoint_tensors
from .sam3d_flow import sam3d_classifier_free_guidance, sam3d_flow_time_sequence, sam3d_seeded_normal
from .sam3d_ss_flow import _gelu_tanh, _linear, _silu
from .sam3d_transformer import (
    run_sam3d_timestep_embedder,
    sam3d_layer_norm,
    sam3d_multihead_rms_norm,
    sam3d_scaled_dot_product_attention,
)
from .sparse_conv import sparse_conv_map_vectorized, weighted_sparse_conv_chunked


SAM3D_SLAT_GENERATOR_PREFIX = "_base_models.generator."


@dataclass(frozen=True)
class Sam3dSparseTensor:
    coords: np.ndarray
    feats: mx.array
    layout: tuple[slice, ...]
    spatial_cache: dict[str, object]

    @property
    def token_count(self) -> int:
        return int(self.coords.shape[0])

    def replace(self, feats: mx.array | None = None, coords: np.ndarray | None = None) -> "Sam3dSparseTensor":
        next_coords = self.coords if coords is None else _validate_coords(coords)
        return Sam3dSparseTensor(
            coords=next_coords,
            feats=self.feats if feats is None else feats,
            layout=_layout_for_coords(next_coords),
            spatial_cache=self.spatial_cache,
        )


@dataclass(frozen=True)
class Sam3dSLatFlowConfig:
    model_channels: int = 1024
    num_heads: int = 16
    num_blocks: int = 24
    in_channels: int = 8
    out_channels: int = 8
    io_block_channels: tuple[int, ...] = (128,)
    num_io_res_blocks: int = 2
    cfg_strength: float = 1.0
    cfg_interval: tuple[float, float] = (0.0, 500.0)
    rescale_t: float = 1.0
    time_scale: float = 1000.0
    attention_chunk_size: int = 128


@dataclass(frozen=True)
class Sam3dSLatFlowOutput:
    coords: np.ndarray
    feats: mx.array
    metadata: dict[str, object]


def load_sam3d_slat_generator_tensors(path: str | Path) -> dict[str, mx.array]:
    tensors = load_checkpoint_tensors(path, prefixes=(SAM3D_SLAT_GENERATOR_PREFIX,))
    return {
        key[len(SAM3D_SLAT_GENERATOR_PREFIX) :]: value
        for key, value in tensors.items()
        if key.startswith(SAM3D_SLAT_GENERATOR_PREFIX)
    }


def infer_sam3d_slat_flow_config(
    tensors: dict[str, mx.array],
    *,
    cfg_strength: float = 1.0,
    cfg_interval: tuple[float, float] = (0.0, 500.0),
    rescale_t: float = 1.0,
    attention_chunk_size: int = 128,
) -> Sam3dSLatFlowConfig:
    prefix = "reverse_fn.backbone."
    model_channels = int(tensors[f"{prefix}t_embedder.mlp.2.bias"].shape[0])
    block_indices = {
        int(key.split(".")[3])
        for key in tensors
        if key.startswith(f"{prefix}blocks.") and key.split(".")[3].isdigit()
    }
    input_block_indices = {
        int(key.split(".")[3])
        for key in tensors
        if key.startswith(f"{prefix}input_blocks.") and key.split(".")[3].isdigit()
    }
    out_block_indices = {
        int(key.split(".")[3])
        for key in tensors
        if key.startswith(f"{prefix}out_blocks.") and key.split(".")[3].isdigit()
    }
    gamma = tensors.get(f"{prefix}blocks.0.self_attn.q_rms_norm.gamma")
    return Sam3dSLatFlowConfig(
        model_channels=model_channels,
        num_heads=int(gamma.shape[0]) if gamma is not None else 16,
        num_blocks=max(block_indices) + 1 if block_indices else 0,
        in_channels=int(tensors[f"{prefix}input_layer.weight"].shape[1]),
        out_channels=int(tensors[f"{prefix}out_layer.weight"].shape[0]),
        io_block_channels=(int(tensors[f"{prefix}input_layer.weight"].shape[0]),),
        num_io_res_blocks=max(input_block_indices) + 1 if input_block_indices else 0,
        cfg_strength=float(cfg_strength),
        cfg_interval=cfg_interval,
        rescale_t=float(rescale_t),
        attention_chunk_size=int(attention_chunk_size),
    )


def run_sam3d_slat_flow(
    coords: np.ndarray,
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    seed: int = 42,
    steps: int = 12,
    config: Sam3dSLatFlowConfig | None = None,
    slat_mean: tuple[float, ...] | None = None,
    slat_std: tuple[float, ...] | None = None,
) -> Sam3dSLatFlowOutput:
    """Run the active SAM3D Stage-2 FlowMatching sampler on final SS coords."""

    if condition_tokens.ndim != 3:
        raise ValueError(f"SAM3D SLat condition tokens must have shape [B,T,C], got {tuple(condition_tokens.shape)}")
    cfg = config or infer_sam3d_slat_flow_config(tensors)
    coords_np = _validate_coords(coords)
    feats = sam3d_seeded_normal((coords_np.shape[0], cfg.in_channels), seed=seed + 10_000)
    t_seq = sam3d_flow_time_sequence(steps, rescale_t=cfg.rescale_t)
    for index in range(len(t_seq) - 1):
        t0 = float(t_seq[index])
        t1 = float(t_seq[index + 1])
        t_scaled = t0 * cfg.time_scale
        conditional = _run_slat_velocity(
            coords_np,
            feats,
            condition_tokens,
            tensors,
            t_scaled=t_scaled,
            config=cfg,
        )
        if cfg.cfg_interval[0] <= t_scaled <= cfg.cfg_interval[1] and cfg.cfg_strength:
            unconditional = _run_slat_velocity(
                coords_np,
                feats,
                mx.zeros_like(condition_tokens),
                tensors,
                t_scaled=t_scaled,
                config=cfg,
            )
            velocity = sam3d_classifier_free_guidance(
                conditional,
                unconditional,
                strength=cfg.cfg_strength,
                interval=cfg.cfg_interval,
                t_scaled=t_scaled,
            )
        else:
            velocity = conditional
        feats = feats + (t1 - t0) * velocity
        mx.eval(feats)
    feats = _denormalize_slat(feats, slat_mean=slat_mean, slat_std=slat_std)
    return Sam3dSLatFlowOutput(
        coords=coords_np,
        feats=feats,
        metadata={
            "coords_shape": tuple(int(value) for value in coords_np.shape),
            "feature_shape": tuple(int(value) for value in feats.shape),
            "steps": int(steps),
            "schedule": tuple(float(value) for value in t_seq.tolist()),
            "cfg_strength": float(cfg.cfg_strength),
            "cfg_interval": tuple(float(value) for value in cfg.cfg_interval),
            "rescale_t": float(cfg.rescale_t),
            "num_blocks": int(cfg.num_blocks),
            "num_heads": int(cfg.num_heads),
            "attention_chunk_size": int(cfg.attention_chunk_size),
            "slat_mean": tuple(float(value) for value in slat_mean) if slat_mean is not None else None,
            "slat_std": tuple(float(value) for value in slat_std) if slat_std is not None else None,
        },
    )


def _run_slat_velocity(
    coords: np.ndarray,
    feats: mx.array,
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    t_scaled: float,
    config: Sam3dSLatFlowConfig,
) -> mx.array:
    prefix = "reverse_fn.backbone."
    tensor = Sam3dSparseTensor(coords=coords, feats=_sparse_linear(feats, tensors, prefix=f"{prefix}input_layer."), layout=_layout_for_coords(coords), spatial_cache={})
    t_emb = run_sam3d_timestep_embedder(
        mx.array([t_scaled], dtype=mx.float32),
        tensors,
        prefix=f"{prefix}t_embedder.",
    )
    skips: list[mx.array] = []
    for index in range(_count_blocks(tensors, f"{prefix}input_blocks.")):
        tensor = _sparse_resblock(
            tensor,
            t_emb,
            tensors,
            prefix=f"{prefix}input_blocks.{index}.",
            downsample=_has_sparse_skip(tensors, f"{prefix}input_blocks.{index}."),
        )
        skips.append(tensor.feats)

    tensor = tensor.replace(tensor.feats + _absolute_position_embedding(tensor.coords[:, 1:], config.model_channels).astype(tensor.feats.dtype))
    for index in range(config.num_blocks):
        tensor = _sparse_transformer_cross_block(
            tensor,
            t_emb,
            condition_tokens,
            tensors,
            prefix=f"{prefix}blocks.{index}.",
            config=config,
        )
        mx.eval(tensor.feats)

    for index in range(_count_blocks(tensors, f"{prefix}out_blocks.")):
        skip = skips[-1 - index]
        if int(tensor.feats.shape[0]) != int(skip.shape[0]):
            raise ValueError("SAM3D SLat skip token count mismatch before out block")
        tensor = tensor.replace(mx.concatenate((tensor.feats, skip), axis=1))
        tensor = _sparse_resblock(
            tensor,
            t_emb,
            tensors,
            prefix=f"{prefix}out_blocks.{index}.",
            upsample=index % max(config.num_io_res_blocks, 1) == 0,
        )
    tensor = tensor.replace(sam3d_layer_norm(tensor.feats, eps=1e-5))
    return _sparse_linear(tensor.feats.astype(feats.dtype), tensors, prefix=f"{prefix}out_layer.")


def _sparse_resblock(
    tensor: Sam3dSparseTensor,
    emb: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    downsample: bool = False,
    upsample: bool = False,
) -> Sam3dSparseTensor:
    if downsample and upsample:
        raise ValueError("SAM3D sparse resblock cannot downsample and upsample together")
    if downsample:
        tensor = _sparse_downsample(tensor, factor=2)
    elif upsample:
        tensor = _sparse_upsample(tensor, factor=2)
    emb_out = _linear(_silu(emb), tensors[f"{prefix}emb_layers.1.weight"], tensors[f"{prefix}emb_layers.1.bias"])
    scale, shift = mx.split(emb_out, 2, axis=1)
    residual = _sparse_linear(tensor.feats, tensors, prefix=f"{prefix}skip_connection.") if _has_sparse_skip(tensors, prefix) else tensor.feats
    hidden = sam3d_layer_norm(tensor.feats, tensors[f"{prefix}norm1.weight"], tensors[f"{prefix}norm1.bias"])
    hidden = _silu(hidden)
    hidden = _sparse_conv3d(tensor.coords, hidden, tensors, prefix=f"{prefix}conv1.conv.")
    hidden = sam3d_layer_norm(hidden)
    hidden = hidden * (1.0 + scale.astype(hidden.dtype)) + shift.astype(hidden.dtype)
    hidden = _silu(hidden)
    hidden = _sparse_conv3d(tensor.coords, hidden, tensors, prefix=f"{prefix}conv2.conv.")
    return tensor.replace(hidden + residual)


def _sparse_transformer_cross_block(
    tensor: Sam3dSparseTensor,
    emb: mx.array,
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSLatFlowConfig,
) -> Sam3dSparseTensor:
    mod = _linear(_silu(emb), tensors[f"{prefix}adaLN_modulation.1.weight"], tensors[f"{prefix}adaLN_modulation.1.bias"])
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=1)
    hidden = sam3d_layer_norm(tensor.feats)
    hidden = hidden * (1.0 + scale_msa.astype(hidden.dtype)) + shift_msa.astype(hidden.dtype)
    hidden = _sparse_self_attention(hidden, tensors, prefix=f"{prefix}self_attn.", config=config)
    feats = tensor.feats + hidden * gate_msa.astype(hidden.dtype)
    hidden = sam3d_layer_norm(feats, tensors[f"{prefix}norm2.weight"], tensors[f"{prefix}norm2.bias"])
    hidden = _sparse_cross_attention(hidden, condition_tokens, tensors, prefix=f"{prefix}cross_attn.", config=config)
    feats = feats + hidden
    hidden = sam3d_layer_norm(feats)
    hidden = hidden * (1.0 + scale_mlp.astype(hidden.dtype)) + shift_mlp.astype(hidden.dtype)
    hidden = _feed_forward(hidden, tensors, prefix=f"{prefix}mlp.")
    return tensor.replace(feats + hidden * gate_mlp.astype(hidden.dtype))


def _sparse_self_attention(
    feats: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSLatFlowConfig,
) -> mx.array:
    tokens, channels = int(feats.shape[0]), int(feats.shape[1])
    head_dim = channels // config.num_heads
    qkv = _linear(feats, tensors[f"{prefix}to_qkv.weight"], tensors[f"{prefix}to_qkv.bias"])
    qkv = mx.reshape(qkv, (1, tokens, 3, config.num_heads, head_dim))
    query = sam3d_multihead_rms_norm(qkv[:, :, 0, :, :], tensors[f"{prefix}q_rms_norm.gamma"])
    key = sam3d_multihead_rms_norm(qkv[:, :, 1, :, :], tensors[f"{prefix}k_rms_norm.gamma"])
    value = qkv[:, :, 2, :, :]
    attended = sam3d_scaled_dot_product_attention(query, key, value, chunk_size=config.attention_chunk_size)
    return _linear(mx.reshape(attended, (tokens, channels)), tensors[f"{prefix}to_out.weight"], tensors[f"{prefix}to_out.bias"])


def _sparse_cross_attention(
    feats: mx.array,
    condition_tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSLatFlowConfig,
) -> mx.array:
    tokens, channels = int(feats.shape[0]), int(feats.shape[1])
    head_dim = channels // config.num_heads
    context = condition_tokens.astype(feats.dtype)
    query = _linear(feats, tensors[f"{prefix}to_q.weight"], tensors[f"{prefix}to_q.bias"])
    kv = _linear(context, tensors[f"{prefix}to_kv.weight"], tensors[f"{prefix}to_kv.bias"])
    query = mx.reshape(query, (1, tokens, config.num_heads, head_dim))
    kv = mx.reshape(kv, (1, int(context.shape[1]), 2, config.num_heads, head_dim))
    attended = sam3d_scaled_dot_product_attention(
        query,
        kv[:, :, 0, :, :],
        kv[:, :, 1, :, :],
        chunk_size=config.attention_chunk_size,
    )
    return _linear(mx.reshape(attended, (tokens, channels)), tensors[f"{prefix}to_out.weight"], tensors[f"{prefix}to_out.bias"])


def _sparse_downsample(tensor: Sam3dSparseTensor, *, factor: int) -> Sam3dSparseTensor:
    coords = tensor.coords.copy()
    coords[:, 1:] //= int(factor)
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    sums = np.zeros((unique.shape[0], int(tensor.feats.shape[1])), dtype=np.float32)
    counts = np.zeros((unique.shape[0], 1), dtype=np.float32)
    feats_np = np.array(tensor.feats, dtype=np.float32)
    np.add.at(sums, inverse, feats_np)
    np.add.at(counts, inverse, 1.0)
    cache = dict(tensor.spatial_cache)
    cache[_upsample_cache_key(factor, "coords")] = tensor.coords
    cache[_upsample_cache_key(factor, "idx")] = inverse.astype(np.int32, copy=False)
    return Sam3dSparseTensor(
        coords=unique.astype(np.int32, copy=False),
        feats=mx.array(sums / np.maximum(counts, 1.0), dtype=tensor.feats.dtype),
        layout=_layout_for_coords(unique),
        spatial_cache=cache,
    )


def _sparse_upsample(tensor: Sam3dSparseTensor, *, factor: int) -> Sam3dSparseTensor:
    coords = tensor.spatial_cache.get(_upsample_cache_key(factor, "coords"))
    inverse = tensor.spatial_cache.get(_upsample_cache_key(factor, "idx"))
    if coords is None or inverse is None:
        raise ValueError("SAM3D SLat SparseUpsample cache not found")
    inverse_np = np.asarray(inverse, dtype=np.int32)
    return Sam3dSparseTensor(
        coords=np.asarray(coords, dtype=np.int32),
        feats=tensor.feats[mx.array(inverse_np, dtype=mx.int32)],
        layout=_layout_for_coords(np.asarray(coords, dtype=np.int32)),
        spatial_cache=tensor.spatial_cache,
    )


def _sparse_conv3d(
    coords: np.ndarray,
    feats: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
) -> mx.array:
    weight = tensors[f"{prefix}weight"].astype(mx.float32)
    kernel_size = tuple(int(value) for value in weight.shape[1:4])
    kernel_count = int(np.prod(kernel_size))
    kernel_weights = mx.reshape(
        mx.contiguous(mx.transpose(weight, (1, 2, 3, 4, 0))),
        (kernel_count, int(weight.shape[-1]), int(weight.shape[0])),
    )
    outputs = []
    offset = 0
    for layout in _layout_for_coords(coords):
        batch_coords = coords[layout, 1:]
        spatial_shape = tuple(int(value) for value in np.maximum(batch_coords.max(axis=0) + 1, 1))
        map_rows = sparse_conv_map_vectorized(mx.array(batch_coords, dtype=mx.int32), spatial_shape, kernel_size=kernel_size)
        outputs.append(
            weighted_sparse_conv_chunked(
                feats[offset : offset + batch_coords.shape[0]].astype(mx.float32),
                map_rows,
                kernel_weights,
                target_count=int(batch_coords.shape[0]),
            )
        )
        offset += int(batch_coords.shape[0])
    conv = mx.concatenate(outputs, axis=0) if outputs else mx.zeros((0, int(weight.shape[0])), dtype=mx.float32)
    return conv + tensors[f"{prefix}bias"].astype(mx.float32)


def _sparse_linear(feats: mx.array, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    weight = tensors.get(f"{prefix}weight")
    bias = tensors.get(f"{prefix}bias")
    if weight is None:
        return feats
    return _linear(feats, weight, bias)


def _absolute_position_embedding(coords: np.ndarray, channels: int) -> mx.array:
    flat = coords.astype(np.float32).reshape(-1)
    freq_dim = int(channels) // 3 // 2
    freqs = 1.0 / (10000 ** (np.arange(freq_dim, dtype=np.float32) / max(freq_dim, 1)))
    emb = np.outer(flat, freqs)
    emb = np.concatenate((np.sin(emb), np.cos(emb)), axis=-1).reshape(coords.shape[0], -1)
    if emb.shape[1] < channels:
        emb = np.pad(emb, ((0, 0), (0, channels - emb.shape[1])))
    return mx.array(emb[:, :channels].astype(np.float32, copy=False))


def _feed_forward(feats: mx.array, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    hidden = _linear(feats, tensors[f"{prefix}mlp.0.weight"], tensors[f"{prefix}mlp.0.bias"])
    hidden = _gelu_tanh(hidden)
    return _linear(hidden, tensors[f"{prefix}mlp.2.weight"], tensors[f"{prefix}mlp.2.bias"])


def _denormalize_slat(
    feats: mx.array,
    *,
    slat_mean: tuple[float, ...] | None,
    slat_std: tuple[float, ...] | None,
) -> mx.array:
    if slat_mean is None or slat_std is None:
        return feats
    mean = mx.array(slat_mean, dtype=feats.dtype)
    std = mx.array(slat_std, dtype=feats.dtype)
    return feats * std[None, :] + mean[None, :]


def _count_blocks(tensors: dict[str, mx.array], prefix: str) -> int:
    indices = {
        int(key[len(prefix) :].split(".", 1)[0])
        for key in tensors
        if key.startswith(prefix) and key[len(prefix) :].split(".", 1)[0].isdigit()
    }
    return max(indices) + 1 if indices else 0


def _has_sparse_skip(tensors: dict[str, mx.array], prefix: str) -> bool:
    return f"{prefix}skip_connection.weight" in tensors


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    array = np.asarray(coords, dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"SAM3D sparse coords must have shape (N,4), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("SAM3D sparse coords must not be empty")
    if np.any(array[:, 1:] < 0):
        raise ValueError("SAM3D sparse coords must be non-negative")
    return array.astype(np.int32, copy=False)


def _layout_for_coords(coords: np.ndarray) -> tuple[slice, ...]:
    coords_np = _validate_coords(coords)
    batch_count = int(coords_np[:, 0].max()) + 1
    layout = []
    start = 0
    for batch in range(batch_count):
        count = int(np.sum(coords_np[:, 0] == batch))
        layout.append(slice(start, start + count))
        start += count
    return tuple(layout)


def _upsample_cache_key(factor: int, name: str) -> str:
    return f"upsample_{(int(factor), int(factor), int(factor))}_{name}"
