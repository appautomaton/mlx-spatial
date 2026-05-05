"""Shared SAM 3D Objects transformer math for the MLX port."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def sam3d_timestep_embedding(t: mx.array | np.ndarray, dim: int, *, max_period: int = 10_000) -> mx.array:
    """Match official sinusoidal `TimestepEmbedder.timestep_embedding`."""

    values = mx.array(t, dtype=mx.float32) if not isinstance(t, mx.array) else t.astype(mx.float32)
    if values.ndim == 0:
        values = values[None]
    half = dim // 2
    freqs = mx.exp(-np.log(float(max_period)) * mx.arange(0, half, dtype=mx.float32) / float(half))
    args = values[:, None] * freqs[None, :]
    emb = mx.concatenate((mx.cos(args), mx.sin(args)), axis=-1)
    if dim % 2:
        emb = mx.concatenate((emb, mx.zeros_like(emb[:, :1])), axis=-1)
    return emb


def run_sam3d_timestep_embedder(t: mx.array | np.ndarray, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    """Run official timestep MLP: Linear, SiLU, Linear."""

    hidden = sam3d_timestep_embedding(t, int(tensors[f"{prefix}mlp.0.weight"].shape[1]))
    hidden = _linear(hidden, tensors[f"{prefix}mlp.0.weight"], tensors[f"{prefix}mlp.0.bias"])
    hidden = hidden * mx.sigmoid(hidden)
    return _linear(hidden, tensors[f"{prefix}mlp.2.weight"], tensors[f"{prefix}mlp.2.bias"])


def sam3d_scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    chunk_size: int | None = None,
) -> mx.array:
    """Scaled dot-product attention for `[B, L, H, D]` tensors."""

    batch, q_len, heads, head_dim = tuple(int(value) for value in query.shape)
    if int(key.shape[0]) != batch or int(value.shape[0]) != batch or int(key.shape[2]) != heads:
        raise ValueError("SAM3D attention q/k/v batch and head dimensions must match")
    scale = head_dim**-0.5
    key_t = mx.transpose(key, (0, 2, 3, 1))
    value_h = mx.transpose(value, (0, 2, 1, 3))
    query_h = mx.transpose(query, (0, 2, 1, 3))
    if chunk_size is None:
        weights = mx.softmax((query_h @ key_t) * scale, axis=-1)
        out = weights @ value_h
    else:
        chunks = []
        for start in range(0, q_len, chunk_size):
            stop = min(start + chunk_size, q_len)
            weights = mx.softmax((query_h[:, :, start:stop, :] @ key_t) * scale, axis=-1)
            chunks.append(weights @ value_h)
        out = mx.concatenate(chunks, axis=2)
    return mx.transpose(out, (0, 2, 1, 3))


def run_sam3d_multihead_attention(
    x: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    num_heads: int,
    context: mx.array | None = None,
    qk_rms_norm: bool = False,
    chunk_size: int | None = None,
) -> mx.array:
    """Run official dense self/cross multi-head attention for SS transformer blocks."""

    batch, tokens, channels = tuple(int(value) for value in x.shape)
    head_dim = channels // num_heads
    if context is None:
        qkv = _linear(x, tensors[f"{prefix}to_qkv.weight"], tensors[f"{prefix}to_qkv.bias"])
        qkv = mx.reshape(qkv, (batch, tokens, 3, num_heads, head_dim))
        query, key, value = (qkv[:, :, index, :, :] for index in range(3))
    else:
        ctx = context.astype(x.dtype)
        query = _linear(x, tensors[f"{prefix}to_q.weight"], tensors[f"{prefix}to_q.bias"])
        kv = _linear(ctx, tensors[f"{prefix}to_kv.weight"], tensors[f"{prefix}to_kv.bias"])
        query = mx.reshape(query, (batch, tokens, num_heads, head_dim))
        kv = mx.reshape(kv, (batch, int(ctx.shape[1]), 2, num_heads, head_dim))
        key, value = kv[:, :, 0, :, :], kv[:, :, 1, :, :]
    if qk_rms_norm:
        query = sam3d_multihead_rms_norm(query, tensors[f"{prefix}q_rms_norm.gamma"])
        key = sam3d_multihead_rms_norm(key, tensors[f"{prefix}k_rms_norm.gamma"])
    attended = sam3d_scaled_dot_product_attention(query, key, value, chunk_size=chunk_size)
    merged = mx.reshape(attended, (batch, tokens, channels))
    return _linear(merged, tensors[f"{prefix}to_out.weight"], tensors[f"{prefix}to_out.bias"])


def sam3d_multihead_rms_norm(values: mx.array, gamma: mx.array) -> mx.array:
    """Match official `F.normalize(x, dim=-1) * gamma * sqrt(dim)`."""

    head_dim = int(values.shape[-1])
    norm = mx.sqrt(mx.sum(values.astype(mx.float32) * values.astype(mx.float32), axis=-1, keepdims=True))
    normalized = values / mx.maximum(norm, mx.array(1e-12, dtype=values.dtype))
    return (normalized * gamma.astype(values.dtype)[None, None, :, :] * (head_dim**0.5)).astype(values.dtype)


def sam3d_layer_norm(values: mx.array, weight: mx.array | None = None, bias: mx.array | None = None, *, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(values.astype(mx.float32), axis=-1, keepdims=True)
    centered = values.astype(mx.float32) - mean
    var = mx.mean(centered * centered, axis=-1, keepdims=True)
    out = centered * mx.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.astype(out.dtype)
    if bias is not None:
        out = out + bias.astype(out.dtype)
    return out.astype(values.dtype)


def sam3d_feed_forward(values: mx.array, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    hidden = _linear(values, tensors[f"{prefix}mlp.0.weight"], tensors[f"{prefix}mlp.0.bias"])
    hidden = nn.gelu(hidden, approximate="tanh")
    return _linear(hidden, tensors[f"{prefix}mlp.2.weight"], tensors[f"{prefix}mlp.2.bias"])


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    out = values @ mx.transpose(weight.astype(values.dtype))
    if bias is not None:
        out = out + bias.astype(out.dtype)
    return out
