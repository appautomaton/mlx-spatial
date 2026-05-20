"""Extracted HY-World 2.0 foundation layer functions for MLX inference.

These functions are extracted from hyworld2_worldmirror.py into separate
named APIs for per-layer parity testing. The WorldMirror module continues
to use its internal implementations; this module provides the same
functionality under stable public names.

Gap IDs: HW-02, HW-04, HW-05, HW-06, HW-07, HW-08.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def apply_layer_scale(hidden_states: mx.array, gamma: mx.array | None) -> mx.array:
    """Apply learnable per-channel scaling (LayerScale).

    HW-07: Matches ``models/layers/layer_scale.py`` — ``LayerScale.forward()``.
    At inference, gamma is a fixed vector; if None, this is identity.

    Args:
        hidden_states: Input tensor ``(..., dim)``.
        gamma: Per-channel scaling vector ``(dim,)`` or None for identity.

    Returns:
        Scaled tensor with the same shape as *hidden_states*.
    """
    if gamma is None:
        return hidden_states
    return hidden_states * gamma.astype(hidden_states.dtype)


def linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    """Linear projection without a module.

    Matches ``nn.Linear`` forward: ``values @ weight.T + bias``.

    Args:
        values: Input tensor ``(..., in_features)``.
        weight: Weight matrix ``(out_features, in_features)``.
        bias: Bias vector ``(out_features,)`` or None.

    Returns:
        Projected tensor ``(..., out_features)``.
    """
    output = values @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float = 1e-5) -> mx.array:
    """Layer normalization without a module.

    Matches ``nn.LayerNorm`` forward for a given eps.

    Args:
        values: Input tensor ``(..., dim)``.
        weight: Gamma parameter ``(dim,)``.
        bias: Beta parameter ``(dim,)``.
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor with the same shape as *values*.
    """
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight + bias


def scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
) -> mx.array:
    """Exact scaled dot-product attention.

    HW-02: Matches ``models/layers/attention.py`` — ``Attention.forward()``
    for the non-Flash, non-distributed path.

    Args:
        query: ``(B, heads, query_len, head_dim)``
        key: ``(B, heads, key_len, head_dim)``
        value: ``(B, heads, key_len, head_dim)``
        scale: Multiplicative scale (typically ``head_dim ** -0.5``).

    Returns:
        Attention output ``(B, heads, query_len, head_dim)``.
    """
    logits = (query @ mx.transpose(key, (0, 1, 3, 2))) * scale
    weights = mx.softmax(logits.astype(mx.float32), axis=-1)
    return weights @ value


def apply_1d_rope(values: mx.array, coordinates: mx.array, rope_base: float) -> mx.array:
    """Apply 1D rotary position embedding.

    HW-06: Matches ``models/layers/rope.py`` —
    ``RotaryPositionEmbedding2D._apply_1d_rope()``.

    The coordinates tensor provides per-token positions; ``rope_base`` controls
    the frequency band.

    Args:
        values: ``(..., tokens, dim)`` where ``dim`` must be even.
        coordinates: ``(batch, tokens)`` integer or float positions.
        rope_base: Base frequency (e.g. 100.0).

    Returns:
        Tensor with the same shape as *values* with RoPE applied.
    """
    axis_dim = int(values.shape[-1])
    pair_count = axis_dim // 2
    exponents = mx.arange(pair_count, dtype=mx.float32) * (2.0 / axis_dim)
    inv_freq = mx.exp(-mx.log(mx.array(float(rope_base), dtype=mx.float32)) * exponents)
    angles = coordinates[:, None, :, None] * inv_freq[None, None, None, :]
    pair_values = mx.reshape(values, (*tuple(int(dim) for dim in values.shape[:-1]), pair_count, 2))
    even = pair_values[..., 0]
    odd = pair_values[..., 1]
    rotated = mx.stack(
        (
            even * mx.cos(angles) - odd * mx.sin(angles),
            even * mx.sin(angles) + odd * mx.cos(angles),
        ),
        axis=-1,
    )
    return mx.reshape(rotated, values.shape)


def apply_2d_rope(
    query: mx.array,
    key: mx.array,
    positions: mx.array,
    rope_base: float,
) -> tuple[mx.array, mx.array]:
    """Apply 2D rotary position embedding to query and key.

    HW-06: Matches ``models/layers/rope.py`` and
    ``models/layers/norm_rope.py`` — ``RotaryPositionEmbedding2D.forward()``
    and ``NormalizedRotaryPositionEmbedding2D.forward()``.

    Splits the head dimension in half: first half gets row positions,
    second half gets column positions, each with 1D RoPE independently.

    For the normalized variant (DINOv3-style), positions should already be
    normalized to ``[0, 1]`` before calling this function.

    Args:
        query: ``(B, heads, tokens, head_dim)`` with ``head_dim % 4 == 0``.
        key: ``(B, heads, tokens, head_dim)``.
        positions: ``(B, tokens, 2)`` — row and column positions.
        rope_base: Base frequency.

    Returns:
        ``(roped_query, roped_key)`` with the same shapes.
    """
    head_dim = int(query.shape[-1])
    axis_dim = head_dim // 2
    row_values = query[..., :axis_dim]
    col_values = query[..., axis_dim:]
    row_key = key[..., :axis_dim]
    col_key = key[..., axis_dim:]
    roped_query = mx.concatenate(
        (
            apply_1d_rope(row_values, positions[:, :, 0].astype(mx.float32), rope_base),
            apply_1d_rope(col_values, positions[:, :, 1].astype(mx.float32), rope_base),
        ),
        axis=-1,
    )
    roped_key = mx.concatenate(
        (
            apply_1d_rope(row_key, positions[:, :, 0].astype(mx.float32), rope_base),
            apply_1d_rope(col_key, positions[:, :, 1].astype(mx.float32), rope_base),
        ),
        axis=-1,
    )
    return roped_query, roped_key


def block_mlp(
    hidden_states: mx.array,
    up_weight: mx.array,
    up_bias: mx.array,
    down_weight: mx.array,
    down_bias: mx.array,
) -> mx.array:
    """Feed-forward MLP: ``down(gelu(up(x)))``.

    HW-04: Matches ``models/layers/mlp.py`` — ``Mlp.forward()``.

    Args:
        hidden_states: Input tensor ``(..., in_features)``.
        up_weight: First linear weight ``(hidden_features, in_features)``.
        up_bias: First linear bias ``(hidden_features,)``.
        down_weight: Second linear weight ``(in_features, hidden_features)``.
        down_bias: Second linear bias ``(in_features,)``.

    Returns:
        Projected tensor ``(..., in_features)``.
    """
    return linear(nn.gelu(linear(hidden_states, up_weight, up_bias)), down_weight, down_bias)


def block_swiglu_ffn(
    hidden_states: mx.array,
    w12_weight: mx.array,
    w12_bias: mx.array | None,
    w3_weight: mx.array,
    w3_bias: mx.array | None,
) -> mx.array:
    """SwiGLU feed-forward network: ``w3(silu(w12_x[:, :half]) * w12_x[:, half:])``.

    HW-04: Matches ``models/layers/swiglu_ffn.py`` — ``SwiGLUFFN.forward()``.

    The ``w12`` projection produces a 2×hidden tensor that is split in half;
    the first half is passed through SiLU and multiplied element-wise with
    the second half, then projected back by ``w3``.

    Args:
        hidden_states: Input tensor ``(..., in_features)``.
        w12_weight: Fused gate+up projection ``(2*hidden_features, in_features)``.
        w12_bias: Fused gate+up bias ``(2*hidden_features,)`` or None.
        w3_weight: Down projection ``(in_features, hidden_features)``.
        w3_bias: Down projection bias ``(in_features,)`` or None.

    Returns:
        Output tensor ``(..., in_features)``.
    """
    x12 = linear(hidden_states, w12_weight, w12_bias)
    x1, x2 = mx.split(x12, 2, axis=-1)
    hidden = nn.silu(x1) * x2
    return linear(hidden, w3_weight, w3_bias)


def head_layer_norm(
    values: mx.array,
    weight: mx.array | None,
    bias: mx.array | None,
    *,
    eps: float = 1e-6,
) -> mx.array:
    """Layer norm applied per-head (QK-norm).

    HW-02: Matches the QK-norm path in ``models/layers/attention.py`` —
    ``Attention.forward()`` with ``qk_norm=True``.

    If weight or bias is None, returns values unchanged (identity).

    Args:
        values: ``(B, heads, tokens, head_dim)``
        weight: Gamma ``(head_dim,)`` or None.
        bias: Beta ``(head_dim,)`` or None.

    Returns:
        Normalized tensor with the same shape as *values*.
    """
    if weight is None or bias is None:
        return values
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight.astype(values.dtype) + bias.astype(values.dtype)
