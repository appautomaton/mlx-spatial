"""Transformer block functions for HY-World 2.0 VGT.

Gap IDs: HW-03, HW-09. Matches
``vendors/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/layers/block.py``
(NestedTensorBlock) and the VGT block loop in ``models/models/visual_transformer.py``.

This module provides the functional equivalents of the vendor's ``Block``,
``DistBlock``, and ``NestedTensorBlock`` classes, adapted to the tensor-dict
pattern used throughout the MLX port.

Each block follows the standard transformer block structure:
    LN → Self-Attention (+ optional QK-norm + optional RoPE) → LayerScale → Residual
    LN → MLP → LayerScale → Residual

The VGT runs alternating frame-local and cross-frame-global blocks,
reflected in the ``mode`` parameter ("frame" or "global").
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx

from .hyworld2_layers import (
    apply_2d_rope,
    apply_layer_scale,
    block_mlp,
    block_swiglu_ffn,
    head_layer_norm,
    layer_norm,
    linear,
    scaled_dot_product_attention,
)


_FLOAT32_BYTES = 4


@dataclass(frozen=True)
class Block:
    """Callable MLX facade for a HY-World transformer block.

    The vendor implementation stores weights in a PyTorch module. The MLX port
    keeps checkpoint tensors in a flat dict, so this class binds that dict to
    the same block name while delegating execution to :func:`run_vgt_block`.
    """

    config: object
    tensors: dict[str, mx.array]
    layer_index: int = 0
    mode: str = "frame"

    def forward(
        self,
        hidden_states: mx.array,
        pos: mx.array | None = None,
    ) -> tuple[mx.array, object | None]:
        return run_vgt_block(
            hidden_states,
            self.config,
            self.tensors,
            layer_index=self.layer_index,
            mode=self.mode,
            rope_positions=pos,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        pos: mx.array | None = None,
    ) -> tuple[mx.array, object | None]:
        return self.forward(hidden_states, pos=pos)


@dataclass(frozen=True)
class DistBlock(Block):
    """Inference-only facade for the vendor distributed block.

    HY-World's distributed arguments are training/multi-device concerns. The
    MLX inference path is single-device, so the extra arguments are accepted
    for API parity and ignored.
    """

    def forward(
        self,
        hidden_states: mx.array,
        pos: mx.array | None = None,
        sp_size: int = 1,
        sp_group: object | None = None,
        padding_tokens: int = 0,
        block_type: str | None = None,
        token_shape: tuple[int, ...] | None = None,
    ) -> tuple[mx.array, object | None]:
        del sp_size, sp_group, padding_tokens, block_type, token_shape
        return super().forward(hidden_states, pos=pos)


@dataclass(frozen=True)
class NestedTensorBlock(Block):
    """NestedTensorBlock facade for MLX inference.

    The xFormers nested-list path in the vendor code is not used for the
    single-device inference port. Lists are still accepted and processed
    element-wise so callers can keep the vendor name without a separate route.
    """

    def forward(
        self,
        hidden_states: mx.array | list[mx.array],
        pos: mx.array | list[mx.array] | None = None,
    ) -> tuple[mx.array | list[mx.array], object | None]:
        if not isinstance(hidden_states, list):
            rope_positions = pos if not isinstance(pos, list) else pos[0]
            return super().forward(hidden_states, pos=rope_positions)

        outputs: list[mx.array] = []
        positions = pos if isinstance(pos, list) else [pos] * len(hidden_states)
        for item, item_pos in zip(hidden_states, positions, strict=True):
            output, blocker = super().forward(item, pos=item_pos)
            if blocker is not None:
                return hidden_states, blocker
            outputs.append(output)
        return outputs, None


def run_vgt_block(
    hidden_states: mx.array,
    config: object,
    tensors: dict[str, mx.array],
    *,
    layer_index: int,
    mode: str,
    rope_positions: mx.array | None = None,
) -> tuple[mx.array, object | None]:
    """Run a single VGT transformer block (frame or global attention).

    This is the functional equivalent of the vendor ``NestedTensorBlock.forward``
    adapted to the MLX tensor-dict convention.

    Args:
        hidden_states: ``(B, N, D)`` input token sequence.
        config: VisualGeometryTransformerConfig with ``embed_dim``, ``num_heads``,
            ``head_dim``, ``layer_norm_eps``, ``rope_base``, etc.
        tensors: Weight dictionary with keys like
            ``{mode}_blocks.{i}.norm1.weight``, ``{mode}_blocks.{i}.attn.qkv.weight``, etc.
        layer_index: Block index within the sequence.
        mode: ``"frame"`` or ``"global"`` — determines the key prefix.
        rope_positions: ``(B, N, 2)`` RoPE coordinates, or None.

    Returns:
        ``(output_hidden_states, blocker)`` where blocker is ``None`` on success.
    """
    embed_dim = _int_or(config.embed_dim)
    num_heads = _int_or(config.num_heads)
    head_dim = _int_or(config.head_dim)
    layer_norm_eps = getattr(config, "layer_norm_eps", 1e-5)
    rope_base = getattr(config, "rope_base", 100.0)

    if embed_dim % num_heads:
        return hidden_states, _make_blocker(
            "visual-transformer",
            "HY-World attention head dimension validation",
            f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}",
            {"embed_dim": embed_dim, "num_heads": num_heads},
        )

    layer = f"{mode}_blocks.{layer_index}"
    required = _transformer_block_required_keys(layer, tensors)
    missing = tuple(key for key in required if key not in tensors)
    if missing:
        return hidden_states, _make_blocker(
            "visual-transformer",
            "HY-World transformer block tensor lookup",
            f"missing tensor for VisualGeometryTransformer block: {missing[0]}",
            {"missing": missing, "layer_index": layer_index},
        )

    residual = hidden_states
    normalized = layer_norm(
        hidden_states,
        tensors[f"{layer}.norm1.weight"],
        tensors[f"{layer}.norm1.bias"],
        eps=layer_norm_eps,
    )
    attended, attn_blocker = _self_attention(
        normalized,
        config,
        tensors,
        layer_index=layer_index,
        mode=mode,
        rope_positions=rope_positions,
    )
    if attn_blocker is not None:
        return hidden_states, attn_blocker
    hidden_states = residual + apply_layer_scale(
        attended, tensors.get(f"{layer}.ls1.gamma")
    )

    residual = hidden_states
    normalized = layer_norm(
        hidden_states,
        tensors[f"{layer}.norm2.weight"],
        tensors[f"{layer}.norm2.bias"],
        eps=layer_norm_eps,
    )
    mlp_out = _block_mlp_with_dict(normalized, layer, tensors)
    hidden_states = residual + apply_layer_scale(
        mlp_out, tensors.get(f"{layer}.ls2.gamma")
    )
    return hidden_states, None


def run_dino_block(
    hidden_states: mx.array,
    config: object,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> tuple[mx.array, object | None]:
    """Run a single DINO ViT transformer block (patch embed backbone).

    Uses key prefix ``patch_embed.blocks.{i}`` and does NOT apply RoPE.
    Otherwise structurally identical to :func:`run_vgt_block`.

    Args:
        hidden_states: ``(B, N, D)`` input token sequence.
        config: VisualGeometryTransformerConfig.
        tensors: Weight dictionary with ``patch_embed.blocks.{i}.*`` keys.
        block_index: Block index within the DINO backbone.

    Returns:
        ``(output_hidden_states, blocker)`` where blocker is ``None`` on success.
    """
    layer = f"patch_embed.blocks.{block_index}"
    required = _transformer_block_required_keys(layer, tensors)
    missing = tuple(key for key in required if key not in tensors)
    if missing:
        return hidden_states, _make_blocker(
            "visual-transformer",
            "HY-World DINO transformer block tensor lookup",
            f"missing tensor for DINO block: {missing[0]}",
            {"missing": missing, "block_index": block_index},
        )

    residual = hidden_states
    normalized = layer_norm(
        hidden_states,
        tensors[f"{layer}.norm1.weight"],
        tensors[f"{layer}.norm1.bias"],
        eps=1e-6,
    )
    attended, attn_blocker = _self_attention(
        normalized,
        config,
        tensors,
        layer_index=block_index,
        mode="patch_embed.blocks",
        rope_positions=None,
    )
    if attn_blocker is not None:
        return hidden_states, attn_blocker
    hidden_states = residual + apply_layer_scale(
        attended, tensors.get(f"{layer}.ls1.gamma")
    )

    residual = hidden_states
    normalized = layer_norm(
        hidden_states,
        tensors[f"{layer}.norm2.weight"],
        tensors[f"{layer}.norm2.bias"],
        eps=1e-6,
    )
    mlp_out = _block_mlp_with_dict(normalized, layer, tensors)
    hidden_states = residual + apply_layer_scale(
        mlp_out, tensors.get(f"{layer}.ls2.gamma")
    )
    return hidden_states, None


def _self_attention(
    hidden_states: mx.array,
    config: object,
    tensors: dict[str, mx.array],
    *,
    layer_index: int,
    mode: str,
    rope_positions: mx.array | None,
) -> tuple[mx.array, object | None]:
    """QKV projection → optional QK-norm → optional 2D RoPE → attention → output projection."""
    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
    head_dim = _int_or(config.head_dim)
    num_heads = _int_or(config.num_heads)
    layer = _block_prefix(mode, layer_index)

    qkv = linear(
        hidden_states,
        tensors[f"{layer}.attn.qkv.weight"],
        tensors[f"{layer}.attn.qkv.bias"],
    )
    q_part, k_part, v_part = mx.split(qkv, 3, axis=-1)
    query = _reshape_heads(q_part, num_heads)
    key = _reshape_heads(k_part, num_heads)
    value = _reshape_heads(v_part, num_heads)

    qk_norm_eps = 1e-6 if mode == "patch_embed.blocks" else float(getattr(config, "qk_norm_eps", 1e-5))
    query = _maybe_head_layer_norm(query, tensors, f"{layer}.attn.q_norm", eps=qk_norm_eps)
    key = _maybe_head_layer_norm(key, tensors, f"{layer}.attn.k_norm", eps=qk_norm_eps)

    if rope_positions is not None:
        rope_base = getattr(config, "rope_base", 100.0)
        if rope_positions.ndim != 3:
            return hidden_states, _make_blocker(
                "visual-transformer",
                "HY-World 2D RoPE position validation",
                f"expected RoPE positions shape [B,tokens,2], got {tuple(rope_positions.shape)}",
                {"positions": tuple(rope_positions.shape)},
            )
        head_dim_val = _int_or(config.head_dim)
        expected_positions = (batch, token_count, 2)
        if tuple(int(dim) for dim in rope_positions.shape) != expected_positions:
            return hidden_states, _make_blocker(
                "visual-transformer",
                "HY-World 2D RoPE position validation",
                f"expected RoPE positions shape {expected_positions}, got {tuple(rope_positions.shape)}",
                {"expected": expected_positions, "actual": tuple(rope_positions.shape)},
            )
        if head_dim_val % 4 != 0:
            return hidden_states, _make_blocker(
                "visual-transformer",
                "HY-World 2D RoPE head dimension validation",
                f"head_dim={head_dim_val} must be divisible by 4 for exact 2D RoPE",
                {"head_dim": head_dim_val, "num_heads": _int_or(config.num_heads), "embed_dim": _int_or(config.embed_dim)},
            )
        if getattr(config, "normalized_rope", False):
            normalize_coords = getattr(config, "rope_normalize_coords", "separate")
            query = _apply_normalized_2d_rope_to_heads(
                query,
                rope_positions,
                base=float(rope_base),
                normalize_coords=str(normalize_coords),
            )
            key = _apply_normalized_2d_rope_to_heads(
                key,
                rope_positions,
                base=float(rope_base),
                normalize_coords=str(normalize_coords),
            )
        else:
            roped_query, roped_key = apply_2d_rope(query, key, rope_positions, rope_base)
            query = roped_query
            key = roped_key

    attended, attention_blocker = _exact_full_attention(
        query,
        key,
        value,
        max_attention_bytes=int(getattr(config, "max_attention_bytes", 4_000_000_000)),
        query_chunk_size=getattr(config, "query_chunk_size", None),
        operation=f"HY-World exact {mode} full attention",
    )
    if attention_blocker is not None:
        return hidden_states, attention_blocker

    merged = mx.reshape(
        mx.transpose(attended, (0, 2, 1, 3)),
        (batch, token_count, _int_or(config.embed_dim)),
    )
    proj_weight, proj_bias = _attention_projection_tensors(layer, tensors)
    output = linear(merged, proj_weight, proj_bias)
    return output, None


def _apply_normalized_2d_rope_to_heads(
    values: mx.array,
    positions: mx.array,
    *,
    base: float,
    normalize_coords: str,
) -> mx.array:
    """Apply HY-World/DINOv3 normalized 2D RoPE to ``(B, heads, N, head_dim)``."""
    batch, _, token_count, head_dim = tuple(int(dim) for dim in values.shape)
    quarter_dim = head_dim // 4
    half_dim = head_dim // 2
    pos = positions.astype(mx.int32)
    height = int(mx.max(pos[..., 0])) + 1
    width = int(mx.max(pos[..., 1])) + 1
    ddtype = mx.float32

    if normalize_coords == "max":
        denom = float(max(height, width))
        coords_h = (mx.arange(height, dtype=ddtype) + 0.5) / denom
        coords_w = (mx.arange(width, dtype=ddtype) + 0.5) / denom
    elif normalize_coords == "min":
        denom = float(min(height, width))
        coords_h = (mx.arange(height, dtype=ddtype) + 0.5) / denom
        coords_w = (mx.arange(width, dtype=ddtype) + 0.5) / denom
    elif normalize_coords == "separate":
        coords_h = (mx.arange(height, dtype=ddtype) + 0.5) / float(height)
        coords_w = (mx.arange(width, dtype=ddtype) + 0.5) / float(width)
    else:
        coords_h = (mx.arange(height, dtype=ddtype) + 0.5) / float(height)
        coords_w = (mx.arange(width, dtype=ddtype) + 0.5) / float(width)

    rows = mx.broadcast_to(coords_h[:, None], (height, width))
    cols = mx.broadcast_to(coords_w[None, :], (height, width))
    coords = mx.stack((mx.reshape(rows, (-1,)), mx.reshape(cols, (-1,))), axis=-1)
    coords = 2.0 * coords - 1.0

    exponents = 2.0 * mx.arange(quarter_dim, dtype=ddtype) / float(half_dim)
    periods = mx.exp(mx.log(mx.array(base, dtype=ddtype)) * exponents)
    angles = (2.0 * math.pi * coords[:, :, None]) / periods[None, None, :]
    angles = mx.reshape(angles, (height * width, half_dim))
    angles = mx.concatenate((angles, angles), axis=-1)

    flat_indices = mx.reshape(pos[..., 0] * width + pos[..., 1], (-1,))
    gathered_cos = mx.reshape(mx.cos(angles)[flat_indices], (batch, 1, token_count, head_dim))
    gathered_sin = mx.reshape(mx.sin(angles)[flat_indices], (batch, 1, token_count, head_dim))
    gathered_cos = gathered_cos.astype(values.dtype)
    gathered_sin = gathered_sin.astype(values.dtype)
    return values * gathered_cos + _rotate_half(values) * gathered_sin


def _rotate_half(values: mx.array) -> mx.array:
    first, second = mx.split(values, 2, axis=-1)
    return mx.concatenate((-second, first), axis=-1)


def _exact_full_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    max_attention_bytes: int,
    query_chunk_size: int | None = None,
    operation: str = "HY-World exact full attention",
) -> tuple[mx.array, object | None]:
    blocker = _validate_attention_inputs(query, key, value)
    if blocker is not None:
        return query, blocker

    batch, heads, query_count, head_dim = tuple(int(dim) for dim in query.shape)
    key_count = int(key.shape[2])
    chunk_size = query_count if query_chunk_size is None else int(query_chunk_size)
    if chunk_size <= 0:
        return query, _make_blocker(
            "visual-transformer",
            "HY-World attention chunk validation",
            f"query_chunk_size must be positive, got {query_chunk_size}",
            {"query_chunk_size": query_chunk_size},
        )
    allocation_blocker = _guard_attention_allocation(
        batch=batch,
        heads=heads,
        query_count=min(query_count, chunk_size),
        key_count=key_count,
        max_attention_bytes=max_attention_bytes,
        operation=operation,
    )
    if allocation_blocker is not None:
        return query, allocation_blocker

    if query_chunk_size is None or chunk_size >= query_count:
        return (
            scaled_dot_product_attention(query, key, value, scale=head_dim**-0.5),
            None,
        )

    chunks = []
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        chunks.append(
            scaled_dot_product_attention(
                query[:, :, start:stop, :],
                key,
                value,
                scale=head_dim**-0.5,
            )
        )
    return mx.concatenate(chunks, axis=2), None


def _validate_attention_inputs(
    query: mx.array,
    key: mx.array,
    value: mx.array,
) -> object | None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return _make_blocker(
            "visual-transformer",
            "HY-World attention tensor validation",
            "query, key, and value must have shape [B,heads,tokens,head_dim]",
            {"query": tuple(query.shape), "key": tuple(key.shape), "value": tuple(value.shape)},
        )
    if tuple(query.shape[:2]) != tuple(key.shape[:2]) or tuple(key.shape) != tuple(value.shape):
        return _make_blocker(
            "visual-transformer",
            "HY-World attention tensor validation",
            "query, key, and value batch/head/key shapes must match for full attention",
            {"query": tuple(query.shape), "key": tuple(key.shape), "value": tuple(value.shape)},
        )
    if int(query.shape[-1]) != int(key.shape[-1]):
        return _make_blocker(
            "visual-transformer",
            "HY-World attention tensor validation",
            "query and key head dimensions must match",
            {"query": tuple(query.shape), "key": tuple(key.shape)},
        )
    return None


def _guard_attention_allocation(
    *,
    batch: int,
    heads: int,
    query_count: int,
    key_count: int,
    max_attention_bytes: int,
    operation: str,
) -> object | None:
    estimated = batch * heads * query_count * key_count * _FLOAT32_BYTES
    if estimated > max_attention_bytes:
        return _make_blocker(
            "visual-transformer",
            operation,
            (
                "HY-World exact full attention would exceed the configured activation guard "
                f"({estimated} > {max_attention_bytes} bytes)"
            ),
            {
                "estimated_attention_bytes": estimated,
                "max_attention_bytes": max_attention_bytes,
                "batch": batch,
                "heads": heads,
                "query_count": query_count,
                "key_count": key_count,
                "exact_attention": True,
            },
        )
    return None


def _block_mlp_with_dict(
    hidden_states: mx.array,
    layer: str,
    tensors: dict[str, mx.array],
) -> mx.array:
    """MLP sub-block: GELU(up) → down, dispatching to SwiGLU if needed."""
    if f"{layer}.mlp.fc1.weight" in tensors:
        return block_mlp(
            hidden_states,
            tensors[f"{layer}.mlp.fc1.weight"],
            tensors[f"{layer}.mlp.fc1.bias"],
            tensors[f"{layer}.mlp.fc2.weight"],
            tensors[f"{layer}.mlp.fc2.bias"],
        )
    if f"{layer}.mlp.w12.weight" in tensors:
        return block_swiglu_ffn(
            hidden_states,
            tensors[f"{layer}.mlp.w12.weight"],
            tensors.get(f"{layer}.mlp.w12.bias"),
            tensors[f"{layer}.mlp.w3.weight"],
            tensors.get(f"{layer}.mlp.w3.bias"),
        )
    up_weight = tensors[f"{layer}.mlp.up.weight"]
    up_bias = tensors[f"{layer}.mlp.up.bias"]
    down_weight = tensors[f"{layer}.mlp.down.weight"]
    down_bias = tensors[f"{layer}.mlp.down.bias"]
    return block_mlp(hidden_states, up_weight, up_bias, down_weight, down_bias)


def _transformer_block_required_keys(
    layer: str, tensors: dict[str, mx.array]
) -> tuple[str, ...]:
    projection = (
        f"{layer}.attn.proj" if f"{layer}.attn.proj.weight" in tensors
        else f"{layer}.attn.out"
    )
    mlp_up = (
        f"{layer}.mlp.fc1" if f"{layer}.mlp.fc1.weight" in tensors
        else f"{layer}.mlp.w12" if f"{layer}.mlp.w12.weight" in tensors
        else f"{layer}.mlp.up"
    )
    mlp_down = (
        f"{layer}.mlp.fc2" if f"{layer}.mlp.fc2.weight" in tensors
        else f"{layer}.mlp.w3" if f"{layer}.mlp.w3.weight" in tensors
        else f"{layer}.mlp.down"
    )
    return (
        f"{layer}.norm1.weight",
        f"{layer}.norm1.bias",
        f"{layer}.norm2.weight",
        f"{layer}.norm2.bias",
        f"{layer}.attn.qkv.weight",
        f"{layer}.attn.qkv.bias",
        f"{projection}.weight",
        f"{projection}.bias",
        f"{mlp_up}.weight",
        f"{mlp_up}.bias",
        f"{mlp_down}.weight",
        f"{mlp_down}.bias",
    )


def _attention_projection_tensors(
    layer: str, tensors: dict[str, mx.array]
) -> tuple[mx.array, mx.array]:
    if f"{layer}.attn.proj.weight" in tensors:
        return tensors[f"{layer}.attn.proj.weight"], tensors[f"{layer}.attn.proj.bias"]
    return tensors[f"{layer}.attn.out.weight"], tensors[f"{layer}.attn.out.bias"]


def _block_prefix(mode: str, layer_index: int) -> str:
    if mode in {"frame", "global"}:
        return f"{mode}_blocks.{layer_index}"
    if mode == "patch_embed.blocks":
        return f"patch_embed.blocks.{layer_index}"
    return f"{mode}.{layer_index}"


def _reshape_heads(values: mx.array, num_heads: int) -> mx.array:
    """Reshape ``(B, N, D)`` → ``(B, heads, N, head_dim)``."""
    batch, token_count, _ = tuple(int(dim) for dim in values.shape)
    head_dim = _int_or(values.shape[-1]) // num_heads
    return mx.transpose(
        mx.reshape(values, (batch, token_count, num_heads, head_dim)),
        (0, 2, 1, 3),
    )


def _maybe_head_layer_norm(
    values: mx.array, tensors: dict[str, mx.array], prefix: str, *, eps: float
) -> mx.array:
    """Per-head layer norm (QK-norm), optional."""
    weight = tensors.get(f"{prefix}.weight")
    bias = tensors.get(f"{prefix}.bias")
    if weight is None or bias is None:
        return values
    return head_layer_norm(values, weight, bias, eps=eps)


class _SimpleBlocker:
    """Minimal blocker sentinel for transformer block errors."""

    def __init__(self, stage: str, operation: str, reason: str, metadata: dict):
        self.stage = stage
        self.operation = operation
        self.reason = reason
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"_SimpleBlocker({self.stage!r}, {self.operation!r}, {self.reason!r})"


def _make_blocker(stage: str, operation: str, reason: str, metadata: dict) -> _SimpleBlocker:
    return _SimpleBlocker(stage, operation, reason, metadata)


def _int_or(value) -> int:
    """Convert an int-like value (mlx array or python int) to int."""
    if isinstance(value, int):
        return value
    return int(value)
