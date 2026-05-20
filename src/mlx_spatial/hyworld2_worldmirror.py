"""MLX VisualGeometryTransformer core for HY-World-2.0 WorldMirror."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .hyworld2_assets import HyWorld2ModelConfig
from .hyworld2_transformer import run_dino_block, run_vgt_block
from .hyworld2_vit import interpolate_dino_pos_embed, run_dino_vit


_FLOAT32_BYTES = 4
_MAX_DEFAULT_FIXTURE_BYTES = 64_000_000
_RESNET_MEAN = (0.485, 0.456, 0.406)
_RESNET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class HyWorld2WorldMirrorBlocker:
    """Structured blocker for exact WorldMirror backbone execution."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualGeometryTransformerConfig:
    """Small configurable subset of the official VisualGeometryTransformer contract."""

    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_register_tokens: int = 4
    enable_cond: bool = True
    rope_base: float = 100.0
    normalized_rope: bool = True
    enable_rope: bool = True
    layer_norm_eps: float = 1e-5
    max_tokens: int = 8192
    max_attention_bytes: int = 4_000_000_000
    max_fixture_bytes: int = _MAX_DEFAULT_FIXTURE_BYTES
    query_chunk_size: int | None = None
    intermediate_layers: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @classmethod
    def from_model_config(
        cls,
        config: HyWorld2ModelConfig,
        *,
        max_tokens: int = 8192,
        max_attention_bytes: int = 4_000_000_000,
        max_fixture_bytes: int = _MAX_DEFAULT_FIXTURE_BYTES,
        query_chunk_size: int | None = None,
        intermediate_layers: Sequence[int] = (),
    ) -> "VisualGeometryTransformerConfig":
        return cls(
            img_size=config.img_size,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_register_tokens=config.num_register_tokens,
            enable_cond=config.enable_cond,
            rope_base=config.rope_base,
            normalized_rope=config.normalized_rope,
            max_tokens=max_tokens,
            max_attention_bytes=max_attention_bytes,
            max_fixture_bytes=max_fixture_bytes,
            query_chunk_size=query_chunk_size,
            intermediate_layers=tuple(intermediate_layers),
        )


@dataclass(frozen=True)
class HyWorld2TokenAssembly:
    """Patch/special token assembly result."""

    tokens: mx.array | None = None
    patch_tokens: mx.array | None = None
    rope_positions: mx.array | None = None
    patch_grid: tuple[int, int] | None = None
    patch_start_idx: int | None = None
    frame_token_count: int | None = None
    blocker: HyWorld2WorldMirrorBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.tokens is not None and self.patch_start_idx is not None and self.blocker is None


@dataclass(frozen=True)
class HyWorld2AttentionOutput:
    hidden_states: mx.array | None = None
    blocker: HyWorld2WorldMirrorBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.hidden_states is not None and self.blocker is None


@dataclass(frozen=True)
class HyWorld2BackboneOutput:
    """Backbone output needed by later camera and dense-head slices."""

    tokens: mx.array | None = None
    intermediate_tokens: tuple[mx.array, ...] = ()
    intermediate_full_tokens: tuple[mx.array, ...] = ()
    patch_start_idx: int | None = None
    patch_grid: tuple[int, int] | None = None
    frame_token_count: int | None = None
    attention_modes: tuple[str, ...] = ()
    blocker: HyWorld2WorldMirrorBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.tokens is not None and self.patch_start_idx is not None and self.blocker is None


def default_visual_geometry_tensors(config: VisualGeometryTransformerConfig) -> dict[str, mx.array]:
    """Create deterministic tiny-checkpoint tensors for fixture execution."""

    fixture_guard = guard_fixture_tensor_allocation(config)
    if fixture_guard is not None:
        raise ValueError(fixture_guard.reason)

    hidden = config.embed_dim
    intermediate = int(hidden * config.mlp_ratio)
    tensors: dict[str, mx.array] = {
        "patch_embed.weight": mx.ones(
            (hidden, 3, config.patch_size, config.patch_size),
            dtype=mx.float32,
        ),
        "patch_embed.bias": mx.zeros((hidden,), dtype=mx.float32),
        "cam_token": mx.zeros((1, 2, 1, hidden), dtype=mx.float32),
        "reg_token": mx.zeros((1, 2, config.num_register_tokens, hidden), dtype=mx.float32),
    }
    eye = mx.eye(hidden, dtype=mx.float32)
    qkv_weight = mx.concatenate((eye, eye, eye), axis=0)
    qkv_bias = mx.zeros((hidden * 3,), dtype=mx.float32)
    for layer_index in range(config.depth):
        for block_name in ("frame_blocks", "global_blocks"):
            layer = f"{block_name}.{layer_index}"
            tensors.update(
                {
                    f"{layer}.norm1.weight": mx.ones((hidden,), dtype=mx.float32),
                    f"{layer}.norm1.bias": mx.zeros((hidden,), dtype=mx.float32),
                    f"{layer}.norm2.weight": mx.ones((hidden,), dtype=mx.float32),
                    f"{layer}.norm2.bias": mx.zeros((hidden,), dtype=mx.float32),
                    f"{layer}.attn.qkv.weight": qkv_weight,
                    f"{layer}.attn.qkv.bias": qkv_bias,
                    f"{layer}.attn.out.weight": eye,
                    f"{layer}.attn.out.bias": mx.zeros((hidden,), dtype=mx.float32),
                    f"{layer}.mlp.up.weight": mx.zeros((intermediate, hidden), dtype=mx.float32),
                    f"{layer}.mlp.up.bias": mx.zeros((intermediate,), dtype=mx.float32),
                    f"{layer}.mlp.down.weight": mx.zeros((hidden, intermediate), dtype=mx.float32),
                    f"{layer}.mlp.down.bias": mx.zeros((hidden,), dtype=mx.float32),
                }
            )
    return tensors


def assemble_worldmirror_tokens(
    image_tensor: mx.array,
    config: VisualGeometryTransformerConfig,
    tensors: dict[str, mx.array] | None = None,
) -> HyWorld2TokenAssembly:
    """Assemble camera/register/patch tokens from `[B,S,3,H,W]` images."""

    blocker = _validate_image_tensor(image_tensor, config)
    if blocker is not None:
        return HyWorld2TokenAssembly(blocker=blocker)
    if tensors is None:
        fixture_guard = guard_fixture_tensor_allocation(config)
        if fixture_guard is not None:
            return HyWorld2TokenAssembly(blocker=fixture_guard)
        tensors = default_visual_geometry_tensors(config)

    batch, frames, channels, height, width = tuple(int(dim) for dim in image_tensor.shape)
    patch_grid = (height // config.patch_size, width // config.patch_size)
    patch_count = patch_grid[0] * patch_grid[1]
    condition_token_count = 2 if config.enable_cond else 0
    patch_start_idx = 1 + config.num_register_tokens + condition_token_count
    frame_token_count = patch_start_idx + patch_count
    guard = _guard_token_count(
        batch=batch,
        frames=frames,
        frame_token_count=frame_token_count,
        config=config,
        operation="assemble HY-World camera/register/patch tokens",
    )
    if guard is not None:
        return HyWorld2TokenAssembly(blocker=guard)

    patch_required = (
        ("patch_embed.patch_embed.proj.weight", "patch_embed.patch_embed.proj.bias")
        if _has_official_dino_patch_embed(tensors)
        else ("patch_embed.weight", "patch_embed.bias")
    )
    missing = tuple(
        key for key in (*patch_required, "cam_token", "reg_token")
        if key not in tensors
    )
    if missing:
        return HyWorld2TokenAssembly(
            blocker=_blocker(
                "model-construction",
                "HY-World VisualGeometryTransformer token tensor lookup",
                f"missing tensor for token assembly: {missing[0]}",
                {"missing": missing},
            )
        )

    if _has_official_dino_patch_embed(tensors):
        dino = _official_dino_patch_tokens(image_tensor, config, tensors, patch_grid=patch_grid)
        if dino.blocker is not None or dino.patch_tokens is None:
            return HyWorld2TokenAssembly(blocker=dino.blocker)
        patch_tokens = dino.patch_tokens
    else:
        reshaped = mx.reshape(image_tensor, (batch * frames, channels, height, width))
        nhwc = mx.transpose(reshaped, (0, 2, 3, 1))
        weight = tensors["patch_embed.weight"]
        expected_weight = (config.embed_dim, 3, config.patch_size, config.patch_size)
        if tuple(weight.shape) != expected_weight:
            return HyWorld2TokenAssembly(
                blocker=_blocker(
                    "model-construction",
                    "HY-World patch embedding shape validation",
                    f"expected patch_embed.weight shape {expected_weight}, got {tuple(weight.shape)}",
                    {"expected": expected_weight, "actual": tuple(weight.shape)},
                )
            )
        hwio_weight = mx.transpose(weight, (0, 2, 3, 1))
        embedded = mx.conv2d(nhwc, hwio_weight, stride=config.patch_size) + tensors["patch_embed.bias"]
        patch_tokens = mx.reshape(embedded, (batch, frames, patch_count, config.embed_dim))

    special_blocker = _validate_special_tokens(tensors, config)
    if special_blocker is not None:
        return HyWorld2TokenAssembly(blocker=special_blocker)

    camera, registers = _select_frame_special_tokens(
        tensors["cam_token"],
        tensors["reg_token"],
        batch=batch,
        frames=frames,
        config=config,
    )
    token_parts = [camera, registers]
    if config.enable_cond:
        token_parts.append(mx.zeros((batch * frames, 2, config.embed_dim), dtype=mx.float32))
    frame_patches = mx.reshape(patch_tokens, (batch * frames, patch_count, config.embed_dim))
    token_parts.append(frame_patches)
    frame_tokens = mx.concatenate(token_parts, axis=1)
    tokens = mx.reshape(frame_tokens, (batch, frames * frame_token_count, config.embed_dim))
    rope_positions = build_worldmirror_rope_positions(
        frames=frames,
        patch_grid=patch_grid,
        patch_start_idx=patch_start_idx,
        normalized=config.normalized_rope,
    )

    return HyWorld2TokenAssembly(
        tokens=tokens,
        patch_tokens=patch_tokens,
        rope_positions=rope_positions,
        patch_grid=patch_grid,
        patch_start_idx=patch_start_idx,
        frame_token_count=frame_token_count,
    )


def build_worldmirror_rope_positions(
    *,
    frames: int,
    patch_grid: tuple[int, int],
    patch_start_idx: int,
    normalized: bool,
) -> mx.array:
    """Build vendor-compatible per-frame RoPE position indices.

    HY-World shifts patch positions by +1 when special tokens are present so
    special tokens occupy RoPE index 0 and image patches start at index 1.
    Normalized RoPE still receives integer indices; normalization happens in
    the RoPE module itself.
    """

    grid_h, grid_w = patch_grid
    rows = mx.broadcast_to(mx.arange(grid_h, dtype=mx.float32)[:, None], (grid_h, grid_w))
    cols = mx.broadcast_to(mx.arange(grid_w, dtype=mx.float32)[None, :], (grid_h, grid_w))
    if patch_start_idx > 0:
        rows = rows + 1
        cols = cols + 1
    patch_positions = mx.stack((mx.reshape(rows, (-1,)), mx.reshape(cols, (-1,))), axis=-1)
    special_positions = mx.zeros((patch_start_idx, 2), dtype=mx.float32)
    frame_positions = mx.concatenate((special_positions, patch_positions), axis=0)
    return mx.broadcast_to(frame_positions[None, :, :], (frames, int(frame_positions.shape[0]), 2))


def exact_full_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    max_attention_bytes: int,
    query_chunk_size: int | None = None,
    operation: str = "HY-World exact full attention",
) -> HyWorld2AttentionOutput:
    """Run exact full attention, optionally chunked by query blocks."""

    blocker = _validate_attention_inputs(query, key, value)
    if blocker is not None:
        return HyWorld2AttentionOutput(blocker=blocker)

    batch, heads, query_count, head_dim = tuple(int(dim) for dim in query.shape)
    key_count = int(key.shape[2])
    chunk_size = query_count if query_chunk_size is None else int(query_chunk_size)
    if chunk_size <= 0:
        return HyWorld2AttentionOutput(
            blocker=_blocker(
                "visual-transformer",
                "HY-World attention chunk validation",
                f"query_chunk_size must be positive, got {query_chunk_size}",
                {"query_chunk_size": query_chunk_size},
            )
        )
    guard = guard_attention_allocation(
        batch=batch,
        heads=heads,
        query_count=min(query_count, chunk_size),
        key_count=key_count,
        max_attention_bytes=max_attention_bytes,
        operation=operation,
    )
    if guard is not None:
        return HyWorld2AttentionOutput(blocker=guard)

    if query_chunk_size is None or chunk_size >= query_count:
        return HyWorld2AttentionOutput(
            hidden_states=_scaled_dot_product_attention(
                query,
                key,
                value,
                scale=head_dim**-0.5,
            )
        )

    chunks = []
    for start in range(0, query_count, chunk_size):
        stop = min(start + chunk_size, query_count)
        chunks.append(
            _scaled_dot_product_attention(
                query[:, :, start:stop, :],
                key,
                value,
                scale=head_dim**-0.5,
            )
        )
    return HyWorld2AttentionOutput(hidden_states=mx.concatenate(chunks, axis=2))


def run_visual_geometry_transformer(
    image_tensor: mx.array,
    config: VisualGeometryTransformerConfig,
    tensors: dict[str, mx.array] | None = None,
) -> HyWorld2BackboneOutput:
    """Execute the fixture-scale WorldMirror VisualGeometryTransformer core."""

    image_blocker = _validate_image_tensor(image_tensor, config)
    if image_blocker is not None:
        return HyWorld2BackboneOutput(blocker=image_blocker)
    if tensors is None:
        fixture_guard = guard_fixture_tensor_allocation(config)
        if fixture_guard is not None:
            return HyWorld2BackboneOutput(blocker=fixture_guard)
        tensors = default_visual_geometry_tensors(config)
    assembly = assemble_worldmirror_tokens(image_tensor, config, tensors)
    if assembly.blocker is not None:
        return HyWorld2BackboneOutput(blocker=assembly.blocker)
    if (
        assembly.tokens is None
        or assembly.patch_start_idx is None
        or assembly.frame_token_count is None
    ):
        return HyWorld2BackboneOutput(
            blocker=_blocker(
                "visual-transformer",
                "HY-World token assembly",
                "token assembly returned no tokens",
                {},
            )
        )

    batch = int(image_tensor.shape[0])
    frames = int(image_tensor.shape[1])
    hidden = assembly.tokens
    intermediates: list[mx.array] = []
    full_intermediates: list[mx.array] = []
    modes: list[str] = []
    frame_rope_positions: mx.array | None = None
    global_rope_positions: mx.array | None = None
    if config.enable_rope and assembly.rope_positions is not None:
        frame_rope_positions = mx.reshape(
            mx.broadcast_to(
                assembly.rope_positions[None, :, :, :],
                (batch, frames, assembly.frame_token_count, 2),
            ),
            (batch * frames, assembly.frame_token_count, 2),
        )
        global_rope_positions = mx.reshape(
            mx.broadcast_to(
                assembly.rope_positions[None, :, :, :],
                (batch, frames, assembly.frame_token_count, 2),
            ),
            (batch, frames * assembly.frame_token_count, 2),
        )

    for layer_index in range(config.depth):
        modes.append("frame")
        layer_input = mx.reshape(
            hidden,
            (batch * frames, assembly.frame_token_count, config.embed_dim),
        )
        local_block = _run_transformer_block(
            layer_input,
            config,
            tensors,
            layer_index=layer_index,
            mode="frame",
            rope_positions=frame_rope_positions,
        )
        if local_block.blocker is not None or local_block.hidden_states is None:
            return HyWorld2BackboneOutput(
                patch_start_idx=assembly.patch_start_idx,
                patch_grid=assembly.patch_grid,
                frame_token_count=assembly.frame_token_count,
                attention_modes=tuple(modes),
                blocker=local_block.blocker,
            )
        local_hidden = mx.reshape(
            local_block.hidden_states,
            (batch, frames * assembly.frame_token_count, config.embed_dim),
        )

        modes.append("global")
        global_block = _run_transformer_block(
            local_hidden,
            config,
            tensors,
            layer_index=layer_index,
            mode="global",
            rope_positions=global_rope_positions,
        )
        if global_block.blocker is not None or global_block.hidden_states is None:
            return HyWorld2BackboneOutput(
                patch_start_idx=assembly.patch_start_idx,
                patch_grid=assembly.patch_grid,
                frame_token_count=assembly.frame_token_count,
                attention_modes=tuple(modes),
                blocker=global_block.blocker,
            )
        hidden = global_block.hidden_states

        if layer_index in config.intermediate_layers:
            full_tokens = mx.concatenate(
                (
                    _reshape_frame_tokens(local_hidden, frames, assembly.frame_token_count),
                    _reshape_frame_tokens(hidden, frames, assembly.frame_token_count),
                ),
                axis=-1,
            )
            full_intermediates.append(full_tokens)
            intermediates.append(full_tokens[:, :, assembly.patch_start_idx :, :])

    return HyWorld2BackboneOutput(
        tokens=hidden,
        intermediate_tokens=tuple(intermediates),
        intermediate_full_tokens=tuple(full_intermediates),
        patch_start_idx=assembly.patch_start_idx,
        patch_grid=assembly.patch_grid,
        frame_token_count=assembly.frame_token_count,
        attention_modes=tuple(modes),
    )


def guard_attention_allocation(
    *,
    batch: int,
    heads: int,
    query_count: int,
    key_count: int,
    max_attention_bytes: int,
    operation: str,
) -> HyWorld2WorldMirrorBlocker | None:
    estimated = batch * heads * query_count * key_count * _FLOAT32_BYTES
    if estimated > max_attention_bytes:
        return _blocker(
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


def estimate_fixture_tensor_bytes(config: VisualGeometryTransformerConfig) -> int:
    hidden = config.embed_dim
    intermediate = int(hidden * config.mlp_ratio)
    patch = hidden * 3 * config.patch_size * config.patch_size + hidden
    special = hidden + (2 * config.num_register_tokens * hidden)
    shared_attention = (hidden * hidden) + (3 * hidden * hidden) + (hidden * 4)
    per_block = (
        (4 * hidden)
        + (intermediate * hidden)
        + intermediate
        + (hidden * intermediate)
        + hidden
    )
    return (patch + special + shared_attention + (config.depth * 2 * per_block)) * _FLOAT32_BYTES


def guard_fixture_tensor_allocation(
    config: VisualGeometryTransformerConfig,
) -> HyWorld2WorldMirrorBlocker | None:
    estimated = estimate_fixture_tensor_bytes(config)
    if estimated > config.max_fixture_bytes:
        return _blocker(
            "model-construction",
            "HY-World deterministic fixture tensor allocation",
            (
                "deterministic fixture tensors would exceed the configured fixture guard "
                f"({estimated} > {config.max_fixture_bytes} bytes); provide checkpoint tensors "
                "for exact WorldMirror execution"
            ),
            {
                "estimated_fixture_bytes": estimated,
                "max_fixture_bytes": config.max_fixture_bytes,
                "embed_dim": config.embed_dim,
                "depth": config.depth,
                "num_heads": config.num_heads,
                "mlp_ratio": config.mlp_ratio,
            },
        )
    return None


def _has_official_dino_patch_embed(tensors: dict[str, mx.array]) -> bool:
    return "patch_embed.patch_embed.proj.weight" in tensors


def _official_dino_patch_tokens(
    image_tensor: mx.array,
    config: VisualGeometryTransformerConfig,
    tensors: dict[str, mx.array],
    *,
    patch_grid: tuple[int, int],
) -> HyWorld2TokenAssembly:
    patch_tokens, blocker = run_dino_vit(
        image_tensor,
        config,
        tensors,
        patch_grid=patch_grid,
    )
    if blocker is not None or patch_tokens is None:
        return HyWorld2TokenAssembly(blocker=_coerce_blocker(blocker))
    return HyWorld2TokenAssembly(patch_tokens=patch_tokens)


def _official_dino_pos_embed(
    pos_embed: mx.array,
    patch_grid: tuple[int, int],
) -> tuple[mx.array | None, HyWorld2WorldMirrorBlocker | None]:
    result, blocker = interpolate_dino_pos_embed(pos_embed, patch_grid)
    return result, _coerce_blocker(blocker)


def _fit_resized_pos_embed(values: np.ndarray, patch_grid: tuple[int, int]) -> np.ndarray:
    height, width = patch_grid
    output = values
    if output.shape[0] < height or output.shape[1] < width:
        pad_h = max(height - output.shape[0], 0)
        pad_w = max(width - output.shape[1], 0)
        output = np.pad(output, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return output[:height, :width, :]


def _run_dino_transformer_block(
    hidden_states: mx.array,
    config: VisualGeometryTransformerConfig,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
) -> HyWorld2AttentionOutput:
    result, blocker = run_dino_block(
        hidden_states,
        config,
        tensors,
        block_index=block_index,
    )
    if blocker is not None:
        return HyWorld2AttentionOutput(blocker=_coerce_blocker(blocker))
    return HyWorld2AttentionOutput(hidden_states=result)


def _run_transformer_block(
    hidden_states: mx.array,
    config: VisualGeometryTransformerConfig,
    tensors: dict[str, mx.array],
    *,
    layer_index: int,
    mode: str,
    rope_positions: mx.array | None,
) -> HyWorld2AttentionOutput:
    result, blocker = run_vgt_block(
        hidden_states,
        config,
        tensors,
        layer_index=layer_index,
        mode=mode,
        rope_positions=rope_positions,
    )
    if blocker is not None:
        return HyWorld2AttentionOutput(blocker=_coerce_blocker(blocker))
    return HyWorld2AttentionOutput(hidden_states=result)


def _self_attention(
    hidden_states: mx.array,
    config: VisualGeometryTransformerConfig,
    tensors: dict[str, mx.array],
    *,
    layer_index: int,
    mode: str,
    rope_positions: mx.array | None,
) -> HyWorld2AttentionOutput:
    batch, token_count, _ = tuple(int(dim) for dim in hidden_states.shape)
    head_dim = config.head_dim
    layer = _block_prefix(mode, layer_index)

    qkv = _linear(
        hidden_states,
        tensors[f"{layer}.attn.qkv.weight"],
        tensors[f"{layer}.attn.qkv.bias"],
    )
    query, key, value = (_heads(part, config) for part in mx.split(qkv, 3, axis=-1))
    query = _maybe_head_layer_norm(query, tensors, f"{layer}.attn.q_norm")
    key = _maybe_head_layer_norm(key, tensors, f"{layer}.attn.k_norm")
    if rope_positions is not None:
        roped_query, roped_key, rope_blocker = _apply_worldmirror_2d_rope(
            query,
            key,
            rope_positions,
            config,
        )
        if rope_blocker is not None:
            return HyWorld2AttentionOutput(blocker=rope_blocker)
        query = roped_query
        key = roped_key

    attended = exact_full_attention(
        query,
        key,
        value,
        max_attention_bytes=config.max_attention_bytes,
        query_chunk_size=config.query_chunk_size,
        operation=f"HY-World exact {mode} full attention",
    )
    if attended.blocker is not None or attended.hidden_states is None:
        return attended

    merged = mx.reshape(
        mx.transpose(attended.hidden_states, (0, 2, 1, 3)),
        (batch, token_count, config.embed_dim),
    )
    proj_weight, proj_bias = _attention_projection_tensors(layer, tensors)
    output = _linear(merged, proj_weight, proj_bias)
    return HyWorld2AttentionOutput(hidden_states=output)


def _block_prefix(mode: str, layer_index: int) -> str:
    if mode in {"frame", "global"}:
        return f"{mode}_blocks.{layer_index}"
    if mode == "patch_embed.blocks":
        return f"patch_embed.blocks.{layer_index}"
    return f"{mode}.{layer_index}"


def _transformer_block_required_keys(layer: str, tensors: dict[str, mx.array]) -> tuple[str, ...]:
    projection = f"{layer}.attn.proj" if f"{layer}.attn.proj.weight" in tensors else f"{layer}.attn.out"
    mlp_up = f"{layer}.mlp.fc1" if f"{layer}.mlp.fc1.weight" in tensors else f"{layer}.mlp.up"
    mlp_down = f"{layer}.mlp.fc2" if f"{layer}.mlp.fc2.weight" in tensors else f"{layer}.mlp.down"
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


def _attention_projection_tensors(layer: str, tensors: dict[str, mx.array]) -> tuple[mx.array, mx.array]:
    if f"{layer}.attn.proj.weight" in tensors:
        return tensors[f"{layer}.attn.proj.weight"], tensors[f"{layer}.attn.proj.bias"]
    return tensors[f"{layer}.attn.out.weight"], tensors[f"{layer}.attn.out.bias"]


def _block_mlp(hidden_states: mx.array, layer: str, tensors: dict[str, mx.array]) -> mx.array:
    if f"{layer}.mlp.fc1.weight" in tensors:
        up_weight = tensors[f"{layer}.mlp.fc1.weight"]
        up_bias = tensors[f"{layer}.mlp.fc1.bias"]
        down_weight = tensors[f"{layer}.mlp.fc2.weight"]
        down_bias = tensors[f"{layer}.mlp.fc2.bias"]
    else:
        up_weight = tensors[f"{layer}.mlp.up.weight"]
        up_bias = tensors[f"{layer}.mlp.up.bias"]
        down_weight = tensors[f"{layer}.mlp.down.weight"]
        down_bias = tensors[f"{layer}.mlp.down.bias"]
    return _linear(nn.gelu(_linear(hidden_states, up_weight, up_bias)), down_weight, down_bias)


def _apply_layer_scale(hidden_states: mx.array, gamma: mx.array | None) -> mx.array:
    if gamma is None:
        return hidden_states
    return hidden_states * gamma.astype(hidden_states.dtype)


def _maybe_head_layer_norm(values: mx.array, tensors: dict[str, mx.array], prefix: str) -> mx.array:
    weight = tensors.get(f"{prefix}.weight")
    bias = tensors.get(f"{prefix}.bias")
    if weight is None or bias is None:
        return values
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + 1e-6) * weight.astype(values.dtype) + bias.astype(values.dtype)


def _heads(values: mx.array, config: VisualGeometryTransformerConfig) -> mx.array:
    batch, token_count, _ = tuple(int(dim) for dim in values.shape)
    return mx.transpose(
        mx.reshape(values, (batch, token_count, config.num_heads, config.head_dim)),
        (0, 2, 1, 3),
    )


def _apply_worldmirror_2d_rope(
    query: mx.array,
    key: mx.array,
    positions: mx.array,
    config: VisualGeometryTransformerConfig,
) -> tuple[mx.array, mx.array, HyWorld2WorldMirrorBlocker | None]:
    if positions.ndim != 3:
        return query, key, _blocker(
            "visual-transformer",
            "HY-World 2D RoPE position validation",
            f"expected RoPE positions shape [B,tokens,2], got {tuple(positions.shape)}",
            {"positions": tuple(positions.shape)},
        )
    batch, _, token_count, head_dim = tuple(int(dim) for dim in query.shape)
    if tuple(positions.shape) != (batch, token_count, 2):
        return query, key, _blocker(
            "visual-transformer",
            "HY-World 2D RoPE position validation",
            (
                f"expected RoPE positions shape {(batch, token_count, 2)}, "
                f"got {tuple(positions.shape)}"
            ),
            {"expected": (batch, token_count, 2), "actual": tuple(positions.shape)},
        )
    if head_dim % 4:
        return query, key, _blocker(
            "visual-transformer",
            "HY-World 2D RoPE head dimension validation",
            f"head_dim={head_dim} must be divisible by 4 for exact 2D RoPE",
            {"head_dim": head_dim, "num_heads": config.num_heads, "embed_dim": config.embed_dim},
        )
    if config.rope_base <= 0:
        return query, key, _blocker(
            "visual-transformer",
            "HY-World 2D RoPE base validation",
            f"rope_base must be positive, got {config.rope_base}",
            {"rope_base": config.rope_base},
        )
    return (
        _apply_2d_rope_to_heads(query, positions, config.rope_base),
        _apply_2d_rope_to_heads(key, positions, config.rope_base),
        None,
    )


def _apply_2d_rope_to_heads(values: mx.array, positions: mx.array, rope_base: float) -> mx.array:
    head_dim = int(values.shape[-1])
    axis_dim = head_dim // 2
    row_values = values[..., :axis_dim]
    col_values = values[..., axis_dim:]
    return mx.concatenate(
        (
            _apply_1d_rope(row_values, positions[:, :, 0], rope_base),
            _apply_1d_rope(col_values, positions[:, :, 1], rope_base),
        ),
        axis=-1,
    )


def _apply_1d_rope(values: mx.array, coordinates: mx.array, rope_base: float) -> mx.array:
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


def _scaled_dot_product_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    *,
    scale: float,
) -> mx.array:
    logits = (query @ mx.transpose(key, (0, 1, 3, 2))) * scale
    weights = mx.softmax(logits.astype(mx.float32), axis=-1)
    return weights @ value


def _extract_patch_tokens(
    hidden_states: mx.array,
    frames: int,
    frame_token_count: int,
    patch_start_idx: int,
) -> mx.array:
    return _reshape_frame_tokens(hidden_states, frames, frame_token_count)[:, :, patch_start_idx:, :]


def _reshape_frame_tokens(
    hidden_states: mx.array,
    frames: int,
    frame_token_count: int,
) -> mx.array:
    batch, _, hidden = tuple(int(dim) for dim in hidden_states.shape)
    return mx.reshape(hidden_states, (batch, frames, frame_token_count, hidden))


def _validate_image_tensor(
    image_tensor: mx.array,
    config: VisualGeometryTransformerConfig,
) -> HyWorld2WorldMirrorBlocker | None:
    if image_tensor.ndim != 5:
        return _blocker(
            "visual-transformer",
            "HY-World image tensor validation",
            f"expected image tensor shape [B,S,3,H,W], got {tuple(image_tensor.shape)}",
            {"shape": tuple(image_tensor.shape)},
        )
    _, _, channels, height, width = tuple(int(dim) for dim in image_tensor.shape)
    if channels != 3:
        return _blocker(
            "visual-transformer",
            "HY-World image tensor validation",
            f"expected RGB channel count 3, got {channels}",
            {"shape": tuple(image_tensor.shape)},
        )
    if height % config.patch_size or width % config.patch_size:
        return _blocker(
            "visual-transformer",
            "HY-World patch grid validation",
            (
                f"image spatial size {(height, width)} is not divisible by "
                f"patch_size={config.patch_size}"
            ),
            {"height": height, "width": width, "patch_size": config.patch_size},
        )
    return None


def _validate_special_tokens(
    tensors: dict[str, mx.array],
    config: VisualGeometryTransformerConfig,
) -> HyWorld2WorldMirrorBlocker | None:
    expected_cam = (1, 2, 1, config.embed_dim)
    expected_reg = (1, 2, config.num_register_tokens, config.embed_dim)
    actual_cam = tuple(tensors["cam_token"].shape)
    actual_reg = tuple(tensors["reg_token"].shape)
    if actual_cam != expected_cam:
        return _blocker(
            "model-construction",
            "HY-World camera token shape validation",
            f"expected cam_token shape {expected_cam}, got {actual_cam}",
            {"expected": expected_cam, "actual": actual_cam},
        )
    if actual_reg != expected_reg:
        return _blocker(
            "model-construction",
            "HY-World register token shape validation",
            f"expected reg_token shape {expected_reg}, got {actual_reg}",
            {"expected": expected_reg, "actual": actual_reg},
        )
    return None


def _select_frame_special_tokens(
    cam_token: mx.array,
    reg_token: mx.array,
    *,
    batch: int,
    frames: int,
    config: VisualGeometryTransformerConfig,
) -> tuple[mx.array, mx.array]:
    cam_frames = mx.concatenate(
        tuple(
            cam_token[:, _frame_slot(frame) : _frame_slot(frame) + 1, :, :]
            for frame in range(frames)
        ),
        axis=1,
    )
    reg_frames = mx.concatenate(
        tuple(
            reg_token[:, _frame_slot(frame) : _frame_slot(frame) + 1, :, :]
            for frame in range(frames)
        ),
        axis=1,
    )
    camera = mx.broadcast_to(cam_frames, (batch, frames, 1, config.embed_dim))
    registers = mx.broadcast_to(
        reg_frames,
        (batch, frames, config.num_register_tokens, config.embed_dim),
    )
    return (
        mx.reshape(camera, (batch * frames, 1, config.embed_dim)),
        mx.reshape(registers, (batch * frames, config.num_register_tokens, config.embed_dim)),
    )


def _frame_slot(frame_index: int) -> int:
    return 0 if frame_index == 0 else 1


def _validate_attention_inputs(
    query: mx.array,
    key: mx.array,
    value: mx.array,
) -> HyWorld2WorldMirrorBlocker | None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return _blocker(
            "visual-transformer",
            "HY-World attention tensor validation",
            "query, key, and value must have shape [B,heads,tokens,head_dim]",
            {"query": tuple(query.shape), "key": tuple(key.shape), "value": tuple(value.shape)},
        )
    if tuple(query.shape[:2]) != tuple(key.shape[:2]) or tuple(key.shape) != tuple(value.shape):
        return _blocker(
            "visual-transformer",
            "HY-World attention tensor validation",
            "query, key, and value batch/head/key shapes must match for full attention",
            {"query": tuple(query.shape), "key": tuple(key.shape), "value": tuple(value.shape)},
        )
    if int(query.shape[-1]) != int(key.shape[-1]):
        return _blocker(
            "visual-transformer",
            "HY-World attention tensor validation",
            "query and key head dimensions must match",
            {"query": tuple(query.shape), "key": tuple(key.shape)},
        )
    return None


def _guard_token_count(
    *,
    batch: int,
    frames: int,
    frame_token_count: int,
    config: VisualGeometryTransformerConfig,
    operation: str,
) -> HyWorld2WorldMirrorBlocker | None:
    total_tokens = batch * frames * frame_token_count
    if total_tokens > config.max_tokens:
        return _blocker(
            "visual-transformer",
            operation,
            (
                "HY-World token count would exceed configured guard "
                f"({total_tokens} > {config.max_tokens})"
            ),
            {
                "token_count": total_tokens,
                "max_tokens": config.max_tokens,
                "batch": batch,
                "frames": frames,
                "frame_token_count": frame_token_count,
            },
        )
    return None


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight)
    if bias is not None:
        output = output + bias
    return output


def _layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight + bias


def _conv2d_nchw(
    values: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    values_nhwc = mx.transpose(values, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight.astype(values.dtype), (0, 2, 3, 1))
    output = mx.conv2d(values_nhwc, weight_ohwi, stride=stride, padding=padding)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _blocker(
    stage: str,
    operation: str,
    reason: str,
    metadata: dict[str, object],
) -> HyWorld2WorldMirrorBlocker:
    return HyWorld2WorldMirrorBlocker(
        stage=stage,
        operation=operation,
        reason=reason,
        metadata=metadata,
    )


def _coerce_blocker(blocker: object | None) -> HyWorld2WorldMirrorBlocker | None:
    if blocker is None:
        return None
    if isinstance(blocker, HyWorld2WorldMirrorBlocker):
        return blocker
    return _blocker(
        str(getattr(blocker, "stage", "visual-transformer")),
        str(getattr(blocker, "operation", "HY-World WorldMirror integration")),
        str(getattr(blocker, "reason", "unknown HY-World integration blocker")),
        dict(getattr(blocker, "metadata", {})),
    )
