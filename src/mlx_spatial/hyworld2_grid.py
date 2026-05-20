"""HY-World 2.0 positional grid utilities."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def create_hyworld2_uv_grid(
    width: int,
    height: int,
    *,
    aspect_ratio: float | None = None,
    dtype=mx.float32,
) -> mx.array:
    """Create the normalized HY-World UV grid from the vendor formula."""

    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {(width, height)}")
    effective_aspect = float(width) / float(height) if aspect_ratio is None else float(aspect_ratio)
    diag_factor = (effective_aspect * effective_aspect + 1.0) ** 0.5
    span_x = effective_aspect / diag_factor
    span_y = 1.0 / diag_factor
    x_coords = np.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        num=width,
        dtype=np.float32,
    )
    y_coords = np.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        num=height,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(x_coords, y_coords, indexing="xy")
    return mx.array(np.stack((grid_x, grid_y), axis=-1), dtype=dtype)


def hyworld2_position_grid_to_embed(
    pos_grid: mx.array | np.ndarray,
    embed_dim: int,
    *,
    omega_0: float = 100.0,
    dtype=mx.float32,
) -> mx.array:
    """Convert an ``(H,W,2)`` position grid to HY-World sinusoidal embeddings."""

    if embed_dim <= 0 or embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be positive and divisible by 4, got {embed_dim}")
    grid = np.asarray(pos_grid, dtype=np.float32)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"pos_grid must have shape (H, W, 2), got {grid.shape}")
    h, w, _ = grid.shape
    pos_flat = grid.reshape(-1, 2)
    omega = np.arange(embed_dim // 4, dtype=np.float32)
    omega = omega / (float(embed_dim) / 4.0)
    omega = 1.0 / (float(omega_0) ** omega)

    out_x = pos_flat[:, 0:1] * omega[None, :]
    out_y = pos_flat[:, 1:2] * omega[None, :]
    emb_x = np.concatenate((np.sin(out_x), np.cos(out_x)), axis=1)
    emb_y = np.concatenate((np.sin(out_y), np.cos(out_y)), axis=1)
    return mx.array(np.concatenate((emb_x, emb_y), axis=1).reshape(h, w, embed_dim), dtype=dtype)


def hyworld2_patch_rope_positions(
    *,
    frames: int,
    patch_grid: tuple[int, int],
    patch_start_idx: int = 0,
) -> mx.array:
    """Build frame-repeated HY-World patch RoPE positions."""

    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    grid_h, grid_w = int(patch_grid[0]), int(patch_grid[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"patch_grid dimensions must be positive, got {patch_grid}")
    if patch_start_idx < 0:
        raise ValueError(f"patch_start_idx must be non-negative, got {patch_start_idx}")
    rows = mx.broadcast_to(mx.arange(grid_h, dtype=mx.float32)[:, None], (grid_h, grid_w))
    cols = mx.broadcast_to(mx.arange(grid_w, dtype=mx.float32)[None, :], (grid_h, grid_w))
    if patch_start_idx > 0:
        rows = rows + 1
        cols = cols + 1
    patch_positions = mx.stack((mx.reshape(rows, (-1,)), mx.reshape(cols, (-1,))), axis=-1)
    special_positions = mx.zeros((patch_start_idx, 2), dtype=mx.float32)
    frame_positions = mx.concatenate((special_positions, patch_positions), axis=0)
    return mx.broadcast_to(frame_positions[None, :, :], (int(frames), int(frame_positions.shape[0]), 2))
