"""Small MLX grid primitives used to verify the bootstrap environment."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul

import mlx.core as mx
import numpy as np


def regular_grid(shape: Sequence[int]) -> mx.array:
    """Return a dense integer grid with the requested shape."""
    dims = tuple(int(dim) for dim in shape)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("shape must contain positive dimensions")

    size = reduce(mul, dims, 1)
    return mx.arange(size).reshape(dims)


def create_uv_grid(
    width: int,
    height: int,
    *,
    normalize: bool = True,
    dtype=mx.float32,
) -> mx.array:
    """Create a (height, width, 2) UV coordinate grid.

    Args:
        width: Grid width in texels.
        height: Grid height in texels.
        normalize: If True, output coordinates are in [-1, 1] x [-1, 1].
            If False, coordinates are in [0, width-1] x [0, height-1].
        dtype: MLX output dtype (default float32).

    Returns:
        An MLX array with shape (height, width, 2) where the last axis
        holds (u, v) coordinates in xy order.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {(width, height)}")

    if normalize:
        u_coords = np.linspace(-1.0, 1.0, num=width, dtype=np.float32)
        v_coords = np.linspace(1.0, -1.0, num=height, dtype=np.float32)
    else:
        u_coords = np.arange(width, dtype=np.float32)
        v_coords = np.arange(height, dtype=np.float32)

    grid_u, grid_v = np.meshgrid(u_coords, v_coords, indexing="xy")
    return mx.array(np.stack((grid_u, grid_v), axis=-1), dtype=dtype)


def fourier_embeddings(
    positions: mx.array | np.ndarray,
    *,
    embed_dim: int,
    omega_0: float = 1.0,
    dtype=mx.float32,
) -> mx.array:
    """Compute sinusoidal Fourier feature embeddings for N-D positions.

    Follows the NeRF-style positional encoding: for each position
    coordinate, generates sin/cos pairs at logarithmically-spaced
    frequency bands.  The total output dimension is ``2 * ndim * bands``
    where ``bands = embed_dim // (2 * ndim)``.

    Args:
        positions: Array with shape ``(..., ndim)``.
        embed_dim: Total output embedding dimension (must be divisible
            by ``2 * ndim``).
        omega_0: Base frequency multiplier (default 1.0).
        dtype: MLX output dtype (default float32).

    Returns:
        An MLX array with shape ``(*positions.shape[:-1], embed_dim)``.
    """
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim == 0:
        raise ValueError("positions must have at least one dimension")
    ndim = int(pos.shape[-1])
    if embed_dim <= 0 or embed_dim % (2 * ndim) != 0:
        raise ValueError(
            f"embed_dim must be positive and divisible by {2 * ndim}, got {embed_dim}"
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must contain only finite values")

    bands = embed_dim // (2 * ndim)
    freqs = np.arange(bands, dtype=np.float32)
    freqs = freqs / max(float(bands) - 1.0, 1.0)
    freqs = np.pi * (float(omega_0) ** freqs)

    flat = pos.reshape(-1, ndim)
    encoded = []
    for idx in range(ndim):
        angle = flat[:, idx : idx + 1] * freqs[None, :]
        encoded.append(np.sin(angle))
        encoded.append(np.cos(angle))

    result = np.concatenate(encoded, axis=1)
    return mx.array(result.reshape(*pos.shape[:-1], embed_dim), dtype=dtype)


def fourier_embeddings_np(
    positions: np.ndarray,
    *,
    embed_dim: int,
    omega_0: float = 1.0,
) -> np.ndarray:
    """NumPy-only Fourier embedding (for use in pure-NumPy code paths).

    Args:
        positions: Array with shape ``(..., ndim)``.
        embed_dim: Total output embedding dimension.
        omega_0: Base frequency multiplier.

    Returns:
        A float32 NumPy array with shape ``(*positions.shape[:-1], embed_dim)``.
    """
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim == 0:
        raise ValueError("positions must have at least one dimension")
    ndim = int(pos.shape[-1])
    if embed_dim <= 0 or embed_dim % (2 * ndim) != 0:
        raise ValueError(
            f"embed_dim must be positive and divisible by {2 * ndim}, got {embed_dim}"
        )
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must contain only finite values")

    bands = embed_dim // (2 * ndim)
    freqs = np.arange(bands, dtype=np.float32)
    freqs = freqs / max(float(bands) - 1.0, 1.0)
    freqs = np.pi * (float(omega_0) ** freqs)

    flat = pos.reshape(-1, ndim)
    encoded = []
    for idx in range(ndim):
        angle = flat[:, idx : idx + 1] * freqs[None, :]
        encoded.append(np.sin(angle))
        encoded.append(np.cos(angle))

    return np.concatenate(encoded, axis=1).reshape(*pos.shape[:-1], embed_dim).astype(np.float32, copy=False)
