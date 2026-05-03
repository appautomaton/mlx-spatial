"""Small MLX grid primitives used to verify the bootstrap environment."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul

import mlx.core as mx


def regular_grid(shape: Sequence[int]) -> mx.array:
    """Return a dense integer grid with the requested shape."""
    dims = tuple(int(dim) for dim in shape)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("shape must contain positive dimensions")

    size = reduce(mul, dims, 1)
    return mx.arange(size).reshape(dims)
