"""Sparse voxel topology helpers for MLX spatial model code.

Coordinates use `(z, y, x)` order and row-major dense indexing for a grid
shape `(depth, height, width)`. All public helpers return plain MLX arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

from .ovoxel import flatten_coordinates


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    dims = tuple(int(dim) for dim in shape)
    if len(dims) != 3 or any(dim <= 0 for dim in dims):
        raise ValueError("shape must contain three positive dimensions")
    return dims


def _coords_list(coordinates: mx.array, shape: Sequence[int]) -> list[tuple[int, int, int]]:
    dims = _shape3(shape)
    if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")

    coords = [tuple(int(value) for value in row) for row in coordinates.tolist()]
    seen: set[tuple[int, int, int]] = set()
    for coord in coords:
        if coord in seen:
            raise ValueError("coordinates must not contain duplicates")
        seen.add(coord)
        if any(value < 0 or value >= dims[axis] for axis, value in enumerate(coord)):
            raise ValueError("coordinates must be inside shape bounds")
    return coords


def neighbor_offsets_26() -> mx.array:
    """Return 26 neighbor offsets in lexicographic `(dz, dy, dx)` order.

    Returns:
        An `int32` MLX array with shape `(26, 3)`, excluding `[0, 0, 0]`.
    """
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ]
    return mx.array(offsets, dtype=mx.int32)


def adjacency_pairs_26(coordinates: mx.array, shape: Sequence[int]) -> mx.array:
    """Return active 26-neighbor adjacency pairs for sparse coordinates.

    Args:
        coordinates: Integer MLX array with shape `(n, 3)` in `(z, y, x)` order.
        shape: Positive `(depth, height, width)` grid shape.

    Returns:
        An `int32` MLX array with shape `(m, 2)`. Each row is
        `(source_index, target_index)` into the input coordinate array. Rows are
        ordered by source input order, then lexicographic neighbor offset order.
    """
    coords = _coords_list(coordinates, shape)
    lookup = {coord: index for index, coord in enumerate(coords)}
    offsets = [tuple(int(value) for value in row) for row in neighbor_offsets_26().tolist()]

    pairs: list[tuple[int, int]] = []
    for source_index, coord in enumerate(coords):
        for offset in offsets:
            neighbor = tuple(coord[axis] + offset[axis] for axis in range(3))
            target_index = lookup.get(neighbor)
            if target_index is not None:
                pairs.append((source_index, target_index))

    return mx.array(pairs, dtype=mx.int32).reshape((-1, 2))


def grid_edges(shape: Sequence[int]) -> mx.array:
    """Return dense grid axis-aligned edge endpoint indices.

    Edges use row-major dense indices and are ordered by axis `z`, then `y`,
    then `x`. Within each axis group, start coordinates are row-major.

    Returns:
        An `int32` MLX array with shape `(num_edges, 2)`. This is a topology
        relationship helper, not mesh extraction.
    """
    depth, height, width = _shape3(shape)
    edges: list[tuple[int, int]] = []

    for z in range(depth - 1):
        for y in range(height):
            for x in range(width):
                edges.append(_dense_pair((z, y, x), (z + 1, y, x), (depth, height, width)))
    for z in range(depth):
        for y in range(height - 1):
            for x in range(width):
                edges.append(_dense_pair((z, y, x), (z, y + 1, x), (depth, height, width)))
    for z in range(depth):
        for y in range(height):
            for x in range(width - 1):
                edges.append(_dense_pair((z, y, x), (z, y, x + 1), (depth, height, width)))

    return mx.array(edges, dtype=mx.int32).reshape((-1, 2))


def grid_cells(shape: Sequence[int]) -> mx.array:
    """Return dense grid cell corner indices.

    Cells use row-major dense indices. Cell rows are ordered by row-major start
    coordinate. The 8 corners within each row are lexicographic local offsets
    `(dz, dy, dx)` from `(0, 0, 0)` to `(1, 1, 1)`.

    Returns:
        An `int32` MLX array with shape `(num_cells, 8)`. This is a topology
        relationship helper, not mesh extraction.
    """
    depth, height, width = _shape3(shape)
    cells: list[list[int]] = []

    for z in range(depth - 1):
        for y in range(height - 1):
            for x in range(width - 1):
                corners = [
                    _dense_index((z + dz, y + dy, x + dx), (depth, height, width))
                    for dz in (0, 1)
                    for dy in (0, 1)
                    for dx in (0, 1)
                ]
                cells.append(corners)

    return mx.array(cells, dtype=mx.int32).reshape((-1, 8))


def _dense_index(coord: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
    return int(flatten_coordinates(mx.array(coord, dtype=mx.int32), shape).item())


def _dense_pair(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> tuple[int, int]:
    return (_dense_index(start, shape), _dense_index(end, shape))
