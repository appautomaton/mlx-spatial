"""Sparse convolution map helpers for MLX spatial model code.

These helpers construct sparse convolution maps and move features across those
maps. They include correctness-first reference compute, not optimized neural
layers.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from .topology import _coords_list, _shape3


_INTEGER_DTYPES = (
    mx.int8,
    mx.int16,
    mx.int32,
    mx.int64,
    mx.uint8,
    mx.uint16,
    mx.uint32,
    mx.uint64,
)


def _kernel_size3(kernel_size: Sequence[int]) -> tuple[int, int, int]:
    dims = tuple(int(dim) for dim in kernel_size)
    if len(dims) != 3 or any(dim <= 0 or dim % 2 == 0 for dim in dims):
        raise ValueError("kernel_size must contain three positive odd dimensions")
    return dims


def _map_rows_list(map_rows: mx.array) -> list[tuple[int, int, int]]:
    if len(map_rows.shape) != 2 or map_rows.shape[1] != 3:
        raise ValueError("map_rows must have shape (m, 3)")
    if map_rows.dtype not in _INTEGER_DTYPES:
        raise ValueError("map_rows must use an integer dtype")
    return [tuple(int(value) for value in row) for row in map_rows.tolist()]


def _feature_shape(features: mx.array, name: str) -> tuple[int, int]:
    if len(features.shape) != 2:
        raise ValueError(f"{name} must have shape (n, channels)")
    return int(features.shape[0]), int(features.shape[1])


def _target_count(target_count: int) -> int:
    count = int(target_count)
    if count < 0:
        raise ValueError("target_count must be non-negative")
    return count


def _kernel_weight_shape(kernel_weights: mx.array) -> tuple[int, int, int]:
    if len(kernel_weights.shape) != 3:
        raise ValueError("kernel_weights must have shape (kernel_count, in_channels, out_channels)")
    return int(kernel_weights.shape[0]), int(kernel_weights.shape[1]), int(kernel_weights.shape[2])


def kernel_offsets(kernel_size: Sequence[int]) -> mx.array:
    """Return sparse convolution kernel offsets in lexicographic order.

    Args:
        kernel_size: Positive odd `(kz, ky, kx)` kernel size.

    Returns:
        An `int32` MLX array with shape `(kz * ky * kx, 3)`. Offsets are
        lexicographic `(dz, dy, dx)` from negative radius to positive radius,
        including center `[0, 0, 0]`.
    """
    kz, ky, kx = _kernel_size3(kernel_size)
    rz, ry, rx = kz // 2, ky // 2, kx // 2
    offsets = [
        (dz, dy, dx)
        for dz in range(-rz, rz + 1)
        for dy in range(-ry, ry + 1)
        for dx in range(-rx, rx + 1)
    ]
    return mx.array(offsets, dtype=mx.int32)


def sparse_conv_map(
    coordinates: mx.array,
    shape: Sequence[int],
    kernel_size: Sequence[int] = (3, 3, 3),
) -> mx.array:
    """Return same-grid stride-1 sparse convolution map rows.

    Args:
        coordinates: Integer MLX array with shape `(n, 3)` in `(z, y, x)` order.
        shape: Positive `(depth, height, width)` grid shape.
        kernel_size: Positive odd `(kz, ky, kx)` kernel size.

    Returns:
        An `int32` MLX array with shape `(m, 3)`. Each row is
        `(target_index, source_index, kernel_index)`. Rows are ordered by target
        input order, then kernel offset order. The source coordinate is computed
        as `target_coordinate + kernel_offsets(kernel_size)[kernel_index]`.
        Missing source neighbors are omitted.
    """
    _shape3(shape)
    coords = _coords_list(coordinates, shape)
    lookup = {coord: index for index, coord in enumerate(coords)}
    offsets = [tuple(int(value) for value in row) for row in kernel_offsets(kernel_size).tolist()]

    rows: list[tuple[int, int, int]] = []
    for target_index, target in enumerate(coords):
        for kernel_index, offset in enumerate(offsets):
            source = tuple(target[axis] + offset[axis] for axis in range(3))
            source_index = lookup.get(source)
            if source_index is not None:
                rows.append((target_index, source_index, kernel_index))

    return mx.array(rows, dtype=mx.int32).reshape((-1, 3))


def sparse_conv_map_vectorized(
    coordinates: mx.array,
    shape: Sequence[int],
    kernel_size: Sequence[int] = (3, 3, 3),
    *,
    row_chunk_size: int = 65536,
) -> mx.array:
    """Return same-grid sparse convolution rows using vectorized NumPy lookup.

    The output contract and row order match :func:`sparse_conv_map`: target
    input order first, kernel offset order second. The implementation chunks
    target coordinates so large decoder grids do not need an `(N, K, 3)` array
    for every active voxel at once.
    """

    dims = _shape3(shape)
    if len(coordinates.shape) != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    if coordinates.dtype not in _INTEGER_DTYPES:
        raise ValueError("coordinates must use an integer dtype")
    coords = np.array(coordinates, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coordinates must have shape (n, 3)")
    upper = np.array(dims, dtype=np.int64)
    if np.any((coords < 0) | (coords >= upper)):
        raise ValueError("coordinates must be inside shape bounds")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("coordinates must not contain duplicates")
    offsets = np.array(kernel_offsets(kernel_size).tolist(), dtype=np.int64)
    chunk_size = int(row_chunk_size)
    if chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if coords.size == 0:
        return mx.array([], dtype=mx.int32).reshape((0, 3))

    strides = np.array((dims[1] * dims[2], dims[2], 1), dtype=np.int64)
    flat_coords = coords @ strides
    order = np.argsort(flat_coords)
    sorted_flat = flat_coords[order]

    rows: list[np.ndarray] = []
    kernel_indices = np.arange(offsets.shape[0], dtype=np.int64)
    for start in range(0, coords.shape[0], chunk_size):
        end = min(start + chunk_size, coords.shape[0])
        targets = coords[start:end]
        candidates = targets[:, None, :] + offsets[None, :, :]
        valid = np.all((candidates >= 0) & (candidates < upper), axis=2)
        candidate_flat = candidates @ strides
        positions = np.searchsorted(sorted_flat, candidate_flat)
        found = positions < sorted_flat.shape[0]
        safe_positions = np.where(found, positions, 0)
        found &= sorted_flat[safe_positions] == candidate_flat
        valid &= found
        if not np.any(valid):
            continue

        target_indices = np.broadcast_to(np.arange(start, end, dtype=np.int64)[:, None], valid.shape)[valid]
        source_indices = order[safe_positions[valid]]
        selected_kernel_indices = np.broadcast_to(kernel_indices[None, :], valid.shape)[valid]
        rows.append(np.stack((target_indices, source_indices, selected_kernel_indices), axis=1))

    if not rows:
        return mx.array([], dtype=mx.int32).reshape((0, 3))
    return mx.array(np.concatenate(rows, axis=0).astype(np.int32, copy=False), dtype=mx.int32).reshape((-1, 3))


def gather_sparse_features(source_features: mx.array, map_rows: mx.array) -> mx.array:
    """Gather source feature rows in sparse map-row order.

    Args:
        source_features: Feature matrix with shape `(source_count, channels)`.
        map_rows: Integer rows shaped `(m, 3)` using the sparse convolution map
            contract `(target_index, source_index, kernel_index)`.

    Returns:
        Feature matrix with shape `(m, channels)`. Row `i` is
        `source_features[map_rows[i, 1]]`.
    """
    source_count, channels = _feature_shape(source_features, "source_features")
    rows = _map_rows_list(map_rows)

    gathered: list[list[float]] = []
    for _, source_index, _ in rows:
        if source_index < 0 or source_index >= source_count:
            raise ValueError("map_rows source_index is out of bounds")
        gathered.append(source_features[source_index].tolist())

    return mx.array(gathered, dtype=source_features.dtype).reshape((len(rows), channels))


def scatter_sparse_features(
    row_features: mx.array,
    map_rows: mx.array,
    target_count: int,
) -> mx.array:
    """Accumulate map-row features into target feature slots.

    Args:
        row_features: Feature matrix with shape `(m, channels)`, where each row
            corresponds to the same row in `map_rows`.
        map_rows: Integer rows shaped `(m, 3)` using the sparse convolution map
            contract `(target_index, source_index, kernel_index)`.
        target_count: Number of target feature rows to produce.

    Returns:
        Feature matrix with shape `(target_count, channels)`. All row features
        with the same `target_index` are summed, and targets with no rows remain
        zero.
    """
    row_count, channels = _feature_shape(row_features, "row_features")
    rows = _map_rows_list(map_rows)
    count = _target_count(target_count)
    if row_count != len(rows):
        raise ValueError("row_features must have the same row count as map_rows")

    accumulated = [[0 for _ in range(channels)] for _ in range(count)]
    feature_rows = row_features.tolist()
    for row_index, (target_index, _, _) in enumerate(rows):
        if target_index < 0 or target_index >= count:
            raise ValueError("map_rows target_index is out of bounds")
        for channel in range(channels):
            accumulated[target_index][channel] += feature_rows[row_index][channel]

    return mx.array(accumulated, dtype=row_features.dtype).reshape((count, channels))


def weighted_sparse_conv(
    source_features: mx.array,
    map_rows: mx.array,
    kernel_weights: mx.array,
    target_count: int,
) -> mx.array:
    """Apply a reference weighted sparse convolution over map rows.

    Args:
        source_features: Feature matrix with shape `(source_count, in_channels)`.
        map_rows: Integer rows shaped `(m, 3)` using the sparse convolution map
            contract `(target_index, source_index, kernel_index)`.
        kernel_weights: Kernel weights with shape
            `(kernel_count, in_channels, out_channels)`.
        target_count: Number of target feature rows to produce.

    Returns:
        Feature matrix with shape `(target_count, out_channels)`. For each map
        row, this adds `source_features[source_index] @
        kernel_weights[kernel_index]` into `target_index`.
    """
    source_count, in_channels = _feature_shape(source_features, "source_features")
    kernel_count, weight_in_channels, out_channels = _kernel_weight_shape(kernel_weights)
    if in_channels != weight_in_channels:
        raise ValueError("source_features channels must match kernel_weights in_channels")

    rows = _map_rows_list(map_rows)
    count = _target_count(target_count)
    if not rows:
        return mx.zeros((count, out_channels), dtype=kernel_weights.dtype)

    row_array = np.array(rows, dtype=np.int64)
    for target_index, source_index, kernel_index in rows:
        if target_index < 0 or target_index >= count:
            raise ValueError("map_rows target_index is out of bounds")
        if source_index < 0 or source_index >= source_count:
            raise ValueError("map_rows source_index is out of bounds")
        if kernel_index < 0 or kernel_index >= kernel_count:
            raise ValueError("map_rows kernel_index is out of bounds")

    accumulated = mx.zeros((count, out_channels), dtype=mx.float32)
    source_float = source_features.astype(mx.float32)
    weights_float = kernel_weights.astype(mx.float32)
    for kernel_index in range(kernel_count):
        mask = row_array[:, 2] == kernel_index
        if not np.any(mask):
            continue
        selected_rows = row_array[mask]
        target_indices = mx.array(selected_rows[:, 0], dtype=mx.int32)
        source_indices = mx.array(selected_rows[:, 1], dtype=mx.int32)
        values = source_float[source_indices] @ weights_float[kernel_index]
        accumulated = accumulated.at[target_indices].add(values)

    return accumulated.astype(kernel_weights.dtype).reshape((count, out_channels))


def weighted_sparse_conv_chunked(
    source_features: mx.array,
    map_rows: mx.array,
    kernel_weights: mx.array,
    target_count: int,
    *,
    row_chunk_size: int = 65536,
) -> mx.array:
    """Apply weighted sparse convolution with bounded gather/matmul chunks."""

    source_count, in_channels = _feature_shape(source_features, "source_features")
    kernel_count, weight_in_channels, out_channels = _kernel_weight_shape(kernel_weights)
    if in_channels != weight_in_channels:
        raise ValueError("source_features channels must match kernel_weights in_channels")

    count = _target_count(target_count)
    chunk_size = int(row_chunk_size)
    if chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    if len(map_rows.shape) != 2 or map_rows.shape[1] != 3:
        raise ValueError("map_rows must have shape (m, 3)")
    if map_rows.dtype not in _INTEGER_DTYPES:
        raise ValueError("map_rows must use an integer dtype")
    if int(map_rows.shape[0]) == 0:
        return mx.zeros((count, out_channels), dtype=kernel_weights.dtype)

    row_array = np.array(map_rows, dtype=np.int64)
    if np.any((row_array[:, 0] < 0) | (row_array[:, 0] >= count)):
        raise ValueError("map_rows target_index is out of bounds")
    if np.any((row_array[:, 1] < 0) | (row_array[:, 1] >= source_count)):
        raise ValueError("map_rows source_index is out of bounds")
    if np.any((row_array[:, 2] < 0) | (row_array[:, 2] >= kernel_count)):
        raise ValueError("map_rows kernel_index is out of bounds")

    accumulated = mx.zeros((count, out_channels), dtype=mx.float32)
    source_float = source_features.astype(mx.float32)
    weights_float = kernel_weights.astype(mx.float32)
    for kernel_index in range(kernel_count):
        selected = row_array[row_array[:, 2] == kernel_index]
        for start in range(0, selected.shape[0], chunk_size):
            chunk = selected[start : start + chunk_size]
            if chunk.shape[0] == 0:
                continue
            target_indices = mx.array(chunk[:, 0], dtype=mx.int32)
            source_indices = mx.array(chunk[:, 1], dtype=mx.int32)
            values = source_float[source_indices] @ weights_float[kernel_index]
            accumulated = accumulated.at[target_indices].add(values)
            mx.eval(accumulated)

    return accumulated.astype(kernel_weights.dtype).reshape((count, out_channels))


def sparse_nearest_interpolate(
    query_coordinates: mx.array,
    source_coordinates: mx.array,
    source_features: mx.array,
    *,
    shape: Sequence[int] | None = None,
    return_valid_mask: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Sample sparse features at nearest integer grid coordinates.

    Coordinates use `(z, y, x)` order, or `(batch, z, y, x)` when a batch column
    is present. Missing sparse voxels produce zero features and an invalid mask
    entry when `return_valid_mask=True`.
    """

    source_coords, query_coords, features, spatial_rank, has_batch = _sparse_interpolation_inputs(
        query_coordinates,
        source_coordinates,
        source_features,
        shape=shape,
    )
    rounded = np.floor(query_coords[:, -spatial_rank:] + 0.5).astype(np.int64)
    prefix = query_coords[:, :1].astype(np.int64) if has_batch else None
    sampled, valid = _lookup_sparse_features(
        rounded,
        source_coords,
        features,
        spatial_rank=spatial_rank,
        shape=shape,
        batch_prefix=prefix,
    )
    values = mx.array(sampled, dtype=source_features.dtype)
    if not return_valid_mask:
        return values
    return values, mx.array(valid, dtype=mx.bool_)


def sparse_trilinear_interpolate(
    query_coordinates: mx.array,
    source_coordinates: mx.array,
    source_features: mx.array,
    *,
    shape: Sequence[int] | None = None,
    require_all_corners: bool = False,
    return_valid_mask: bool = False,
) -> mx.array | tuple[mx.array, mx.array]:
    """Sample sparse features with dense-grid trilinear interpolation semantics.

    Coordinates use `(z, y, x)` order, or `(batch, z, y, x)` when a batch column
    is present. Missing corners contribute zero; `require_all_corners=True`
    raises if any non-zero-weight corner is absent or out of bounds.
    """

    source_coords, query_coords, features, spatial_rank, has_batch = _sparse_interpolation_inputs(
        query_coordinates,
        source_coordinates,
        source_features,
        shape=shape,
    )
    if spatial_rank != 3:
        raise ValueError("sparse_trilinear_interpolate requires three spatial coordinate dimensions")

    spatial_queries = query_coords[:, -spatial_rank:]
    base = np.floor(spatial_queries).astype(np.int64)
    frac = spatial_queries - base.astype(np.float32)
    prefix = query_coords[:, :1].astype(np.int64) if has_batch else None
    sampled = np.zeros((query_coords.shape[0], features.shape[1]), dtype=np.float32)
    present_weight = np.zeros((query_coords.shape[0],), dtype=np.float32)
    missing = np.zeros((query_coords.shape[0],), dtype=bool)
    eps = 1e-6

    for dz in (0, 1):
        wz = frac[:, 0] if dz else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dx in (0, 1):
                wx = frac[:, 2] if dx else 1.0 - frac[:, 2]
                weights = (wz * wy * wx).astype(np.float32, copy=False)
                active = weights > eps
                if not np.any(active):
                    continue
                corners = base + np.array([dz, dy, dx], dtype=np.int64).reshape(1, 3)
                corner_values, valid = _lookup_sparse_features(
                    corners,
                    source_coords,
                    features,
                    spatial_rank=spatial_rank,
                    shape=shape,
                    batch_prefix=prefix,
                )
                missing |= active & ~valid
                usable = active & valid
                if np.any(usable):
                    sampled[usable] += corner_values[usable] * weights[usable, None]
                    present_weight[usable] += weights[usable]

    valid_mask = present_weight > eps
    if require_all_corners:
        invalid = missing | ~valid_mask
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                "sparse trilinear interpolation requires all non-zero interpolation corners; "
                f"query {first} is missing required sparse voxel data"
            )

    values = mx.array(sampled, dtype=source_features.dtype)
    if not return_valid_mask:
        return values
    return values, mx.array(valid_mask, dtype=mx.bool_)


def _sparse_interpolation_inputs(
    query_coordinates: mx.array,
    source_coordinates: mx.array,
    source_features: mx.array,
    *,
    shape: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    source_coords = np.asarray(source_coordinates, dtype=np.int64)
    query_coords = np.asarray(query_coordinates, dtype=np.float32)
    features = np.asarray(source_features, dtype=np.float32)
    if source_coords.ndim != 2 or source_coords.shape[1] not in (3, 4):
        raise ValueError("source_coordinates must have shape (n, 3) or (n, 4)")
    if query_coords.ndim != 2 or query_coords.shape[1] != source_coords.shape[1]:
        raise ValueError("query_coordinates must have the same coordinate rank as source_coordinates")
    if features.ndim != 2 or features.shape[0] != source_coords.shape[0]:
        raise ValueError("source_features must have shape (source_count, channels)")
    if features.shape[1] == 0:
        raise ValueError("source_features must contain at least one channel")
    if not np.all(np.isfinite(query_coords)) or not np.all(np.isfinite(features)):
        raise ValueError("query_coordinates and source_features must contain only finite values")
    if np.unique(source_coords, axis=0).shape[0] != source_coords.shape[0]:
        raise ValueError("source_coordinates must not contain duplicates")
    has_batch = source_coords.shape[1] == 4
    spatial_rank = source_coords.shape[1] - int(has_batch)
    if shape is not None:
        dims = tuple(int(dim) for dim in shape)
        if len(dims) != spatial_rank or any(dim <= 0 for dim in dims):
            raise ValueError("shape must contain one positive entry per spatial coordinate")
        spatial = source_coords[:, -spatial_rank:]
        upper = np.array(dims, dtype=np.int64)
        if np.any((spatial < 0) | (spatial >= upper)):
            raise ValueError("source_coordinates must be inside shape bounds")
    return source_coords, query_coords, features, spatial_rank, has_batch


def _lookup_sparse_features(
    spatial_queries: np.ndarray,
    source_coords: np.ndarray,
    source_features: np.ndarray,
    *,
    spatial_rank: int,
    shape: Sequence[int] | None,
    batch_prefix: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_prefix is not None:
        keys = np.concatenate((batch_prefix, spatial_queries), axis=1)
    else:
        keys = spatial_queries
    valid = np.ones((keys.shape[0],), dtype=bool)
    if shape is not None:
        upper = np.array(tuple(int(dim) for dim in shape), dtype=np.int64)
        spatial = keys[:, -spatial_rank:]
        valid &= np.all((spatial >= 0) & (spatial < upper), axis=1)

    lookup = {tuple(int(value) for value in coord): index for index, coord in enumerate(source_coords)}
    sampled = np.zeros((keys.shape[0], source_features.shape[1]), dtype=np.float32)
    for row, key in enumerate(keys):
        if not valid[row]:
            continue
        source_index = lookup.get(tuple(int(value) for value in key))
        if source_index is None:
            valid[row] = False
            continue
        sampled[row] = source_features[source_index]
    return sampled, valid
