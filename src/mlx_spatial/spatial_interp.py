"""Sparse spatial interpolation helpers.

Nearest-neighbour and trilinear interpolation over sparse voxel grids.
Coordinates use `(z, y, x)` order, or `(batch, z, y, x)` when a batch
column is present.  Missing sparse voxels produce zero features; callers
can use `return_valid_mask` to distinguish missing from sampled zeros.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np


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
