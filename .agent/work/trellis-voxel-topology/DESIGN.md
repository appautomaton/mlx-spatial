# DESIGN: Sparse Voxel Topology Helpers

## API Shape

Add `mlx_spatial.topology` as the public module for sparse voxel topology. Keep `mlx_spatial.ovoxel` for coordinate/index primitives.

## Data Contracts

- Sparse coordinates are MLX arrays with shape `(n, 3)` and integer dtype.
- Grid shape is a positive 3-tuple `(depth, height, width)` following existing row-major conventions.
- Neighbor offsets are a deterministic `int32` array with shape `(26, 3)`.
- Adjacency pairs are deterministic `int32` arrays with shape `(m, 2)` where each row is `(source_index, target_index)` into the input coordinate array.
- Edge/cell relationship helpers return plain MLX arrays, not custom containers.

## Ordering

- Neighbor offsets are lexicographic over `(dz, dy, dx)` from `-1` to `1`, excluding `(0, 0, 0)`.
- Adjacency rows are ordered by source coordinate input order, then neighbor offset order.
- Duplicate input coordinates are invalid to keep adjacency deterministic.

## Primitive Set

- `neighbor_offsets_26()` returns the 26 offsets.
- `adjacency_pairs_26(coordinates, shape)` returns active neighbor pairs for sparse coordinates inside the grid.
- `grid_edges(shape)` returns deterministic axis-aligned dense grid edge endpoint indices for topology tests.
- `grid_cells(shape)` returns deterministic 8-corner dense cell index relationships for topology tests.

## Out Of Scope

- Sparse convolution maps.
- Mesh extraction.
- Surface reconstruction.
- Model-specific containers.
- Checkpoint/model loading.
