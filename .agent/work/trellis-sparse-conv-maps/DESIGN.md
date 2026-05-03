# DESIGN: Sparse Convolution Map Primitives

## API Shape

Add `mlx_spatial.sparse_conv` for sparse convolution map construction. This module depends on the existing coordinate and topology conventions but does not perform convolution math.

## Data Contracts

- Coordinates are MLX arrays shaped `(n, 3)` in `(z, y, x)` order.
- Grid shape is a positive 3-tuple `(depth, height, width)`.
- Kernel size is a positive odd 3-tuple `(kz, ky, kx)`.
- Kernel offsets are an `int32` array shaped `(kz * ky * kx, 3)`.
- Map rows are an `int32` array shaped `(m, 3)` with columns `(target_index, source_index, kernel_index)`.

## Ordering

- Kernel offsets are lexicographic `(dz, dy, dx)` from negative radius to positive radius.
- The center offset is included because sparse convolution maps need a self slot.
- Map rows are ordered by target coordinate input order, then kernel offset order.
- Duplicate coordinates are invalid.

## Initial Scope

- Same-grid stride-1 maps: input and output active coordinates are the same coordinate set.
- Missing neighbors are omitted from the map.
- Returned values are plain MLX arrays.

## Out Of Scope

- Convolution compute, weights, features, batching, dilation, stride, transposed convolution, and neural modules.
