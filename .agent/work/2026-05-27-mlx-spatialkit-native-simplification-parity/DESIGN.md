# mlx-spatialkit Native Simplification Parity Design

## Current Failure

`simplify_mesh` is currently deterministic face sampling:

```text
8M source faces
  -> take every Nth face
  -> compact referenced vertices
  -> valid-ish GLB, fragmented geometry
```

This is fast, but it is not simplification. It discards most topology and cannot be a dependable export backend.

## First Replacement

Use native spatial vertex clustering as the first geometry-aware backend:

```text
source mesh
  -> compute bounds
  -> choose grid resolution from target face budget
  -> cluster vertices by spatial cell
  -> average vertex positions per cell
  -> remap all source faces through clusters
  -> drop degenerates / duplicate faces
  -> compact mesh
  -> metrics + diagnostics
```

Why this first:

- O(vertices + faces), suitable for multi-million-face fixtures.
- Native-only hot path.
- Uses all source geometry, unlike face stride.
- Produces an exportable mesh and useful diagnostics quickly.

Why it is not final production parity:

- It does not optimize quadric error.
- It does not preserve boundaries like a production remesher.
- It can still need follow-up cleanup/hole-fill/unwrap work.

## Diagnostics

The simplifier should report:

- `backend`: `spatial-cluster`
- `algorithm`: `native_spatial_vertex_clustering`
- `quality_tier`: `geometry_aware_preview`
- `production_ready`: `false`
- source/target/final face counts
- source/final vertex counts
- cluster count and grid resolution
- degenerate and duplicate faces removed
- target reached flag

Production readiness remains false until a later phase proves parity thresholds.

## Reference Metrics

The checked-in Pixal3D reference trace is a comparison target, not a byte-parity requirement:

- final faces: `212542`
- raw coverage: about `0.413`
- final coverage: `1.0`
- unwrap backend: `xatlas-parallel-spatial`
- bake backend: `xatlas-kdtree`

Spatialkit should record where it stands against those values so the next gap is explicit.
