# mlx-spatialkit Native Atlas Coverage Parity Design

## Current Problem

The current native face atlas uses one triangular face per rectangular tile:

```text
tile
  lower-left triangle = used
  upper-right triangle = empty
```

With reference-target exports this creates a hard utilization ceiling before the
Metal bake even starts. The last verified heavy run had good face-count parity
and clean topology, but final global coverage stayed around `0.269`.

## Native Paired Atlas

Pack two unrelated triangle faces into one tile:

```text
tile N
  face 2N     -> lower-left half
  face 2N + 1 -> upper-right half
```

The output GLB still has the same face count and duplicated per-face vertices,
but the atlas uses about half as many tiles. This raises available texture
surface area without adding xatlas or Python hot loops.

## Diagnostics

The UV stage should report:

- `backend`: `face-atlas`
- `packing`: `paired-triangles`
- `faces_per_tile`: `2`
- `atlas_tiles`
- `atlas_cols`
- `atlas_rows`
- `estimated_tile_utilization`
- source/output vertex and face counts

The texture stage already records global coverage, UV-surface coverage, timing,
and memory samples; those become the real fixture proof.

## Readiness Boundary

This atlas is still a native preview/quality-improvement path, not xatlas parity.
Production readiness remains controlled by the existing threshold report. If
coverage improves but the backend tier remains preview, the diagnostics should
show exactly that.
