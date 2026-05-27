# mlx-spatialkit

Native C++ and Metal spatial export primitives for `mlx-spatial`.

This package is intentionally independent from MLX. It accepts ordinary Python
buffers, NumPy arrays, and files at the binding boundary, while native code owns
the expensive mesh, texture, and export stages.

Pixal3D decoded export entry point:

```python
from mlx_spatialkit import export_pixal3d_glb

result = export_pixal3d_glb(
    "inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr",
    "/tmp/mlx-spatialkit-pixal3d",
)
print(result.glb.path)
print(result.diagnostics_path)
```

To run the reference-target quality gate, use:

```python
result = export_pixal3d_glb(
    "inputs/mlx-spatialkit/pixal3d-1024-cascade-decoded-pbr",
    "/tmp/mlx-spatialkit-pixal3d-reference",
    quality_preset="reference-target",
)
print(result.diagnostics["quality"]["production_thresholds"])
```

The real fixture test is opt-in with `pytest -m heavy` and writes generated
GLB/diagnostic artifacts under `/tmp`.

## Quality Tier

`mlx-spatialkit` currently produces a preview-quality Pixal3D GLB. The native
texture path uses Metal exact sparse-voxel sampling, bounded sparse-neighbor
fallback, and native UV-surface dilation. Diagnostics record raw exact coverage,
fallback-filled texels, final visible base-color coverage, runtime, and RSS
samples.

Mesh simplification is currently labeled `spatial-cluster` with
`quality_tier=geometry_aware_preview`. The native C++ path clusters vertices
spatially, remaps source faces, removes degenerate/duplicate output faces, and
rejects faces that would create nonmanifold edges. A written GLB can be
`artifact_ready=true` while `production_quality_ready=false`; production
remeshing parity is still a later phase.

The UV stage uses a native paired-triangle face atlas: two unrelated triangle
faces share one atlas tile in complementary halves. This is still not xatlas
charting, but it reduces atlas waste and lets the Metal texture bake report
`atlas_faces_per_tile=2`.

`quality_preset="reference-target"` resolves the face target from the checked-in
Pixal3D reference trace when available and records threshold checks for topology,
face-count ratio, raw/final texture coverage, and backend tier. The current
reference-target heavy fixture passes face-count, topology, and final global
coverage thresholds, with final visible coverage around `0.602` versus the older
one-triangle atlas baseline of about `0.269`. Production readiness still fails
because the active simplifier is `geometry_aware_preview`, so
`production_quality_ready=false` remains the correct result.
