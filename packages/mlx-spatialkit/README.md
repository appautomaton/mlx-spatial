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

`quality_preset="reference-target"` resolves the face target from the checked-in
Pixal3D reference trace when available and records threshold checks for topology,
face-count ratio, raw/final texture coverage, and backend tier. The current
reference-target run passes face-count and topology thresholds, but fails
production readiness because the active simplifier is still preview-tier and
global final texture coverage is below the production threshold.
