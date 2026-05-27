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

The real fixture test is opt-in with `pytest -m heavy` and writes generated
GLB/diagnostic artifacts under `/tmp`.
