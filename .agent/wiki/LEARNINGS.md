# Learnings

Durable project facts execution paid to learn. Format: `- Fact. Evidence: path or command (change-slug)`.

- Native Pixal3D→GLB pipeline (remesh → QEM simplify → native UV unwrap → texture bake w/ native Telea inpaint) is complete and merged to main. Evidence: `export_pixal3d_glb` in `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, git 743cb13 (2026-06-12-mlx-spatialkit-texture-postprocess).
- Open geometry defect — watertightness: ~10 residual boundary loops (open edges) from QEM repair input-prep; fix = non-manifold-tolerant QEM, or higher remesh resolution. Evidence: `.agent/work/2026-05-28-pixal3d-glb-quality-rebaseline/spec/legacy-2026-05-27-rollup.md`.
- Open geometry defect — thin-feature loss: thin geometry (e.g. the violin string) collapsed by QEM edge-collapse; fix = feature-aware simplification. Evidence: same rollup.
