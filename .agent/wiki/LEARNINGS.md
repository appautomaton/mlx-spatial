# Learnings

Durable project facts execution paid to learn. Format: `- Fact. Evidence: path or command (change-slug)`.

- Native Pixal3D→GLB pipeline (remesh → QEM simplify → native UV unwrap → texture bake w/ native Telea inpaint) is complete and merged to main. Evidence: `export_pixal3d_glb` in `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, git 743cb13 (2026-06-12-mlx-spatialkit-texture-postprocess).
- Watertightness on the reference-target QEM path: both fixtures (main + violin) are geometrically watertight — **0 genuine open boundaries** after a position-weld; all residual boundary edges are coincident duplicate-position seams (UV-atlas + nonmanifold-repair vertex-splitting at `cpp/remesh.cpp:45-50`), and both render solid (no visible holes). `boundary_loop_count` counts coincident seams, like the reference (25k boundary edges). The earlier "visible holes" were a bad-param checkpoint (remesh=False/4096), not this path. Evidence: `/tmp/watertight_robust.py` position-weld + `render_glb_visual_parity.cjs` (change 2026-06-14-mlx-spatialkit-watertightness, Slice 1).
- Open geometry defect — thin-feature loss: thin geometry (e.g. the violin string) collapsed by QEM edge-collapse; fix = feature-aware simplification. Evidence: same rollup.
