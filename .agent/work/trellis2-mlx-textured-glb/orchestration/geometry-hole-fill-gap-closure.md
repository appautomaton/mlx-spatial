# Geometry Hole-Fill Gap Closure

## Gap Identified

The live 512 OBJ was Blender-readable but still had visible topology holes. Measurement on the regenerated shape OBJ before this fix:

- `boundary_edges=35858`
- `nonmanifold_edges=82909`

Upstream TRELLIS.2 calls `m.fill_holes()` after shape decode before constructing `MeshWithVoxel`. The MLX path was exporting the raw FlexiDualGrid mesh without that postprocess step.

## Changes

- Added a CPU/NumPy bounded hole-fill helper for clean manifold boundary loops.
- The helper uses the upstream default intent: `max_hole_perimeter=3e-2`.
- The helper skips large, open, or complex boundary components instead of inventing geometry.
- Wired the helper before shape OBJ export.
- Wired the helper before texture baking and GLB export.
- Added unit coverage for filling a small clean loop and leaving a large loop open.

## Verification

- Focused tests: `uv run pytest -q tests/test_ovoxel.py tests/test_trellis2_inference.py tests/test_trellis2_export.py tests/test_trellis2_tools.py` -> `77 passed`.
- Full suite: `uv run pytest -q` -> `237 passed, 5 skipped`.
- `git diff --check` -> passed.
- Shape command:

```sh
uv run mlx-spatial-trellis2 generate-shape weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-512-rmbg-shape.obj --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --decoder-token-limit 1200000
```

- Shape artifact: `outputs/trellis2/image-512-rmbg-shape.obj`
- Shape bytes: `81075396`
- Shape trace output includes `shape_mesh_hole_fill`.
- Blender shape stats after fix: `verts=1052725`, `faces=2137852`, `boundary_edges=18068`, `nonmanifold_edges=65119`.

Textured command:

```sh
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-512-textured.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --slat-steps 1 --decoder-token-limit 1000000 --texture-size 1024
```

- Textured artifact: `outputs/trellis2/image-512-textured.glb`
- Textured bytes: `173065480`
- Textured trace output includes `texture_mesh_hole_fill`.
- Blender GLB import: `GLB_OK 1 1 2`.

## Remaining Gap

This closes the upstream `fill_holes()` gap for clean small loops. It does not yet implement a full `cumesh` equivalent for complex/open boundary components, non-manifold repair, connected-component cleanup, simplification, or xatlas-equivalent UV packing.
