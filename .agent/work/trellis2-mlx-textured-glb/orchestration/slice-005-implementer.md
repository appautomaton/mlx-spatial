# Slice 5 Implementer: Mesh/Voxel Coupling And Baking Fixtures

## Scope

Implemented the deterministic Mac-native mesh/voxel baking surface for the textured TRELLIS.2 path. This slice intentionally stops before GLB writing.

## Changes

- Added `Trellis2TextureBakeResult` and PBR layout metadata in `src/mlx_spatial/trellis2_export.py`.
- Added deterministic per-face UV atlas generation with duplicated triangle vertices.
- Added pure NumPy/Pillow texture-field baking from decoder-style `(batch,z,y,x)` coordinates plus 6-channel attributes:
  - channels `0:3`: base color
  - channel `3`: metallic
  - channel `4`: roughness
  - channel `5`: alpha
- Added PNG payload encoding helper for baked image fixtures.
- Added early public API guards for texture pixels, texel/voxel pair count, coordinate shape/range, duplicate coordinates, and mesh validity.
- Made nearest voxel sampling deterministic under sparse-coordinate permutation by lexicographically sorting coordinates and using stable tie-breaks.
- Wired `generate_textured_glb` through FlexiDualGrid mesh extraction and texture baking after texture decoder success.
- Updated the textured path blocker from Slice 5 to Slice 6: `textured GLB writer`.
- Exported new helpers through `mlx_spatial.__init__`.

## Verification

- `uv run pytest -q tests/test_trellis2_export.py tests/test_trellis2_inference.py` -> `44 passed in 0.31s`
- `git diff --check` -> passed

