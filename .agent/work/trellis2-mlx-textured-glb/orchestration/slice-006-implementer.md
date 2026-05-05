# Slice 6 Implementer: GLB Writer And Blender Fixture Verification

## Scope

Implemented a dependency-free internal GLB 2.0 writer for baked TRELLIS.2 texture fixtures. This slice verifies fixture export with Blender but does not yet connect live `generate-textured` to write the final artifact.

## Changes

- Added `trellis2_textured_glb_payload` in `src/mlx_spatial/trellis2_export.py`.
- Added `write_trellis2_textured_glb` with `.glb` and `outputs/` path policy.
- Embedded:
  - mesh positions
  - mesh UVs
  - triangle indices
  - base-color RGBA PNG
  - metallic-roughness PNG
  - glTF PBR material and texture references
- Added validation for mesh/UV/face shapes, finite geometry, texture dtypes, and output path ordering.
- Exported new helpers through `mlx_spatial.__init__`.
- Added GLB structure tests and Blender fixture import test.
- Wrote fixture output: `outputs/trellis2/fixture-textured.glb`.

## Verification

- `uv run pytest -q tests/test_trellis2_export.py tests/test_trellis2_inference.py` -> `51 passed in 0.87s`
- `git diff --check` -> passed
- `blender --background --python-expr ... outputs/trellis2/fixture-textured.glb ...` -> `GLB_OK 1 1 2`

