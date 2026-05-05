# Slice 8 Summary

## Status

- Slice: `slice-8-live-512-textured-glb-verification`
- Route: direct
- Result: complete

## What Changed

- Fixed guided texture C2S subdivision handling in `src/mlx_spatial/trellis2_decode.py`: texture decoder guide tensors are now thresholded with `> 0`, matching upstream `SparseResBlockC2S3d` behavior for shape decoder subdivision logits.
- Added explicit guide shape parity errors before C2S expansion so future texture decoder blockers name the guide index and expected token count.
- Replaced the live texture bake dense all-pairs fallback in `src/mlx_spatial/trellis2_export.py` with a deterministic sparse-grid neighbor sampler when the dense texel/voxel pair count exceeds the guard.
- Added regression coverage in `tests/test_trellis2_decode.py` and `tests/test_trellis2_export.py`.

## Live Evidence

- First live command reached `texture-decoder` blocker: `C2S subdivision must have shape (23448, 8), got (13031, 8)`.
- After guide thresholding, live command reached `mesh-export` blocker: `texture bake would compare up to 70338674688 texel/voxel pairs, above guard 8000000`.
- Final live command:
  - `uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-textured.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --slat-steps 1 --decoder-token-limit 1000000`
  - Result: completed through `mesh-export`
  - Artifact: `outputs/trellis2/image-textured.glb`
  - Bytes: `169310492`
- Blender GLB import:
  - Command reported `GLB_OK 1 1 2`

## Regression Evidence

- Shape command:
  - `uv run mlx-spatial-trellis2 generate-shape weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-512-rmbg-shape.obj --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --decoder-token-limit 1200000`
  - Result: completed through `mesh-export`
  - Artifact: `outputs/trellis2/image-512-rmbg-shape.obj`
  - Bytes: `80562246`
- Blender OBJ import reported `OBJ_OK 1`.
- `uv run pytest -q` reported `235 passed, 5 skipped`.
- `git diff --check` passed.

## Follow-Ups

- The generated GLB is Blender-readable and textured, but the UV atlas and bake are still internal deterministic export machinery. Upstream-quality UV unwrapping, hole filling, and texture refinement remain the next parity layer.
