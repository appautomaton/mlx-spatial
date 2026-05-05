# Slice 008 Summary

## Scope

Gaussian Attribute Stage Contract.

## Result

- Added the staged `gs` route on top of the MLX dense-head fixture path.
- Exported Gaussian attributes to `gaussian/attributes.npz` plus `gaussian/metadata.json`.
- Kept Gaussian rendering/export explicit as a `gaussian-export` blocker.
- Trace metadata now distinguishes Gaussian attributes from rendering.
- No CUDA `gsplat` runtime import is used by the shipped MLX path.

## Evidence

- `uv run pytest -q tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py` -> 29 passed.
- `git diff --check -- src/mlx_spatial/hyworld2_heads.py src/mlx_spatial/hyworld2_inference.py src/mlx_spatial/hyworld2_export.py tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py` -> passed.

## Next

Slice 9: Live 518 Reconstruction Verification.
