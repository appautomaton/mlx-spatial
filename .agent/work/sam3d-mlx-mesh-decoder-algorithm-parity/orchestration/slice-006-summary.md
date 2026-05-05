STATUS: COMPLETE

SUMMARY:
- Slice 6 is complete and reviewed.
- `mlx-spatial-sam3d reconstruct --glb-output` now has a real mesh decoder -> FlexiCubes -> GLB path.
- Gaussian-only reconstruct remains independent of mesh decoder assets.

VERIFICATION:
- `uv run pytest -q tests/test_sam3d_assets.py tests/test_sam3d_contract.py tests/test_sam3d_tools.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py tests/test_sam3d_mesh.py` -> `59 passed`

NEXT:
- Slice 7: run live official sample and broader regression checks.
