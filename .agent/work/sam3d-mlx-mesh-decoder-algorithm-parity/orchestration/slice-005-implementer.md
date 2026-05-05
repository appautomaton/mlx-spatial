STATUS: DONE

SUMMARY:
- Implemented `Sam3dMeshExtractResult`, FlexiCubes non-training triangulation, high-level dense-field/feature mesh extraction, optional sigmoid color propagation, and structured mesh blockers.
- Added GLB writer acceptance coverage for extracted mesh outputs.

FILES:
- `src/mlx_spatial/sam3d_mesh.py`
- `tests/test_sam3d_mesh.py`
- `tests/test_sam3d_export.py`

VERIFICATION:
- `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_export.py` -> `26 passed` before review fixes; `28 passed` after review fixes.
- `python -m py_compile src/mlx_spatial/sam3d_mesh.py tests/test_sam3d_mesh.py tests/test_sam3d_export.py` -> passed
