STATUS: DONE

SUMMARY:
- Implemented inference-only FlexiCubes surface-core primitives in `src/mlx_spatial/sam3d_mesh.py`.
- Added repo-local official FlexiCubes tables in `src/mlx_spatial/sam3d_flexicubes_tables.py`.
- Added focused tests for surface cube/edge detection, official weight activation, case IDs, and dual vertex candidate generation.

FILES:
- `src/mlx_spatial/sam3d_mesh.py`
- `src/mlx_spatial/sam3d_flexicubes_tables.py`
- `tests/test_sam3d_mesh.py`

VERIFICATION:
- `uv run pytest -q tests/test_sam3d_mesh.py` -> `13 passed`
- `python -m py_compile src/mlx_spatial/sam3d_mesh.py src/mlx_spatial/sam3d_flexicubes_tables.py tests/test_sam3d_mesh.py` -> passed
