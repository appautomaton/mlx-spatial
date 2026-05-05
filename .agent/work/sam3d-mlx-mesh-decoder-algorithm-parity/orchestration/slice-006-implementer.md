STATUS: DONE

SUMMARY:
- Replaced the `--glb-output` blocker with real mesh decoder feature execution, FlexiCubes mesh extraction, and basic GLB export.
- Gaussian-only reconstruction now skips mesh decoder asset requirements.
- Trace now records SLat shape, mesh decoder config/feature metadata, extraction metadata, GLB stats, and blockers.

FILES:
- `src/mlx_spatial/sam3d_assets.py`
- `src/mlx_spatial/sam3d_inference.py`
- `src/mlx_spatial/sam3d_export.py`
- `tests/test_sam3d_assets.py`
- `tests/test_sam3d_tools.py`
- `tests/test_sam3d_export.py`

VERIFICATION:
- `uv run pytest -q tests/test_sam3d_assets.py tests/test_sam3d_tools.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py` -> `35 passed`
- `python -m py_compile src/mlx_spatial/sam3d_assets.py src/mlx_spatial/sam3d_inference.py src/mlx_spatial/sam3d_export.py tests/test_sam3d_assets.py tests/test_sam3d_tools.py tests/test_sam3d_export.py` -> passed
