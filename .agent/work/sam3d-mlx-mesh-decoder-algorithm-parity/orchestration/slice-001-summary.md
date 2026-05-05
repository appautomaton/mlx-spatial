# Slice 1 Summary: Mesh Decoder Contract And Torso Split

Status: completed
Route: direct

Files changed:
- `src/mlx_spatial/sam3d_decoder.py`: added explicit Gaussian and mesh decoder targets, mesh decoder config parsing, mesh tensor loading, and a shared decoder torso function.
- `tests/test_sam3d_decoder.py`: added fixture coverage for mesh decoder config parsing, mesh tensor prefix loading, and routing the shared torso into distinct output heads.

Verification:
- `uv run pytest -q tests/test_sam3d_decoder.py tests/test_sam3d_contract.py tests/test_sam3d_gaussian.py`
- Result: 11 passed in 0.27s

Notes:
- Existing Gaussian decoder behavior remains covered by the original network test and focused Gaussian tests.
