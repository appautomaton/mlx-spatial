STATUS: COMPLETE

SUMMARY:
- Slice 5 is complete and reviewed.
- Dense mesh fields can now produce non-empty vertices/faces through FlexiCubes inference triangulation.
- Empty or guarded extraction returns structured `mesh-decoder` blockers instead of fake geometry.

VERIFICATION:
- `uv run pytest -q tests/test_sam3d_mesh.py tests/test_sam3d_export.py` -> `28 passed`
- `python -m py_compile src/mlx_spatial/sam3d_mesh.py tests/test_sam3d_mesh.py tests/test_sam3d_export.py` -> passed

NEXT:
- Slice 6: integrate mesh decode/extract/export into `reconstruct --glb-output` trace.
