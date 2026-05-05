STATUS: APPROVED

SUMMARY:
- Slice 6 meets the requested integration scope.
- `reconstruct --glb-output` now uses `slat_decoder_mesh` features through extraction and GLB export.
- Trace shape metadata comes from actual SLat coords/features.
- Fixture tests prove GLB vertices/faces come from extracted mesh output.

ISSUES:
- none after fixes

EVIDENCE:
- `uv run pytest -q tests/test_sam3d_tools.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py` -> `23 passed`
