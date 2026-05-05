STATUS: COMPLETE

SUMMARY:
- Live SAM3D human-object reconstruction succeeded with gaussian PLY plus basic GLB.
- Trace reached `glb-export` with no blocker.
- Blender headless import succeeded.
- Focused SAM3D tests, full pytest, and diff check passed.

LIVE COMMAND:
- `uv run mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx vendors/sam-3d-objects/notebook/images/human_object/image.png --mask vendors/sam-3d-objects/notebook/images/human_object/0.png --moge-root weights/moge-vitl-mlx --output outputs/sam3d/human-object/gaussians.ply --glb-output outputs/sam3d/human-object/mesh.glb --seed 42 --memory-profile large --trace-output outputs/sam3d/human-object/trace.json` -> exit `0`

ARTIFACTS:
- `outputs/sam3d/human-object/gaussians.ply`: 166304 vertices, binary official-field PLY, 11309088 bytes
- `outputs/sam3d/human-object/mesh.glb`: 156005 mesh vertices, 311508 faces, 7483504 bytes
- `outputs/sam3d/human-object/trace.json`: blocker `null`

VERIFICATION:
- Blender headless import -> `BLENDER_IMPORT_OK objects=4 meshes=2 faces=311514 verts=156013`
- `uv run pytest -q tests/test_sam3d_assets.py tests/test_sam3d_condition.py tests/test_sam3d_decoder.py tests/test_sam3d_export.py tests/test_sam3d_gaussian.py tests/test_sam3d_mesh.py tests/test_sam3d_tools.py` -> `63 passed`
- `uv run pytest -q` -> `425 passed, 5 skipped`
- `git diff --check` -> passed
