# Execution Summary — Production Pipeline Parity

All 7 slices executed 2026-05-20. Multi-agent parallel dispatch for parallel-safe group A.

## Slice 1: GS/SH Completion (G1.1, G1.2, G1.3)
- **Status:** DONE
- **Route:** subagent
- **Changes:** hyworld2_sh.py (fixed eval_sh_numpy bugs for degrees 1-4), hyworld2_export.py (unconditional GS rendering + default camera), tests/test_hyworld2_utils.py (SH index fix)
- **Verification:** 155/155 HY-World tests pass

## Slice 2: Sparse Interpolation (G2.3)
- **Status:** DONE
- **Route:** subagent
- **Changes:** spatial_interp.py (created, 4 functions), sparse_conv.py (refactored re-export), __init__.py (updated), test_spatial_interp.py (27 tests)
- **Verification:** 27 new tests + 13 regression tests pass

## Slice 3: mesh_to_flexible_dual_grid (G2.1)
- **Status:** DONE (feasibility confirmed)
- **Route:** subagent
- **Changes:** mesh_to_fdg.py (created, 348 lines), test_mesh_to_fdg.py (15 tests)
- **Verification:** 15/15 pass, 613/613 regression pass. Round-trip works. Conservative intersected flag noted as caveat.
- **Checkpoint:** Discharged — forward conversion works.

## Slice 4: Trellis2TexturingPipeline (G2.2)
- **Status:** DONE
- **Route:** subagent
- **Changes:** trellis2_texturing.py (created, 280 lines), __init__.py (7 exports), test_trellis2_texturing.py (18 tests)
- **Verification:** 101/101 pass. Pipeline produces textured GLB with fixture assets.

## Slice 5: Mesh Postprocessing (G3.1, G3.2, G3.3, G3.4)
- **Status:** DONE
- **Route:** subagent
- **Changes:** sam3d_render.py (+130), sam3d_mesh.py (+180), ovoxel.py (+220), grid.py (+130), __init__.py, 4 test files
- **Verification:** 193/193 pass

## Slice 6: Integration and Regression
- **Status:** PASSED
- **Route:** direct
- **Verification:** `uv run pytest tests/ -v` — 631 passed, 5 skipped, 0 failures

## Slice 7: Gap Matrix Refresh
- **Status:** DONE
- **Route:** direct
- **Changes:** spec/gap-matrix.md — appended Closure Status section with per-gap evidence

## New Modules Created
- src/mlx_spatial/spatial_interp.py
- src/mlx_spatial/mesh_to_fdg.py
- src/mlx_spatial/trellis2_texturing.py

## Open Caveats (non-blocking)
- GS rasterizer: per-pixel loop, no tile binning (performance deferred)
- mesh_to_fdg: conservative intersected flag at low grid sizes
- Dual-contouring: CPU-only, no GPU replication of cumesh
- No CI; parity verification needs local PyTorch checkout

## SAM3D Follow-Up: PointPatch Resize Parity
- **Status:** DONE 2026-05-21
- **Issue:** `sam3d_condition._resize_nearest_bchw` used center-based nearest indices, while official `PointPatchEmbed.resize_input` calls PyTorch `F.interpolate(..., mode="nearest")`, which uses floor-index mapping.
- **Changes:** Patched `_resize_nearest_bchw` to floor-index mapping and added `test_resize_nearest_bchw_matches_torch_floor_indices`.
- **Verification:** `uv run pytest tests/test_sam3d_condition.py tests/test_sam3d_preprocess.py tests/test_sam3d_gaussian.py tests/test_sam3d_decoder.py -q` - 25 passed. Direct helper check against `torch.nn.functional.interpolate` produced `max_abs=0.0`.
- **Artifact:** Regenerated plant output at `outputs/sam3d/colorful-interior-condition-fixed/gaussians.ply` with `trace.json` and `debug-orthographic-panels.png`.
- **Stats:** Vertex count 131616, visible alpha>0.5 count 29202, alpha mean 0.2661, bounds extent approximately `[1.0015, 0.9674, 0.5806]`.

## SAM3D Follow-Up: Official Default Sampling Steps
- **Status:** DONE 2026-05-21
- **Issue:** CLI/API defaults used fast `2/12` stage steps from generator YAMLs, but the official notebook path leaves inference steps unset and `InferencePipeline` defaults to `25/25`. The fast run produced visibly incomplete/missing geometry on a clean stool mask.
- **Changes:** Updated `mlx-spatial-sam3d reconstruct` and `Sam3dInferencePipeline.generate_gaussians_ply` defaults to `25/25`; users can still pass `--stage1-steps 2 --stage2-steps 12` for quick low-quality checks.
- **Verification:** `uv run pytest tests/test_sam3d_tools.py tests/test_sam3d_condition.py tests/test_sam3d_preprocess.py -q` - 26 passed.
- **Artifact:** Official-default stool output at `outputs/sam3d/colorful-interior-mask-35-steps25/gaussians.ply`, with trace and `debug-alpha-threshold-panels.png`.
