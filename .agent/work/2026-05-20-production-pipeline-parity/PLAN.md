# Plan: Production Pipeline Parity

**Goal:** Close all remaining inference gaps across SAM 3D, TRELLIS.2, and HY-World 2.0. Spec: `SPEC.md`.

## Ordered Slice Sequence

### Slice 1: GS/SH Completion (G1.1, G1.2, G1.3)

**Objective:** Integrate higher-degree SH into the rasterizer, complete SH degree 4 evaluation, and wire HY-World GS rasterization into export.
**Acceptance criteria:**
- `_colors_from_sh_features()` accepts `sh_degree > 0` and uses `hyworld2_sh.eval_sh()` for degrees 1-3
- `hyworld2_sh.eval_sh(deg=4, ...)` produces correct output matching vendor reference
- `hyworld2_export.py` renders GS images via `rasterize_gaussians()` instead of exporting raw params only
- Metadata no longer states "no CUDA rasterization"
**Verification:** `uv run pytest tests/test_gs_rasterize.py tests/test_hyworld2_utils.py -v` passes; existing HY-World suite (155 tests) still passes. *Correction: SH tests live in test_hyworld2_utils.py, not test_hyworld2_sh.py.*
**Execution:** subagent recommended
**Touches:** `gs_rasterize.py`, `hyworld2_sh.py`, `hyworld2_export.py`, tests
**Covers:** AC-01, AC-02, AC-03

### Slice 2: Sparse Interpolation (G2.3)

**Objective:** Implement `sparse_nearest_interpolate` and `sparse_trilinear_interpolate` matching vendor `spatial2channel.py` behavior.
**Acceptance criteria:**
- `sparse_nearest_interpolate` produces output matching vendor reference within float32 tolerance
- `sparse_trilinear_interpolate` produces output matching vendor reference within float32 tolerance
**Verification:** `uv run pytest tests/test_spatial_interp.py -v` passes (new test file)
**Execution:** subagent recommended
**Touches:** new `spatial_interp.py`, new `tests/test_spatial_interp.py`
**Covers:** AC-06

### Slice 3: mesh_to_flexible_dual_grid (G2.1)

**Objective:** Implement pure Python/NumPy conversion from triangle mesh to FlexiDualGrid representation (voxel indices + dual vertices + intersected flags), replacing `o_voxel.convert.mesh_to_flexible_dual_grid()`.
**Acceptance criteria:**
- `mesh_to_flexible_dual_grid(vertices, faces, grid_size)` returns voxel indices, dual vertices, and intersected flags
- Round-trip through existing `flexible_dual_grid_to_mesh_np()` produces a valid mesh
- Output format matches the interface expected by `run_flexi_dual_grid_vae_encoder()` in `trellis2_decode.py`
**Verification:** `uv run pytest tests/test_mesh_to_fdg.py -v` passes (new test file)
**Execution:** subagent recommended
**Touches:** new `mesh_to_fdg.py`, new `tests/test_mesh_to_fdg.py`
**Checkpoint after:** decision
**Checkpoint reason:** If forward conversion proves infeasible at spec quality, user must choose: (a) accept documented limitation and keep texturing gated on image-to-3D path only, or (b) invest in alternative approach. Provide feasibility assessment with test evidence.
**Covers:** AC-04

### Slice 4: Trellis2TexturingPipeline (G2.2)

**Objective:** Build standalone texturing-only pipeline class: preprocess image → preprocess mesh → encode mesh to shape SLat via FlexiDualGrid encoder → sample texture SLat via FlowEuler → decode to PBR voxels → bake texture → export textured GLB.
**Acceptance criteria:**
- `Trellis2TexturingPipeline.run(image_path, mesh_path)` produces a textured GLB file
- Texture bake output contains base_color, metallic, roughness, alpha channels on a UV-unwrapped mesh
- Pipeline reuses existing modules (encoder in `trellis2_decode.py`, flow in `trellis2_slat.py`, export in `trellis2_export.py`)
**Verification:** `uv run pytest tests/test_trellis2_texturing.py -v` passes; existing TRELLIS.2 suite (200 tests) still passes
**Depends on:** Slice 3
**Touches:** new `trellis2_texturing.py`, `trellis2_inference.py`, `__init__.py`, tests
**Covers:** AC-05

### Slice 5: Mesh Postprocessing (G3.1, G3.2, G3.3, G3.4)

**Objective:** Implement layout post-optimization, multi-view hole-filling, dual-contouring remeshing, and grid utilities — replacing PyTorch3D/Open3D/igraph/cumesh/scipy with Mac-native alternatives.
**Acceptance criteria:**
- Layout post-optimization improves scene layout quality (render-and-compare score vs unoptimized)
- Multi-view hole-filling removes interior faces without removing valid exterior faces
- Dual-contouring remeshing produces watertight mesh from FlexiDualGrid fields
- Grid utilities provide UV grid creation and Fourier embeddings matching vendor output
**Verification:** `uv run pytest tests/test_sam3d_render.py tests/test_sam3d_mesh.py tests/test_ovoxel.py tests/test_grid.py -v` passes; existing SAM 3D suite (124 tests) and TRELLIS.2 suite (200 tests) still pass
**Execution:** subagent recommended
**Touches:** `sam3d_render.py`, `sam3d_mesh.py`, `ovoxel.py`, `grid.py`, `export_utils.py`, tests
**Covers:** AC-07, AC-08, AC-09, AC-10

### Slice 6: Integration and Regression

**Objective:** Verify all slices integrate cleanly with no regressions across the full test suite.
**Acceptance criteria:**
- All existing test suites pass unchanged (HY-World 151, SAM 3D 124, TRELLIS.2 200, GS rasterizer 9, sparse interp)
- New test suites for all 10 gaps pass
- No import errors, no module conflicts
**Verification:** `uv run pytest tests/ -v` passes with 0 failures
**Depends on:** Slices 1-5

### Slice 7: Gap Matrix Refresh

**Objective:** Update `spec/gap-matrix.md` to mark all 10 gaps as closed, with verification notes and closed dates.
**Acceptance criteria:**
- All G1.x, G2.x, G3.x gaps marked closed with verification evidence
- SAM-MOT and SAM-SHORTCUT explicitly noted as already ported
**Verification:** Gap matrix has no open P1 or P2 gaps; `numeric_parity_verified` flags set where applicable
**Depends on:** Slice 6

## Execution Routing and Topology

Default: direct, serial, continuation after verification.

Overrides:
- Slice 1: subagent recommended (3 gaps across rasterizer internals, SH math, and export integration — crosses math/graphics/export boundaries)
- Slice 2: subagent recommended (new module with reference-dependent parity tests)
- Slice 3: subagent recommended (high-risk algorithm development, implements mesh voxelization from scratch)
- Slice 5: subagent recommended (4 independent gaps touching 5 files across mesh/render/export subsystems)

Parallel-safe groups (disjoint write sets, no shared state):
- **Group A:** Slices 1, 2, 3, 5 — all write to different files. Slice 1 (gs_rasterize/hyworld2_sh/hyworld2_export), Slice 2 (new spatial_interp.py), Slice 3 (new mesh_to_fdg.py), Slice 5 (sam3d_render/sam3d_mesh/ovoxel/grid/export_utils). Zero overlap.
- Slice 4 must run after Slice 3 (depends on mesh_to_flexible_dual_grid).
- Slice 6 must run after Slices 1-5 (integration gate).
- Slice 7 must run after Slice 6 (gap matrix refresh).

Checkpoints:
- Slice 3: decision (feasibility gate for mesh_to_flexible_dual_grid — forward conversion is highest-risk gap)

## Requirement Traceability

| AC | Gap(s) | Slice |
|---|---|---|
| AC-01 | G1.1 (higher-degree SH in rasterizer) | Slice 1 |
| AC-02 | G1.2 (SH degree 4) | Slice 1 |
| AC-03 | G1.3 (HY-World GS rasterization) | Slice 1 |
| AC-04 | G2.1 (mesh_to_flexible_dual_grid) | Slice 3 |
| AC-05 | G2.2 (Trellis2TexturingPipeline) | Slice 4 |
| AC-06 | G2.3 (sparse interpolation) | Slice 2 |
| AC-07 | G3.1 (layout post-optimization) | Slice 5 |
| AC-08 | G3.2 (multi-view hole-filling) | Slice 5 |
| AC-09 | G3.3 (dual-contouring remeshing) | Slice 5 |
| AC-10 | G3.4 (grid utilities) | Slice 5 |
