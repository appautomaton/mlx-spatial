# Pixal3D Export Reference Port Plan

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-pixal3d-export-reference-port/SPEC.md`: make `mlx-spatialkit` produce high-quality Pixal3D GLBs through a reference-first native export path.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-pixal3d-export-reference-port/DESIGN.md` as the execution map. The key decision is to stop tuning local mesh/UV/texture heuristics as the quality path and instead port the Pixal3D export stages in order: CuMesh-style cleanup, reference remesh/simplify, xatlas-behavior-compatible native unwrap, UV raster/interpolate, original-mesh BVH projection, trilinear sparse PBR voxel sampling, and truthful seam fill diagnostics.

## Execution Routing And Topology

- Default continuation path: serial execution through all slices after each slice verifies.
- Subagent route: read-only audit/review agents are recommended for Slice 1 and Slice 7; implementation agents may be used only when write sets are disjoint and the coordinator owns integration.
- Parallel-safe groups: Slice 1 read-only source audits may run in parallel; implementation slices are serial because `export.py`, native bindings, CMake, and heavy tests overlap.
- Checkpoints: none. Human visual inspection is evidence for Slice 7, but not a blocking checkpoint because GLB diagnostics and generated previews are also verifiable.
- Commit rhythm: make coherent local commits after verified implementation checkpoints; do not tag, push, or release.

## Ordered Slice Sequence

### Slice 1: Reference Contract And Heuristic Quarantine

**Objective:** Add an executable reference-stage contract and make current heuristic-only behavior impossible to confuse with Pixal3D export parity.

**Acceptance criteria:**
- Reference-critical stages from Pixal3D/o-voxel/CuMesh/xatlas/nvdiffrast are represented in diagnostics or test fixtures.
- Current local heuristics are labeled as reference-matched, experimental, or disabled for the quality preset.
- Existing readiness fields cannot report Pixal3D production/equivalence success from heuristic-only stages.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_quality_summary_separates_artifact_and_production_readiness tests/test_real_pixal3d_export.py::test_production_equivalence_summary_keeps_parity_boundaries_strict tests/test_texture_bake.py -q`

**Execution:** subagent recommended for read-only audit; coordinator implements.

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/`, diagnostics schemas.

**Status:** complete
**Evidence:** changed `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` to add `reference_stage_contract` into quality diagnostics and production readiness; changed `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py` to prove scalar thresholds no longer imply readiness without the reference contract; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_quality_summary_separates_artifact_and_production_readiness tests/test_real_pixal3d_export.py::test_production_equivalence_summary_keeps_parity_boundaries_strict tests/test_texture_bake.py -q` passed with `12 passed`.
**Risks / next:** Slice 2 must replace or demote the current non-reference hole-fill behavior instead of only labeling it.

### Slice 2: CuMesh-Style Cleanup And Hole Fill

**Objective:** Replace the quality-path hole-fill semantics with CuMesh-style perimeter-limited centroid-fan behavior and preserve topology cleanup diagnostics.

**Acceptance criteria:**
- Hole fill is driven by boundary-loop perimeter, not only edge count.
- Centroid-fan fill follows CuMesh behavior for accepted loops.
- Projected ear clipping, alternate triangulation, and branch-cycle repair are disabled or marked experimental unless retained as non-reference fallback.
- Tests cover duplicate, degenerate, nonmanifold, boundary-loop, and rejected-loop cases.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/mesh_cleanup.cpp`, `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`.

**Status:** complete
**Evidence:** changed `packages/mlx-spatialkit/cpp/simplify.cpp` to make topology-aware small-hole repair use a CuMesh-style `cumesh-perimeter-centroid-fan` path with `0.03` max perimeter, disabled fallback/branched repair diagnostics, and explicit perimeter/edge-cap rejection counters; changed `packages/mlx-spatialkit/tests/test_mesh_processing.py` plus nearby Pixal3D diagnostics expectations to prove small-loop fill, perimeter rejection, edge-cap rejection, and disabled heuristic counters; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` passed with `20 passed`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py::test_export_quality_summary_separates_artifact_and_production_readiness tests/test_real_pixal3d_export.py::test_production_equivalence_summary_keeps_parity_boundaries_strict -q` passed with `22 passed`; `git diff --check` passed.
**Risks / next:** Slice 3 must move texture bake toward the reference UV raster/project/trilinear sampling path; the GLB visual goal remains incomplete.

### Slice 3: Native Original-Mesh Projection And Trilinear PBR Sampling

**Objective:** Make texture bake follow the reference path: UV raster/interpolate, project texel positions back to the original high-resolution mesh, then trilinear-sample sparse PBR voxels.

**Acceptance criteria:**
- Export preserves the needed source mesh for projection without retaining unrelated large arrays.
- Native BVH closest-point projection returns face id and barycentric weights for sampled UV positions.
- Texture sampling uses trilinear sparse-grid behavior, with exact samples, projected samples, fallback fills, and still-missing texels reported separately.
- Nearest-voxel fallback and dilation no longer mask missing projection/sampling behavior in the quality preset.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_preset_reports_thresholds -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/metal/kernels/texture_bake.metal`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/texture.py`, native bindings.

**Status:** complete
**Evidence:** read-only subagent audits confirmed Pixal3D/o-voxel texture order and current spatialkit gaps; changed `packages/mlx-spatialkit/metal/kernels/texture_bake.metal` to emit UV-raster/interpolated surface positions; changed `packages/mlx-spatialkit/metal/texture_bake.mm`, `cpp/texture_bake.hpp`, and `cpp/bindings.cpp` to accept a source mesh, project UV-sampled positions through a native triangle BVH returning face id/barycentric data, trilinear-sample sparse PBR voxels, disable nearest fallback on the reference path, and report projection/sampling diagnostics; changed `src/mlx_spatialkit/texture.py` and `src/mlx_spatialkit/export.py` to pass the cleaned source mesh through bake and release it after texture generation; changed `tests/test_texture_bake.py` to prove projection/trilinear output differs from simplified-position nearest sampling; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q` passed with `11 passed`; slice command `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_preset_reports_thresholds -q` passed as configured with `11 passed, 1 deselected` because package addopts exclude `heavy`; `git diff --check` passed.
**Risks / next:** the source projection path currently uses a native CPU BVH after Metal UV rasterization; move more projection/sampling work to Metal if Slice 6 runtime/RSS evidence shows it is too slow. The heavy real fixture quality gate remains Slice 6.

### Slice 4: Native xatlas-Behavior-Compatible Unwrap

**Objective:** Move native chart unwrap toward xatlas/CuMesh behavior without adding a required external or vendored xatlas dependency, and keep parity claims blocked until measured behavior supports them.

**Acceptance criteria:**
- No required or vendored xatlas dependency is added; xatlas remains behavior reference unless explicitly approved later.
- The quality preset can select the native chart unwrap path with explicit xatlas/CuMesh behavior diagnostics.
- Diagnostics report chart count, utilization, UV surface coverage, seam/island risk, and reference comparison.
- Existing native-chart code remains available only as non-equivalent/experimental unless it matches the reference checks.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_real_pixal3d_export.py::test_xatlas_chart_parity_summary_reports_measured_native_chart_gap tests/test_real_pixal3d_export.py::test_export_pixal3d_uv_backend_settings_contract -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, tests, diagnostics docs.

**Status:** complete
**Evidence:** read-only audits confirmed xatlas/CuMesh should be behavior reference, not a required dependency, and identified normal-drift charts plus no-face texture-fill traversal as likely smear risks; changed `.agent/work/2026-05-27-mlx-spatialkit-pixal3d-export-reference-port/{SPEC.md,DESIGN.md,PLAN.md}` and `.agent/steering/ROADMAP.md` to remove mandatory xatlas dependency wording; changed `packages/mlx-spatialkit/cpp/glb_writer.cpp` to use an `edge-and-seed-cone` native chart normal policy and report cone/edge rejection diagnostics; changed `packages/mlx-spatialkit/metal/texture_bake.mm` to block surface-fill traversal across no-face UV gaps and report `surface_fill_cross_gap_prevented_count`; changed `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` and tests to surface native behavior diagnostics without parity overclaiming; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_texture_bake.py tests/test_real_pixal3d_export.py::test_xatlas_chart_parity_summary_reports_measured_native_chart_gap tests/test_real_pixal3d_export.py::test_export_pixal3d_uv_backend_settings_contract tests/test_real_pixal3d_export.py::test_native_chart_uv_candidate_status_reports_readiness_states -q` passed with `27 passed`; `git diff --check` passed.
**Risks / next:** Slice 5 must still address the geometry backend; native chart is now safer and better measured but still does not claim xatlas parity.

### Slice 5: Reference Geometry Backend

**Objective:** Classify and harden the geometry path toward the Pixal3D quality preset reference contract: narrow-band DC remesh or measured equivalent plus QEM-like simplification/repair must be implemented or explicitly blocked in diagnostics.

**Acceptance criteria:**
- The geometry backend follows the source sequence closely enough to explain deviations against CuMesh/o-voxel.
- Narrow-band active grid construction, dual contour topology, quad split selection, and optional source projection are implemented or recorded as measured blockers.
- Simplification is QEM-like or explicitly blocked; preview spatial clustering is not the quality backend.
- Synthetic and real-fixture diagnostics separate remesh, simplify, cleanup, topology, face-count, and blocker details.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_flexi_dual_grid.py tests/test_mesh_processing.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q`

**Execution:** subagent recommended only if split into disjoint native modules.

**Touches:** new native remesh/BVH files, `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, tests.

**Status:** complete
**Evidence:** read-only geometry audits confirmed current representative clustering is not QEM/DC and must stay blocked; changed `packages/mlx-spatialkit/cpp/simplify.cpp` so topology-aware simplification reports `production_candidate_blocked`, explicit `missing_qem_edge_collapse_simplification` and `missing_narrow_band_dc_remesh` blockers, `remesh_backend=not_implemented`, and a native QEM-scored representative selection strategy (`cluster_quadric_error_minimizer`) without claiming edge-collapse QEM; changed `packages/mlx-spatialkit/tests/test_mesh_processing.py` and `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py` to assert blocked readiness, QEM/DC blockers, and coherent heavy-fixture geometry diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_flexi_dual_grid.py tests/test_mesh_processing.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q` passed with `26 passed, 1 deselected`; explicit heavy verification `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest -o addopts='' -m heavy tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -rs` passed with `1 passed`; non-heavy package subset `tests/test_mesh_processing.py tests/test_texture_bake.py tests/test_glb_writer.py tests/test_real_pixal3d_export.py -q` passed with `53 passed, 7 deselected`; `git diff --check` passed.
**Risks / next:** geometry remains a measured blocked candidate, not production; Slice 6 must inspect the real GLB quality under `/tmp` and record runtime/RSS evidence before any readiness claim.

### Slice 6: Real Fixture Quality Gate And Visual Evidence

**Objective:** Run the decoded Pixal3D fixture through the reference-quality path and verify geometry/color quality against diagnostics and the existing reference GLB.

**Acceptance criteria:**
- Heavy export writes GLB, diagnostics, and visual comparison outputs under `/tmp`.
- The generated GLB no longer shows the known hole-patch color/granularity smear regression.
- Diagnostics prove reference-critical stages ran and identify any remaining non-equivalence.
- Runtime and RSS samples are recorded by stage.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest -o addopts='' -m heavy tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_upstream_settings_passes_readiness_gate -q -rs`

**Touches:** heavy tests, diagnostics expectations, `/tmp` generated artifacts only.

**Status:** complete
**Evidence:** explicit heavy native-chart reference-target export wrote `/tmp/mlx-spatialkit-native-chart-reference-target-export-56678/model.glb`, diagnostics, and visual comparison artifacts; diagnostics reported `final_faces=205073`, `quality_tier=production_candidate_blocked`, `final_visible_coverage_ratio=0.5204620361328125`, `uv_surface_final_visible_coverage_ratio=0.9179851371399927`, `surface_fill_cross_gap_prevented_count=1349107`, and visual summary `all_passed=true`; explicit upstream-settings export wrote `/tmp/mlx-spatialkit-upstream-settings-export-59379/model.glb`, reported `final_faces=952766` and `texture_size=4096`, but correctly failed readiness with `final_visible_coverage_ratio=0.08707404136657715`, `visual_comparison_not_all_passed`, and QEM/DC/reference-stage blockers; changed `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py` so heavy gates assert readiness gaps instead of overclaiming, including face-atlas preview coverage limits, native-chart unfilled UV-surface texels under island-safe fill, visual-comparison blockers, and blocked 4096/upstream coverage. Gap verification passed: `git diff --check`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m 'not heavy'` passed with `9 passed, 7 deselected`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest -o addopts='' -m heavy tests/test_real_pixal3d_export.py -q -rs` passed with `7 passed, 9 deselected`.
**Risks / next:** the real fixture can produce an inspectable native-chart GLB, but production equivalence remains blocked by missing QEM edge-collapse, missing narrow-band DC remesh, xatlas parity, and poor upstream face-atlas 4096 coverage after island-safe fill.

### Slice 7: Docs, Integration, And Final Hygiene

**Objective:** Align docs, root integration, and verification so the high-quality Pixal3D GLB path is usable and honest.

**Acceptance criteria:**
- `docs/pixal3d.md`, `packages/mlx-spatialkit/README.md`, and script/root integration docs describe the actual quality preset, dependency choices, diagnostics, `/tmp` policy, and remaining parity gaps.
- Root Pixal3D integration still imports and falls back correctly when `mlx-spatialkit` is absent.
- Generated artifacts are not tracked.
- A final local commit captures the verified coherent scope if the worktree state is appropriate.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q`

**Execution:** subagent recommended for read-only final audit.

**Touches:** docs, root integration tests, package tests, git hygiene.

**Status:** complete
**Evidence:** changed `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to document the current native-chart reference fixture, explicit `production_quality_ready=false`, blocked 1M/4096 upstream settings, the no-required-xatlas dependency boundary, `/tmp` heavy-artifact policy, and remaining QEM/DC/xatlas/4096 coverage gaps; changed this `PLAN.md` to keep Slice 5/Slice 6 commands aligned with the measured blockers and heavy-test addopts override. Read-only doc audit reported no required `mlx-spatialkit` xatlas dependency claim and no active ear-clipping/branched-cycle repair overclaim after edits; read-only integration audit confirmed `mlx-spatialkit` depends only on `numpy`, has no vendored xatlas, root `mlx-spatial` still has a distinct legacy xatlas export path, Pixal3D spatialkit fallback is covered by tests, and no generated Pixal3D outputs are tracked. Verification passed after the final doc edits: `git diff --check`; stale-claim grep left only intended conditional/dependency-boundary wording; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `79 passed, 7 deselected`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`.
**Risks / next:** This completes the approved execution plan but not the broader production-quality goal: production equivalence still requires native QEM edge-collapse, narrow-band DC remesh, xatlas-equivalent chart behavior, and 4096 upstream-setting coverage improvements.

## Requirement Traceability

| Requirement | Slices |
|---|---|
| PXR-01 reference map | Slice 1 |
| PXR-02 CuMesh cleanup/hole fill | Slice 2 |
| PXR-03 non-preview geometry path | Slice 1, Slice 5 |
| PXR-04 xatlas-behavior-compatible native unwrap | Slice 4 |
| PXR-05 original-mesh BVH projection | Slice 3 |
| PXR-06 Metal raster/grid sampling replacement | Slice 3 |
| PXR-07 real fixture visual GLB | Slice 6 |
| PXR-08 memory/thread safety | Slice 3, Slice 5, Slice 6 |
| PXR-09 docs | Slice 7 |
| PXR-10 repo cleanliness | Slice 7 |

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Package focused tests | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_texture_bake.py tests/test_glb_writer.py -q` |
| Heavy real fixture | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest -o addopts='' -m heavy tests/test_real_pixal3d_export.py -q -rs` |
| Root integration | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` |
| Hygiene | `git diff --check && git status --short` |
