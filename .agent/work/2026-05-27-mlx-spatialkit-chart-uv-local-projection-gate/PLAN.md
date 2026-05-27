# mlx-spatialkit Chart UV Local Projection Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-local-projection-gate/SPEC.md`: replace fixed-axis native chart projection with deterministic local-frame/PCA projection and prove real chart export occupancy improves.

## Architecture Approach

Keep chart grouping and shelf packing unchanged. For each chart, build a deterministic tangent frame from the averaged chart normal, project chart vertices into that plane, evaluate a bounded set of rotations around the PCA principal axis, choose the minimum rectangle area with deterministic tie-breaking, and emit normalized local UVs into the existing shelf packer. Add diagnostics for the projection backend, candidate count, and aggregate chart rectangle fill estimate.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Local Projection

**Objective:** Implement deterministic local-frame/PCA chart projection in `make_native_chart_uvs`.

**Acceptance criteria:**
- Focused chart tests pass with projection diagnostics.
- Coplanar square still produces one chart and reuses four vertices.
- Hard crease still produces multiple charts.
- A diagonal coplanar rectangular chart reports high chart rect fill under local projection.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`

**Status:** complete
**Evidence:** implemented local-frame/PCA projection, projection diagnostics, stable near-tie chart sort, and native NaN parameter guards; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_texture_bake.py::test_bake_pbr_texture_metal_uses_binned_path_for_native_chart_uvs -q` -> 11 passed.
**Risks / next:** Slice 2 must prove the real fixture improves over the Phase 16 shelf-packing baseline.

### Slice 2: Real Occupancy Proof

**Objective:** Prove the real Pixal3D native-chart export improves UV-surface occupancy over the Phase 16 shelf baseline.

**Acceptance criteria:**
- Heavy chart fixture writes GLB and diagnostics under `/tmp`.
- `quality.native_chart_uv_candidate.uv_surface_occupancy_ratio > 0.3310232162475586`.
- Chart export still bakes through `metal-uv-binned-nearest`.
- Readiness diagnostics remain truthful if quality thresholds still fail.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** tightened heavy fixture assertions to the Phase 16 baseline; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics show `projection=local-frame-pca`, `chart_rect_fill=0.5651956666422095`, `uv_surface_occupancy=0.38708019256591797`, `global_coverage=0.24212932586669922`, still `status=quality_blocked`.
**Risks / next:** local projection improved occupancy without reaching the 0.50 chart readiness floors.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document local projection and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe local projection as native chart candidate improvement, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe deterministic local-frame/PCA projection and the non-xatlas boundary; final verify reran `git diff --check`, package tests -> 61 passed, 5 deselected, root Pixal3D tests -> 35 passed, `/tmp` build succeeded, and artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** none for this phase.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| CLPG-01 | Slice 1 |
| CLPG-02 | Slice 1 |
| CLPG-03 | Slice 2 |
| CLPG-04 | Slice 3 |
| CLPG-05 | Slice 3 |

## Execution Notes

- Do not change chart grouping thresholds or Metal bake behavior in this phase.
- Do not switch the default UV backend.
- Keep generated and heavy artifacts under `/tmp`.
