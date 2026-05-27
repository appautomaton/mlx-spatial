# mlx-spatialkit Native Chart Split Axis Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-split-axis-gate/SPEC.md`: improve native low-fill chart splitting by bounded two-axis search and verify real fixture progress.

## Architecture Approach

Keep the current low-fill split threshold and max depth from Phase 26. For each eligible low-fill chart, evaluate median splits along both local centroid axes, choose the split with the best child fill, and accept it only when it beats the parent by the existing improvement margin. Add diagnostics so the bounded search policy is visible.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Two-Axis Low-Fill Split Search

**Objective:** Implement bounded two-axis split search and prove reference-target native-chart fill advances.

**Acceptance criteria:**
- Focused chart test passes with axis-search diagnostics.
- Reference-target native-chart heavy gate reports chart rect fill above `0.5647715200751198` or xatlas utilization ratio above `0.6131533138496904`.
- Parity diagnostics remain measured and non-ready.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py::test_make_native_chart_uvs_splits_low_fill_l_shape_deterministically -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** implemented bounded two-axis low-fill split search with diagnostics; focused split test passed; reference-target native-chart heavy gate passed. Latest diagnostics under `/tmp` show chart rect fill `0.5670824417746222` vs Phase 26 baseline `0.5647715200751198`, UV-surface occupancy `0.5153675079345703` vs `0.5095109939575195`, and xatlas utilization ratio `0.6202011322387381` vs `0.6131533138496904`, while `parity_ready=false`.
**Risks / next:** explicit 1M/4096 gate still needs verification after axis search.

### Slice 2: Upstream Gate And Docs

**Objective:** Keep explicit 1M/4096 native-chart readiness intact and document bounded split-axis search.

**Acceptance criteria:**
- Explicit 1M/4096 native-chart heavy gate still passes upstream-setting and native-chart readiness.
- Docs mention bounded split-axis search without claiming xatlas equivalence.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** explicit 1M/4096 native-chart heavy gate still passes after two-axis search; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy` -> 1 passed. Latest diagnostics show upstream-setting readiness true, native-chart quality-ready, rect fill `0.5551323155092972`, UV-surface occupancy `0.50787752866745`, two-axis candidate count `40390`, and parity still measured/non-ready.
**Risks / next:** full regression/build verification remains.

### Slice 3: Regression Hygiene

**Objective:** Verify package/root/build stability and artifact cleanliness.

**Acceptance criteria:**
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Status:** complete
**Evidence:** `git diff --check` -> passed; docs grep found two-axis split-search language and diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 64 passed, 7 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify should inspect latest real-fixture diagnostics before marking verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCAS-01 | Slice 1 |
| NCAS-02 | Slice 1 |
| NCAS-03 | Slice 1 |
| NCAS-04 | Slice 2 |
| NCAS-05 | Slice 2 |
| NCAS-06 | Slice 3 |

## Execution Notes

- Do not relax thresholds or parity readiness.
- Keep generated and heavy artifacts under `/tmp`.
- If real fixture metrics do not improve, revert the algorithm change instead of committing churn.
