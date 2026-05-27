# mlx-spatialkit Chart UV Readiness Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-readiness-gate/SPEC.md`: make native chart UV export diagnostics separate artifact readiness from quality readiness and surface coverage/utilization blockers.

## Architecture Approach

Keep the current chart export path unchanged. Strengthen only the quality summary around it: compute chart artifact checks, coverage/utilization checks, quality blockers, and a chart-specific warning after texture bake stats are available. This keeps performance behavior stable while making the current quality gap explicit.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Chart Readiness Summary Contract

**Objective:** Replace the generic chart candidate status with artifact/quality readiness checks and blockers.

**Acceptance criteria:**
- `_native_chart_uv_candidate_status` reports not-requested, artifact-blocked, quality-blocked, and quality-ready states.
- Summary includes chart stats, texture backend, UV-bin diagnostics, global coverage, UV-surface occupancy, UV-surface visible coverage, checks, and blockers.
- `export_pixal3d_glb` appends `native_chart_uv_candidate_quality_blocked` to quality warnings when the chart candidate is artifact-ready but quality-blocked.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_native_chart_uv_candidate_status_reports_readiness_states -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added chart readiness checks for artifact and quality states; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_native_chart_uv_candidate_status_reports_readiness_states -q` -> 1 passed.
**Risks / next:** Slice 2 must prove the real fixture now reports quality-blocked instead of generic candidate.

### Slice 2: Real Fixture Readiness Proof

**Objective:** Prove the current real chart export is artifact-ready but quality-blocked, with the actual low-coverage diagnostics visible.

**Acceptance criteria:**
- Heavy chart export reports `artifact_ready=true`, `quality_ready=false`, `status=quality_blocked`.
- Diagnostics include failed global coverage and UV-surface occupancy checks.
- `result.quality_warnings` includes `native_chart_uv_candidate_quality_blocked`.
- Existing xatlas chart parity deferral remains present.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** updated heavy chart fixture assertions; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics report `status=quality_blocked`, `artifact_ready=true`, `quality_ready=false`, `quality_blockers=['global_coverage_floor', 'uv_surface_occupancy_floor']`, `global_coverage=0.14284706115722656`, `uv_surface_occupancy=0.23263072967529297`.
**Risks / next:** chart packing remains a future quality-improvement phase.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document the chart readiness gate and verify package/root/build stability.

**Acceptance criteria:**
- Docs explain chart readiness diagnostics and distinguish artifact readiness from quality readiness.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe chart `artifact_ready` versus `quality_ready`; `git diff --check` passed; package tests -> 59 passed, 5 deselected; root Pixal3D tests -> 35 passed; `/tmp` build succeeded; artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** final verify must rerun plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| CUVR-01 | Slice 1 |
| CUVR-02 | Slice 2 |
| CUVR-03 | Slice 2 |
| CUVR-04 | Slice 3 |
| CUVR-05 | Slice 3 |

## Execution Notes

- Do not change chart packing or Metal bake behavior in this phase.
- Do not switch the default UV backend.
- Keep generated and heavy artifacts under `/tmp`.
