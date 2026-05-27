# mlx-spatialkit Xatlas Parity Diagnostics Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-xatlas-parity-diagnostics-gate/SPEC.md`: make the remaining xatlas chart parity boundary quantitative and test-covered.

## Architecture Approach

This is diagnostics hardening, not a charting algorithm replacement. Extend reference trace loading and export quality summaries with structured `quality.xatlas_chart_parity`; keep `parity_ready=false` until native charting is actually equivalent to the xatlas reference path.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Structured Xatlas Parity Diagnostics

**Objective:** Add `quality.xatlas_chart_parity` diagnostics and focused helper tests.

**Acceptance criteria:**
- Reference trace loading includes xatlas chart count and utilization fields.
- Native-chart exports report reference/native chart metrics, ratios, checks, `parity_ready=false`, `xatlas_chart_parity=false`, and `deferred_boundary="not_xatlas_chart_parity"`.
- Focused tests cover native-chart, non-native-chart, and missing-reference states.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_xatlas_chart_parity_summary_reports_measured_native_chart_gap -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added structured `quality.xatlas_chart_parity` diagnostics and focused helper coverage; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_xatlas_chart_parity_summary_reports_measured_native_chart_gap -q` -> 1 passed. The helper reports measured native-chart gaps, non-native-chart not-requested state, and missing-reference state while keeping `parity_ready=false`.
**Risks / next:** real fixture assertions must prove the diagnostics are populated from the checked-in Pixal3D reference trace.

### Slice 2: Real Fixture Assertions And Docs

**Objective:** Assert the structured diagnostics in heavy native-chart gates and document the measured-but-deferred boundary.

**Acceptance criteria:**
- Reference-target native-chart heavy gate asserts structured xatlas parity diagnostics.
- Explicit 1M/4096 native-chart heavy gate asserts structured xatlas parity diagnostics.
- Docs explain measured xatlas diagnostics without claiming parity or adding xatlas as a package dependency.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** asserted structured xatlas parity diagnostics in both native-chart real-fixture gates and documented the measured-but-deferred boundary; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy` -> 2 passed. The gates assert reference unwrap backend, xatlas chart count/utilization, native chart metrics, ratios, false backend-equivalence check, and `parity_ready=false`.
**Risks / next:** this measures xatlas parity but does not implement native xatlas-equivalent charting.

### Slice 3: Regression Hygiene

**Objective:** Verify package/root/build stability and artifact cleanliness.

**Acceptance criteria:**
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Status:** complete
**Evidence:** `git diff --check` -> passed; docs grep found `quality.xatlas_chart_parity`, xatlas utilization/count, and `parity_ready=false` language; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 64 passed, 7 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify should inspect the latest real-fixture diagnostics before marking verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| XPDG-01 | Slice 1 |
| XPDG-02 | Slice 1 |
| XPDG-03 | Slice 1 |
| XPDG-04 | Slice 2 |
| XPDG-05 | Slice 2 |
| XPDG-06 | Slice 3 |

## Execution Notes

- Do not add xatlas to the spatialkit package dependencies.
- Do not turn a ratio threshold into parity readiness in this phase.
- Keep generated and heavy artifacts under `/tmp`.
