# mlx-spatialkit Xatlas Deficit Diagnostics Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-xatlas-deficit-diagnostics-gate/SPEC.md`: make the remaining native-chart versus xatlas utilization deficit explicit and test-covered.

## Architecture Approach

This is diagnostics hardening, not an unwrap algorithm replacement. Add a named utilization-equivalence threshold and deficit fields to `quality.xatlas_chart_parity`; keep backend equivalence and overall parity false until native charting is actually equivalent to the xatlas reference path.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Deficit Diagnostics Contract

**Objective:** Add explicit xatlas utilization-deficit fields and a failed utilization-equivalence check to the parity helper.

**Acceptance criteria:**
- `quality.xatlas_chart_parity.deficits` reports utilization deficit and ratio gap fields when reference/native measurements exist.
- `quality.xatlas_chart_parity.checks.xatlas_utilization_equivalence` exposes the target and fails for the current measured gap.
- Existing no-parity fields remain false and the not-requested/missing-reference states remain stable.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_xatlas_chart_parity_summary_reports_measured_native_chart_gap -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added explicit xatlas utilization deficit fields, a `0.95` utilization-equivalence target, and `checks.xatlas_utilization_equivalence`; focused helper coverage asserts measured, missing-reference, and not-requested states. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_xatlas_chart_parity_summary_reports_measured_native_chart_gap -q` -> 1 passed.
**Risks / next:** Real fixture assertions must prove this remains visible when scalar native-chart quality passes.

### Slice 2: Real Fixture Assertions And Docs

**Objective:** Prove the deficit contract on real native-chart fixture paths and document the boundary.

**Acceptance criteria:**
- Reference-target native-chart heavy gate asserts native-chart quality readiness while xatlas utilization equivalence fails.
- Explicit 1M/4096 native-chart heavy gate asserts native-chart quality readiness while xatlas utilization equivalence fails.
- Docs describe the deficit fields, the utilization-equivalence target, and the continuing xatlas non-equivalence.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Heavy native-chart reference-target and explicit 1M/4096 gates now assert deficit fields and failed `xatlas_utilization_equivalence` while native-chart quality readiness remains true. Docs describe the deficit fields, `0.95` utilization target, and non-equivalence boundary. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy` -> 2 passed in 56.06s.
**Risks / next:** Full regression and artifact inspection still need to verify package/root/build hygiene.

### Slice 3: Regression Hygiene

**Objective:** Verify package/root/build stability and artifact cleanliness.

**Acceptance criteria:**
- Package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Status:** complete
**Evidence:** Final verify reran the plan-level checks: `git diff --check` -> passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 64 passed, 7 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed in 13.72s; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** none.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| XDDG-01 | Slice 1 |
| XDDG-02 | Slices 1, 2 |
| XDDG-03 | Slice 1 |
| XDDG-04 | Slice 2 |
| XDDG-05 | Slice 2 |
| XDDG-06 | Slice 3 |

## Execution Notes

- Do not add xatlas to the spatialkit package dependencies.
- Do not turn utilization equivalence into full parity readiness.
- Keep generated and heavy artifacts under `/tmp`.
