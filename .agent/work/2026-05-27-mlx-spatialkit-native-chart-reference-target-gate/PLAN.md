# mlx-spatialkit Native Chart Reference-Target Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-reference-target-gate/SPEC.md`: codify the reference-target native-chart Pixal3D path as a verified real-fixture gate.

## Architecture Approach

This is evidence hardening, not a runtime algorithm change. Reuse the existing native chart backend, texture surface fill, visual comparison, and production threshold diagnostics; add one explicit heavy fixture gate and docs that distinguish readiness from xatlas/upstream-setting parity.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Reference-Target Native-Chart Gate

**Objective:** Add a heavy real-fixture test for `quality_preset="reference-target"` with `uv_backend="native-chart"`.

**Acceptance criteria:**
- Test writes GLB and diagnostics under `/tmp`.
- Test asserts `production_quality_ready=true`, `native_chart_uv_candidate.status=quality_ready`, visual comparison `all_passed=true`, UV-bin guard passing, and xatlas/1M parity boundaries deferred.
- Test asserts raw/exact/final coverage and surface-fill diagnostics remain separate.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy`

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added heavy reference-target native-chart gate; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy` -> 1 passed. The gate asserts production readiness, native chart quality readiness, raw/exact/final coverage separation, visual comparison pass, and deferred xatlas/1M parity boundaries.
**Risks / next:** this closes reference-target native-chart readiness only; 1M/4096 native-chart remains deferred.

### Slice 2: Docs And Regression Hygiene

**Objective:** Document the reference-target native-chart readiness boundary and verify package/root/build stability.

**Acceptance criteria:**
- Docs mention reference-target native-chart readiness but do not claim xatlas chart parity or 1M/4096 upstream-setting parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** documented reference-target native-chart readiness and parity boundaries; `git diff --check` -> passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 63 passed, 6 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify must rerun plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCRTG-01 | Slice 1 |
| NCRTG-02 | Slice 1 |
| NCRTG-03 | Slice 1 |
| NCRTG-04 | Slice 1 |
| NCRTG-05 | Slice 2 |
| NCRTG-06 | Slice 2 |

## Execution Notes

- Do not change runtime behavior unless Slice 1 exposes a real bug.
- Keep generated and heavy artifacts under `/tmp`.
