# mlx-spatialkit Native Chart Upstream Settings Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-upstream-settings-gate/SPEC.md`: codify explicit 1M/4096 native-chart Pixal3D export readiness as a verified real-fixture gate.

## Architecture Approach

This is evidence hardening unless the gate exposes a misleading diagnostic. Reuse the existing native chart UV backend, binned Metal bake, upstream-setting readiness summary, visual comparison, and memory telemetry. Keep the 1024-reference visual mismatch explicit rather than turning it into a false pass.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native-Chart 1M/4096 Gate

**Objective:** Add a heavy real-fixture test for explicit `target_faces=1000000`, `texture_size=4096`, `uv_backend="native-chart"`.

**Acceptance criteria:**
- Test writes GLB and diagnostics under `/tmp`.
- Test asserts artifact readiness, upstream-setting readiness, native chart quality readiness, UV-bin guard passing, GLB viewer compatibility, and memory diagnostics.
- Test asserts visual comparison is not all-passed because the reference GLB is 1024 and lower-face, while deferred parity boundaries retain only xatlas chart parity.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy`

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added heavy explicit 1M/4096 native-chart gate; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy` -> 1 passed. The gate asserts artifact readiness, upstream-setting readiness, native chart quality readiness, UV-bin guard, GLB viewer compatibility, memory diagnostics, expected 1024-reference visual mismatch, and xatlas-only deferred parity.
**Risks / next:** this closes explicit 1M/4096 native-chart readiness, not xatlas chart parity.

### Slice 2: Docs And Regression Hygiene

**Objective:** Document the native-chart 1M/4096 readiness boundary and verify package/root/build stability.

**Acceptance criteria:**
- Docs mention native-chart 1M/4096 readiness without claiming xatlas chart parity or 1024-reference visual parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** documented explicit native-chart 1M/4096 readiness and visual/reference mismatch semantics; `git diff --check` -> passed; docs grep found native-chart 1M/4096 and 1024-reference boundary language; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 63 passed, 7 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify must rerun or inspect plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCUSG-01 | Slice 1 |
| NCUSG-02 | Slice 1 |
| NCUSG-03 | Slice 1 |
| NCUSG-04 | Slice 1 |
| NCUSG-05 | Slice 2 |
| NCUSG-06 | Slice 2 |

## Execution Notes

- Do not relax coverage thresholds.
- Keep generated and heavy artifacts under `/tmp`.
- Keep visual mismatch semantics explicit for 1M/4096 versus the checked-in 1024 reference GLB.
