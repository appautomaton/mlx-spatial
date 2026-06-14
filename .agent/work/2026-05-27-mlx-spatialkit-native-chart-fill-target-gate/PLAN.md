# mlx-spatialkit Native Chart Fill Target Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-fill-target-gate/SPEC.md`: improve native chart UV fill toward the measured xatlas reference while preserving honest parity diagnostics.

## Architecture Approach

The measured bottleneck is chart-internal rect fill, not shelf packing or public chart angle. Tune the existing deterministic low-fill chart splitting policy in native C++, keep the split bounded, and verify the real fixture advances against the Phase 25 baseline.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Bounded Low-Fill Split Improvement

**Objective:** Tune native low-fill chart splitting so real reference-target chart fill advances beyond the Phase 25 baseline.

**Acceptance criteria:**
- Focused chart tests pass with updated bounded split metadata.
- Reference-target native-chart heavy gate reports chart rect fill above `0.5637785177491498` or xatlas utilization ratio above `0.6074138349759521`.
- Xatlas parity diagnostics remain measured and non-ready.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py::test_make_native_chart_uvs_splits_low_fill_l_shape_deterministically -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** tuned bounded low-fill splitting from threshold `0.65`/depth `2` to threshold `0.70`/depth `3`; focused split test passed; reference-target native-chart heavy gate passed. Latest diagnostics under `/tmp` show chart rect fill `0.5647715200751198` vs Phase 25 baseline `0.5637785177491498`, UV-surface occupancy `0.5095109939575195` vs `0.5047416687011719`, and xatlas utilization ratio `0.6131533138496904` vs `0.6074138349759521`, while `parity_ready=false`.
**Risks / next:** improvement is deliberately modest and bounded; explicit 1M/4096 gate still needs verification after the tuning.

### Slice 2: Upstream Gate And Docs

**Objective:** Keep explicit 1M/4096 native-chart readiness intact and document the measured fill improvement.

**Acceptance criteria:**
- Explicit 1M/4096 native-chart heavy gate still passes upstream-setting and native-chart readiness.
- Docs mention native chart fill improved toward xatlas metrics without claiming parity.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** explicit 1M/4096 native-chart heavy gate still passes after tuning; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy` -> 1 passed. Latest diagnostics show upstream-setting readiness true, native-chart quality-ready, rect fill `0.5528597490837637`, UV-surface occupancy `0.5047669410705566`, and parity still measured/non-ready.
**Risks / next:** docs updated; full regression/build verification remains.

### Slice 3: Regression Hygiene

**Objective:** Verify package/root/build stability and artifact cleanliness.

**Acceptance criteria:**
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Status:** complete
**Evidence:** `git diff --check` -> passed; docs grep found low-fill splitter and parity boundary language; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 64 passed, 7 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify should inspect latest real-fixture diagnostics before marking verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCFG-01 | Slice 1 |
| NCFG-02 | Slice 1 |
| NCFG-03 | Slice 1 |
| NCFG-04 | Slice 2 |
| NCFG-05 | Slice 2 |
| NCFG-06 | Slice 3 |

## Execution Notes

- Do not relax production readiness or xatlas parity diagnostics.
- Do not add xatlas as a spatialkit package dependency.
- Keep heavy artifacts under `/tmp`.
