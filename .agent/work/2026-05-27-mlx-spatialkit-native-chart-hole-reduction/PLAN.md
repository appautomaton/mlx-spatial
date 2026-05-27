# mlx-spatialkit Native Chart Hole Reduction Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-hole-reduction/SPEC.md`: reduce native-chart UV holes with real-fixture proof while keeping xatlas non-equivalence explicit.

## Architecture Approach

Treat this as one quality-improvement program with small execution slices. The first implementation target is chart-local fill, because current diagnostics show shelf packing is already near full and chart-angle probing did not beat the `45.0` degree baseline.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Hole Baseline And Lever Selection

**Objective:** Record the real-fixture hole baseline and select the next bounded implementation lever.

**Acceptance criteria:**
- PLAN records the chart-angle probe result.
- The active blocker is identified as chart-local fill rather than shelf packing or public chart angle.
- Implementation slice targets a bounded native C++ chart policy change.

**Verification:** inspect `/tmp/mlx-spatialkit-chart-angle-probe` diagnostics and the latest native reference-target diagnostics; confirm `45.0` remains best among probed angles and `shelf_packing_efficiency > 0.99`.

**Touches:** `.agent/work/2026-05-27-mlx-spatialkit-native-chart-hole-reduction/PLAN.md`

**Status:** complete
**Evidence:** `/tmp/mlx-spatialkit-chart-angle-probe` compared `30,45,60,75,90` degrees. `45.0` remained best: `uv_surface_occupancy_ratio=0.5693244934082031`, `xatlas_utilization_ratio=0.685133792850289`, and `shelf_packing_efficiency=0.9952422592195517`. Other angles ranged from `0.5475645065307617` to `0.5622367858886719` occupancy. The active blocker is chart-local fill (`chart_rect_fill_ratio=0.5764121465152018`), not public chart angle or shelf packing.
**Risks / next:** Slice 2 should improve projection/splitting policy and gate on real UV-surface occupancy, not just local rect fill.

### Slice 2: Native Chart Fill Improvement

**Objective:** Implement one bounded native chart-generation policy or C++ improvement that raises real-fixture UV-surface occupancy or xatlas-utilization ratio.

**Acceptance criteria:**
- Focused tests cover the changed chart policy deterministically.
- Heavy reference-target native-chart export improves beyond baseline: `uv_surface_occupancy_ratio > 0.5693244934082031` or `xatlas_utilization_ratio > 0.685133792850289`.
- Existing readiness and honesty contracts hold: `production_quality_ready=true`, `parity_ready=false`, `xatlas_utilization_equivalence=false`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m 'not heavy or heavy'`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Rejected two local-metric traps: deeper five-position low-fill splitting raised `chart_rect_fill_ratio` to `0.5848910188914018` but lowered real occupancy to `0.5640239715576172`; finer projection search raised rect fill to `0.5795091805206093` but lowered occupancy to `0.5610494613647461`. Kept the bounded native-chart padding policy instead: default `tile_padding` changed from `0.005` to `0.001`, focused resolver test passed, and heavy reference-target gate passed. Latest `/tmp/mlx-spatialkit-native-chart-reference-target-export-75645/diagnostics.json` reports `uv_surface_occupancy_ratio=0.5768346786499023`, `xatlas_utilization_ratio=0.6941716645020964`, `visual_comparison.summary.all_passed=true`, and `production_quality_ready=true`. Slice verification `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m 'not heavy or heavy'` -> 13 passed in 12.50s.
**Risks / next:** This reduces padding holes but does not solve chart-local fill; xatlas parity remains false by design.

### Slice 3: Docs And Regression Hygiene

**Objective:** Document the improved chart policy and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs describe the new bounded chart policy and still-open xatlas boundary.
- Package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Final verify reran from `stage=verify`. Docs now state native-chart default padding is `0.001`. Focused resolver test -> 1 passed; slice verification -> 13 passed in 18.17s; `git diff --check` -> passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 64 passed, 7 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed in 13.62s; `/tmp` package build succeeded; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** none.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCHR-01 | Slice 1 |
| NCHR-02 | Slice 1 |
| NCHR-03 | Slice 2 |
| NCHR-04 | Slice 2 |
| NCHR-05 | Slice 2 |
| NCHR-06 | Slice 3 |
| NCHR-07 | Slice 3 |

## Execution Notes

- Do not add xatlas to `packages/mlx-spatialkit` dependencies.
- Do not claim xatlas parity from a bounded occupancy improvement.
- Keep generated and heavy artifacts under `/tmp`.
