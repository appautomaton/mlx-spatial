# mlx-spatialkit Chart UV Large Chart Split Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-large-chart-split-gate/SPEC.md`: split oversized native UV charts deterministically and prove the real chart candidate advances beyond the Phase 17 boundary.

## Architecture Approach

Keep normal-threshold chart grouping, local-frame/PCA projection, and shelf packing. After source chart grouping, split any chart above a fixed face target by sorting faces on a deterministic 3D centroid axis and chunking into bounded subcharts. This creates explicit native seams for oversized smooth regions while preserving all source faces and keeping the algorithm allocation-bounded.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Large-Chart Split

**Objective:** Implement deterministic oversized-chart splitting and split diagnostics in `make_native_chart_uvs`.

**Acceptance criteria:**
- Focused chart tests pass.
- Existing coplanar square and hard-crease tests remain unchanged.
- Synthetic oversized coplanar chart splits into multiple charts, preserves face count, and emits in-range UVs.
- Stats include pre-split chart count, post-split chart count, split target, and split count.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`

**Status:** complete
**Evidence:** implemented deterministic centroid-axis chunk splitting for oversized source charts and split diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q` -> 11 passed.
**Risks / next:** Slice 2 must prove the real fixture quality boundary advances.

### Slice 2: Real Boundary Proof

**Objective:** Prove the real Pixal3D native-chart export improves occupancy or moves the active blocker away from oversized charts.

**Acceptance criteria:**
- Heavy chart fixture writes GLB and diagnostics under `/tmp`.
- `quality.native_chart_uv_candidate.uv_surface_occupancy_ratio > 0.38708019256591797` or `stages.uv.stats.max_chart_faces <= stages.uv.stats.chart_split_max_faces`.
- Chart export still bakes through `metal-uv-binned-nearest`.
- Readiness diagnostics remain truthful if quality thresholds still fail.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** tightened heavy fixture assertions to the Phase 17 baseline and split diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics show `chart_count=9604`, `chart_split_count=16`, `oversized_source_chart_count=5`, `max_chart_faces=512`, `uv_bin_max_candidate_faces=460`, `uv_surface_occupancy=0.38801002502441406`, `global_coverage=0.24304580688476562`, still `status=quality_blocked`.
**Risks / next:** splitting reduced max chart size and candidate load, with only modest occupancy movement; the next blocker is chart fill/overlap quality, not oversized chart size.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document large-chart splitting and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe splitting as native chart candidate improvement, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe deterministic oversized-chart splitting and the non-xatlas boundary; final verify reran `git diff --check`, package tests -> 62 passed, 5 deselected, root Pixal3D tests -> 35 passed, `/tmp` build succeeded, and artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** none for this phase.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| LCSG-01 | Slice 1 |
| LCSG-02 | Slice 1 |
| LCSG-03 | Slice 2 |
| LCSG-04 | Slice 3 |
| LCSG-05 | Slice 3 |

## Execution Notes

- Do not change Metal bake behavior.
- Do not switch the default UV backend.
- Keep generated and heavy artifacts under `/tmp`.
