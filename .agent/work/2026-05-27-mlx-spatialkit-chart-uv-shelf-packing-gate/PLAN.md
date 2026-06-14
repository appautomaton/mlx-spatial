# mlx-spatialkit Chart UV Shelf Packing Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-shelf-packing-gate/SPEC.md`: replace equal-grid native chart UV packing with deterministic aspect-aware shelf packing and prove real chart export occupancy improves.

## Architecture Approach

Keep chart grouping unchanged. During native chart UV construction, collect each chart's projected bounding rectangle and local vertex map first, then sort charts by normalized rectangle height and pack them into horizontal shelves using a binary-searched global scale. Emit UVs after packing, with stats for row count, packed bounds, packing efficiency, and the legacy grid dimensions removed from the chart path.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Shelf Packer

**Objective:** Implement deterministic aspect-aware shelf packing in `make_native_chart_uvs`.

**Acceptance criteria:**
- Focused chart tests pass with `packing=aspect-shelf-charts`.
- Coplanar square still produces one chart and reuses four vertices.
- Hard crease still produces multiple charts.
- Stats include shelf row count, packing efficiency, packed bounds, and duplicate ratio.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`

**Status:** complete
**Evidence:** replaced equal-grid chart packing with deterministic aspect-aware shelf packing and shelf diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q` -> 8 passed.
**Risks / next:** Slice 2 must prove real fixture occupancy improves over the Phase 15 baseline.

### Slice 2: Real Occupancy Proof

**Objective:** Prove the real Pixal3D native-chart export improves UV-surface occupancy over the Phase 15 baseline.

**Acceptance criteria:**
- Heavy chart fixture writes a GLB and diagnostics under `/tmp`.
- `quality.native_chart_uv_candidate.uv_surface_occupancy_ratio > 0.23263072967529297`.
- Chart export still bakes through `metal-uv-binned-nearest`.
- Readiness diagnostics remain truthful if quality thresholds still fail.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** heavy chart fixture passed with shelf packing; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics show `packing=aspect-shelf-charts`, `uv_surface_occupancy=0.3310232162475586`, `global_coverage=0.20439910888671875`, `shelf_packing_efficiency=0.990489573026458`, still `status=quality_blocked`.
**Risks / next:** shelf packing improves utilization but does not yet meet the 0.50 chart readiness floors.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document shelf packing and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe shelf packing as native chart candidate improvement, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe aspect-aware shelf packing and the non-xatlas boundary; `git diff --check` passed; package tests -> 59 passed, 5 deselected; root Pixal3D tests -> 35 passed; `/tmp` build succeeded; artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** final verify must rerun plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| CUVP-01 | Slice 1 |
| CUVP-02 | Slice 1 |
| CUVP-03 | Slice 2 |
| CUVP-04 | Slice 3 |
| CUVP-05 | Slice 3 |

## Execution Notes

- Do not change chart grouping thresholds or Metal bake behavior in this phase.
- Do not switch the default UV backend.
- Keep generated and heavy artifacts under `/tmp`.
