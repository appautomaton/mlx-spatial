# mlx-spatialkit Chart Rect Fill Rotation Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-chart-rect-fill-rotation-gate/SPEC.md`: improve native chart rect fill with a finer bounded projection rotation search and prove real fixture quality advances.

## Architecture Approach

Keep chart grouping, oversized-chart splitting, and shelf packing unchanged. Replace the fixed 7-angle projection candidate array with a deterministic 19-candidate PCA-centered search at 5-degree steps across the half-quadrant. Preserve deterministic tie-breaking and expose the candidate count and step size in UV stats.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Bounded Rotation Search

**Objective:** Implement 19-candidate deterministic projection rotation search and focused test updates.

**Acceptance criteria:**
- Focused chart tests pass with expanded projection diagnostics.
- Coplanar, hard-crease, rotated-rectangle, and oversized-chart tests remain valid.
- Stats include projection candidate count and rotation step degrees.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`

**Status:** complete
**Evidence:** replaced 7-candidate projection search with deterministic 19-candidate 5-degree PCA-centered search and stats; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q` -> 11 passed.
**Risks / next:** Slice 2 must prove the real fixture advances over the Phase 19 baseline.

### Slice 2: Real Rect-Fill Proof

**Objective:** Prove the real Pixal3D native-chart export improves chart rect fill or global coverage.

**Acceptance criteria:**
- Heavy chart fixture writes GLB and diagnostics under `/tmp`.
- `stages.uv.stats.chart_rect_fill_ratio > 0.5649183023244753` or `quality.native_chart_uv_candidate.global_coverage_ratio > 0.36844539642333984`.
- Native chart still clears the UV occupancy floor and keeps truthful remaining blockers.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** tightened heavy fixture assertions to the Phase 19 baseline; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics show `projection_rotation_candidates=19`, `chart_rect_fill=0.5727071617508422`, `uv_surface_occupancy=0.5145282745361328`, `global_coverage=0.3883981704711914`, and remaining blockers `["global_coverage_floor"]`.
**Risks / next:** improvement is measurable but global coverage remains below the 0.50 floor.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document bounded rotation search and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe bounded projection rotation search as a native candidate improvement, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** documented 19-candidate 5-degree bounded projection rotation search and the native-chart parity boundary; `git diff --check` -> passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 62 passed, 5 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify must rerun plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| CRFG-01 | Slice 1 |
| CRFG-02 | Slice 1 |
| CRFG-03 | Slice 2 |
| CRFG-04 | Slice 3 |
| CRFG-05 | Slice 3 |

## Execution Notes

- Do not change Metal bake behavior.
- Do not change chart readiness floors.
- Keep generated and heavy artifacts under `/tmp`.
