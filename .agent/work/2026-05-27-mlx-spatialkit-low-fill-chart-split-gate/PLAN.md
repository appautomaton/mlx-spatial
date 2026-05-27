# mlx-spatialkit Low-Fill Chart Split Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-low-fill-chart-split-gate/SPEC.md`: improve native chart rect fill with deterministic low-fill chart splitting and prove real fixture quality advances.

## Architecture Approach

Keep chart grouping, oversized-chart splitting, local-frame/PCA projection, rotation search, and shelf packing intact. Add a bounded low-fill splitting pass for charts whose projected triangle area under-fills their local rectangle. Split such charts deterministically by projected/centroid footprint, expose pre/post split diagnostics, and keep UV-bin guardrails visible.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Low-Fill Split Policy

**Objective:** Implement deterministic bounded low-fill chart splitting with focused test coverage.

**Acceptance criteria:**
- Low-fill chart stats expose policy threshold, minimum face count, source/split counts, and pre/post rect-fill diagnostics.
- Focused low-fill chart test proves splitting occurs, output is deterministic on repeated calls, and rect fill improves versus the pre-split diagnostic.
- Existing coplanar, hard-crease, rotated-rectangle, and oversized-chart tests remain valid.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`

**Status:** complete
**Evidence:** implemented deterministic bounded low-fill chart splitting with pre/post rect-fill stats and split guard diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q` -> 12 passed.
**Risks / next:** Slice 2 must prove the real fixture improves without hiding remaining blockers.

### Slice 2: Real Fixture Quality Proof

**Objective:** Prove the real Pixal3D native-chart export improves chart rect fill or global coverage beyond the Phase 20 baseline.

**Acceptance criteria:**
- Heavy chart fixture writes GLB and diagnostics under `/tmp`.
- `stages.uv.stats.chart_rect_fill_ratio > 0.5727071617508422` or `quality.native_chart_uv_candidate.global_coverage_ratio > 0.3883981704711914`.
- Native chart still keeps UV-bin guard passing and reports remaining blockers without threshold relaxation.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** tightened heavy fixture assertions to the Phase 20 baseline and low-fill split diagnostics; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics show `low_fill_chart_split_count=392`, `chart_rect_fill=0.576915029519614`, `uv_surface_occupancy=0.5232105255126953`, `global_coverage=0.3953571319580078`, `uv_bin_max_candidate_faces=479`, and remaining blockers `["global_coverage_floor"]`.
**Risks / next:** global coverage is measurably improved but still below the 0.50 floor.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document low-fill splitting and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe low-fill splitting as a native chart candidate improvement, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** documented bounded low-fill chart splitting and the native-chart parity boundary; `git diff --check` -> passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 63 passed, 5 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify must rerun plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| LFCS-01 | Slice 1 |
| LFCS-02 | Slice 1 |
| LFCS-03 | Slice 2 |
| LFCS-04 | Slice 2 |
| LFCS-05 | Slice 3 |
| LFCS-06 | Slice 3 |

## Execution Notes

- Do not change Metal bake behavior.
- Do not change chart readiness floors.
- Keep generated and heavy artifacts under `/tmp`.
