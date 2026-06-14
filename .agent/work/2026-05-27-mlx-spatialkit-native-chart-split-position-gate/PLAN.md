# mlx-spatialkit Native Chart Split Position Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-split-position-gate/SPEC.md`: add bounded split-position search to native low-fill chart splitting and verify real fixture progress.

## Architecture Approach

Keep the existing low-fill threshold, minimum child size, and max depth. For each eligible chart, evaluate three deterministic split positions on each local centroid axis, choose the best improving split, and expose both axis and partition candidate counts in diagnostics.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Bounded Split-Position Search

**Objective:** Implement fixed-position low-fill split search and prove the reference-target native-chart boundary advances.

**Acceptance criteria:**
- Focused chart test passes with split-position and partition-candidate diagnostics.
- Reference-target native-chart heavy gate reports chart rect fill above `0.5670824417746222` or xatlas utilization ratio above `0.6202011322387381`.
- Parity diagnostics remain measured and non-ready.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py::test_make_native_chart_uvs_splits_low_fill_l_shape_deterministically -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** implemented bounded three-position split search with diagnostics; focused split test passed; reference-target native-chart heavy gate passed. Named `/tmp` probe reports chart rect fill `0.5764121465152018` vs Phase 27 `0.5670824417746222`, UV-surface occupancy `0.5396347045898438`, xatlas utilization ratio `0.6494046474199308` vs Phase 27 `0.6202011322387381`, partition candidates `36186`, evaluated partitions `31932`, and `parity_ready=false`.
**Risks / next:** explicit 1M/4096 gate and full package hygiene remain.

### Slice 2: Upstream Gate, Docs, And Package Hygiene

**Objective:** Keep explicit 1M/4096 native-chart readiness intact, document the split-position boundary, and verify package hygiene.

**Acceptance criteria:**
- Explicit 1M/4096 native-chart heavy gate still passes upstream-setting and native-chart readiness.
- Docs mention bounded split-position search without claiming xatlas equivalence.
- Full package tests, root Pixal3D tests, build, and artifact inspection pass with generated outputs under `/tmp`.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** `git diff --check` passed; explicit 1M/4096 native-chart heavy gate passed in `45.22s`; package tests passed `64 passed, 7 deselected`; root Pixal3D tests passed `35 passed`; `/tmp` build produced wheel and sdist; artifact inspection reported wheel `bad 0` and sdist `bad 0`; docs grep found split-position and partition-candidate language.
**Risks / next:** final verify should re-run the plan checks before marking verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCSP-01 | Slice 1 |
| NCSP-02 | Slice 1 |
| NCSP-03 | Slice 1 |
| NCSP-04 | Slice 2 |
| NCSP-05 | Slice 2 |
| NCSP-06 | Slice 2 |

## Execution Notes

- Keep generated and heavy artifacts under `/tmp`.
- Do not relax thresholds or parity readiness.
- If real fixture metrics do not improve, revert the algorithm change rather than preserving noisy diagnostics.
