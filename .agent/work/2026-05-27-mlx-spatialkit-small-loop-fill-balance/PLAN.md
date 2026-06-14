# mlx-spatialkit Small Loop Fill Balance Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-small-loop-fill-balance/SPEC.md`: tune small-loop repair to triangular loops so UV quality improves versus the cap-4 repair while geometry still improves versus no repair.

## Architecture Approach

Keep the existing native repair implementation and tighten the default loop cap from 4 edges to 3 edges. This preserves the deterministic target-face and topology guards while narrowing repair to the smallest closed holes.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Triangular Loop Fill Policy

**Objective:** Change the bounded repair cap to triangular loops and update deterministic focused coverage.

**Acceptance criteria:**
- Native stats report `small_boundary_loop_fill_max_edges == 3`.
- Focused tests prove one triangular hole is filled.
- Focused tests prove a 4-edge hole remains unfilled.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`

**Status:** complete
**Evidence:** Tightened native repair cap to triangular loops (`small_boundary_loop_fill_max_edges=3`). Focused tests now prove a triangular interior loop is filled and a 4-edge loop is preserved. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` -> 12 passed in 0.21s.
**Risks / next:** Fills fewer holes than cap 4 by design; heavy gate must prove the UV balance improves.

### Slice 2: Heavy Balance Gate And Docs

**Objective:** Prove the cap-3 policy improves UV utilization versus cap 4 while retaining real topology improvement.

**Acceptance criteria:**
- Heavy reference-target export reports `boundary_loop_count < 2594`.
- Heavy reference-target export reports `xatlas_utilization_ratio > 0.680372125614308`.
- Production-quality readiness and deterministic visual comparison pass.
- Xatlas parity/equivalence remain false.
- Docs and roadmap explain the cap-3 balance.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy && git diff --check`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`

**Status:** complete
**Evidence:** Heavy reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-89495/diagnostics.json` report `small_boundary_loops_filled=722`, `small_boundary_loop_faces_added=722`, `boundary_loop_count=1872 < 2594`, `nonmanifold_edges=0`, `visual_all=true`, and xatlas utilization ratio `0.6828063257125282 > 0.680372125614308`. Heavy gate `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy` -> 1 passed in 8.13s. Final hygiene: `git diff --check` -> passed; package tests -> 66 passed, 7 deselected; root Pixal3D tests -> 35 passed in 9.30s; `/tmp` package build succeeded with wheel 10 entries bad 0 and sdist 36 entries bad 0.
**Risks / next:** Still below the no-repair UV utilization; xatlas parity remains false.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| SLFB-01 | Slice 1 |
| SLFB-02 | Slice 1 |
| SLFB-03 | Slice 2 |
| SLFB-04 | Slice 2 |
| SLFB-05 | Slice 2 |
| SLFB-06 | Slice 2 |
