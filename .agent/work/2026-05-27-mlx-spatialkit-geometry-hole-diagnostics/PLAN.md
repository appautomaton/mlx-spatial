# mlx-spatialkit Geometry Hole Diagnostics Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-geometry-hole-diagnostics/SPEC.md`: make geometry-boundary hole risk measurable without changing export output or weakening UV/xatlas honesty.

## Architecture Approach

Add boundary-loop accounting inside the existing native `mesh_metrics` implementation. The export pipeline already records `source_metrics` and `export_metrics`, so the new fields should propagate without a separate Python topology pass.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Boundary Loop Metrics

**Objective:** Extend native mesh metrics with bounded boundary-loop diagnostics.

**Acceptance criteria:**
- `mesh_metrics` reports `boundary_loop_count`, `boundary_open_chain_count`, `boundary_small_loop_count`, `boundary_small_loop_edge_count`, and `boundary_max_loop_edges`.
- Focused tests cover a square boundary loop, an open-chain/nonmanifold case, and existing blocker behavior.
- Metrics remain native C++ and deterministic.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/mesh_metrics.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`

**Status:** complete
**Evidence:** Added native boundary topology accounting to `mesh_metrics` and focused assertions for closed boundary loops plus nonmanifold/open-chain behavior. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` -> 10 passed in 0.32s.
**Risks / next:** Diagnostics only; repair policy remains deferred until loop evidence is understood.

### Slice 2: Export Contract And Docs

**Objective:** Preserve geometry-loop metrics in Pixal3D export diagnostics and document the current boundary versus UV distinction.

**Acceptance criteria:**
- Heavy reference-target native-chart export asserts boundary-loop metrics under `export_metrics.metrics`.
- Existing production-quality and xatlas non-equivalence assertions remain intact.
- ROADMAP current state reflects the latest verified change and metrics.
- README/Pixal3D docs mention boundary-loop diagnostics as the current geometry-hole evidence.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy && git diff --check`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `.agent/steering/ROADMAP.md`

**Status:** complete
**Evidence:** Heavy reference-target diagnostics now expose final topology: `/tmp/mlx-spatialkit-native-chart-reference-target-export-81671/diagnostics.json` reports `boundary_edges=23822`, `boundary_vertices=21851`, `boundary_loop_count=2594`, `boundary_open_chain_count=808`, `boundary_small_loop_count=2594`, `boundary_max_loop_edges=18`, `boundary_max_component_edges=205`, and `nonmanifold_edges=0`. Combined verification `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m 'not heavy or heavy'` -> 11 passed in 14.09s. Final hygiene: `git diff --check` -> passed; package tests -> 64 passed, 7 deselected; root Pixal3D tests -> 35 passed in 14.02s; `/tmp` wheel/sdist build succeeded with wheel 10 entries bad 0 and sdist 36 entries bad 0.
**Risks / next:** Geometry evidence is now measurable; actual hole repair and UV normal-drift chart splitting remain future quality work.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| GHD-01 | Slice 1 |
| GHD-02 | Slice 1 |
| GHD-03 | Slice 2 |
| GHD-04 | Slice 2 |
| GHD-05 | Slice 2 |
| GHD-06 | Slices 1-2 |
