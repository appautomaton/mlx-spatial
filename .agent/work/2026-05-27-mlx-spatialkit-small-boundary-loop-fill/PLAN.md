# mlx-spatialkit Small Boundary Loop Fill Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-small-boundary-loop-fill/SPEC.md`: add bounded native small-loop geometry repair and prove it improves final Pixal3D export topology.

## Architecture Approach

Add a post-simplification repair step inside the topology-aware native simplifier. It will inspect boundary-edge components, accept only closed loops up to a small edge-count cap, triangulate them with existing vertices, and reject any patch that would exceed the remaining target-face budget or create degenerate, duplicate, or nonmanifold faces.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Small Loop Fill

**Objective:** Implement bounded small closed boundary-loop fill in the topology-aware simplifier with deterministic stats.

**Acceptance criteria:**
- Repair runs only for the topology-aware backend.
- Stats report enablement, max loop edges, face budget, considered/filled/rejected loop counts, and faces added.
- Focused tests show an interior small loop is filled while the large outer boundary remains open.
- Final focused metrics have fewer boundary loops and no nonmanifold edges.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`

**Status:** complete
**Evidence:** Added bounded small closed boundary-loop fill to the topology-aware simplifier. Focused mesh test fills one 4-edge interior loop, leaves the 24-edge outer boundary open, adds 2 faces, and keeps `nonmanifold_edges=0`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` -> 11 passed in 0.31s.
**Risks / next:** Repair is intentionally limited to 4-edge closed loops; larger/open boundaries remain deferred.

### Slice 2: Real Fixture Gate And Docs

**Objective:** Prove the repair improves Pixal3D reference-target final topology and update contracts/docs.

**Acceptance criteria:**
- Heavy reference-target export reports `boundary_loop_count < 2594`, `nonmanifold_edges=0`, and `final_faces <= 212542`.
- Production-quality readiness remains true.
- Xatlas chart parity/equivalence remain false.
- Docs and roadmap describe bounded small-loop repair and the remaining geometry/UV boundaries.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy && git diff --check`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`

**Status:** complete
**Evidence:** Heavy reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-87106/diagnostics.json` report `small_boundary_loops_filled=1115`, `small_boundary_loop_faces_added=1508`, `final_faces=200126 <= 212542`, `boundary_loop_count=1479 < 2594`, `boundary_edges=20084 < 23822`, and `nonmanifold_edges=0`. Combined verification `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m 'not heavy or heavy'` -> 12 passed in 13.91s. Xatlas parity remains false; utilization ratio is explicitly tracked at `0.680372125614308`. Final hygiene: `git diff --check` -> passed; package tests -> 65 passed, 7 deselected; root Pixal3D tests -> 35 passed in 14.02s; `/tmp` package build succeeded with wheel 10 entries bad 0 and sdist 36 entries bad 0.
**Risks / next:** Geometry topology improves with a known UV-utilization tradeoff; xatlas chart parity and larger/open-boundary repair remain open.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| SBLF-01 | Slice 1 |
| SBLF-02 | Slice 1 |
| SBLF-03 | Slice 2 |
| SBLF-04 | Slice 2 |
| SBLF-05 | Slice 2 |
| SBLF-06 | Slices 1-2 |
