# PLAN: mlx-spatialkit Ear-Clip Small-Hole Repair

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-earclip-hole-repair/SPEC.md` by improving native bounded hole repair and proving it against focused tests plus the real Pixal3D fixture.

## Ordered Slice Sequence

### Slice 1: Native Ear-Clipping Repair

**Objective:** Replace fan-only boundary-loop patching with projected ear-clipping while preserving topology safety checks.
**Acceptance criteria:**
- Small-loop repair triangulates non-tri/quad closed loops without creating degenerate, duplicate, or nonmanifold faces.
- Stats expose the fill algorithm and continue to report considered, filled, rejected, budget-limited, and faces-added counts.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Added projected ear-clipping repair in native C++ with a `small_boundary_loop_fill_algorithm` diagnostic and focused tests for 8-edge concave loop fill, cap behavior, disabled fill, and invalid cap validation. Focused verification passed: `16 passed`.
**Risks / next:** Larger or invalid loops are still rejected; this is bounded small-hole repair, not full remeshing.

### Slice 2: Pixal3D Default Cap And Docs

**Objective:** Widen the Pixal3D default repair cap to 8 edges only after the real fixture stays manifold and shows fewer boundary holes.
**Acceptance criteria:**
- Export defaults, diagnostics assertions, README, Pixal3D docs, and scripts docs describe `small_boundary_loop_fill_max_edges=8`.
- Real-fixture evidence is recorded in this plan and shows no export-blocking topology regression.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/mesh.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Updated Python/native-binding defaults and docs to `small_boundary_loop_fill_max_edges=8`. Real-fixture cap probe at `/tmp/mlx-spatialkit-earclip-hole-fill-cap-probe.json` showed cap 8 reduced boundary loops to `1089`, boundary edges to `18348`, and kept nonmanifold edges at `0`; heavy native-chart reference-target verification passed with tightened thresholds. Package verification passed: `73 passed, 7 deselected`; build artifacts were written under `/tmp/mlx-spatialkit-dist-earclip-hole-repair`.
**Risks / next:** Production equivalence still remains blocked on xatlas chart parity and upstream 1M/4096 parity where applicable.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none. The subagent pool is currently unavailable, and the C++ repair plus tests are tightly coupled.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-earclip-hole-repair`
