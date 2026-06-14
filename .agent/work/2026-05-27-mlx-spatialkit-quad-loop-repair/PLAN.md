# PLAN: mlx-spatialkit Quad Loop Repair Default

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-quad-loop-repair/SPEC.md` by making bounded triangle/quad loop repair the default topology-aware native export policy.

## Ordered Slice Sequence

### Slice 1: Default Repair Contract

**Objective:** Align Python/export/native defaults and unit tests around cap-4 small-loop repair.
**Acceptance criteria:**
- Defaults in export, Python wrapper, and nanobind binding are `4`.
- Default topology-aware simplification fills a 4-edge loop.
- Explicit cap `3` preserves the 4-edge loop.
- Disable and invalid-cap behavior remain covered.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/mesh.py`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Defaults now resolve to cap `4` in export, Python wrapper, and native binding. Mesh tests prove default quad filling, explicit cap-3 preservation, disable behavior, and no nonmanifold blockers. Verification passed: `15 passed`.
**Risks / next:** none.

### Slice 2: Real Fixture And Docs Alignment

**Objective:** Prove the cap-4 policy improves real Pixal3D geometry diagnostics and document the bounded scope.
**Acceptance criteria:**
- Native-chart reference-target heavy fixture asserts cap `4`, fewer final boundary loops than the cap-3 baseline, and no nonmanifold export blockers.
- README, Pixal3D docs, and script docs describe triangle/quad repair and keep full remesh/xatlas parity out of scope.
- Automaton roadmap/status remains compact and verified.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k "reference_target_native_chart_backend_reports_readiness"`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Real native-chart reference-target fixture reports cap `4`, `1115` loops filled, `1479` final boundary loops, and `nonmanifold_edges=0`; docs and roadmap describe bounded triangle/quad repair. Verification passed: `1 passed, 15 deselected`; package suite passed: `71 passed, 7 deselected`; `/tmp/mlx-spatialkit-dist-quad-loop` wheel/sdist built and archive checks passed.
**Risks / next:** arbitrary N-gon repair and xatlas parity remain deferred.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none. The policy name, default, diagnostics, and docs need one consistent vocabulary.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k "reference_target_native_chart_backend_reports_readiness"`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-quad-loop`
