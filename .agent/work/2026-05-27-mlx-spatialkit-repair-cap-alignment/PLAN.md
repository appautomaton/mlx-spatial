# PLAN: mlx-spatialkit Repair Cap Alignment

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-repair-cap-alignment/SPEC.md` by aligning native small-boundary repair caps and verifying the real Pixal3D fixture.

## Ordered Slice Sequence

### Slice 1: Native Repair Cap Contract

**Objective:** Make closed-loop fallback and branched-cycle repair obey a coherent effective-cap policy.
**Acceptance criteria:**
- `small_boundary_loop_fill_max_edges` bounds branch-cycle extraction.
- Centroid fallback policy cap is 8 edges.
- Branched-cycle policy cap is 6 edges.
- Diagnostics report policy and effective caps.
- Synthetic tests cover cap-limited branch cycles and expanded fallback behavior.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Updated native effective-cap policy, added policy/effective cap diagnostics, added synthetic cap-limited branched quad and 8-edge centroid-fallback fixtures. Focused verification passed: `19 passed`.
**Risks / next:** Larger repaired patches remain guarded by duplicate/nonmanifold checks; real fixture gate decides whether the policy stays.

### Slice 2: Real Fixture Gate And Docs

**Objective:** Keep the policy change only if the real Pixal3D fixture improves topology without production-quality regression.
**Acceptance criteria:**
- Heavy fixture has lower open-boundary component and edge counts than the current baseline.
- `nonmanifold_edges=0`, no quality warnings, and `production_quality_ready=true` remain true.
- Docs describe policy/effective cap behavior without claiming open-chain repair, xatlas parity, or production equivalence.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-24906/diagnostics.json` showed boundary edges `10070`, open components `216`, open-boundary edges `4133`, branch vertices `403`, branch-cycle fills `1427`, nonmanifold edges `0`, xatlas-utilization ratio `0.7024394521058578`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed. Package verification passed: `76 passed, 7 deselected`; build artifacts were written under `/tmp/mlx-spatialkit-dist-repair-cap-alignment`.
**Risks / next:** UV occupancy moved slightly lower while open-boundary topology improved; xatlas parity and 1M export parity remain deferred.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-repair-cap-alignment`
