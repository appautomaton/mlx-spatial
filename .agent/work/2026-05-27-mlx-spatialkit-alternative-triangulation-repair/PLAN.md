# PLAN: mlx-spatialkit Alternative Triangulation Repair

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-alternative-triangulation-repair/SPEC.md` by adding guarded alternate triangulation for topology-blocked small-loop patches.

## Ordered Slice Sequence

### Slice 1: Native Alternative Triangulation

**Objective:** Add bounded alternate ear-clipping triangulation search behind existing topology guards.
**Acceptance criteria:**
- Alternate triangulation is attempted only after primary projected ear-clipping is rejected for duplicate or nonmanifold topology.
- Alternate patches still use the same degenerate, duplicate, nonmanifold, and face-budget guards.
- Diagnostics report alternative-triangulation attempt and fill counts.
- Synthetic tests cover a topology-blocked primary diagonal repaired by an alternate diagonal.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Added bounded alternate ear-clipping variants, reused existing degenerate/duplicate/nonmanifold guards, exposed attempt/fill diagnostics, and added a topology-blocked quad fixture. Focused verification passed: `20 passed`.
**Risks / next:** The search remains capped at 256 variants per loop; real-fixture verification decides whether the quality/UV tradeoff is acceptable.

### Slice 2: Real Fixture Gate And Docs

**Objective:** Keep the alternate triangulation path only if the real Pixal3D fixture improves topology without production-quality regression.
**Acceptance criteria:**
- Heavy fixture has lower boundary-loop and open-boundary counts than the current baseline.
- `nonmanifold_edges=0`, no quality warnings, and `production_quality_ready=true` remain true.
- Docs describe alternate triangulation as a guarded repair path, not remesh, open-chain repair, xatlas parity, or production equivalence.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-28243/diagnostics.json` showed boundary edges `8882`, boundary loops `946`, open components `185`, open-boundary edges `3599`, branch vertices `329`, alternative-triangulation fills `249`, nonmanifold edges `0`, xatlas-utilization ratio `0.6971257574077158`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed. Package verification passed: `77 passed, 7 deselected`; build artifacts were written under `/tmp/mlx-spatialkit-dist-alternative-triangulation-repair`.
**Risks / next:** UV occupancy moved lower while topology improved; xatlas parity and 1M export parity remain deferred.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-alternative-triangulation-repair`
