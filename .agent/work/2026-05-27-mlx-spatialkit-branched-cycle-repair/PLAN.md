# PLAN: mlx-spatialkit Branched-Cycle Repair

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-branched-cycle-repair/SPEC.md` by adding bounded native repair for small simple cycles inside branched boundary components.

## Ordered Slice Sequence

### Slice 1: Native Branched-Cycle Repair

**Objective:** Extract small simple cycles from branched boundary components and apply existing guarded patching to those cycles.
**Acceptance criteria:**
- Existing closed-loop repair behavior remains intact.
- Branched-cycle candidates are deterministic and bounded.
- Stats expose branch-cycle candidate, fill, reject, and budget-limited counts.
- A synthetic pinched/branched boundary fixture repairs without nonmanifold regressions.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Added bounded 4-edge branch-cycle extraction, guarded patch application, two-pass repair aggregation, and a synthetic pinched-hole fixture. Focused verification passed: `17 passed`.
**Risks / next:** Candidate extraction remains conservative; rejected and remaining branch topology stays visible in diagnostics.

### Slice 2: Real Fixture Gate And Docs

**Objective:** Keep the repair only if the real Pixal3D native-chart fixture improves topology without readiness regressions.
**Acceptance criteria:**
- Heavy fixture has lower branched open-boundary counts or lower open-boundary edge totals than the current baseline.
- `nonmanifold_edges=0`, no quality warnings, and `production_quality_ready=true` remain true.
- README, Pixal3D docs, scripts docs, and roadmap state that this is branched-cycle repair, not open-chain repair or production equivalence.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-21751/diagnostics.json` showed boundary edges `12016`, open components `353`, open-boundary edges `6340`, branch vertices `715`, branch-cycle fills `1065`, nonmanifold edges `0`, xatlas-utilization ratio `0.7037718926122386`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed. Package verification passed: `74 passed, 7 deselected`; build artifacts were written under `/tmp/mlx-spatialkit-dist-branched-cycle-repair`.
**Risks / next:** Closed-loop count rises as branch components are decomposed; remaining closed loops and branch rejections are still visible and should be handled separately.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-branched-cycle-repair`
