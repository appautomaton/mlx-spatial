# PLAN: mlx-spatialkit Open-Boundary Diagnostics

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-open-boundary-diagnostics/SPEC.md` by extending native topology metrics enough to guide the next open-boundary repair decision.

## Ordered Slice Sequence

### Slice 1: Native Open-Boundary Metrics

**Objective:** Extend native `mesh_metrics` with deterministic open-boundary component classifications.
**Acceptance criteria:**
- Closed-loop metrics keep their existing values on closed boundary fixtures.
- Open components report edge count, max open-component edges, endpoint count, branch-vertex count, simple open-chain count, branched open-component count, and small-open-component count/edge totals.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/mesh_metrics.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Added native open-boundary metrics and focused assertions for closed-loop and branched-open fixtures. Focused verification passed: `16 passed`.
**Risks / next:** This is diagnostic contract work only; it does not change repair behavior.

### Slice 2: Real Fixture Contract And Docs

**Objective:** Prove the Pixal3D real fixture carries the new open-boundary metrics and keep docs honest about what remains unrepaired.
**Acceptance criteria:**
- Heavy native-chart reference-target fixture records the new fields under `export_metrics.metrics`.
- Production-quality readiness, quality warnings, and existing topology thresholds do not regress.
- README, Pixal3D docs, scripts docs, and roadmap explain the open-boundary diagnostics without claiming repair.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-20068/diagnostics.json` showed `808` open components, `12622` open-boundary edges, `753` small open components, `0` simple open chains, `808` branched open components, `1894` branch vertices, nonmanifold edges `0`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed. Package verification passed: `73 passed, 7 deselected`; build artifacts were written under `/tmp/mlx-spatialkit-dist-open-boundary-diagnostics`.
**Risks / next:** The fixture shows branched open components rather than simple open chains, so repair should remain conservative and be framed separately.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-open-boundary-diagnostics`
