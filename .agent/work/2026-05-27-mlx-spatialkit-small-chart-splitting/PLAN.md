# PLAN: mlx-spatialkit Small-Chart UV Splitting

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-small-chart-splitting/SPEC.md` by tightening native-chart low-fill splitting around small charts and proving the real fixture improves without readiness regressions.

## Ordered Slice Sequence

### Slice 1: Small-Chart Split Policy

**Objective:** Let native-chart low-fill splitting evaluate 4- and 5-face charts with 2-face minimum children.
**Acceptance criteria:**
- Native C++ split thresholds change from 6/3 to 4/2 and diagnostics report the new values.
- Focused GLB writer tests cover deterministic low-fill splitting and the new threshold values.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_glb_writer.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`
**Status:** complete
**Evidence:** Updated native-chart low-fill thresholds to `low_fill_split_min_faces=4` and `low_fill_split_min_child_faces=2`; focused GLB writer verification passed: `12 passed`.
**Risks / next:** More split charts increase duplicated vertices, so the real fixture gate must prove occupancy/readiness benefit.

### Slice 2: Real Fixture And Docs

**Objective:** Prove the real Pixal3D native-chart export benefits from the smaller split threshold and document the measured boundary.
**Acceptance criteria:**
- Heavy native-chart reference-target fixture remains artifact-ready and production-quality-ready.
- Xatlas-utilization ratio improves above the current `0.6826` baseline and real-fixture tests assert the stronger threshold.
- Docs and roadmap explain this as chart-utilization progress, not xatlas parity.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-11875/diagnostics.json` showed chart count `39213`, chart fill `0.591078906593391`, xatlas-utilization ratio `0.6973151223375631`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed. Package verification passed: `73 passed, 7 deselected`; native build artifacts were written under `/tmp/mlx-spatialkit-dist-small-chart-splitting`.
**Risks / next:** Xatlas chart parity still remains false; this is measured progress against the gap, not equivalence.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none. The native-chart policy and fixture thresholds need serial tuning against real output.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_glb_writer.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-small-chart-splitting`
