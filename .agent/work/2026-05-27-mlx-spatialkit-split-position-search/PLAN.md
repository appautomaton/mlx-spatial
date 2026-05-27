# PLAN: mlx-spatialkit Split-Position Search

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-split-position-search/SPEC.md` by expanding native-chart low-fill split-position search and proving whether the real fixture benefits.

## Ordered Slice Sequence

### Slice 1: Five-Position Split Search

**Objective:** Expand low-fill split-position fractions from three to five deterministic positions.
**Acceptance criteria:**
- Native C++ uses five split-position fractions and diagnostics report `low_fill_split_position_candidates=5`.
- Focused GLB writer tests pass with updated deterministic partition-count expectations.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_glb_writer.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`
**Status:** complete
**Evidence:** Expanded native-chart split-position fractions to five deterministic positions and updated focused diagnostics assertions. Focused GLB writer verification passed: `12 passed`.
**Risks / next:** The denser search increases partition evaluations, so the real fixture must carry the quality justification.

### Slice 2: Real Fixture Benefit And Docs

**Objective:** Keep the denser search only if the real Pixal3D native-chart export improves utilization without readiness regressions.
**Acceptance criteria:**
- Heavy native-chart reference-target fixture remains artifact-ready and production-quality-ready.
- Xatlas-utilization ratio improves above `0.6973`, and tests assert the stronger measured threshold.
- Docs and roadmap state that this is chart-utilization progress, not xatlas parity.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-14723/diagnostics.json` showed chart count `39811`, chart fill `0.5964736059946982`, xatlas-utilization ratio `0.7004861241628259`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed.
**Risks / next:** This is measured native-chart progress, not xatlas equivalence; production equivalence remains blocked by xatlas parity and 1M setting parity where applicable.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none. This is a serial tuning change; the real fixture decides whether the code remains.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_glb_writer.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-split-position-search-verify`
