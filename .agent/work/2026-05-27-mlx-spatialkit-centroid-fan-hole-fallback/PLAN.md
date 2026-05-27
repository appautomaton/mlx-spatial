# PLAN: mlx-spatialkit Centroid-Fan Hole Fallback

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-centroid-fan-hole-fallback/SPEC.md` by adding a guarded native fallback for rejected small closed-loop patches and proving the real fixture benefits.

## Ordered Slice Sequence

### Slice 1: Native Fallback And Diagnostics

**Objective:** Add centroid-fan fallback patching after projected ear-clipping fails and expose method/rejection diagnostics.
**Acceptance criteria:**
- Projected ear-clipping remains the primary repair path.
- Centroid-fan fallback can add a center vertex and patch a closed loop only when topology guards pass and the loop is at most 6 edges.
- Stats expose ear-clipping fill count, centroid-fan fill count, fallback attempts, and rejection reason counts.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`
**Status:** complete
**Evidence:** Added guarded centroid-fan fallback after projected ear-clipping failure, capped fallback at 6 edges, and exposed method/rejection diagnostics. Focused verification passed: `16 passed`.
**Risks / next:** The fallback is intentionally conservative; larger or nonmanifold-prone rejected loops remain visible through rejection counts.

### Slice 2: Real Fixture Gate And Docs

**Objective:** Keep the fallback only if the Pixal3D reference-target fixture improves topology without readiness regressions.
**Acceptance criteria:**
- Real fixture boundary loop/edge counts improve versus the cap-only baseline.
- Heavy native-chart reference-target fixture remains production-quality-ready with no quality warnings.
- README, Pixal3D docs, scripts docs, and roadmap describe the fallback without claiming open-chain repair or production equivalence.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target diagnostics at `/tmp/mlx-spatialkit-native-chart-reference-target-export-18535/diagnostics.json` showed boundary loops `723`, boundary edges `16521`, open chains `808`, nonmanifold edges `0`, centroid-fan fills `366`, xatlas-utilization ratio `0.7001911739387605`, no quality warnings, and `production_quality_ready=true`; heavy fixture passed. Package verification passed: `73 passed, 7 deselected`; build artifacts were written under `/tmp/mlx-spatialkit-dist-centroid-fan-hole-fallback`.
**Risks / next:** Open boundary chains and remaining rejected closed loops are still present and should stay visible as a separate geometry-quality gap.

## Execution Routing And Topology

Default route: direct. The subagent pool is currently unavailable, and the C++ repair path plus real-fixture gate are tightly coupled.

Parallel-safe groups: none.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_mesh_processing.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k reference_target_native_chart_backend_reports_readiness`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-centroid-fan-hole-fallback`
