# PLAN: mlx-spatialkit Readiness Semantics

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-readiness-semantics/SPEC.md` by adding a stricter Pixal3D production-equivalence diagnostics contract.

## Ordered Slice Sequence

### Slice 1: Equivalence Diagnostics Contract

**Objective:** Add production-equivalence diagnostics without changing scalar `production_quality_ready` semantics.
**Acceptance criteria:**
- `quality.production_equivalence` and `result.production_equivalence_ready` exist.
- Remaining parity boundaries include xatlas and setting/visual gaps when applicable.
- Existing scalar quality, native-chart, xatlas parity, and upstream-setting diagnostics remain present.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -k "quality_summary or production_equivalence or xatlas_chart_parity"`
**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`
**Status:** complete
**Evidence:** Added `quality.production_equivalence`, `result.production_equivalence_ready`, remaining-boundary/blocker diagnostics, and unit coverage. Verification passed: `3 passed, 13 deselected`.
**Risks / next:** none.

### Slice 2: Real Fixture And Docs Alignment

**Objective:** Prove the stricter contract on real Pixal3D diagnostics and document the distinction.
**Acceptance criteria:**
- Native-chart reference-target heavy fixture asserts scalar readiness can pass while production equivalence remains false.
- README and Pixal3D docs explain the difference between scalar quality readiness and production equivalence.
- Roadmap remains compact and reflects this active change.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k "reference_target_native_chart_backend_reports_readiness"`
**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Heavy native-chart reference-target fixture now asserts scalar readiness with `production_equivalence_ready=false`; docs and roadmap describe scalar readiness versus production equivalence. Verification passed: `1 passed, 15 deselected`; package check passed: `70 passed, 7 deselected`; `git diff --check` passed; `/tmp/mlx-spatialkit-dist-readiness` wheel/sdist built and archive checks passed.
**Risks / next:** xatlas parity itself remains deferred by design.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none. The code, tests, and docs all touch the same diagnostics vocabulary, so serial edits are clearer.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -k "quality_summary or production_equivalence or xatlas_chart_parity"`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_real_pixal3d_export.py -q -m heavy -k "reference_target_native_chart_backend_reports_readiness"`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_glb_compare.py tests/test_texture_bake.py tests/test_real_pixal3d_export.py -q -m "not heavy"`
