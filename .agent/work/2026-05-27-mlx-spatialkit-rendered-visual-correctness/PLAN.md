# Pixal3D Rendered Visual Correctness Plan

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-rendered-visual-correctness/SPEC.md`: make the 1024 Pixal3D native-chart GLB render coherently by fixing texture postprocess, sparse sampling, normals/hole readiness, and the visual gate.

## Execution Routing And Topology

Default continuation path: direct execution through all slices in order.

Parallel-safe groups: none. The work touches shared export diagnostics and the same native texture/GLB pipeline, so parallel writes would create merge risk. Read-only subagent audit is complete and folded into this plan.

## Ordered Slice Sequence

### Slice 1: Visual Failure Gate

**Objective:** Make the current bad GLB fail for the actual color/gloss/coverage symptoms.

**Acceptance criteria:**
- Candidate/reference GLB comparison reports base-color alpha coverage, visible RGB coverage, dark/gray visible RGB ratios, roughness mean, and roughness-low ratio.
- The known bad native-chart artifact no longer passes `visual_comparison.summary.all_passed` when those signals are bad.
- The gate remains renderer-independent enough for CI and can still write `/tmp` visual artifacts.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, related tests.

### Slice 2: Texture Bake Parity Fixes

**Objective:** Stop texture bake from producing dim, half-transparent, overly glossy GLB textures.

**Acceptance criteria:**
- Sparse trilinear sampling normalizes over present sparse-grid corners.
- Render texture fill copies alpha for no-face gutter padding or otherwise produces a render-ready opaque base-color texture while preserving separate raw/UV coverage diagnostics.
- Metallic/roughness texture stats reflect reference-like roughness coverage instead of large low-roughness gaps.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_compare.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/tests/test_texture_bake.py`, comparison tests as needed.

### Slice 3: Normals And Small-Hole Readiness

**Objective:** Reduce avoidable fragmented highlights and make visible holes affect quality readiness.

**Acceptance criteria:**
- GLB normals are not recomputed in a way that unnecessarily breaks smoothness at UV chart seam duplicates.
- Small-hole or boundary diagnostics are included in visual/quality readiness so holes cannot pass as rendered-ready.
- Existing production-equivalence blockers for QEM/DC/xatlas remain explicit.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_writer.py tests/test_real_pixal3d_export.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, tests.

### Slice 4: Real 1024 GLB Regeneration

**Objective:** Regenerate the real Pixal3D 1024 native-chart GLB under `/tmp` and inspect it in Preview/browser evidence.

**Acceptance criteria:**
- The GLB opens from `/tmp` and the turtle/city object has coherent color, materially less shiny fragmented noise, and fewer obvious holes/white seams than the failing artifact.
- Diagnostics report `artifact_ready=true`, an honest rendered-visual status, and remaining parity blockers only for unresolved production-equivalence work.
- No heavy generated artifacts are left in the repo.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q && git diff --check && git status --short`

**Touches:** runtime only under `/tmp`, diagnostics, and tests if the real fixture boundary needs tightening.

**Status:** complete
**Evidence:** Added diagnostic texture-bake isolation knobs for source projection, source-projection fallback distance/neighbors/mode, and render padding without changing default behavior. Fixed native texture sparse lookup to treat Pixal3D texture coordinates as `batch-x-y-z`; regenerated `/tmp/mlx-spatialkit-coordinate-order-fix-98893/model.glb`, with `trilinear_sampled_texel_count=595807`, `source_projection_nearest_fallback_texel_count=6`, `surface_unfilled_texel_count=0`, and `uv_surface_exact_coverage_ratio=0.99999`. Added source-grounded pre-simplify clean-boundary loop fill in native simplify, matching the Pixal3D reference order (`max_edges=64`, `max_perimeter=0.03`) before clustering; fresh GLB `/tmp/mlx-spatialkit-pre-simplify-holefill/model.glb` fills `12815` pre-simplify loops / `79000` faces before simplification, close to reference trace `12876` / `79203`. Browser proof `/tmp/mlx-spatialkit-pre-simplify-holefill/browser_render/comparison.png` passes render checks and aligns candidate/reference across iso/front/top; Preview inspection shows improved color, shell, and bottom coherence. Verified `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_compare.py tests/test_glb_writer.py tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` -> 67 passed, 7 deselected; heavy `tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy` -> 1 passed; `git diff --check` -> passed. Also ran the same export on `outputs/pixal3d/real-smoke-moge-balanced-decoders1100k` to prove path/metadata independence; it produced the same readiness stats.
**Risks / next:** This is not a turtle-specific fix: it ports the generic Pixal3D reference hole-fill contract and has synthetic topology coverage for >8-edge clean loops. True object-general proof still needs a second independent decoded Pixal3D fixture. Residual Preview artifacts remain around lower shell/legs because the native backend still uses topology-aware clustering, not QEM edge-collapse / narrow-band remesh / xatlas parity; `rendered_visual_ready` remains honestly false with `boundary_open_chain_count=242`.

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Native package focus | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_compare.py tests/test_glb_writer.py -q` |
| Real Pixal3D fixture | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q` |
| Hygiene | `git diff --check && git status --short` |
