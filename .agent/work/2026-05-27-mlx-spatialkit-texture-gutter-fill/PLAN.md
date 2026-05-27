# mlx-spatialkit Texture Gutter Fill Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-texture-gutter-fill/SPEC.md`: add bounded native texture gutter fill without changing UV-surface coverage semantics or xatlas parity claims.

## Architecture Approach

Add a native postprocess in `texture_bake.mm` after existing UV-surface fill. It copies RGB plus metallic/roughness from visible neighboring texels into no-face gutter texels for a bounded number of passes, while keeping alpha and coverage status unchanged. This reduces linear-filter seam risk without making diagnostics count gutter texels as UV surface or visible coverage.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Gutter Fill And Focused Contract

**Objective:** Implement bounded no-face texture gutter fill in the native bake postprocess and prove coverage semantics stay honest.

**Acceptance criteria:**
- No-face texels adjacent to visible texels receive nonzero RGB/MR gutter values.
- Gutter fill leaves alpha and coverage status unchanged for no-face texels.
- Visual comparison separates raw RGB footprint from visible RGB coverage.
- Stats expose enabled flag, pass count, max passes, and filled texel count.
- Existing UV-surface and visible-alpha coverage calculations remain unchanged by gutter-only fill.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/tests/test_glb_compare.py`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** Added bounded native no-face gutter RGB/MR fill after surface fill while preserving alpha and coverage status; tightened visual comparison to separate raw RGB footprint from visible RGB coverage; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_texture_bake.py -q` -> `13 passed`.
**Risks / next:** verify on real Pixal3D export and document the seam-quality boundary.

### Slice 2: Pixal3D Export Gate And Docs

**Objective:** Propagate gutter-fill diagnostics through real Pixal3D export evidence and document the visual-quality boundary.

**Acceptance criteria:**
- Reference-target native-chart heavy test asserts gutter fill stats are present and nonzero.
- Existing quality, visual comparison, viewer compatibility, memory, and xatlas non-parity assertions remain intact.
- Spatialkit/Pixal3D docs explain gutter fill as seam robustness, not xatlas parity or coverage inflation.
- Roadmap current state is updated after verification.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy && git diff --check`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`

**Status:** complete
**Evidence:** Heavy reference-target native-chart gate asserts nonzero gutter stats and preserved quality/xatlas-honesty checks; latest `/tmp` diagnostics showed `gutter_filled_texel_count=453197`, `uv_surface_texel_count=594952`, and unchanged `final_visible_coverage_ratio=0.5673904418945312`. Verification command -> `1 passed`; `git diff --check` -> passed.
**Risks / next:** none for this slice.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| TGF-01 | Slice 1 |
| TGF-02 | Slice 1 |
| TGF-03 | Slice 1 |
| TGF-04 | Slice 1 |
| TGF-05 | Slice 1 |
| TGF-06 | Slice 2 |
| TGF-07 | Slice 2 |
| TGF-08 | Slices 1-2 |

## Aggregate Verification

- Package suite: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q`
- Root Pixal3D integration: `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q`
- Build hygiene: `rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`, then inspect wheel/sdist for generated artifact leakage.

## Verification Evidence

- Focused texture/visual suite: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_texture_bake.py -q` -> `13 passed`.
- Heavy Pixal3D gate: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy` -> `1 passed`.
- Package suite: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> `69 passed, 7 deselected`.
- Root Pixal3D integration: `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> `35 passed`.
- Hygiene: `git diff --check` passed; `/tmp/mlx-spatialkit-dist` build inspection reported wheel `10` entries, bad `0`, and sdist `36` entries, bad `0`.
