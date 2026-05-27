# mlx-spatialkit UV Surface Fill Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-uv-surface-fill-gate/SPEC.md`: add bounded native UV-surface fill and prove the real native-chart fixture clears the global coverage floor.

## Architecture Approach

Keep the Metal sampling kernel, UV-bin construction, and chart UV generation unchanged. Add a native post-bake surface-fill pass that propagates visible PBR texels into remaining UV-surface missing/out-of-grid texels, never into no-face texels. Preserve exact sampled, nearest-voxel fallback, dilation, surface-fill, and final-visible diagnostics as separate counters.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Surface Fill

**Objective:** Implement bounded native UV-surface fill and focused texture diagnostics.

**Acceptance criteria:**
- Texture stats expose surface-fill enabled/filled counts separately from sampled and fallback counts.
- Focused texture tests prove no-face texels remain unfilled and remaining surface texels can be filled.
- Existing texture bake tests still pass.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** added bounded native UV-surface fill with distinct coverage status and diagnostics for seed, filled, and remaining surface texels; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q` -> 9 passed.
**Risks / next:** Slice 2 must prove final fill clears real native-chart coverage while preserving raw/exact coverage diagnostics.

### Slice 2: Real Coverage Readiness Proof

**Objective:** Prove the real Pixal3D native-chart export clears global coverage without hiding raw/exact diagnostics.

**Acceptance criteria:**
- Heavy chart fixture writes GLB and diagnostics under `/tmp`.
- Native chart reports `quality_ready=true`, `global_coverage_ratio >= 0.50`, UV-bin guard passing, and `xatlas_chart_parity=false`.
- Texture stats expose surface-filled texels and preserve raw/exact coverage ratios.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** native-chart heavy fixture now reports `status=quality_ready`, `global_coverage_ratio=0.5232105255126953`, `uv_surface_final_visible_coverage_ratio=1.0`, `surface_filled_texel_count=134064`, `surface_unfilled_texel_count=0`, `raw_coverage_ratio=0.015173912048339844`, and `uv_surface_exact_coverage_ratio=0.02900154203409973`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed.
**Risks / next:** final fill clears scalar coverage but xatlas chart parity remains explicitly deferred.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document UV-surface fill and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe UV-surface fill as a native texture candidate improvement, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** documented bounded UV-surface fill and the native-chart parity boundary; `git diff --check` -> passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> 63 passed, 5 deselected; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> 35 passed; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` -> built wheel and sdist; artifact inspection -> wheel 10 entries bad 0, sdist 36 entries bad 0.
**Risks / next:** final verify must rerun plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| USFG-01 | Slice 1 |
| USFG-02 | Slice 1 |
| USFG-03 | Slice 2 |
| USFG-04 | Slice 2 |
| USFG-05 | Slice 3 |
| USFG-06 | Slice 3 |

## Execution Notes

- Do not change Metal sampling kernel behavior.
- Do not change chart readiness floors.
- Keep generated and heavy artifacts under `/tmp`.
