# mlx-spatialkit Upstream Settings Readiness Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-upstream-settings-readiness-gate/SPEC.md`: make explicit Pixal3D 1M/4096 export settings pass native upstream-setting readiness without claiming xatlas chart parity.

## Architecture Approach

Keep the fix native and diagnostic-driven:

- Increase bounded 4096 high-density atlas fallback/dilation floors in `texture_bake.mm`.
- Add a separate upstream export-setting readiness diagnostic in `export.py` instead of overloading checked-in 1024 reference visual parity.
- Filter `not_1m_face_export_setting_parity` only when the upstream-setting readiness check passes.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: High-Density Atlas Fill

**Objective:** Raise bounded native fallback/dilation budgets for 4096 dense face-atlas exports.

**Acceptance criteria:**
- 1024 and non-atlas behavior remains unchanged.
- 4096 dense atlas diagnostics can report `fallback_radius >= 24` and `dilation_max_passes >= 26`.
- Focused texture bake tests remain green.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** Added high-resolution atlas floors to the native texture bake budget resolution (`texture_size / 171` for fallback, capped at 24; `texture_size / 160` for dilation, capped at 64) while preserving non-atlas defaults. Focused atlas test now asserts coherent capped adaptive values. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q` passed with `6 passed`.
**Risks / next:** Slice 2 must prove the 1M/4096 real fixture reaches coverage readiness with these budgets.

### Slice 2: Upstream Settings Readiness Contract

**Objective:** Add diagnostics and real fixture proof for explicit 1M/4096 upstream-setting readiness.

**Acceptance criteria:**
- Diagnostics include `quality.upstream_export_settings` with checks for target faces, texture size, backend tier, target reach, face retention, artifact readiness, and final coverage.
- Default reference-target visual comparison keeps `not_1m_face_export_setting_parity`.
- Explicit `target_faces=1000000`, `texture_size=4096` heavy export reports upstream-setting readiness passing and removes only `not_1m_face_export_setting_parity`.
- `not_xatlas_chart_parity` remains present.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added `quality.upstream_export_settings` diagnostics with explicit checks for target faces, texture size, backend tier, target reach, face retention, artifact readiness, and final coverage. Export visual comparison now removes `not_1m_face_export_setting_parity` only when those checks pass. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m 'not heavy'` passed with `4 passed, 4 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `4 passed, 4 deselected` in `217.08s`. Fresh `/tmp/mlx-spatialkit-upstream-settings-export-58084/diagnostics.json` reports `quality.upstream_export_settings.all_passed=true`, `final_visible_coverage_ratio=0.5604206919670105`, `fallback_radius=24`, `dilation_max_passes=26`, `final_faces=911927`, peak observed RSS `4658708480`, and deferred visual boundaries `["not_xatlas_chart_parity"]`.
**Risks / next:** Slice 3 must document that this closes upstream 1M/4096 setting readiness, not xatlas chart or CUDA remesh parity.

### Slice 3: Docs And Full Hygiene

**Objective:** Document the upstream settings readiness gate and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs explain that 1M/4096 setting readiness is separate from xatlas chart parity and CUDA remesh parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to document explicit upstream-style 1M/4096 setting readiness, its separation from checked-in 1024 reference parity, and the remaining xatlas/CUDA remesh boundaries. `git diff --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `50 passed, 4 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the sdist and wheel into `/tmp`; artifact inspection found 36 sdist entries, 10 wheel entries, and no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, or pytest cache paths.
**Risks / next:** Continue into auto-verify; xatlas chart/CUDA remesh parity remains the true deferred production boundary.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| USET-01 | Slice 1, Slice 2 |
| USET-02 | Slice 2 |
| USET-03 | Slice 2 |
| USET-04 | Slice 2 |
| USET-05 | Slice 3 |

## Execution Notes

- Do not implement xatlas chart parity in this phase.
- Do not claim upstream CUDA/cuMesh remesh parity.
- Keep generated and heavy artifacts under `/tmp`.
