# mlx-spatialkit 4096 Texture Coverage Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-4096-texture-coverage-gate/SPEC.md`: make native Pixal3D reference-target export pass production coverage at `texture_size=4096`.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-4096-texture-coverage-gate/DESIGN.md`. Keep the fix native and dependency-free by adapting the existing Metal nearest-fallback radius and C++ post-bake dilation budget from atlas tile geometry.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after the full verify gate; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Adaptive Fill Budgets

**Objective:** Replace fixed low texture fill budgets with adaptive atlas-size-aware fallback and dilation budgets.

**Acceptance criteria:**
- Native texture bake computes bounded adaptive nearest-fallback and dilation budgets.
- Diagnostics report the resolved budgets in `fallback_radius` and `dilation_max_passes`.
- Focused texture bake tests prove atlas-backed baking reports coherent adaptive fill stats.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** Replaced the fixed fill budgets with `resolve_fallback_radius(...)` and `resolve_dilation_max_passes(...)`, both scaled from face-atlas tile pixels and bounded (`fallback_radius` clamps to `12..24`; `dilation_max_passes` clamps to `8..64`). Texture bake diagnostics report both resolved budgets. Added focused coverage proving atlas-backed texture bake can use `fallback_radius > 12` and `dilation_max_passes > 8` while non-atlas UVs retain the lower defaults. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q` passed with `6 passed`.
**Risks / next:** Slice 2 must prove the real 4096 Pixal3D fixture now passes production coverage without threshold changes.

### Slice 2: Real 4096 Fixture Gate

**Objective:** Add and pass a heavy real Pixal3D 4096 texture coverage gate.

**Acceptance criteria:**
- Existing 1024 reference-target heavy test remains passing.
- New 4096 heavy test writes generated artifacts under `/tmp`.
- 4096 diagnostics report `texture_size=4096`, `dilation_max_passes > 8`, `final_visible_coverage_ratio >= 0.50`, `production_quality_ready=true`, and stage peak memory for `texture_bake` and `write_glb`.
- Visual comparison may honestly record texture-resolution mismatch versus the checked-in 1024 reference GLB.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added `test_export_pixal3d_glb_reference_target_4096_texture_passes_coverage_gate` to run the real decoded fixture under `/tmp` with `texture_size=4096`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `3 passed, 3 deselected` in `111.53s`. Latest `/tmp/mlx-spatialkit-reference-target-4096-export-49006/diagnostics.json` reports `production_quality_ready=true`, `final_visible_coverage_ratio=0.5642194151878357`, `fallback_radius=24`, `dilation_max_passes=26`, `dilation_pass_count=10`, peak observed RSS `4313841664`, and texture-resolution mismatch remains explicit versus the checked-in 1024 reference GLB.
**Risks / next:** Slice 3 must document the high-resolution runtime cost and remaining xatlas/1M-face boundaries.

### Slice 3: Docs And Full Verification

**Objective:** Document 4096 texture coverage readiness and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs explain adaptive dilation, 4096 coverage readiness, and remaining xatlas/1M-face boundaries.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to document adaptive atlas-size-aware Metal fallback/native dilation budgets, 4096 coverage readiness, expected 4096-vs-1024 texture-resolution mismatch in visual comparison, and remaining xatlas/1M-face boundaries. `git diff --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `49 passed, 3 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the sdist and wheel into `/tmp`; artifact inspection found 36 sdist entries, 10 wheel entries, and no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, or pytest cache paths.
**Risks / next:** Continue into auto-verify; the broad native-backend goal still has separate xatlas chart parity and 1M-face export-setting work after this phase.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| T4096-01 | Slice 1 |
| T4096-02 | Slice 2 |
| T4096-03 | Slice 2 |
| T4096-04 | Slice 3 |
| T4096-05 | Slice 3 |

## Execution Notes

- Do not relax coverage thresholds.
- Keep heavy artifacts under `/tmp`.
- Do not claim xatlas chart parity or 1M-face setting parity.
