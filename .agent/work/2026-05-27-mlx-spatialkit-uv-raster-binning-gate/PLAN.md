# mlx-spatialkit UV Raster Binning Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-uv-raster-binning-gate/SPEC.md`: make arbitrary non-atlas UV texture baking use a bounded Metal face-bin lookup path instead of an all-faces-per-pixel scan.

## Architecture Approach

Keep the face-atlas fast path unchanged. For non-atlas UVs, build a CPU-side UV bin index from triangle UV bounding boxes, upload `bin_offsets` and `bin_faces` to Metal, and have each texel scan only the faces referenced by its UV bin. Diagnostics report bin grid shape, reference counts, candidate limits, and guard status.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native UV Bin Index And Metal Lookup

**Objective:** Implement CPU UV-bin construction and Metal binned face lookup for non-atlas texture baking.

**Acceptance criteria:**
- Non-atlas UV bakes report `backend=metal-uv-binned-nearest`.
- Face-atlas bakes keep the existing `metal-face-atlas-nearest` backend and behavior.
- Bin diagnostics include grid dimensions, bin count, face-reference count, max/average candidates per bin, and guard status.
- Invalid or pathological bin-reference counts fail before Metal allocation with a structured error.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/metal/kernels/texture_bake.metal`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** Added CPU UV-bin construction with a 64M face-reference guard and Metal bin-offset/bin-face buffers for non-atlas bakes. Non-atlas bakes now report `metal-uv-binned-nearest` with bin counts/candidate diagnostics while face-atlas bakes remain `metal-face-atlas-nearest`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q` passed with `8 passed`.
**Risks / next:** Slice 2 must prove the existing real Pixal3D face-atlas exports and parity diagnostics remain intact.

### Slice 2: Pixal3D Regression Gate

**Objective:** Prove the binned arbitrary-UV path does not regress the existing face-atlas real Pixal3D export path or parity diagnostics.

**Acceptance criteria:**
- Heavy real Pixal3D fixture still passes reference-target and upstream-setting gates.
- Face-atlas texture diagnostics remain unchanged for real exports.
- `not_xatlas_chart_parity` remains deferred.
- Memory telemetry remains present for texture bake and write stages.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Heavy real Pixal3D export regression passed with `4 passed, 5 deselected` in `224.84s`. Fresh `/tmp/mlx-spatialkit-upstream-settings-export-68413/diagnostics.json` reports `texture_bake.backend=metal-face-atlas-nearest`, `atlas_faces_per_tile=2`, `uv_bin_count=0`, upstream readiness true, GLB viewer compatibility true, peak RSS `5140529152`, and deferred visual boundaries `["not_xatlas_chart_parity"]`.
**Risks / next:** Slice 3 must document binned UV rasterization as an arbitrary-chart prerequisite, not chart generation itself.

### Slice 3: Docs And Full Hygiene

**Objective:** Document UV raster binning as a prerequisite to chart parity and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs explain that non-atlas UV bakes are now binned and scalable, while native xatlas chart generation remains deferred.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated package, Pixal3D, and scripts docs to explain `metal-uv-binned-nearest` as scalable arbitrary-UV raster baking and not native xatlas chart generation. `git diff --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `54 passed, 4 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the sdist and wheel into `/tmp`; artifact inspection found 36 sdist entries, 10 wheel entries, and no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, or pytest cache paths.
**Risks / next:** none for this phase; final verification should re-run the planned checks from current state.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| UVBIN-01 | Slice 1 |
| UVBIN-02 | Slice 1 |
| UVBIN-03 | Slice 1 |
| UVBIN-04 | Slice 2 |
| UVBIN-05 | Slice 3 |

## Execution Notes

- Do not implement native xatlas chart generation in this phase.
- Do not switch real Pixal3D exports from face-atlas UVs to arbitrary chart UVs yet.
- Keep generated and heavy artifacts under `/tmp`.
