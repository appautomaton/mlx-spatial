# mlx-spatialkit Native Chart UV Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-uv-gate/SPEC.md`: add an opt-in native chart UV generator and prove it through the binned Metal texture path without changing the default Pixal3D export backend.

## Architecture Approach

Add a C++ `make_native_chart_uvs` binding alongside `make_face_atlas_uvs`. It will group faces through edge adjacency when normals are within a configurable threshold, duplicate source vertices per chart, project each chart to a stable 2D plane, pack charts into a deterministic grid, and return `NativeUvMesh` plus stats. The texture bake uses the existing non-atlas binned path because chart stats intentionally do not expose face-atlas lookup fields.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Chart UV API

**Objective:** Implement `make_native_chart_uvs` in native C++ and expose it through the Python package.

**Acceptance criteria:**
- Public import works from `mlx_spatialkit`.
- Coplanar square fixture produces one chart and reuses four vertices instead of duplicating six face vertices.
- Hard-crease/disconnected fixtures produce multiple charts.
- Stats include backend, chart count, output/source counts, duplicate ratio, threshold, and chart packing grid.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/cpp/glb_writer.hpp`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/__init__.py`, `packages/mlx-spatialkit/tests/test_glb_writer.py`

**Status:** complete
**Evidence:** added native binding, Python facade, top-level export, and focused chart UV tests; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py -q` -> 8 passed.
**Risks / next:** Slice 2 still needs Metal binned bake proof.

### Slice 2: Binned Texture Bake Proof

**Objective:** Prove native chart UV meshes bake through `metal-uv-binned-nearest`.

**Acceptance criteria:**
- Chart UV fixture bakes with `backend=metal-uv-binned-nearest`.
- Binned diagnostics are present and nonzero.
- Existing face-atlas texture tests still pass.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** added chart-generated UV bake proof; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_texture_bake.py -q` -> 9 passed.
**Risks / next:** Slice 3 still needs docs and full regression/build gates.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document the native chart candidate boundary and verify existing Pixal3D export behavior remains stable.

**Acceptance criteria:**
- Docs explain native chart UV generation as opt-in and not xatlas parity.
- Heavy real Pixal3D fixture still passes with face-atlas backend and `not_xatlas_chart_parity` deferred.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe opt-in native chart-candidate UV generation without changing Pixal3D defaults; `git diff --check` passed; package tests -> 57 passed, 4 deselected; heavy Pixal3D -> 4 passed, 5 deselected; root Pixal3D -> 35 passed; `/tmp` build succeeded; artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** final verify must rerun the plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| CUV-01 | Slice 1 |
| CUV-02 | Slice 1 |
| CUV-03 | Slice 1 |
| CUV-04 | Slice 2 |
| CUV-05 | Slice 3 |

## Execution Notes

- Do not switch `export_pixal3d_glb` from face atlas to chart UVs in this phase.
- Do not claim xatlas or CUDA/cuMesh parity.
- Keep generated and heavy artifacts under `/tmp`.
