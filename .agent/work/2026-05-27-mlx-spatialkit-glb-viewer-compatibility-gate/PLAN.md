# mlx-spatialkit GLB Viewer Compatibility Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-glb-viewer-compatibility-gate/SPEC.md`: make native Pixal3D GLBs carry normals, avoid large uint32-indexed primitives, and report strict-viewer compatibility without claiming xatlas chart parity.

## Architecture Approach

Keep the hot path native:

- Generate finite normalized vertex normals inside `glb_writer.cpp`.
- Emit chunk-local GLB primitives with local position/normal/UV/index accessors, using `UNSIGNED_SHORT` indices for every primitive.
- Extend GLB inspection and Pixal3D diagnostics so compatibility readiness is a checked contract, not a visual guess.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native GLB Normals And Chunked Primitives

**Objective:** Update the native GLB writer so every primitive has normals and large meshes are split into uint16-indexed primitive chunks.

**Acceptance criteria:**
- Small writer fixtures include `NORMAL` and still embed the two PBR textures.
- A synthetic large mesh produces multiple primitives.
- Every index accessor uses `componentType=5123` and has max local index <= `65535`.
- Geometry/image validation continues to reject invalid inputs before writing payloads.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_glb_compare.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/tests/test_glb_compare.py`

**Status:** complete
**Evidence:** Added native normal generation and chunk-local GLB primitive emission in `glb_writer.cpp`; inspection now reports normal/index details. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_glb_compare.py -q` passed with `9 passed`.
**Risks / next:** Slice 2 must prove the Pixal3D export diagnostics and real fixture path consume the chunked GLB correctly.

### Slice 2: Pixal3D Export Compatibility Diagnostics

**Objective:** Add a `quality.glb_viewer_compatibility` diagnostic gate to Pixal3D exports and prove it on focused and real fixture paths.

**Acceptance criteria:**
- Diagnostics check parseability, material/texture presence, normals on all primitives, uint16-only indices, max local index, and primitive chunking for large meshes.
- Default/reference-target visual deferrals still keep `not_xatlas_chart_parity`.
- Heavy real Pixal3D export under `/tmp` passes the compatibility gate.
- Browser render proof remains compatible with the chunked GLB.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added `quality.glb_viewer_compatibility` from native GLB inspection and asserted normals, uint16 indices, local index bounds, material/texture presence, and chunking. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m 'not heavy'` passed with `5 passed, 4 deselected`; heavy real fixture passed with `4 passed, 5 deselected` in `223.93s`. Fresh `/tmp/mlx-spatialkit-upstream-settings-export-63737/diagnostics.json` reports `glb_viewer_compatibility.all_passed=true`, `primitive_count=42`, `total_vertices=2735781`, `total_faces=911927`, uint16-only indices, max local index `65534`, normals on every primitive, upstream readiness true, and deferred boundaries `["not_xatlas_chart_parity"]`. Browser render proof against the chunked reference-target GLB passed with all three visible-pixel ratios around `1.01`.
**Risks / next:** Slice 3 must document this as strict-viewer GLB hardening, not xatlas/cuMesh parity.

### Slice 3: Docs And Full Hygiene

**Objective:** Document the compatibility gate and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs explain normals and uint16 primitive chunking as strict-viewer hardening, not xatlas/cuMesh parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Documented native normals, uint16 primitive chunking, `quality.glb_viewer_compatibility`, and the xatlas/cuMesh non-claim in package, Pixal3D, and script docs. `git diff --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `52 passed, 4 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the sdist and wheel into `/tmp`; artifact inspection found 36 sdist entries, 10 wheel entries, and no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, or pytest cache paths.
**Risks / next:** none for this phase; final verification should re-run the acceptance checks from the completed plan.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| GVC-01 | Slice 1 |
| GVC-02 | Slice 1 |
| GVC-03 | Slice 2 |
| GVC-04 | Slice 2 |
| GVC-05 | Slice 3 |

## Execution Notes

- Do not implement xatlas chart parity in this phase.
- Do not claim upstream CUDA/cuMesh remesh parity.
- Keep generated and heavy artifacts under `/tmp`.
