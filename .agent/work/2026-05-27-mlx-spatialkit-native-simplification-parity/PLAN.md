# mlx-spatialkit Native Simplification Parity Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-simplification-parity/SPEC.md`: replace the face-stride preview simplifier with a native geometry-aware baseline and add reference metrics for production parity work.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-native-simplification-parity/DESIGN.md`. Native C++ owns simplification; Python only calls the API, reads diagnostics, and verifies outputs.

## Execution Routing And Topology

- Default execution: direct, serial, continuation after verification.
- Parallel-safe groups: none; simplifier, export diagnostics, and heavy tests share contracts.
- Checkpoints: none.
- Commit rhythm: commit after verified slice groups; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Spatial-Cluster Simplifier

**Objective:** Replace `face-stride-preview` with a native spatial vertex-clustering simplifier.

**Acceptance criteria:**
- `simplify_mesh` reports `backend=spatial-cluster`, `algorithm=native_spatial_vertex_clustering`, and `quality_tier=geometry_aware_preview`.
- C++ implementation remaps all source faces through spatial vertex clusters, drops degenerates/duplicates, and compacts the mesh.
- Stats include source/target/final face counts, source/final vertex counts, cluster count, grid resolution, removed-degenerate count, removed-duplicate count, and target reached flag.
- Synthetic tests prove the output is coherent on structured meshes and fail if the backend reverts to face-stride.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`

**Status:** complete
**Evidence:** Replaced face-stride simplification in `packages/mlx-spatialkit/cpp/simplify.cpp` with native spatial vertex clustering: it computes mesh bounds, clusters vertices by a target-derived grid, averages cluster positions, remaps all source faces, removes degenerate and duplicate remapped faces, compacts the mesh, and reports `backend=spatial-cluster`, `algorithm=native_spatial_vertex_clustering`, `quality_tier=geometry_aware_preview`, cluster/grid stats, removed-face counts, and target status. Updated `packages/mlx-spatialkit/tests/test_mesh_processing.py` to use a structured grid mesh and fail if the backend regresses to face-stride. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` passed with `8 passed`.
**Risks / next:** Slice 2 must run the real fixture because spatial clustering changes final face counts and may expose new texture/unwrap behavior.

### Slice 2: Export Quality Metrics And Reference Trace

**Objective:** Add spatialkit export diagnostics that compare real output against the checked-in Pixal3D reference trace.

**Acceptance criteria:**
- Reference trace helper reads face count, raw coverage, final coverage, unwrap backend, and bake backend from `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json`.
- Export diagnostics include a `reference_comparison` block when the reference trace is available.
- Heavy test asserts the new simplifier backend appears in real fixture diagnostics and compares against reference metrics without claiming byte parity.
- Production readiness remains false unless documented thresholds are met.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added checked-in Pixal3D reference trace loading and `reference_comparison` diagnostics in `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`; updated the real fixture heavy test to assert `spatial-cluster`, reference unwrap/bake metrics, and production readiness remaining false. Fixed the native simplifier to reject clustered faces that would make any edge nonmanifold, keeping export blockers honest while producing an artifact-ready real fixture. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `1 passed, 2 deselected`. Latest `/tmp` diagnostics: `artifact_ready=true`, `production_quality_ready=false`, `export_blocking_reasons=[]`, `final_faces=43632`, `nonmanifold_edges=0`, `nonmanifold_faces_removed=4663`, `final_face_count_ratio=0.20528648455364115`, `final_coverage_ratio_vs_reference=0.21571731567382812`.
**Risks / next:** Slice 3 must update docs to stop describing the active simplifier as `face-stride-preview`, then run full package/root/build checks.

### Slice 3: Docs And Full Verification

**Objective:** Document the new simplifier tier and verify package/root cleanliness.

**Acceptance criteria:**
- Docs explain `spatial-cluster` as geometry-aware preview, not production remesh parity.
- Full package tests pass.
- Root Pixal3D integration tests pass.
- Package artifact check excludes generated/heavy paths.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to describe `spatial-cluster` as `geometry_aware_preview`, explain the nonmanifold guard, and keep production readiness false. Tightened `packages/mlx-spatialkit/pyproject.toml` sdist file selection so bytecode/cache files are not packaged. Verification passed: `git diff --check`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> `40 passed, 1 deselected`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> `35 passed`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` passed; artifact inspection found wheel `bad=[]` and sdist `bad=[]`.
**Risks / next:** The active change is verified for spatial-cluster preview parity metrics. The broader thread goal remains active because production remesh/unwrap/texture parity is still not achieved.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| SKP-01 | Slice 1 |
| SKP-02 | Slice 1 |
| SKP-03 | Slice 1 |
| SKP-04 | Slice 2 |
| SKP-05 | Slice 2 |
| SKP-06 | Slice 2 |
| SKP-07 | Slice 3 |

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Simplifier tests | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` |
| Heavy real fixture | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` |
| Full package tests | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` |
| Root smoke integration | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` |

## Execution Notes

- Keep generated heavy outputs under `/tmp`.
- Do not mark the broad thread goal complete after this change unless reference parity and production readiness are actually achieved.
- If spatial clustering improves over face stride but remains below production thresholds, keep `production_quality_ready=false` and carry the remaining QEM/remesh gap forward.
