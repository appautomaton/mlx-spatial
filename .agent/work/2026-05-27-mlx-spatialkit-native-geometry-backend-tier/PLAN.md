# mlx-spatialkit Native Geometry Backend Tier Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-geometry-backend-tier/SPEC.md`: add a distinct native topology-aware geometry backend for reference-target Pixal3D export and prove the production-tier claim through existing thresholds rather than relaxed checks.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-native-geometry-backend-tier/DESIGN.md`. Keep `spatial-cluster` as the honest preview/default backend; route reference-target exports to the new native backend only after the backend contract and tests exist.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Subagent route: recommended for native backend implementation review after Slice 2 because it touches shared C++/Python contracts and production-readiness semantics.
- Parallel-safe groups: none; slices share the simplifier/export contract.
- Checkpoints: none.
- Commit rhythm: commit after verified slice groups; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Backend Selection Contract

**Objective:** Add an explicit native simplifier backend selection contract without changing production thresholds.

**Acceptance criteria:**
- `simplify_mesh` accepts an explicit backend intent through C++ bindings and the Python wrapper.
- Preview/default exports continue to report `backend=spatial-cluster` and `quality_tier=geometry_aware_preview`.
- Reference-target export records the intended production backend route without relaxing `_production_thresholds()`.
- Invalid backend names fail with a clear public error.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m "not heavy"`

**Touches:** `packages/mlx-spatialkit/cpp/mesh_processing.hpp`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/mesh.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added explicit simplifier backend intent through the C++ binding/header, Python `simplify_mesh(..., backend=...)` wrapper, and Pixal3D export routing. Preview/default requests `spatial-cluster`; reference-target requests `topology-aware` but currently records `fallback_preview_unimplemented` while actual backend remains `spatial-cluster`, so thresholds stay unchanged. Added tests for backend intent stats, invalid backend validation, and reference-target backend request selection. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m "not heavy"` passed with `13 passed, 2 deselected`.
**Risks / next:** Slice 2 must replace the fallback with a distinct native topology-aware backend before any production-tier claim is possible.

### Slice 2: Native Topology-Aware Geometry Backend

**Objective:** Implement a distinct native topology-aware simplifier/remesher candidate that can justify production-tier status only when its own checks pass.

**Acceptance criteria:**
- New backend reports a backend name other than `spatial-cluster` and an algorithm other than `native_spatial_vertex_clustering`.
- Tests would fail if the implementation simply aliases `spatial-cluster`.
- Synthetic structured meshes prove the backend reaches target bounds, consumes source geometry coherently, and emits no degenerate, duplicate, or nonmanifold export blockers.
- Stats include production-readiness fields, production blockers, target/source/final counts, topology cleanup counts, and candidate/collapse work counts.
- Native code owns the hot path; no Python per-face/per-vertex simplification loops are introduced.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Depends on:** Slice 1

**Execution:** subagent recommended

**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, optional new C++ implementation/header files under `packages/mlx-spatialkit/cpp/`, `packages/mlx-spatialkit/CMakeLists.txt`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`

### Slice 3: Reference-Target Production Gate

**Objective:** Route reference-target Pixal3D decoded-fixture export through the new backend and verify whether the real fixture can pass production readiness honestly.

**Acceptance criteria:**
- Heavy reference-target diagnostics show the new backend/algorithm, not `spatial-cluster`.
- `native_geometry_candidate.status` is `candidate` only when backend tier and all other production thresholds pass.
- `production_quality_ready=true` is allowed only when backend tier, topology, face-count, final coverage, raw reporting, preset, and reference checks all pass.
- If the backend fails, diagnostics identify a specific measured blocker and keep `production_quality_ready=false`.
- Memory samples and timings remain present, with generated GLB/diagnostics under `/tmp`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

### Slice 4: Root Bridge And Documentation

**Objective:** Keep the root Pixal3D spatialkit bridge and docs coherent with the new backend boundary.

**Acceptance criteria:**
- Root Pixal3D tests still prove optional spatialkit export routing and fallback behavior.
- Docs explain preview/default versus reference-target backend selection and whether the production gate passes.
- No docs claim xatlas/remesh parity beyond the verified backend gate.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_pipeline.py::test_pixal3d_pipeline_uses_optional_spatialkit_export_backend tests/test_pixal3d_pipeline.py::test_pixal3d_pipeline_falls_back_when_optional_spatialkit_is_missing -q`

**Depends on:** Slice 3

**Touches:** `docs/pixal3d.md`, `packages/mlx-spatialkit/README.md`, `scripts/README.md`, root bridge tests only if behavior changes

### Slice 5: Full Verification And Packaging Hygiene

**Objective:** Verify package/root cleanliness, build artifacts, and heavy-output hygiene before marking the change verified.

**Acceptance criteria:**
- `git diff --check` passes.
- Full package tests pass.
- Root Pixal3D export/pipeline tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, pycache, and pytest cache files.
- `git status --short` contains only intentional source/artifact changes before commit and is clean after commit.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 4

**Touches:** package/build metadata only if verification exposes hygiene gaps

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| GBT-01 | Slice 1, Slice 2 |
| GBT-02 | Slice 1, Slice 3 |
| GBT-03 | Slice 3 |
| GBT-04 | Slice 3 |
| GBT-05 | Slice 4 |
| GBT-06 | Slice 5 |

## Execution Notes

- Do not relax production thresholds.
- Do not mark the broad thread goal complete solely because backend tier passes; visual parity and future export settings may still need follow-up cycles.
- If the new backend cannot pass production readiness, record the measured backend-specific blocker instead of renaming preview behavior.
- Heavy outputs stay under `/tmp`.
