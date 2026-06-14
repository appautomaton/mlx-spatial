# mlx-spatialkit Native Atlas Coverage Parity Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-atlas-coverage-parity/SPEC.md`: improve native atlas utilization for Pixal3D reference-target exports and verify real-fixture coverage improvement without weakening production thresholds.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-native-atlas-coverage-parity/DESIGN.md`. Keep atlas generation in C++; keep production readiness controlled by the existing threshold report.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none; native atlas, texture bake diagnostics, and heavy tests share contracts.
- Checkpoints: none.
- Commit rhythm: commit after verified slice groups; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native Paired Atlas

**Objective:** Change native face-atlas generation to pair two triangle faces per tile and report packing diagnostics.

**Acceptance criteria:**
- `make_face_atlas_uvs` keeps duplicated per-face vertices and valid GLB indices.
- Two triangle faces can occupy complementary halves of one atlas tile.
- UV stats report `packing=paired-triangles`, `faces_per_tile=2`, tile count, and estimated utilization.
- Existing GLB and texture-bake synthetic tests pass.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_texture_bake.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/tests/test_glb_writer.py`, `packages/mlx-spatialkit/tests/test_texture_bake.py`

**Status:** complete
**Evidence:** Updated native `make_face_atlas_uvs` in `packages/mlx-spatialkit/cpp/glb_writer.cpp` to pack two triangle faces per atlas tile using complementary lower-left and upper-right UV halves. UV stats now report `packing=paired-triangles`, `faces_per_tile=2`, `atlas_tiles`, and `estimated_tile_utilization`. Updated GLB and Metal texture tests for the paired layout. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_writer.py tests/test_texture_bake.py -q` passed with `10 passed`.
**Risks / next:** Slice 2 must prove the paired atlas improves the real reference-target fixture and keep threshold failures explicit.

### Slice 2: Reference-Target Coverage Gate

**Objective:** Verify the dense atlas improves real Pixal3D reference-target global coverage and keep production-readiness failures explicit.

**Acceptance criteria:**
- Heavy reference-target diagnostics show `uv.stats.packing=paired-triangles`.
- Heavy reference-target final global coverage exceeds the prior `0.269` baseline and passes or clearly reports the `0.50` threshold.
- `production_quality_ready` remains false if backend tier or coverage thresholds fail.
- Memory samples and timings remain present.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` if threshold details need refinement

**Status:** complete
**Evidence:** Updated the heavy reference-target fixture test to assert paired atlas diagnostics (`packing=paired-triangles`, `faces_per_tile=2`, `atlas_faces_per_tile=2`) and the coverage-threshold bookkeeping. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `2 passed, 3 deselected`. Latest `/tmp/mlx-spatialkit-reference-target-export-22838/diagnostics.json` reports `final_visible_coverage_ratio=0.6019515991210938`, above the prior `0.269` baseline and above the `0.50` threshold; `production_quality_ready=false` remains because `backend_tier=geometry_aware_preview` fails.
**Risks / next:** Coverage blocker is improved for the reference fixture; production readiness still needs the later non-preview remesh/backend tier.

### Slice 3: Docs And Full Verification

**Objective:** Document native paired atlas behavior and verify package/root cleanliness.

**Acceptance criteria:**
- Docs explain paired native atlas improves coverage but is not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Package artifact inspection excludes generated/heavy paths.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to document paired native atlas behavior, the improved reference-target coverage, and the remaining preview-tier backend blocker. `git diff --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `41 passed, 2 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the wheel and sdist under `/tmp`; artifact inspection found no generated outputs, inputs, diagnostics, GLBs, pycache, or pytest cache entries.
**Risks / next:** Phase 3 is verified; broad production readiness still needs the Phase 4 non-preview geometry backend work.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| ACP-01 | Slice 1 |
| ACP-02 | Slice 2 |
| ACP-03 | Slice 2 |
| ACP-04 | Slice 3 |

## Execution Notes

- Heavy outputs stay under `/tmp`.
- Do not mark the broad thread goal complete unless production thresholds pass and current evidence proves the full backend objective.
- If paired atlas improves coverage but backend tier remains preview, keep `production_quality_ready=false`.
