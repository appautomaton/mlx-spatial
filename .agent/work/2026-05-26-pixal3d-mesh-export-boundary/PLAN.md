# Plan: Pixal3D Mesh Export Boundary

## Goal

Execute [SPEC.md](SPEC.md): advance Pixal3D from decoded shape/texture artifacts into shared mesh extraction, texture baking, and textured GLB export.

## Architecture Approach

Pixal3D orchestration remains in `pixal3d_inference.py`; Pixal3D-specific GLB writing lives in `pixal3d_export.py`. The mesh, postprocess, UV unwrap, texture bake, and GLB payload primitives stay shared through `ovoxel.py` and `trellis2_export.py`.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXME-01 | Slice 1 |
| PXME-02 | Slice 1 |
| PXME-03 | Slice 1 |
| PXME-04 | Slice 1 |
| PXME-05 | Slice 2 |

## Ordered Slice Sequence

### Slice 1: Shared Mesh/Bake/GLB Export

**Objective:** Wire Pixal3D decoded fields through shared mesh extraction, texture baking, and Pixal3D-labeled GLB writing.

**Acceptance criteria:**
- Compatible fake decode assets plus explicit LR/HR/texture NAF can return a ready result with the requested `.glb` path.
- Pipeline metadata records mesh extraction, postprocess stats, bake stats, GLB path, bytes written, and export option values.
- Mesh/bake failure returns a structured `mesh-export` blocker with all decoded artifacts preserved.
- GLB writer failure returns a structured `glb-export` blocker with all decoded artifacts preserved.

**Touches:** `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d.py`, `src/mlx_spatial/__init__.py`, `src/mlx_spatial/trellis2_export.py`, `tests/test_pixal3d_export.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py tests/test_trellis2_export.py -q`

**Status:** complete
**Evidence:** added Pixal3D-labeled GLB writing, threaded export options through CLI and runtime, reused shared FlexiDualGrid mesh extraction/postprocess/texture bake helpers after decoded tensors, and covered ready GLB plus writer-failure preservation paths. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py tests/test_trellis2_export.py -q` -> `124 passed`.
**Risks / next:** normal CLI still needs MLX NAF to reach the now-wired decoded/export route without lower-level explicit feature maps.

### Slice 2: Docs And Release Hygiene

**Objective:** Update Pixal3D docs/script descriptions and prove the change is non-regressive.

**Acceptance criteria:**
- Docs and help text describe the Pixal3D GLB path, export knobs, and remaining MLX NAF blocker.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 1

**Touches:** `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-mesh-export-boundary/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py tests/test_trellis2_export.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, architecture, Pixal3D docs, scripts README, script help text, and Pixal3D CLI help for the GLB path, export knobs, and remaining NAF blocker. Verification passed: `uv run pytest tests/test_pixal3d_*.py tests/test_trellis2_export.py -q` -> `107 passed`; `uv run pytest -q` -> `877 passed, 10 skipped, 27 deselected`; Pixal3D runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; Pixal3D CLI and script help checks -> passed; release build/artifact/git hygiene -> produced and checked `mlx_spatial-0.0.3` sdist/wheel.
**Risks / next:** full ordinary Pixal3D generation still needs the MLX NAF feature path and MoGe auto-camera remains separate.
