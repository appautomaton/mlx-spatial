# Plan: Pixal3D Texture SLat Boundary

## Goal

Execute [SPEC.md](SPEC.md): advance Pixal3D from HR shape SLat into texture projection and optional texture SLat probing.

## Architecture Approach

Pixal3D orchestration owns stage order and artifact metadata. Texture SLat execution reuses the shared `trellis2_slat.py` texture probe, with Pixal3D-specific projection selection and normalization handled in `pixal3d_inference.py`.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXTX-01 | Slice 1 |
| PXTX-02 | Slice 1 |
| PXTX-03 | Slice 1 |
| PXTX-04 | Slice 1 |
| PXTX-05 | Slice 2 |

## Ordered Slice Sequence

### Slice 1: Texture Projection And SLat Artifact

**Objective:** Wire texture projection conditioning after HR shape SLat and write `texture_slat.npz` when explicit texture NAF features and compatible texture flow assets are available.

**Acceptance criteria:**
- Existing HR shape SLat path blocks at texture projection conditioning when texture NAF is absent.
- Compatible fake texture SLat assets plus explicit texture NAF complete texture SLat probing.
- Result artifacts include `sparse_projection.npz`, `sparse_structure.npz`, `shape_slat_lr.npz`, `shape_slat_hr_coordinates.npz`, `shape_slat_hr.npz`, and `texture_slat.npz`.
- Runtime metadata records selected texture projected shape, normalized shape concat feature shape, sampled texture feature shape, and the next decode/export blocker.

**Touches:** `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/__init__.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_export.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q`

**Status:** complete
**Evidence:** added `write_pixal3d_texture_slat_npz`, API-only `texture_naf_feature_map`, texture projection conditioning at the selected cascade grid, coordinate-indexed projected texture features, shape-SLat denormalization for texture concat conditioning, shared `probe_texture_slat_forward_boundary` execution, normalized `texture_slat.npz` writing, and structured blockers for missing texture NAF or texture SLat failures. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q` -> `80 passed`.
**Risks / next:** full latent decode still needs shape/texture decoder handoff, PBR bake, mesh extraction, and GLB export.

### Slice 2: Docs And Release Hygiene

**Objective:** Update Pixal3D docs/script descriptions and prove the change is non-regressive.

**Acceptance criteria:**
- Docs describe `texture_slat.npz`, explicit texture NAF boundary, and remaining decode/export blockers.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 1

**Touches:** `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-texture-slat-boundary/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, architecture, Pixal3D docs, scripts README, and script help text to describe `texture_slat.npz`, explicit texture NAF, and the remaining latent decode/export blockers. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `63 passed`; `uv run pytest -q` -> `870 passed, 10 skipped, 27 deselected`; forbidden runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; release build/artifact/git hygiene -> produced and checked `mlx_spatial-0.0.3` sdist/wheel; Pixal3D CLI and script help checks -> passed.
**Risks / next:** full Pixal3D support still needs MLX NAF, full shape/texture decode, PBR baking, mesh extraction, and final GLB export.
