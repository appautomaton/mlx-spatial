# Plan: Pixal3D Latent Decode Boundary

## Goal

Execute [SPEC.md](SPEC.md): advance Pixal3D from texture SLat into shared shape/texture decoder artifacts.

## Architecture Approach

Pixal3D orchestration remains in `pixal3d_inference.py`; decoded artifact serialization remains in `pixal3d_export.py`. Shape and texture decoder math must call `trellis2_decode.py` so Pixal3D stays aligned with the existing TRELLIS.2 MLX decoder surface.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXLD-01 | Slice 1 |
| PXLD-02 | Slice 1 |
| PXLD-03 | Slice 1 |
| PXLD-04 | Slice 1 |
| PXLD-05 | Slice 2 |

## Ordered Slice Sequence

### Slice 1: Shared Decoder Artifacts

**Objective:** Wire shared Pixal3D shape and texture decoder execution after `texture_slat.npz`, writing decoded NPZ artifacts and preserving structured blockers on decode failure.

**Acceptance criteria:**
- Compatible fake decoder assets plus explicit LR/HR/texture NAF complete shape decoder field extraction.
- Guided texture decoder uses shape decoder subdivisions and completes PBR voxel attribute decoding.
- Result artifacts include `shape_decoder_fields.npz` and `texture_decoder_pbr.npz` after the existing six Pixal3D artifacts.
- Runtime metadata records decoder model keys, config/checkpoint paths, decoded shapes, subdivision guide shapes, and decoder token limit.
- Decode failures return structured `shape-decoder` or `texture-decoder` blockers without losing already written artifacts.

**Touches:** `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/__init__.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_export.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q`

**Status:** complete
**Evidence:** added decoded Pixal3D shape and texture NPZ writers, wired `run_shape_decoder_to_fields` and `run_texture_decoder_to_representation` after `texture_slat.npz`, added compatible fake texture decoder fixtures, and covered both missing-texture-decoder and full fake decode paths. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q` -> `84 passed`.
**Risks / next:** Pixal3D still needs MLX NAF for normal CLI shape SLat entry and mesh extraction/PBR baking/GLB export after decoded artifacts.

### Slice 2: Docs And Release Hygiene

**Objective:** Update Pixal3D docs/script descriptions and prove the change is non-regressive.

**Acceptance criteria:**
- Docs describe `shape_decoder_fields.npz`, `texture_decoder_pbr.npz`, explicit NAF boundary, and remaining mesh/export blockers.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 1

**Touches:** `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-latent-decode-boundary/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, architecture, Pixal3D docs, scripts README, and script help text for `shape_decoder_fields.npz`, `texture_decoder_pbr.npz`, and the remaining mesh/export blocker. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `67 passed`; `uv run pytest -q` -> `874 passed, 10 skipped, 27 deselected`; Pixal3D runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; Pixal3D CLI and script help checks -> passed; release build/artifact/git hygiene -> produced and checked `mlx_spatial-0.0.3` sdist/wheel.
**Risks / next:** full Pixal3D support still needs MLX NAF and final Pixal3D mesh/PBR/GLB export.
