# Plan: Pixal3D MLX NAF Bridge

## Goal

Execute [SPEC.md](SPEC.md): replace caller-supplied Pixal3D NAF feature-map dependency with a Torch-free MLX NAF projection bridge.

## Architecture Approach

Add a small `naf.py` runtime module for converted NAF safetensors, image-encoder primitives, RoPE, and coordinate-sampled neighborhood attention. Keep Pixal3D orchestration in `pixal3d_inference.py`; `pixal3d_projection.py` remains the projection math owner and gains selected-coordinate helpers instead of forcing full HR feature maps.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXNAF-01 | Slice 1 |
| PXNAF-02 | Slice 1 |
| PXNAF-03 | Slice 2 |
| PXNAF-04 | Slice 2 |
| PXNAF-05 | Slice 2 |
| PXNAF-06 | Slice 3 |

## Ordered Slice Sequence

### Slice 1: NAF Assets And MLX Runtime

**Objective:** Add converted-NAF asset validation plus a Torch-free MLX NAF runtime capable of coordinate-sampled HR feature projection.

**Acceptance criteria:**
- Converted NAF safetensors are validated by expected tensor names and shapes.
- MLX image encoder, RoPE, and coordinate-sampled local attention run on small deterministic fixtures.
- Runtime import scan over the new NAF module rejects Torch/NATTEN/vendor imports.

**Touches:** `src/mlx_spatial/model_assets.py`, `src/mlx_spatial/naf.py`, `src/mlx_spatial/__init__.py`, `tests/test_naf.py`, `tests/pixal3d_fixtures.py`

**Verification:** `uv run pytest tests/test_naf.py tests/test_pixal3d_projection.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/naf.py`, NAF asset manifest/default exports, fake converted-weight fixtures, and focused NAF runtime tests for release checkpoint shape validation, safetensors loading, image encoder/RoPE output, and coordinate-sampled attention. Verification passed: `uv run pytest tests/test_naf.py tests/test_pixal3d_projection.py -q` -> `17 passed`; conversion smoke passed with `/tmp/mlx-spatial-naf/naf_release.pth` -> `/tmp/mlx-spatial-naf-convert/naf_release.safetensors`, then `load_naf_tensors` loaded 37 tensors with RoPE periods shape `(16,)`.

### Slice 2: Pixal3D Pipeline NAF Integration

**Objective:** Use the MLX NAF bridge for Pixal3D shape/HR/texture projection when explicit NAF maps are absent.

**Acceptance criteria:**
- Fake converted NAF weights let the pipeline advance past the old `shape-projection-conditioning` blocker without `shape_lr_naf_feature_map`.
- Explicit `shape_lr_naf_feature_map`, `shape_hr_naf_feature_map`, and `texture_naf_feature_map` still override runtime NAF for parity tests.
- Missing NAF assets produce a structured blocker with the local expected path and setup command.
- Metadata records NAF source, target size, coordinate count, chunk size, and full-map avoidance.

**Depends on:** Slice 1

**Touches:** `src/mlx_spatial/pixal3d.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d_projection.py`, `tests/test_pixal3d_pipeline.py`, `tests/test_pixal3d_projection.py`

**Verification:** `uv run pytest tests/test_naf.py tests/test_pixal3d_pipeline.py tests/test_pixal3d_projection.py -q`

**Status:** complete
**Evidence:** wired `naf_root` and `naf_coordinate_chunk_size` through `Pixal3DInferencePipeline`, package CLI, and script wrapper; runtime now loads converted NAF safetensors when explicit NAF maps are absent, uses coordinate-sampled projected HR features with full-map avoidance, and keeps explicit feature maps as overrides. Tests cover missing NAF assets returning `naf-assets` and fake converted NAF weights advancing past the previous shape-projection blocker into `shape_slat_lr.npz`. Verification passed: `uv run pytest tests/test_naf.py tests/test_pixal3d_pipeline.py tests/test_pixal3d_projection.py -q` -> `32 passed`.

### Slice 3: Docs And Release Hygiene

**Objective:** Document the NAF setup and prove the change is non-regressive.

**Acceptance criteria:**
- README, Pixal3D docs, architecture docs, and script help describe converted NAF assets and remaining blockers.
- Pixal3D/NAF targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 2

**Touches:** `README.md`, `docs/pixal3d.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-mlx-naf-bridge/PLAN.md`

**Verification:** `uv run pytest tests/test_naf.py tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, Pixal3D docs, architecture docs, scripts README, package CLI help, and script help for converted NAF assets, `--naf-root`, `--naf-coordinate-chunk-size`, full-map avoidance, and remaining MoGe boundary. Verification passed: `uv run pytest tests/test_naf.py tests/test_pixal3d_*.py -q` -> `76 passed`; full `uv run pytest -q` -> `883 passed, 10 skipped, 27 deselected`; Pixal3D/NAF runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; package CLI and script help checks -> passed; release build/artifact/git hygiene -> produced and checked `mlx_spatial-0.0.3` sdist/wheel.
