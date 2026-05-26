# Plan: Pixal3D Shape HR Cascade Boundary

## Goal

Execute [SPEC.md](SPEC.md): advance Pixal3D from 512 shape SLat into guarded HR coordinate planning and optional HR shape SLat probing.

## Architecture Approach

Pixal3D orchestration owns cascade stage order and artifacts. Coordinate upsample reuses the shared `trellis2_decode.py` shape decoder helper; HR SLat execution reuses the shared `trellis2_slat.py` probe.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXHR-01 | Slice 1 |
| PXHR-02 | Slice 1 |
| PXHR-03 | Slice 1 |
| PXHR-04 | Slice 2 |
| PXHR-05 | Slice 2 |
| PXHR-06 | Slice 3 |

## Ordered Slice Sequence

### Slice 1: HR Coordinate Cascade Artifact

**Objective:** Run shape decoder LR-to-HR coordinate upsample after 512 shape SLat and write a guarded HR coordinate artifact.

**Acceptance criteria:**
- Runtime uses `run_shape_decoder_upsample_coordinates` for compatible shape decoder assets.
- HR coordinates are quantized/deduplicated with Pixal3D's max-token guard.
- Artifact metadata records input LR shape, raw upsample shape, selected HR resolution/grid, token count, decoder model paths, and next blocker.
- Existing 512 shape SLat artifact behavior remains intact.

**Touches:** `src/mlx_spatial/pixal3d_camera.py`, `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/__init__.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_trellis2_decode.py -q`

**Status:** complete
**Evidence:** added `pixal3d_select_hr_coordinates`, `write_pixal3d_shape_hr_coordinates_npz`, fake compatible shape decoder assets, and runtime handoff from `shape_slat_lr.npz` through shared `run_shape_decoder_upsample_coordinates`; targeted verification passed inside the larger Pixal3D/decode/SLat set: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q` -> `78 passed`.
**Risks / next:** real Pixal3D upsample may still hit the configured token guard depending on learned subdivision density; the runtime reports that as a structured cascade blocker.

### Slice 2: Optional HR Shape SLat Probe

**Objective:** When explicit HR NAF features are supplied, run the 1024 shape SLat probe on guarded HR coordinates and write `shape_slat_hr.npz`; otherwise return a structured HR projection blocker.

**Acceptance criteria:**
- Existing lower-level LR NAF path blocks at HR projection conditioning when HR NAF is absent.
- Compatible fake 1024 shape SLat assets plus explicit HR NAF complete HR shape SLat probing.
- Result artifacts include `sparse_projection.npz`, `sparse_structure.npz`, `shape_slat_lr.npz`, HR coordinate NPZ, and `shape_slat_hr.npz`.
- Runtime metadata records HR projection selection, sampled HR feature shape, selected HR resolution/grid, and next blocker.

**Depends on:** Slice 1

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q`

**Status:** complete
**Evidence:** added API-only `shape_hr_naf_feature_map`, HR projection conditioning at the selected cascade grid, coordinate-indexed projected features, shared 1024 shape SLat probing, normalized `shape_slat_hr.npz` writing, and structured blockers for missing HR NAF or HR SLat failures. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_camera.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q` -> `78 passed`.
**Risks / next:** texture projection still needs MLX NAF or explicit texture NAF support before texture SLat can run.

### Slice 3: Docs And Release Hygiene

**Objective:** Update Pixal3D docs/script descriptions and prove the change is non-regressive.

**Acceptance criteria:**
- Docs describe the HR coordinate artifact, `shape_slat_hr.npz`, explicit NAF boundaries, and remaining texture/decode/export blockers.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 2

**Touches:** `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-shape-hr-cascade-boundary/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, architecture, Pixal3D docs, scripts README, and script help text to describe `shape_slat_hr_coordinates.npz`, `shape_slat_hr.npz`, explicit LR/HR NAF boundaries, and the remaining texture/decode/export blockers. Verification passed: `uv run pytest tests/test_pixal3d_*.py tests/test_trellis2_slat.py tests/test_trellis2_decode.py -q` -> `99 passed`; `uv run pytest -q` -> `868 passed, 10 skipped, 27 deselected`; forbidden runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; release build/artifact/git hygiene -> produced and checked `mlx_spatial-0.0.3` sdist/wheel; Pixal3D CLI and script help checks -> passed.
**Risks / next:** full Pixal3D support still needs MLX NAF, texture projection and texture SLat, full shape/texture decode, PBR baking, mesh extraction, and final GLB export.
