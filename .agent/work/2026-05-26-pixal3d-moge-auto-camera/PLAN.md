# Plan: Pixal3D MoGe Auto-Camera

## Goal

Execute [SPEC.md](SPEC.md): replace Pixal3D's hard manual-FOV requirement with a Torch-free MLX MoGe auto-camera path.

## Architecture Approach

Keep Pixal3D camera math in `pixal3d_camera.py`. Reuse `sam3d_moge.py` only from `pixal3d_inference.py`, where model execution and blockers already live. Manual FOV stays the cheap deterministic path; auto-camera is opt-in by omission of `manual_fov`.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXMG-01 | Slice 1 |
| PXMG-02 | Slice 2 |
| PXMG-03 | Slice 2 |
| PXMG-04 | Slice 2 |
| PXMG-05 | Slice 3 |

## Ordered Slice Sequence

### Slice 1: Pure Camera Helper

**Objective:** Add and test the MoGe-intrinsics-to-Pixal3D-camera conversion helper.

**Acceptance criteria:**
- Helper computes `camera_angle_x = 2 * atan(width / (2 * fx))` from normalized MoGe `fx`.
- Helper reuses the existing Pixal3D distance formula and records image width/resolution/mesh scale.
- Invalid intrinsics or image width are rejected.

**Touches:** `src/mlx_spatial/pixal3d_camera.py`, `src/mlx_spatial/__init__.py`, `tests/test_pixal3d_camera.py`

**Verification:** `uv run pytest tests/test_pixal3d_camera.py -q`

**Status:** complete

**Evidence:** added `pixal3d_camera_params_from_moge_intrinsics` in `src/mlx_spatial/pixal3d_camera.py`, exported it from `src/mlx_spatial/__init__.py`, and covered upstream normalized-intrinsics FOV math plus invalid input guards in `tests/test_pixal3d_camera.py`. `uv run pytest tests/test_pixal3d_camera.py -q` -> 7 passed.

**Risks / next:** none; continue to pipeline and CLI integration.

### Slice 2: Pipeline And CLI Integration

**Objective:** Run the existing MLX MoGe pointmap/intrinsics stage for Pixal3D camera setup when manual FOV is absent.

**Acceptance criteria:**
- Monkeypatched ready MoGe output lets Pixal3D progress through `camera-setup` without `manual_fov`.
- Missing or blocked MoGe returns a structured Pixal3D blocker with root/profile metadata.
- Manual FOV path keeps previous behavior and does not call MoGe.
- CLI and script expose `--moge-root` and `--moge-memory-profile`.

**Depends on:** Slice 1

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d.py`, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_camera.py tests/test_pixal3d_pipeline.py tests/test_sam3d_moge.py -q`

**Status:** complete

**Evidence:** wired `Pixal3DInferencePipeline.generate` to use the existing MLX MoGe pointmap/intrinsics runtime when `manual_fov` is omitted, added `--moge-root` and `--moge-memory-profile` to the package CLI and script, and added tests for MoGe auto-camera success, MoGe camera blockers, and manual-FOV no-MoGe override. `uv run pytest tests/test_pixal3d_camera.py tests/test_pixal3d_pipeline.py tests/test_sam3d_moge.py -q` -> 31 passed.

**Risks / next:** docs and final hygiene still need to be run.

### Slice 3: Docs And Release Hygiene

**Objective:** Document the auto-camera path and prove the change is non-regressive.

**Acceptance criteria:**
- README, Pixal3D docs, architecture docs, scripts README, and help text describe MoGe root/profile plus manual override.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 2

**Touches:** `README.md`, `docs/pixal3d.md`, `docs/architecture.md`, `scripts/README.md`, `.agent/work/2026-05-26-pixal3d-moge-auto-camera/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete

**Evidence:** updated README, Pixal3D docs, architecture docs, scripts README, package CLI help, and script help for MoGe auto-camera defaults, `--moge-root`, `--moge-memory-profile`, manual-FOV override, and remaining NAF/asset blockers. Verification passed: `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q` -> 82 passed; `uv run pytest -q` -> 888 passed, 10 skipped, 27 deselected; Pixal3D/NAF/MoGe runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; package CLI and script help checks -> passed; stale MoGe/manual-FOV docs scan -> passed; `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` built `dist/mlx_spatial-0.0.3.tar.gz` and `dist/mlx_spatial-0.0.3-py3-none-any.whl`; artifact checker and git hygiene passed.

**Risks / next:** exact upstream MoGe v2 parity is not claimed; broader Pixal3D completion still depends on real-weight end-to-end generation quality and any remaining parity gaps outside this cycle.
