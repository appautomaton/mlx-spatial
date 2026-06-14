# Plan: Pixal3D Shape SLat LR Boundary

## Goal

Execute [SPEC.md](SPEC.md): advance Pixal3D from sparse coordinates into the 512 shape SLat probe and preserve the next blocker honestly.

## Architecture Approach

Pixal3D keeps orchestration, coordinate-indexed projection selection, and artifact serialization. Shape SLat model execution reuses the shared `trellis2_slat.py` MLX probe.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXSLAT-01 | Slice 1 |
| PXSLAT-02 | Slice 2 |
| PXSLAT-03 | Slice 2 |
| PXSLAT-04 | Slice 2 |
| PXSLAT-05 | Slice 3 |

## Ordered Slice Sequence

### Slice 1: Coordinate-Indexed Shape Projection

**Objective:** Add a Pixal3D helper that selects projected image features at sparse decoder coordinates.

**Acceptance criteria:**
- Helper maps `(batch, z, y, x)` coordinates into flattened projected feature rows.
- Helper validates coordinate shape, batch bounds, spatial bounds, grid size, and projected feature shape.
- Existing projection-conditioning behavior remains unchanged.

**Touches:** `src/mlx_spatial/pixal3d_projection.py`, `src/mlx_spatial/__init__.py`, `tests/test_pixal3d_projection.py`

**Verification:** `uv run pytest tests/test_pixal3d_projection.py -q`

**Status:** complete
**Evidence:** added `select_pixal3d_projected_features_at_coordinates`, exported it from the package facade, and covered deterministic `(batch, z, y, x)` feature selection plus bounds validation. Verification passed: `uv run pytest tests/test_pixal3d_projection.py -q` -> `12 passed`.
**Risks / next:** coordinate convention is explicit and internally consistent; later Torch parity can revisit if upstream sparse coordinate labels prove different.

### Slice 2: Shape SLat LR Runtime Handoff

**Objective:** Wire Pixal3D runtime to build shape projection conditioning, run the shared 512 shape SLat probe when NAF features and fake-compatible assets are available, write `shape_slat_lr.npz`, and block at the next cascade/decoder boundary.

**Acceptance criteria:**
- Existing sparse-coordinate root returns a structured NAF/shape projection blocker.
- Valid fake shape SLat assets plus explicit NAF feature map complete 512 shape SLat probing.
- Result artifacts include `sparse_projection.npz`, `sparse_structure.npz`, and `shape_slat_lr.npz`.
- Runtime metadata records shape SLat coordinate/feature shapes and next blocker.

**Touches:** `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/__init__.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_export.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py -q`

**Status:** complete
**Evidence:** added `write_pixal3d_shape_slat_npz`, fake Pixal3D 512 shape SLat assets, deterministic pipeline seeding, shape projection conditioning after sparse coordinates, coordinate-indexed projected features, shared `probe_shape_slat_forward_boundary` execution, normalization, and `shape_slat_lr.npz` artifact writing. Existing sparse-coordinate roots now stop at a structured shape projection/NAF blocker; fake NAF plus fake shape SLat assets reaches `shape-slat-cascade`. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_pixal3d_projection.py tests/test_pixal3d_flow.py tests/test_trellis2_slat.py -q` -> `54 passed`.
**Risks / next:** NAF itself, HR cascade upsample, shape decoder, texture SLat, texture/PBR decode, and GLB export remain incomplete.

### Slice 3: Docs And Release Hygiene

**Objective:** Update Pixal3D docs/script descriptions and prove the change is non-regressive.

**Acceptance criteria:**
- Docs describe `shape_slat_lr.npz`, the NAF feature boundary, and remaining HR cascade/decoder/export blockers.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 2

**Touches:** `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-shape-slat-lr-boundary/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, architecture, Pixal3D docs, script docs, and CLI help text to describe `shape_slat_lr.npz`, the explicit NAF feature-map handoff, the normal CLI NAF blocker, and the remaining HR cascade/decoder/export blockers. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `57 passed`; `uv run pytest -q` -> `864 passed, 10 skipped, 27 deselected`; forbidden runtime import scan -> passed; `uv lock --check` -> passed; `git diff --check` -> passed; release build/artifact/git hygiene -> produced and checked `mlx_spatial-0.0.3` sdist/wheel; `uv run mlx-spatial-pixal3d generate --help` and `uv run python scripts/pixal3d/generate.py --help` -> passed.
**Risks / next:** full Pixal3D support still needs MLX NAF, HR shape cascade/upsample, shape decoder, texture SLat, texture/PBR decode, and final GLB export.
