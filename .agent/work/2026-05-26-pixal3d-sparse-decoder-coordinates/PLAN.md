# Plan: Pixal3D Sparse Decoder Coordinates

## Goal

Execute [SPEC.md](SPEC.md): write a Pixal3D sparse-structure coordinate artifact after successful sparse decoder probing, then block honestly at shape SLat.

## Architecture Approach

Reuse `trellis2_sparse_structure.py` for decoder execution and coordinate extraction. Pixal3D owns only artifact serialization, trace bookkeeping, and blocker translation.

## Execution Routing And Topology

Default route: direct, serial, continuation after verification.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
| --- | --- |
| PXCOORD-01 | Slice 1 |
| PXCOORD-02 | Slice 1 |
| PXCOORD-03 | Slice 1 |
| PXCOORD-04 | Slice 1, Slice 2 |
| PXCOORD-05 | Slice 2 |

## Ordered Slice Sequence

### Slice 1: Sparse Coordinate Artifact

**Objective:** Persist sparse decoder coordinates as a Pixal3D intermediate artifact and preserve the correct next blocker.

**Acceptance criteria:**
- Valid fake sparse-flow plus sparse-decoder assets reach `sparse-structure-decoding`.
- Runtime writes `sparse_structure.npz` with coordinates, decoded shape, target resolution, and pipeline metadata.
- Result artifacts include both sparse projection and sparse structure files.
- Successful sparse coordinate extraction blocks at `shape-slat-sampling`, not `sparse-structure-decoding`.
- Existing sparse-flow failure behavior remains structured.

**Touches:** `src/mlx_spatial/pixal3d_export.py`, `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/__init__.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_pipeline.py`, `tests/test_pixal3d_export.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_trellis2_sparse_structure.py -q`

**Status:** complete
**Evidence:** added `write_pixal3d_sparse_structure_npz`, exported the sparse-structure artifact type/writer, wired `Pixal3DInferencePipeline.generate` to write `sparse_structure.npz` after non-empty sparse decoder coordinates, and added fake sparse-decoder assets plus pipeline/export tests. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_export.py tests/test_trellis2_sparse_structure.py -q` -> `26 passed`.
**Risks / next:** shape SLat remains the intentional next blocker; real Pixal3D weights are not present locally for real-checkpoint validation.

### Slice 2: Docs And Release Hygiene

**Objective:** Update the user-facing Pixal3D docs/script description and prove the change is non-regressive.

**Acceptance criteria:**
- Pixal3D docs and script docs list `sparse_structure.npz` when sparse decoder coordinates are produced.
- Docs state that shape SLat, texture SLat, high-resolution NAF, and GLB export remain incomplete.
- Pixal3D targeted tests, full suite, import scan, lock, diff, build, artifact, and git hygiene pass.

**Depends on:** Slice 1

**Touches:** `docs/pixal3d.md`, `README.md`, `docs/architecture.md`, `scripts/README.md`, `scripts/pixal3d/generate.py`, `.agent/work/2026-05-26-pixal3d-sparse-decoder-coordinates/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated README, Pixal3D docs, architecture notes, script docs, and the Pixal3D script help text to list `sparse_structure.npz` after sparse decoder coordinates and name `shape-slat-sampling` as the next blocker. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `52 passed`; `uv run pytest -q` -> `859 passed, 10 skipped, 27 deselected, 2 warnings`; AST forbidden runtime import scan passed; `uv lock --check`, `git diff --check`, clean `uv build`, artifact checker, git hygiene, and Pixal3D CLI/script help checks passed.
**Risks / next:** this cycle still does not prove real TencentARC Pixal3D checkpoint output quality because local `weights/pixal3d` is not present; the runtime boundary is now shape SLat sampling.
