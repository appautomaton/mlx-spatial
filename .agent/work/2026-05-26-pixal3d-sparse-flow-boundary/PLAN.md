# Plan: Pixal3D Sparse Flow Boundary

## Goal

Execute [SPEC.md](SPEC.md): advance Pixal3D runtime from sparse projection conditioning into sparse-structure FlowEuler probing and sparse-decoder handoff.

## Architecture Approach

Reuse `trellis2_sparse_structure.py` directly. Pixal3D orchestration owns only model-asset lookup, projection-conditioning dict assembly, trace metadata, and blocker translation.

## Execution Routing and Topology

Default route: direct, serial.

Parallel-safe groups: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
|---|---|
| PXSPARSE-01 | Slice 1 |
| PXSPARSE-02 | Slice 1 |
| PXSPARSE-03 | Slice 1 |
| PXSPARSE-04 | Slice 2 |
| PXSPARSE-05 | Slice 2 |

## Ordered Slice Sequence

### Slice 1: Sparse Flow Orchestration

**Objective:** Call the shared sparse-structure FlowEuler probe from Pixal3D runtime after sparse projection conditioning.

**Acceptance criteria:**
- Runtime resolves sparse flow and decoder model paths from Pixal3D config.
- Valid fake sparse-flow assets complete `sparse-structure-flow`.
- Sparse decoder config/checkpoint failure returns a structured `sparse-structure-decoding` blocker.
- Existing invalid fake roots still return a structured `sparse-structure-flow` blocker.

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `tests/pixal3d_fixtures.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_flow.py -q`

**Status:** complete
**Evidence:** updated `Pixal3DInferencePipeline.generate` to resolve Pixal3D sparse flow/decoder model assets, call `read_sparse_structure_flow_config`, pass projection conditioning as `{"global": ..., "proj": ...}` to `probe_sparse_structure_forward_boundary`, and then probe the sparse decoder boundary. Added a valid fake sparse-flow root fixture and runtime test proving `sparse-structure-flow` completion before a structured `sparse-structure-decoding` blocker. Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_flow.py -q` -> `12 passed`.
**Risks / next:** fake sparse decoder config intentionally remains invalid in this slice, so coordinate extraction and shape SLat sampling are still deferred.

### Slice 2: Docs and Verification

**Objective:** Update docs and prove the sparse-flow boundary is non-regressive.

**Acceptance criteria:**
- Pixal3D docs describe the new sparse-flow boundary and remaining decoder/SLat blockers.
- Pixal3D targeted tests and full suite pass.
- AST import scan, lock, diff, build, artifact, and git hygiene checks pass.

**Depends on:** Slice 1

**Touches:** `docs/pixal3d.md`, `README.md`, `.agent/work/2026-05-26-pixal3d-sparse-flow-boundary/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py -q && uv run pytest -q && uv lock --check && git diff --check && rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Status:** complete
**Evidence:** updated Pixal3D docs, README, architecture notes, and script docs to describe sparse FlowEuler probing as implemented and sparse decoder/shape-SLat as the remaining blocker. Verification passed: `uv run pytest tests/test_pixal3d_*.py -q` -> `49 passed`; full `uv run pytest -q` -> `856 passed, 10 skipped, 27 deselected, 2 warnings`; AST forbidden import scan passed; `uv lock --check` and `git diff --check` passed; `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl && python scripts/packaging/check_release_artifacts.py --git-hygiene` built and checked `dist/mlx_spatial-0.0.3.tar.gz` plus `dist/mlx_spatial-0.0.3-py3-none-any.whl`.
**Risks / next:** real Pixal3D weights are not present under `weights/pixal3d`, so this cycle proves orchestration with fake sparse-flow assets and leaves real checkpoint coordinate extraction for the next cycle.
