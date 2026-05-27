# mlx-spatialkit Repair Policy Contract Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-repair-policy-contract/SPEC.md`: expose the verified small-loop repair cap as a native/Python/export contract without changing the default behavior.

## Architecture Approach

Thread one integer setting through the existing native simplifier binding and Pixal3D export orchestration. The default remains `3`; `0` disables repair; negative values fail validation. Diagnostics keep reporting the effective value.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native And Python Contract

**Objective:** Expose `small_boundary_loop_fill_max_edges` in native and Python simplification APIs.

**Acceptance criteria:**
- Native binding accepts the parameter with default `3`.
- Python wrapper accepts the parameter with default `3`.
- `0` disables repair and reports disabled stats.
- Negative values raise a clear error.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/cpp/mesh_processing.hpp`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/mesh.py`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`

### Slice 2: Pixal3D Export Contract And Docs

**Objective:** Expose the repair cap through Pixal3D export settings, diagnostics, tests, and docs.

**Acceptance criteria:**
- `export_pixal3d_glb` accepts `small_boundary_loop_fill_max_edges`.
- Diagnostics record the effective setting.
- Heavy reference-target default remains green and records cap `3`.
- Docs and roadmap explain default `3` and disable value `0`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy && git diff --check`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| RPC-01 | Slice 1 |
| RPC-02 | Slice 1 |
| RPC-03 | Slice 2 |
| RPC-04 | Slice 1 |
| RPC-05 | Slice 2 |
| RPC-06 | Slice 2 |
| RPC-07 | Slices 1-2 |

## Verification Evidence

- Slice 1: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py -q` -> `14 passed`.
- Slice 2 heavy gate: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy` -> `1 passed`.
- Package suite: `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` -> `68 passed, 7 deselected`.
- Root Pixal3D integration: `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` -> `35 passed`.
- Hygiene: `git diff --check` passed; `/tmp/mlx-spatialkit-dist` build inspection reported wheel `10` entries, bad `0`, and sdist `36` entries, bad `0`.
