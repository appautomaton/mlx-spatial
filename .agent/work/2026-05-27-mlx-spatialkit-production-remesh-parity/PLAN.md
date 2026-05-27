# mlx-spatialkit Production Remesh Parity Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-production-remesh-parity/SPEC.md`: add a Pixal3D reference-target export path, threshold-gated production readiness, and native hot-path improvements against the real reference fixture.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-production-remesh-parity/DESIGN.md`. Production readiness is a threshold contract, not a label; native geometry and texture changes must improve that same real-fixture report.

## Execution Routing And Topology

- Default execution: direct, serial, continue after each verified slice.
- Parallel-safe groups: none; slices share export diagnostics and heavy fixture contracts.
- Checkpoints: none.
- Commit rhythm: commit after verified slice groups; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Reference Preset And Threshold Contract

**Objective:** Add a Pixal3D reference-target export preset and production-threshold diagnostics without changing the current preview backend's honesty.

**Acceptance criteria:**
- `export_pixal3d_glb(..., quality_preset="reference-target")` resolves target faces from `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json` when available.
- Diagnostics include `quality_preset`, resolved target settings, reference availability, and a structured production threshold block.
- `production_quality_ready` requires passing topology, face-count, coverage, and non-preview backend-tier checks.
- Preview/default export behavior remains backward-compatible and production readiness remains false.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py tests/test_mesh_processing.py -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Added `quality_preset="reference-target"` setting resolution in `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`; it loads the checked-in Pixal3D reference trace, resolves `target_faces=212542` from `final_faces`, records reference settings, and reports structured production thresholds. Updated `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py` to cover preset resolution, preset aliases, explicit target override, invalid preset validation, and threshold-gated production readiness. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py tests/test_mesh_processing.py -q` passed with `11 passed, 1 deselected`. Real `/tmp` probe with `quality_preset="reference-target"` wrote `/tmp/mlx-spatialkit-reference-preset-slice1-15702/diagnostics.json`; diagnostics show `artifact_ready=true`, `production_quality_ready=false`, `target_faces_source=reference_final_faces`, face-count threshold passed at ratio `0.9344882423238701`, backend-tier threshold failed (`geometry_aware_preview`), and final coverage threshold failed at `0.26881885528564453` vs required `0.5`.
**Risks / next:** Slice 2 must address or explicitly block the native production geometry candidate; Slice 3 must keep the heavy reference-target gate and docs aligned.

### Slice 2: Native Production Geometry Candidate

**Objective:** Add or select a native geometry path for the reference-target preset that is stronger than the current preview spatial-cluster path, while keeping failed production thresholds explicit.

**Acceptance criteria:**
- The production/reference preset reports either a non-preview native simplifier backend or an explicit `native_geometry_candidate_blocked` reason with measured evidence.
- Synthetic native tests prove the candidate consumes all source geometry and does not reintroduce degenerate, duplicate, or nonmanifold export blockers.
- Real-fixture diagnostics show whether face-count parity and topology thresholds pass at the reference target.
- No Python per-face/per-vertex hot loop is added.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m "not heavy"`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/mesh.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_mesh_processing.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

### Slice 3: Real Fixture Production-Preset Gate And Docs

**Objective:** Verify the reference-target preset on the real Pixal3D fixture, document the production-readiness boundary, and keep package/root artifacts clean.

**Acceptance criteria:**
- Heavy test exports the real decoded Pixal3D fixture under `/tmp` with `quality_preset="reference-target"`.
- Heavy diagnostics include reference face-count ratio, raw/final coverage ratios, threshold pass/fail details, timings, and RSS samples.
- Docs explain preview vs reference-target preset behavior and why production readiness can remain false.
- Package tests, root Pixal3D tests, build artifact inspection, and `git status --short` stay clean.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, package tests, root Pixal3D tests as needed

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| RMP-01 | Slice 1 |
| RMP-02 | Slice 2 |
| RMP-03 | Slice 1, Slice 3 |
| RMP-04 | Slice 3 |
| RMP-05 | Slice 1, Slice 3 |
| RMP-06 | Slice 3 |

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Preset/threshold contract | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py tests/test_mesh_processing.py -q` |
| Native geometry candidate | `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m "not heavy"` |
| Full package/root/build | `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` |

## Execution Notes

- Heavy outputs stay under `/tmp`.
- Do not mark the broad thread goal complete unless production thresholds pass and current evidence proves the full backend objective.
- If Slice 2 or Slice 3 proves UV/texture coverage is the dominant blocker, record that blocker instead of weakening the readiness threshold.
