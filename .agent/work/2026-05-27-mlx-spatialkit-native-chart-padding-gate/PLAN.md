# mlx-spatialkit Native Chart Padding Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-padding-gate/SPEC.md`: make native-chart Pixal3D exports use a tighter backend-aware padding default and prove the real fixture clears the UV occupancy floor.

## Architecture Approach

Change `export_pixal3d_glb` to accept `tile_padding=None` as the default and resolve it after UV backend normalization. Keep face-atlas backend default at `0.08`; use `0.02` for native-chart. Store both the resolved float and source string in diagnostics. Pass the resolved value to UV generation.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Backend-Aware Padding Contract

**Objective:** Implement backend-aware tile padding resolution and focused contract tests.

**Acceptance criteria:**
- Resolver tests pass for face-atlas default, native-chart default, and explicit override.
- Invalid/non-finite padding still raises before file validation.
- Diagnostics settings use the resolved padding value and source.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_uv_backend_settings_contract -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added backend-aware tile-padding resolver, diagnostics source field, and focused resolver tests; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_uv_backend_settings_contract -q` -> 1 passed.
**Risks / next:** Slice 2 must prove the real fixture clears the occupancy floor with the default native-chart path.

### Slice 2: Real Native-Chart Occupancy Proof

**Objective:** Prove the default native-chart real fixture clears the UV-surface occupancy floor.

**Acceptance criteria:**
- Heavy chart fixture writes GLB and diagnostics under `/tmp`.
- Diagnostics report `tile_padding=0.02` and `tile_padding_source=backend_default:native-chart`.
- `quality.native_chart_uv_candidate.uv_surface_occupancy_ratio > 0.50`.
- `quality.native_chart_uv_candidate.quality_blockers` does not include `uv_surface_occupancy_floor`.
- Readiness diagnostics remain truthful if global coverage still fails.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** heavy fixture passed with backend default native-chart padding; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Latest `/tmp` diagnostics show `tile_padding=0.02`, `tile_padding_source=backend_default:native-chart`, `uv_surface_occupancy=0.5065326690673828`, `global_coverage=0.36844539642333984`, and quality blockers `["global_coverage_floor"]`.
**Risks / next:** global coverage remains below the 0.50 floor.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document the backend-aware native-chart padding default and verify package/root/build stability.

**Acceptance criteria:**
- Docs describe tighter native-chart padding as native candidate quality setting, not xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe backend-aware native-chart padding and the non-xatlas boundary; final verify reran `git diff --check`, package tests -> 62 passed, 5 deselected, root Pixal3D tests -> 35 passed, `/tmp` build succeeded, and artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** none for this phase.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCPG-01 | Slice 1 |
| NCPG-02 | Slices 1 and 2 |
| NCPG-03 | Slice 2 |
| NCPG-04 | Slice 3 |
| NCPG-05 | Slice 3 |

## Execution Notes

- Do not change the default UV backend.
- Do not change chart readiness floors.
- Keep generated and heavy artifacts under `/tmp`.
