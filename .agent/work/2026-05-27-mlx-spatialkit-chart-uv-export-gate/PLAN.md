# mlx-spatialkit Chart UV Export Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-export-gate/SPEC.md`: add an opt-in native chart UV backend to `export_pixal3d_glb` and prove it through the real Pixal3D decoded fixture without changing the default backend.

## Architecture Approach

Add a small backend resolver in `mlx_spatialkit.export`, then branch only at the UV stage: `face-atlas` continues to call `make_face_atlas_uvs`, while `native-chart` calls `make_native_chart_uvs`. The texture bake already selects the binned Metal path when face-atlas stats are absent, so chart exports should naturally report `metal-uv-binned-nearest`. Diagnostics and GLB metadata will record the requested/resolved UV backend and chart angle.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Export API Contract

**Objective:** Add the `uv_backend` and `chart_angle_degrees` public contract while keeping the default face-atlas path unchanged.

**Acceptance criteria:**
- `export_pixal3d_glb` accepts `uv_backend="face-atlas"` and `uv_backend="native-chart"`.
- Invalid UV backend and invalid chart angle fail before heavy export work.
- Diagnostics settings record requested/resolved UV backend and chart angle.
- Existing default face-atlas tests remain valid.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_rejects_invalid_public_guards tests/test_real_pixal3d_export.py::test_export_pixal3d_uv_backend_settings_contract -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added `uv_backend`/`chart_angle_degrees` validation and diagnostics contract; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_rejects_invalid_public_guards tests/test_real_pixal3d_export.py::test_export_pixal3d_uv_backend_settings_contract -q` -> 2 passed.
**Risks / next:** Slice 2 must prove the chart backend on the real decoded fixture.

### Slice 2: Real Chart Export Proof

**Objective:** Prove the native chart backend writes a real Pixal3D GLB and uses the binned Metal texture bake path.

**Acceptance criteria:**
- Heavy decoded fixture with `uv_backend="native-chart"` writes `model.glb` and `diagnostics.json` under `/tmp`.
- UV diagnostics report `backend=native-chart-atlas`, chart count, chart packing grid, and duplicate ratio.
- Texture bake diagnostics report `backend=metal-uv-binned-nearest`, nonzero UV-bin face references, and sampled texels.
- Visual comparison still contains `not_xatlas_chart_parity`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** added heavy real fixture chart-backend test; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_backend_writes_real_fixture -q -m heavy` -> 1 passed. Observed diagnostics under `/tmp`: `chart_count=9588`, `output_faces=43632`, `texture_backend=metal-uv-binned-nearest`, `uv_bin_face_reference_count=73764`, `coverage=0.14284706115722656`.
**Risks / next:** chart backend is artifact-ready for preview-size real fixture, but not default and not xatlas parity.

### Slice 3: Regression, Docs, And Hygiene

**Objective:** Document the opt-in chart export boundary and verify existing Pixal3D export behavior remains stable.

**Acceptance criteria:**
- Docs explain `uv_backend="native-chart"` as an opt-in candidate, not default or xatlas parity.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** docs describe opt-in `uv_backend="native-chart"` decoded-NPZ export while preserving face-atlas defaults; `git diff --check` passed; package tests -> 58 passed, 5 deselected; root Pixal3D tests -> 35 passed; `/tmp` build succeeded; artifact inspection found bad 0 in wheel and sdist.
**Risks / next:** final verify must rerun the plan gates before marking the change verified.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| CUVX-01 | Slice 1 |
| CUVX-02 | Slice 1, Slice 3 |
| CUVX-03 | Slice 2 |
| CUVX-04 | Slice 2 |
| CUVX-05 | Slice 2, Slice 3 |

## Execution Notes

- Do not switch `export_pixal3d_glb` defaults from `face-atlas`.
- Do not remove `not_xatlas_chart_parity`.
- Keep generated and heavy artifacts under `/tmp`.
