# mlx-spatialkit Parity Boundary Coherence Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-parity-boundary-coherence-gate/SPEC.md`: make visual-comparison deferred parity boundaries match the gates that are still truly open.

## Execution Routing And Topology

- Default execution: direct, serial.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Deferred Boundary Contract

**Objective:** Remove stale closed-gate labels from visual-comparison deferred boundaries while retaining true xatlas and 1M-face labels.

**Acceptance criteria:**
- `compare_textured_glbs` no longer emits `not_4096_texture_parity` or `not_browser_rendered_visual_proof` by default.
- `compare_textured_glbs` still emits `not_xatlas_chart_parity` and `not_1m_face_export_setting_parity`.
- The real reference-target heavy export test asserts the same boundary contract.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q -m 'not heavy' && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/tests/test_glb_compare.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Removed stale default `not_4096_texture_parity` and `not_browser_rendered_visual_proof` labels from `compare_textured_glbs(...)` while retaining `not_xatlas_chart_parity` and `not_1m_face_export_setting_parity`. Updated focused GLB comparison and real Pixal3D heavy assertions for both presence and absence. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` passed with `6 passed, 3 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `3 passed, 3 deselected` in `111.22s`; fresh `/tmp/mlx-spatialkit-reference-target-export-53210/diagnostics.json` reports deferred boundaries `["not_xatlas_chart_parity", "not_1m_face_export_setting_parity"]`.
**Risks / next:** Slice 2 must finish docs and full hygiene verification.

### Slice 2: Docs And Hygiene

**Objective:** Keep docs and packaging verification aligned with the updated diagnostics contract.

**Acceptance criteria:**
- Docs state that 4096 texture coverage and browser-render proof are closed gates.
- Docs keep xatlas chart parity and 1M-face export-setting parity as remaining boundaries.
- Full package/root/build hygiene passes and built artifacts contain no generated output.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` so closed 4096/browser gates are not described as default deferrals and the remaining visual parity boundaries are xatlas chart parity plus 1M-face export-setting parity. `git diff --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `49 passed, 3 deselected`. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the sdist and wheel into `/tmp`; artifact inspection found 36 sdist entries, 10 wheel entries, and no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, browser render artifacts, pycache, or pytest cache paths.
**Risks / next:** Continue into auto-verify; xatlas chart parity and 1M-face export-setting parity remain real deferred production work.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| PBC-01 | Slice 1 |
| PBC-02 | Slice 1 |
| PBC-03 | Slice 1 |
| PBC-04 | Slice 2 |
| PBC-05 | Slice 2 |

## Execution Notes

- Do not implement xatlas chart parity in this phase.
- Do not implement 1M-face export-setting parity in this phase.
- Keep generated and heavy artifacts under `/tmp`.
