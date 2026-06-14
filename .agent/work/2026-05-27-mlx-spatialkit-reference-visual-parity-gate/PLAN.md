# mlx-spatialkit Reference Visual Parity Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-reference-visual-parity-gate/SPEC.md`: add deterministic GLB visual-comparison diagnostics for spatialkit reference-target exports against the checked-in Pixal3D reference GLB.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-reference-visual-parity-gate/DESIGN.md`. Keep comparison post-export and dependency-light: parse GLB/PNG bytes in Python, write sidecar reports under the export output directory, and keep native export hot paths unchanged.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none; comparator and export diagnostics share report contracts.
- Checkpoints: none.
- Commit rhythm: commit after verified slice groups; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: GLB Inspection And PNG Coverage

**Objective:** Add package-level GLB/PNG inspection utilities that can summarize embedded textured GLBs without external image dependencies.

**Acceptance criteria:**
- Parser validates GLB 2.0 JSON/BIN structure and extracts mesh primitive counts, material texture references, and image payloads.
- PNG coverage supports standard filter types used by the reference GLB.
- Unit tests cover unfiltered and filtered PNG rows plus GLB summaries.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/__init__.py`, `packages/mlx-spatialkit/tests/test_glb_compare.py`

**Status:** complete
**Evidence:** Added package-level `glb_compare.py` with GLB 2.0 parsing, mesh/material/image summaries, embedded PNG extraction, and standard PNG row-filter coverage handling. Exported `inspect_glb`, `parse_glb`, and `png_coverage`; added unit tests for PNG filters and native GLB summaries. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py -q` passed with `2 passed`.
**Risks / next:** Slice 2 must add candidate/reference comparison and sidecar report writing.

### Slice 2: Reference Visual Comparison Report

**Objective:** Implement candidate-vs-reference GLB comparison and sidecar report writing.

**Acceptance criteria:**
- Comparison returns candidate/reference summaries, face and vertex ratios, texture dimension checks, alpha/RGB coverage ratios, pass/fail checks, and deferred parity boundaries.
- Report writer emits `visual_parity.json`, extracted base-color PNGs, and a lightweight `index.html`.
- Tests prove report artifacts are written under the requested output directory and no generated artifacts are packaged.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_glb_writer.py -q`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/tests/test_glb_compare.py`

**Status:** complete
**Evidence:** Added `compare_textured_glbs(...)` with candidate/reference mesh, texture, coverage, pass/fail checks, deferred parity boundaries, and sidecar writing for `visual_parity.json`, `index.html`, and extracted base-color PNGs. Unit tests cover report generation and artifact paths. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_glb_compare.py tests/test_glb_writer.py -q` passed with `8 passed`.
**Risks / next:** Slice 3 must integrate the report into reference-target export diagnostics and assert the real reference GLB comparison.

### Slice 3: Export Diagnostics Integration

**Objective:** Wire visual comparison into reference-target Pixal3D export diagnostics when the checked-in reference GLB is available.

**Acceptance criteria:**
- `export_pixal3d_glb` writes `visual_parity/` sidecars next to the output GLB for reference-target exports with an available reference GLB.
- `diagnostics.json` includes compact `visual_comparison` summary and paths.
- Heavy real fixture test asserts report artifacts exist under `/tmp` and checks face ratio, texture resolution match, coverage ratio, and deferred parity boundary fields.
- Existing production readiness assertions remain unchanged.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Integrated `compare_textured_glbs(...)` into reference-target `export_pixal3d_glb` when the checked-in reference `model.glb` is available. Export diagnostics now include compact `visual_comparison` summary/checks/artifact paths, and sidecars are written under `visual_parity/` next to the generated GLB. Heavy fixture verification `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `2 passed, 3 deselected`. Latest `/tmp/mlx-spatialkit-reference-target-export-33741/diagnostics.json` reports visual comparison `all_passed=true`, face ratio `0.9344882423238701`, base-color alpha coverage ratio `0.602107048034668`, RGB coverage ratio `0.6020927528989454`, texture resolution match `1024x1024`, and sidecar JSON/HTML/PNG artifacts under `/tmp/mlx-spatialkit-reference-target-export-33741/visual_parity`.
**Risks / next:** Docs must explain that this is a deterministic texture/GLB comparison aid, not browser-rendered proof or xatlas/4096/1M parity.

### Slice 4: Docs And Full Verification

**Objective:** Document the visual-comparison report and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs explain where visual-comparison artifacts are written and what they do and do not prove.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 3

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to describe `visual_parity/` sidecars, visual-comparison metrics, and the boundary that this is not browser-rendered proof or xatlas/4096/1M parity. Full verification passed: `git diff --check`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `46 passed, 2 deselected`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the wheel and sdist under `/tmp`; artifact inspection found no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, pycache, or pytest cache entries.
**Risks / next:** Ready for independent Automaton verify. Browser-rendered GLB screenshots remain deferred.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| VPG-01 | Slice 1 |
| VPG-02 | Slice 3 |
| VPG-03 | Slice 2, Slice 3 |
| VPG-04 | Slice 3 |
| VPG-05 | Slice 4 |
| VPG-06 | Slice 4 |

## Execution Notes

- Do not relax production thresholds.
- Do not claim browser-rendered visual proof.
- Heavy/generated artifacts stay under `/tmp`.
- Broad thread goal remains open if visual comparison exposes xatlas, texture resolution, or export-setting gaps.
