# Pixal3D GLB Quality Rebaseline Plan

## Goal

Implement [SPEC.md](SPEC.md): make Pixal3D GLB quality evidence honest and source-grounded before more native export fixes are treated as production progress.

## Architecture Approach

Treat every comparison as a manifest-bound A/B/C lineage:

- A: decoded Pixal3D model output and source/preprocess metadata.
- B: native `mlx-spatialkit` GLB plus diagnostics and optional browser proof.
- C: reference/control GLB for the same lineage.

Work proceeds from A-side generation contracts to B-side export attribution. Native remesh/QEM/open-boundary work stays blocked or explicitly scoped until diagnostics prove which layer is responsible.

The 55 `2026-05-27-mlx-spatialkit-*` work dirs are consolidated as historical evidence in [spec/legacy-2026-05-27-rollup.md](spec/legacy-2026-05-27-rollup.md). They are not the active backlog. The follow-on implementation direction is production-quality geometry/export parity, not another readiness or screenshot-tuning loop.

## Regression Guardrails

Every slice must preserve the working signals in [spec/gap-matrix.md](spec/gap-matrix.md): Pixal3D texture coordinate order, PBR channel packing, flexible dual-grid extraction/metrics, GLB writer/viewer compatibility, and existing fixture boundaries. Historical `.agent/work/*` artifacts are evidence only; this change supersedes stale meanings from the active spec/plan instead of rewriting old specs or plans.

## Execution Routing And Topology

Default continuation path: direct execution through all slices in order.

Parallel-safe groups: none for writes. Read-only subagents are useful for Slice 5 or Slice 6 audits, but implementation touches shared readiness and export contracts and should stay serial.

Checkpoint after: Slice 6 only if the evidence says full QEM or narrow-band remesh is required for the current goal.

## Ordered Slice Sequence

### Slice 1: Readiness And Historical Correction

**Objective:** Remove the false-completion path from the prior rendered-visual work and make readiness names impossible to confuse.

**Acceptance criteria:**
- Prior rendered-visual evidence is treated by the active change as a partial checkpoint, not a completed quality result.
- Diagnostics/tests separate `artifact_ready`, `rendered_visual_ready`, `browser_rendered_visual_proof`, `production_quality_ready`, and `production_equivalence_ready`.
- `visual_comparison.summary.all_passed` and browser visible-pixel checks cannot mark visual or production quality by themselves.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q -m 'not heavy'`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, `packages/mlx-spatialkit/tests/`, active slice evidence only.

**Status:** complete
**Evidence:** tightened `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` so production equivalence requires explicit `rendered_visual_ready=True` and no longer falls back from `visual_comparison.summary.all_passed`; updated `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py` to prove scalar visual pass alone is blocked. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` -> 14 passed, 7 deselected.
**Risks / next:** none for Slice 1; Slice 2 owns manifest lineage rather than readiness semantics.

### Slice 2: A/B/C Manifest Lineage

**Objective:** Make artifact provenance explicit enough that candidate/reference comparisons cannot silently pair the wrong outputs.

**Acceptance criteria:**
- Heavy comparison runs write `artifact-manifest.json` under `/tmp` with A/B/C roles, `lineage_id`, source image/preprocess variant, decoded paths, GLB paths, trace paths, command/settings, diagnostics, and browser-proof paths when present.
- Local fixture manifests exist for the base Pixal3D 1024 cascade fixture and the violin/bow lineages under `inputs/mlx-spatialkit`.
- Comparisons fail closed when A and C lineage does not match or when a manifest is missing/ambiguous.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q -m 'not heavy'`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/`, `packages/mlx-spatialkit/tests/`, `inputs/mlx-spatialkit/*/manifest.json`.

**Status:** complete
**Evidence:** added local fixture manifests for the base Pixal3D 1024 cascade, violin/bow raw, and violin/bow preprocessed-black lineages; `export_pixal3d_glb` now discovers and validates A/C fixture lineage, fails closed on mismatched or ambiguous manifests, and writes `/tmp/.../artifact-manifest.json` with A/B/C roles, settings, diagnostics, visual-parity, and browser-proof paths when present. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` -> 17 passed, 7 deselected.
**Risks / next:** run manifests are written by export; browser proof remains optional and is recorded when its augmented path is present.

### Slice 3: Pixal3D Input Preprocessing Parity

**Objective:** Feed Pixal3D generation with the same foreground-isolated, cropped, composited RGB image contract used by the vendor path.

**Acceptance criteria:**
- Pixal3D preprocessing handles RGBA alpha and RGB/RMBG cases with vendor-equivalent crop/composite behavior.
- The preprocessed image is used before MoGe, DINO, and NAF conditioning.
- Trace metadata records raw path, preprocess mode, effective image dimensions, and whether background removal was used.
- A tiny fixture catches the violin-style white-sheet/background leakage before full generation.

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_cli.py tests/test_trellis2_preprocess.py -q`

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/trellis2_preprocess.py` or a Pixal3D-specific helper, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_pipeline.py`, `tests/test_pixal3d_cli.py`.

**Status:** complete
**Evidence:** added `src/mlx_spatial/pixal3d_preprocess.py` with vendor-equivalent RGBA/RMBG crop and black-composite semantics; `Pixal3DInferencePipeline.generate` now preprocesses once and feeds the same RGB image to MoGe, DINO, and NAF, with trace metadata for raw path, mode, dimensions, alpha/RMBG source, crop box, and vendor reference. Added focused preprocessing tests, updated Pixal3D pipeline/CLI expectations, and left TRELLIS.2 preprocessing behavior untouched. `uv run pytest tests/test_pixal3d_preprocess.py tests/test_pixal3d_pipeline.py tests/test_pixal3d_cli.py tests/test_trellis2_preprocess.py -q` -> 41 passed. Exact plan command `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_cli.py tests/test_trellis2_preprocess.py -q` -> 37 passed.
**Risks / next:** CLI still fails closed for RGB/fully opaque images unless a Pixal3D-compatible RMBG background-remover hook is supplied through the Python API; Slice 4 owns stage-specific conditioning resolution, not background removal.

### Slice 4: Stage-Specific Conditioning Contract

**Objective:** Stop 1024 stages from accidentally using 512-resolution conditioning without an explicit blocker.

**Acceptance criteria:**
- `ss` and `shape_512` may use 512-resolution DINO/NAF tensors.
- `shape_1024` and `tex_1024` use 1024-resolution conditioning or fail closed on mismatched patch grids.
- Trace metadata records per-stage image size, patch grid, DINO/NAF source, and whether conditioning was reused.
- Tests cover helper-level patch-grid mismatch and pipeline-level per-stage conditioning requests without full 1024 generation.

**Verification:** `uv run pytest tests/test_pixal3d_projection.py tests/test_pixal3d_pipeline.py -q`

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d_projection.py`, `tests/test_pixal3d_projection.py`, `tests/test_pixal3d_pipeline.py`.

**Status:** complete
**Evidence:** added explicit DINO patch-grid validation to `src/mlx_spatial/pixal3d_projection.py` so upstream 512 stages require 32x32 patches and 1024 stages require 64x64 patches; `Pixal3DInferencePipeline` now treats caller-supplied `projection_hidden_states` as 512-only, accepts `projection_hidden_states_1024` for HR stages, lazily runs 1024 DINO when needed, and records per-stage image size, patch grid, DINO source/reuse, and NAF bridge metadata. Tests cover helper-level 512-vs-1024 mismatch, pipeline-level missing-1024 failure, and NAF bridge metadata. `uv run pytest tests/test_pixal3d_projection.py tests/test_pixal3d_pipeline.py -q` -> 39 passed.
**Risks / next:** supplied NAF feature maps are still trusted by shape rather than full semantic provenance; Slice 5 owns UV/bake/material attribution, not NAF model parity.

### Slice 5: UV, Bake, Material, And Viewer Attribution

**Objective:** Separate texture and surface symptoms into unwrap, sampling, postprocess/render padding, material packing, normals, and viewer-proof buckets.

**Acceptance criteria:**
- UV-only constant-field tests isolate chart occupancy from texture sampling.
- Sampling tests use fixed UVs and assert the full `coverage_status` histogram, not only `coverage_mask`.
- Postprocess tests toggle fill/render-padding stages one at a time and prove whether interior texels move.
- Material/normal tests include unequal roughness/metallic fixtures and keep viewer compatibility orthogonal to texture correctness.
- `compare_textured_glbs` is treated as a coarse heuristic, not spatial proof.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_writer.py tests/test_glb_compare.py -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/texture.py`, `packages/mlx-spatialkit/metal/texture_bake.mm`, `packages/mlx-spatialkit/metal/kernels/texture_bake.metal`, `packages/mlx-spatialkit/cpp/glb_writer.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/glb_compare.py`, related tests.

**Status:** complete
**Evidence:** added explicit coverage status legends/histograms and a native `surface_fill` toggle in `packages/mlx-spatialkit/src/mlx_spatialkit/texture.py`, `packages/mlx-spatialkit/cpp/bindings.cpp`, `packages/mlx-spatialkit/cpp/texture_bake.hpp`, and `packages/mlx-spatialkit/metal/texture_bake.mm`; added tests that isolate UV occupancy from texture values, assert full status histograms, toggle surface fill and render padding independently, preserve unequal metallic/roughness textures, and label `compare_textured_glbs` as coarse heuristic evidence rather than spatial proof. Rebuilt editable native package with `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit sync --reinstall-package mlx-spatialkit`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_writer.py tests/test_glb_compare.py -q` -> 39 passed.
**Risks / next:** comparison remains intentionally non-spatial; Slice 6 owns topology blocker classification rather than texture/material attribution.

### Slice 6: Topology Blocker Map

**Objective:** Decide whether the remaining quality failure is bounded repair work or a true production-remesh/QEM implementation.

**Acceptance criteria:**
- Metrics distinguish clean closed loops, simple open chains, branched open chains, non-manifold edges, heuristic QEM, and missing narrow-band remesh.
- The current real fixture's remaining topology blocker is classified from diagnostics, not screenshots.
- Stale Automaton specs for repair caps, open-boundary diagnostics, and simplification parity are read only as historical evidence and explicitly superseded by this active plan.
- No full QEM or narrow-band remesh implementation starts unless this slice proves it is the correct next scope.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m 'not heavy'`

**Touches:** `packages/mlx-spatialkit/cpp/mesh_metrics.cpp`, `packages/mlx-spatialkit/cpp/simplify.cpp`, `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, related tests.

**Status:** complete
**Evidence:** added `quality.topology_blocker_map` in `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` to classify artifact blockers, clean closed boundary loops, simple open chains, branched open chains, nonmanifold edges, heuristic QEM, and missing narrow-band remesh from `export_metrics` plus `simplify_stats`; added targeted tests for simple open-chain metrics and blocker-map classification. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` -> 36 passed, 7 deselected.
**Risks / next:** no valid Slice 6 checkpoint: diagnostics classify QEM/narrow-band as production backend gaps, but the approved next step is still two-fixture proof before broad remesh/QEM implementation.

### Slice 7: Two-Fixture Heavy Proof

**Objective:** Produce honest `/tmp` quality evidence on two independent Pixal3D fixture lineages after the contracts above are in place.

**Acceptance criteria:**
- Base Pixal3D 1024 cascade and one violin/bow lineage each produce a native GLB, manifest, diagnostics, visual comparison, and browser proof under `/tmp`.
- Readiness outcomes are honest: production blockers remain explicit instead of hidden behind passing structural/render counters.
- Heavy artifacts are not tracked in the repo.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q -m heavy && git diff --check && git status --short`

**Touches:** Runtime artifacts under `/tmp`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`, diagnostics if heavy evidence exposes missing fields.

**Status:** complete
**Evidence:** added a heavy native-chart proof for `inputs/mlx-spatialkit/violin-bow-preprocessed-black/pixal3d-1024-cascade-decoded-pbr` and corrected stale heavy expectations around opaque render-padded texture PNGs versus sampled model coverage. Clean heavy proof: `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q -m heavy` -> 8 passed, 13 deselected in 422.47s. Latest `/tmp` artifacts: turtle/city candidate `/tmp/mlx-spatialkit-native-chart-pixal3d-export-87002/model.glb`, violin/bow candidate `/tmp/mlx-spatialkit-violin-black-native-chart-export-87002/model.glb`; both have `artifact-manifest.json`, `diagnostics.json`, `visual_parity/visual_parity.json`, and browser render proof under `visual_parity/browser_render/`. Browser proof command succeeded for both with `browser_render.summary.all_passed=true`; the script now updates both `visual_parity.json` and `artifact-manifest.json` when `--artifact-manifest` is supplied. Fast regression checks: `uv run pytest tests/test_pixal3d_projection.py tests/test_pixal3d_pipeline.py tests/test_pixal3d_cli.py -q` -> 43 passed; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_texture_bake.py tests/test_glb_writer.py tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` -> 75 passed, 8 deselected; `node --check scripts/spatialkit/render_glb_visual_parity.cjs` -> passed; `git diff --check` -> passed.
**Risks / next:** Slice 7 proves multi-fixture export/readiness instrumentation, not final visual production quality. Both fixture lineages remain `artifact_ready=true` and `production_quality_ready=false`; diagnostics continue to identify open-boundary topology plus missing QEM edge-collapse and narrow-band remesh as the next production-quality gaps.

### Slice 8: Legacy Work Consolidation

**Objective:** Collapse the 2026-05-27 `mlx-spatialkit` micro-spec chain into the active rebaseline without rewriting historical artifacts.

**Acceptance criteria:**
- The 55 historical work dirs are summarized by bucket in the active change.
- Historical specs/plans remain forensic evidence, not active backlog.
- The active spec names production-quality geometry/export parity as the next implementation direction.

**Verification:** `git diff --check`

**Touches:** `.agent/work/2026-05-28-pixal3d-glb-quality-rebaseline/SPEC.md`, `.agent/work/2026-05-28-pixal3d-glb-quality-rebaseline/spec/legacy-2026-05-27-rollup.md`, active slice evidence only.

**Status:** complete
**Evidence:** added [spec/legacy-2026-05-27-rollup.md](spec/legacy-2026-05-27-rollup.md) to bucket the 55 `2026-05-27-mlx-spatialkit-*` work dirs and updated the active spec so remaining GLB quality work is focused on production geometry/export parity rather than readiness/browser/scalar gates.
**Risks / next:** after this rebaseline verifies, start a separate focused production-geometry implementation change if the user wants to pursue actual GLB quality closure.

## Requirement Traceability

| Requirement | Slices |
|---|---|
| PQR-01 | Slice 1 |
| PQR-02 | Slice 1, Slice 5, Slice 7 |
| PQR-03, PQR-10 | Slice 2, Slice 7 |
| PQR-04 | Slice 3 |
| PQR-05 | Slice 4 |
| PQR-06 | Slice 6 |
| PQR-07 | Slice 5 |
| PQR-08 | Slice 7 |
| PQR-09 | Slice 7 and final hygiene checks |
| PQR-11 | All slices through regression guardrails |
| PQR-12, PQR-13 | Slice 8 |

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Root Pixal3D input/conditioning | `uv run pytest tests/test_pixal3d_projection.py tests/test_pixal3d_pipeline.py tests/test_pixal3d_cli.py -q` |
| Spatialkit readiness/texture/geometry | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_glb_compare.py tests/test_texture_bake.py tests/test_glb_writer.py tests/test_mesh_processing.py tests/test_real_pixal3d_export.py -q -m 'not heavy'` |
| Heavy two-fixture proof | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q -m heavy` |
| Hygiene | `git diff --check && git status --short` |
