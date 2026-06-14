# mlx-spatialkit 4096 Texture Coverage Gate Spec

## Bounded Goal

Make the native Pixal3D reference-target export path sustain production texture coverage at `texture_size=4096` by replacing the fixed texture-fill dilation budget with an adaptive budget and verifying the real fixture under `/tmp`.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

The current reference-target path passes at `1024`, but a `4096` probe writes a valid GLB while failing production coverage. This is one of the remaining explicit production boundaries.

## Work Scale And Shape

- Scale: capability hardening
- Shape: quality parity and runtime diagnostics

## Selected Lenses

- **engineering:** Fix the native texture-fill behavior without adding dependencies or Python hot loops.
- **runtime:** Keep the 4096 path memory-bounded and observable through existing peak-memory telemetry.
- **product:** A developer should be able to request `texture_size=4096` and get an artifact-ready, coverage-threshold-passing GLB with honest diagnostics.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating Pixal3D decoded NPZ artifacts into GLB output through the `mlx-spatialkit` companion backend.

## Current Evidence

A `/tmp` probe using `quality_preset="reference-target"` and `texture_size=4096` produced:

- `artifact_ready=true`
- `production_quality_ready=false`
- `texture_size=4096`
- `final_visible_coverage_ratio=0.4189611077308655`
- `dilation_max_passes=8`
- `dilation_pass_count=8`
- `uv_surface_final_visible_coverage_ratio=0.5953780408617827`
- peak observed RSS about `3.89 GB`

The fixed 8-pass post-bake dilation is enough for the current 1024 gate but not for larger atlas tiles at 4096.

## Required Outcome

1. Native texture fill computes adaptive fallback and dilation budgets from atlas geometry and texture size instead of using fixed low budgets.
2. Texture bake diagnostics report the resolved adaptive fallback/dilation budgets and still report actual passes run.
3. Existing 1024 reference-target behavior remains passing.
4. A heavy real fixture gate proves `texture_size=4096` reference-target export reaches the production coverage threshold without changing geometry thresholds.
5. Docs explain that this closes 4096 texture coverage readiness for the native path, while xatlas chart parity and 1M-face setting parity remain separate.

## Coverage Targets

| ID | Target | Required evidence |
|---|---|---|
| T4096-01 | Adaptive fill budgets | Unit/focused test observes atlas-size-aware fallback/dilation budgets for atlas texture baking. |
| T4096-02 | 1024 regression safety | Existing heavy reference-target test remains passing. |
| T4096-03 | 4096 coverage readiness | Heavy 4096 export reports `final_visible_coverage_ratio >= 0.50` and `production_quality_ready=true`. |
| T4096-04 | Memory visibility | Heavy 4096 diagnostics include stage peak memory for `texture_bake` and `write_glb`. |
| T4096-05 | Honesty boundary | Docs keep xatlas chart parity and 1M-face setting parity deferred. |

## Constraints

- Keep heavy/generated artifacts under `/tmp` in tests and development runs.
- Do not add runtime dependencies.
- Do not change model inference or decoded NPZ generation.
- Do not relax production thresholds to make 4096 pass.
- Do not claim local 1024 checked-in reference GLB texture-resolution equality for a 4096 candidate.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Runtime cost risk:** More dilation passes can add CPU work at 4096. Mitigation: adaptive budget is bounded and diagnostics report pass count.
- **False parity risk:** Passing 4096 coverage is not xatlas chart parity. Mitigation: docs keep xatlas/1M-face boundaries explicit.
- **Memory risk:** 4096 textures allocate much larger buffers. Mitigation: existing max-pixel guard and peak-memory telemetry remain active.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| T4096-01 | Adaptive fallback/dilation budgets replace fixed low-budget behavior. | Focused texture bake test asserts atlas-size-aware `fallback_radius` and `dilation_max_passes` plus coherent pass stats. |
| T4096-02 | 1024 reference-target export still passes. | Existing heavy reference-target test remains green. |
| T4096-03 | 4096 reference-target export passes production coverage without threshold relaxation. | Heavy test asserts `texture_size=4096`, `final_visible_coverage_ratio >= 0.50`, `production_quality_ready=true`, and no production threshold warnings. |
| T4096-04 | Docs describe 4096 support and remaining boundaries. | README/Pixal3D/scripts docs mention adaptive dilation and remaining xatlas/1M-face gaps. |
| T4096-05 | Repo/package hygiene holds. | Package tests, root Pixal3D tests, build artifact inspection, and `git status --short` stay clean. |

## Scope Coverage Decisions

- **Included:** adaptive native dilation budget, diagnostics, focused texture bake test, heavy real 4096 export gate, docs, package/root/build verification.
- **Deferred:** xatlas chart parity, 1M-face export setting parity, exact perceptual scoring, changing the checked-in reference GLB, model inference reruns.
- **Anti-goals:** threshold relaxation, dependency additions, repo-checked generated GLBs, claims of exact upstream xatlas parity.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: the immediate 4096 blocker is native surface fill budget because the probe reached valid GLB output while leaving too many surface texels missing after nearest fallback and dilation.
