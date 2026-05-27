# mlx-spatialkit UV Surface Fill Gate Spec

## Bounded Goal

Improve native chart texture coverage by adding bounded native UV-surface hole fill and prove the real Pixal3D native-chart candidate clears the global coverage floor without threshold relaxation.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native texture coverage improvement
- Shape: bounded native post-bake fill hardening

## Selected Lenses

- **engineering:** Fill remaining UV-surface holes in native code while preserving raw/exact coverage diagnostics.
- **runtime:** Keep the fill O(texture pixels), bounded in memory, and independent of Python loops.
- **product:** Move native chart output from quality-blocked toward quality-ready while still deferring xatlas chart parity.

## Current Evidence

- Phase 21 native-chart output reports `chart_rect_fill_ratio=0.576915029519614`, `uv_surface_occupancy_ratio=0.5232105255126953`, `uv_surface_final_visible_coverage_ratio=0.7556368090466`, and `global_coverage_ratio=0.3953571319580078`.
- The only current native-chart quality blocker is `global_coverage_floor`.
- Because UV-surface occupancy already exceeds `0.50`, filling remaining surface holes can clear global coverage if surface visibility approaches full coverage.
- Raw/exact coverage remains low and must stay visible in diagnostics so final fill cannot masquerade as exact model sampling.

## Required Outcome

1. Native texture baking fills remaining UV-surface missing/out-of-grid texels through bounded native logic after exact/fallback/dilation stages.
2. Texture stats report the surface-fill policy and filled texel count separately from exact sampled and nearest-voxel fallback counts.
3. Focused texture tests prove missing surface texels are filled while no-face texels remain unfilled.
4. The real Pixal3D native-chart export clears `global_coverage_floor` with truthful raw/exact/final coverage diagnostics.
5. Docs describe UV-surface fill as a native texture candidate improvement, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| USFG-01 | Surface fill is bounded and native. | Focused texture tests assert surface-fill diagnostics and no-face preservation. |
| USFG-02 | Exact/raw diagnostics remain separate. | Tests assert exact sampled, fallback, surface-filled, and final-visible counts are distinct. |
| USFG-03 | Real native-chart coverage clears the floor. | Heavy fixture reports `quality.native_chart_uv_candidate.global_coverage_ratio >= 0.50` and `quality_ready=true`. |
| USFG-04 | Runtime guardrails remain honest. | Heavy fixture keeps UV-bin guard passing and xatlas parity false. |
| USFG-05 | Default export behavior remains unchanged outside texture fill. | Full package/root Pixal3D tests still pass. |
| USFG-06 | Docs match the parity boundary. | Docs say UV-surface fill improves the native candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** Native UV-surface fill, texture stats, focused texture tests, real fixture proof, docs, regression/build verification.
- **Deferred:** xatlas chart parity, additional chart splitting, default UV backend switch, decoded model output changes, threshold relaxation.
- **Anti-goals:** relaxing chart readiness floors, hiding raw/exact coverage, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native and deterministic for identical inputs.
- Keep runtime bounded by texture pixel count and avoid repeated full-texture copy passes for the final surface fill.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Visual risk:** Surface fill may smear colors across remaining holes. Mitigation: preserve exact/raw coverage stats and report fill counts explicitly.
- **Runtime risk:** Full-texture fill can add memory pressure at 4096. Mitigation: use a bounded queue over texture pixels and avoid Python loops.
- **Regression risk:** Coverage-status semantics change. Mitigation: focused texture tests, real chart heavy fixture, package/root tests, and build checks.

## Blocking Questions Or Assumptions

Assumption: filling remaining UV-surface holes is acceptable as final texture coverage only when raw/exact/fallback/surface-fill diagnostics remain separate.
