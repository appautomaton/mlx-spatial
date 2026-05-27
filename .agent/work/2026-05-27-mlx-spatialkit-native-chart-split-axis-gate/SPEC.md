# mlx-spatialkit Native Chart Split Axis Gate Spec

## Bounded Goal

Improve native-chart low-fill splitting by evaluating both local split axes per candidate and prove the real Pixal3D fixture advances while xatlas parity remains explicit.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: bounded algorithmic parity-gap reduction

## Selected Lenses

- **engineering:** Improve the existing deterministic low-fill splitter instead of adding a new dependency or claiming parity.
- **runtime:** Keep the search bounded to two axes per candidate and preserve current depth/threshold guards.
- **product:** Move native-chart output closer to the intended Pixal3D/xatlas export path while reporting remaining non-equivalence.

## Current Evidence

- Phase 26 improved reference-target chart rect fill to `0.5647715200751198` and xatlas utilization ratio to `0.6131533138496904`.
- Phase 26 kept explicit 1M/4096 native-chart readiness passing with UV-surface occupancy `0.5047669410705566`.
- Current low-fill splitting uses only one `parent.split_axis` per candidate even though each chart has 2D local centroids.

## Required Outcome

1. Native low-fill splitting evaluates both local split axes for each eligible chart and accepts only an improving split.
2. Diagnostics expose the bounded axis-search policy and candidate count.
3. Real fixture tests assert reference-target fill or xatlas utilization ratio exceeds the Phase 26 baseline.
4. Explicit 1M/4096 native-chart readiness remains passing.
5. Docs explain this as bounded split-axis search, not xatlas equivalence.
6. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCAS-01 | Low-fill splitting searches both axes. | Focused test asserts axis-search diagnostics and deterministic output. |
| NCAS-02 | Reference-target native-chart boundary advances. | Heavy test asserts chart fill or xatlas-utilization ratio exceeds Phase 26 baseline. |
| NCAS-03 | Parity remains honest. | Heavy tests assert `parity_ready=false` and `xatlas_chart_parity=false`. |
| NCAS-04 | 1M/4096 native-chart gate remains ready. | Existing heavy 1M/4096 gate passes. |
| NCAS-05 | Docs match the boundary. | Docs state bounded split-axis search improves native chart fill without xatlas parity. |
| NCAS-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Two-axis low-fill split search, diagnostics, focused tests, real-fixture assertions, docs, verification.
- **Deferred:** Full xatlas-equivalent charting, xatlas dependency, default backend switch.
- **Anti-goals:** relaxing thresholds, removing parity deferrals, tagging/pushing/releasing.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Keep algorithm deterministic, native C++, and bounded.
- Do not add `xatlas` to `packages/mlx-spatialkit` dependencies.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Runtime risk:** Evaluating both axes doubles low-fill split evaluation work for candidates. Mitigation: keep depth and candidate guards unchanged and verify heavy runtime remains acceptable.
- **Quality risk:** Axis search may increase chart count without useful occupancy gain. Mitigation: require real fixture boundary improvement over Phase 26.

## Blocking Questions Or Assumptions

Assumption: a two-axis search is low-risk because it stays inside the existing bounded split policy and uses already-computed local chart centroids.
