# mlx-spatialkit Chart Rect Fill Rotation Gate Spec

## Bounded Goal

Improve native chart rect-fill quality with a finer deterministic projection rotation search and prove the real Pixal3D chart candidate advances beyond the Phase 19 global-coverage boundary.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: bounded C++ projection-search hardening

## Selected Lenses

- **engineering:** Improve chart-internal rectangle fill instead of relaxing global coverage thresholds.
- **runtime:** Keep search deterministic and bounded; no global unwrap solve or external dependency.
- **product:** Move the native chart candidate closer to visual parity while diagnostics remain honest about remaining blockers.

## Current Evidence

- Phase 19 native-chart default padding clears the UV occupancy floor with `uv_surface_occupancy_ratio=0.5065326690673828`.
- Global coverage remains blocked at `0.36844539642333984`.
- Additional dilation probes plateau around `0.393`, so post-fill alone cannot solve global coverage.
- Latest diagnostics show high atlas rectangle coverage (`~0.984`) but low `chart_rect_fill_ratio=0.5649183023244753`; chart-internal fill is now the measured bottleneck.
- Current local projection evaluates only 7 rotation candidates per chart.

## Required Outcome

1. `make_native_chart_uvs` uses a finer deterministic bounded rotation search around the chart PCA frame.
2. Chart UV stats report the expanded candidate count and step size.
3. Focused tests prove rotated coplanar chart behavior remains deterministic and existing chart grouping/splitting behavior remains intact.
4. The real Pixal3D native-chart export improves `chart_rect_fill_ratio` or `global_coverage_ratio` over the Phase 19 baseline.
5. Docs describe bounded rotation search as a native candidate improvement, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| CRFG-01 | Projection search is deterministic and bounded. | Focused chart tests assert projection candidate count and step diagnostics. |
| CRFG-02 | Existing chart behavior remains intact. | Existing coplanar, hard-crease, and oversized-chart tests still pass. |
| CRFG-03 | Real chart quality advances. | Heavy fixture reports `chart_rect_fill_ratio > 0.5649183023244753` or `global_coverage_ratio > 0.36844539642333984`. |
| CRFG-04 | Default export behavior remains unchanged outside native chart quality. | Full package/root Pixal3D tests still pass. |
| CRFG-05 | Docs match the parity boundary. | Docs say bounded rotation search improves the native candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** C++ bounded rotation search, stats, focused tests, real fixture proof, docs, regression/build verification.
- **Deferred:** LSCM/ABF unwrap, overlap removal, xatlas chart parity, default UV backend switch, threshold relaxation.
- **Anti-goals:** changing decoded model outputs, changing Metal bake kernels, changing chart readiness floors, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native C++ and deterministic for identical inputs.
- Keep runtime bounded with a fixed small candidate count.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Finer rotation search may only marginally improve non-rectangular chart fill. Mitigation: require real fixture proof and keep blocker diagnostics truthful.
- **Runtime risk:** More candidates increase projection work. Mitigation: fixed 19-candidate search over already-local chart vertices.
- **Regression risk:** UV layout changes can affect texture bake results. Mitigation: focused tests, real chart heavy fixture, package/root tests, and build checks.

## Blocking Questions Or Assumptions

Assumption: a 19-candidate PCA-centered search is a useful next increment because chart rect fill, not packing or dilation, is the current measured blocker.
