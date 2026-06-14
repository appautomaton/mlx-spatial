# mlx-spatialkit Native Chart Minimum Padding Gate Spec

## Bounded Goal

Improve native-chart UV occupancy by tightening the Pixal3D native-chart backend default padding from `0.01` to `0.005` and prove the real fixture advances without claiming xatlas parity.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no hidden half-stubbed behavior behind passing tests.

## Work Scale And Shape

- Scale: native chart export quality improvement
- Shape: bounded settings parity-gap reduction

## Selected Lenses

- **engineering:** Change only the native-chart default padding from measured real-fixture evidence.
- **runtime:** Keep the C++/Metal hot path unchanged; tighter padding should improve occupancy without extra work.
- **product:** Move native-chart output closer to Pixal3D/xatlas utilization while preserving explicit non-equivalence.

## Current Evidence

- Phase 29 default native-chart padding `0.01` reports reference-target UV-surface occupancy `0.5597553253173828` and xatlas utilization ratio `0.6736181097830843`.
- A `/tmp` reference-target probe with explicit padding `0.005` reports UV-surface occupancy `0.5693244934082031`, xatlas utilization ratio `0.685133792850289`, final visible coverage `0.5693244934082031`, surface visible coverage `1.0`, and visual comparison passing.
- A `/tmp` 1M/4096 probe with explicit padding `0.005` reports upstream readiness true, native-chart quality-ready, UV-surface occupancy `0.5488805174827576`, and xatlas utilization ratio `0.660531199902127`.

## Required Outcome

1. The Pixal3D native-chart backend default tile padding is `0.005`.
2. Diagnostics and tests continue to report `settings.tile_padding` and `settings.tile_padding_source`.
3. Real reference-target native-chart tests prove UV-surface occupancy or xatlas-utilization ratio improves beyond the Phase 29 baseline.
4. Explicit 1M/4096 native-chart readiness remains passing.
5. Docs describe the tighter `0.005` native-chart padding boundary without claiming xatlas chart equivalence.
6. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCMP-01 | Native-chart backend default padding is `0.005`. | Resolver/unit and real-fixture tests assert the default value and source. |
| NCMP-02 | Reference-target native-chart occupancy advances. | Heavy test asserts UV-surface occupancy or xatlas-utilization ratio exceeds Phase 29 baseline. |
| NCMP-03 | Parity remains honest. | Heavy tests assert `parity_ready=false` and `xatlas_chart_parity=false`. |
| NCMP-04 | 1M/4096 native-chart gate remains ready. | Existing heavy 1M/4096 gate passes. |
| NCMP-05 | Docs match the boundary. | Docs state `0.005` native-chart default padding without xatlas equivalence. |
| NCMP-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Native-chart default padding, tests, docs, real-fixture assertions, verification.
- **Deferred:** Full xatlas-equivalent charting, xatlas dependency, further split-position changes, default backend switch, release/tag/push work.
- **Anti-goals:** relaxing readiness thresholds, removing parity deferrals, adding Python hot-path behavior, reducing padding below `0.005`.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Keep runtime hot-path behavior deterministic and native.
- Do not add `xatlas` to `packages/mlx-spatialkit` dependencies.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Visual risk:** Less gutter can increase chart bleeding in some viewers. Mitigation: do not go below the probed `0.005`; keep reference-target and 1M/4096 visual/readiness checks.
- **Regression risk:** 1M/4096 output could rely on wider padding. Mitigation: rerun the explicit 1M/4096 native-chart heavy gate.

## Blocking Questions Or Assumptions

Assumption: `0.005` is acceptable because both reference-target and 1M/4096 probes passed current visual/readiness gates and improved occupancy.
