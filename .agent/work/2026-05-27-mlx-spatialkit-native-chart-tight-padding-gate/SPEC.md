# mlx-spatialkit Native Chart Tight Padding Gate Spec

## Bounded Goal

Improve native-chart UV occupancy by tightening the Pixal3D native-chart backend default padding from `0.02` to `0.01` and prove the real fixture advances without claiming xatlas parity.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no hidden half-stubbed behavior behind passing tests.

## Work Scale And Shape

- Scale: native chart export quality improvement
- Shape: bounded settings parity-gap reduction

## Selected Lenses

- **engineering:** Improve the existing native-chart backend default from measured real-fixture evidence rather than adding a dependency or expanding chart splitting.
- **runtime:** Keep the hot path unchanged; tighter padding should improve occupancy without extra native work.
- **product:** Move native-chart output closer to the intended Pixal3D/xatlas export path while preserving explicit non-equivalence.

## Current Evidence

- Phase 28 reference-target default native-chart padding `0.02` reports UV-surface occupancy `0.5396347045898438` and xatlas utilization ratio `0.6494046474199308`.
- A `/tmp` probe with explicit native-chart padding `0.01` reports UV-surface occupancy `0.5597553253173828` and xatlas utilization ratio `0.6736181097830843`, with final visible coverage `0.5597553253173828` and surface visible coverage `1.0`.
- A more aggressive `0.005` probe improves further but leaves less chart gutter, so `0.01` is the conservative default change.
- Five-position split probes improved local rect fill but regressed UV-surface occupancy; local rect fill alone is not a reliable quality target.

## Required Outcome

1. The Pixal3D native-chart backend default tile padding is `0.01`.
2. Diagnostics and tests keep reporting `settings.tile_padding` and `settings.tile_padding_source` so callers can distinguish backend defaults from explicit padding.
3. Real reference-target native-chart tests prove UV-surface occupancy or xatlas-utilization ratio improves beyond the Phase 28 baseline.
4. Explicit 1M/4096 native-chart readiness remains passing.
5. Docs describe the tighter native-chart padding boundary without claiming xatlas chart equivalence.
6. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCTP-01 | Native-chart backend default padding is `0.01`. | Resolver/unit and real-fixture tests assert the default value and source. |
| NCTP-02 | Reference-target native-chart occupancy advances. | Heavy test asserts UV-surface occupancy or xatlas-utilization ratio exceeds Phase 28 baseline. |
| NCTP-03 | Parity remains honest. | Heavy tests assert `parity_ready=false` and `xatlas_chart_parity=false`. |
| NCTP-04 | 1M/4096 native-chart gate remains ready. | Existing heavy 1M/4096 gate passes. |
| NCTP-05 | Docs match the boundary. | Docs state `0.01` native-chart default padding without xatlas equivalence. |
| NCTP-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Native-chart default padding, tests, docs, real-fixture assertions, verification.
- **Deferred:** Full xatlas-equivalent charting, xatlas dependency, further split-position changes, default backend switch, release/tag/push work.
- **Anti-goals:** relaxing readiness thresholds, removing parity deferrals, adding Python hot-path behavior.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Keep runtime hot-path behavior deterministic and native.
- Do not add `xatlas` to `packages/mlx-spatialkit` dependencies.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Visual risk:** Less gutter can increase chart bleeding in some viewers. Mitigation: choose conservative `0.01` instead of `0.005`, keep viewer/visual gates, and preserve explicit caller override.
- **Regression risk:** 1M/4096 output could rely on the wider padding. Mitigation: rerun the explicit 1M/4096 native-chart heavy gate.

## Blocking Questions Or Assumptions

Assumption: `0.01` is the best bounded default for now because it improves measured occupancy materially while preserving more gutter than the more aggressive `0.005` probe.
