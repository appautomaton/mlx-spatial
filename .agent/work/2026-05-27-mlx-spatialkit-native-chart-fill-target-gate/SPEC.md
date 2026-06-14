# mlx-spatialkit Native Chart Fill Target Gate Spec

## Bounded Goal

Improve native-chart Pixal3D UV fill toward the measured xatlas reference by tuning bounded low-fill chart splitting and proving the real fixture advances without relaxing readiness thresholds.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: parity gap reduction

## Selected Lenses

- **engineering:** Use measured xatlas parity diagnostics to improve the current bottleneck instead of adding another label.
- **runtime:** Keep the algorithm deterministic, native, bounded, and safe for Apple Silicon memory/runtime behavior.
- **product:** Move native-chart outputs closer to the intended Pixal3D export path while still refusing to claim xatlas parity.

## Current Evidence

- Phase 25 measured reference xatlas unwrap as `unwrap_chart_count=51953` and `unwrap_utilization=0.8309683442115784`.
- Latest native reference-target chart diagnostics: chart count `34507`, rect fill `0.5637785177491498`, UV-surface occupancy `0.5047416687011719`, parity ready false.
- Latest native 1M/4096 chart diagnostics: chart count `118575`, rect fill `0.5520982533517874`, UV-surface occupancy `0.5028553009033203`, parity ready false.
- `/tmp/mlx-spatialkit-chart-angle-sweep.json` shows 45 degrees is already the best exposed angle in the tested set, with rect fill `0.5637785177491498`; changing only `chart_angle_degrees` is not enough.

## Required Outcome

1. Native low-fill chart splitting advances the real reference-target chart fill or xatlas utilization ratio beyond the Phase 25 baseline.
2. The implementation remains deterministic and bounded, with diagnostics exposing any changed split thresholds/depth.
3. Real fixture tests assert the improved boundary while keeping `xatlas_chart_parity.parity_ready=false`.
4. Docs explain the improvement as native chart gap reduction, not xatlas equivalence.
5. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCFG-01 | Reference-target native-chart fill advances. | Heavy test asserts chart rect fill or xatlas utilization ratio exceeds the Phase 25 baseline. |
| NCFG-02 | Implementation remains bounded. | Focused tests assert deterministic split metadata and finite max-depth/split thresholds. |
| NCFG-03 | Parity remains honest. | Heavy tests still assert `parity_ready=false` and `xatlas_chart_parity=false`. |
| NCFG-04 | 1M/4096 native-chart gate remains ready. | Existing heavy 1M/4096 gate passes with upstream and chart readiness intact. |
| NCFG-05 | Docs match the boundary. | Docs state native chart fill improved toward xatlas metrics without claiming xatlas equivalence. |
| NCFG-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Bounded native low-fill split tuning, diagnostics/test threshold updates, docs, package/root/build verification.
- **Deferred:** Full native xatlas-equivalent charting, xatlas package dependency, default backend switch.
- **Anti-goals:** relaxing production thresholds, claiming xatlas parity, tagging/pushing/releasing.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Keep chart generation deterministic and native C++.
- Do not add `xatlas` to `packages/mlx-spatialkit` dependencies.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Over-splitting risk:** More splits may inflate chart count without meaningful utilization gain. Mitigation: require real fixture boundary improvement and keep diagnostics visible.
- **Performance risk:** Deeper split evaluation could add CPU work. Mitigation: chart generation is currently sub-second on the reference-target simplified mesh; keep bounded depth.

## Blocking Questions Or Assumptions

Assumption: improving bounded low-fill splitting is the most pragmatic next step because exposed chart angle sweeps do not close the measured xatlas gap.
