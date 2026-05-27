# mlx-spatialkit Native Chart Split Position Gate Spec

## Bounded Goal

Improve native-chart low-fill splitting by evaluating bounded split positions per local axis and prove the real Pixal3D fixture advances without claiming xatlas parity.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no hidden half-stubbed behavior behind passing tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: bounded algorithmic parity-gap reduction

## Selected Lenses

- **engineering:** Improve the existing deterministic splitter instead of adding a dependency or relaxing quality gates.
- **runtime:** Keep split search bounded and native C++; expose candidate counts so cost remains visible.
- **product:** Move native-chart output closer to the intended Pixal3D export path while keeping non-equivalence explicit.

## Current Evidence

- Phase 27 improved reference-target chart rect fill to `0.5670824417746222` and xatlas utilization ratio to `0.6202011322387381`.
- Phase 27 kept explicit 1M/4096 native-chart readiness passing with UV-surface occupancy `0.50787752866745`.
- Current low-fill splitting evaluates both local axes but only at the median split position.

## Required Outcome

1. Low-fill chart splitting evaluates a small fixed set of split positions on each eligible local axis.
2. Diagnostics expose the split-position policy and total partition candidate count.
3. Focused tests prove deterministic split-position search on synthetic low-fill charts.
4. Real reference-target fixture tests prove chart fill or xatlas-utilization ratio improves beyond the Phase 27 baseline.
5. Explicit 1M/4096 native-chart readiness remains passing.
6. Docs describe bounded split-position search without claiming xatlas chart equivalence.
7. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCSP-01 | Low-fill splitting searches bounded positions per local axis. | Focused native-chart test asserts split-position diagnostics and deterministic output. |
| NCSP-02 | Reference-target native-chart boundary advances. | Heavy test asserts chart fill or xatlas-utilization ratio exceeds Phase 27 baseline. |
| NCSP-03 | Parity remains honest. | Heavy tests assert `parity_ready=false` and `xatlas_chart_parity=false`. |
| NCSP-04 | 1M/4096 native-chart gate remains ready. | Existing heavy 1M/4096 gate passes. |
| NCSP-05 | Docs match the boundary. | Docs state bounded split-position search improves native chart fill without xatlas parity. |
| NCSP-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Fixed split-position search, diagnostics, focused tests, real-fixture assertions, docs, verification.
- **Deferred:** Full xatlas-equivalent charting, xatlas dependency, default backend switch, release/tag/push work.
- **Anti-goals:** relaxing thresholds, removing parity deferrals, broad remesh rewrites, adding Python hot-path behavior.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Keep algorithm deterministic, native C++, bounded, and memory-conscious.
- Do not add `xatlas` to `packages/mlx-spatialkit` dependencies.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Runtime risk:** More split candidates increase low-fill split evaluation work. Mitigation: use a small fixed position set, retain depth/threshold guards, and verify heavy gates.
- **Quality risk:** More split candidates may increase chart count without useful occupancy gain. Mitigation: require real fixture improvement over Phase 27.

## Blocking Questions Or Assumptions

Assumption: a three-position search per axis is the right first bound because it captures off-center cuts while keeping the per-chart search small and deterministic.
