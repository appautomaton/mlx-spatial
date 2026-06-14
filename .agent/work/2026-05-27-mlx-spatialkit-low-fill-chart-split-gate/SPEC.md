# mlx-spatialkit Low-Fill Chart Split Gate Spec

## Bounded Goal

Improve native chart UV quality by splitting low-fill charts deterministically and prove the real Pixal3D native-chart candidate advances beyond the Phase 20 global-coverage boundary.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: bounded C++ chart-splitting hardening

## Selected Lenses

- **engineering:** Improve chart-internal rectangle fill with bounded deterministic splitting rather than relaxing readiness thresholds.
- **runtime:** Keep chart count, duplicated vertices, and UV-bin references bounded and diagnosable.
- **product:** Move native chart output closer to visual parity while diagnostics remain honest about remaining blockers.

## Current Evidence

- Phase 20 native-chart output reports `projection_rotation_candidates=19` and `projection_rotation_step_degrees=5.0`.
- Latest real fixture diagnostics report `chart_rect_fill_ratio=0.5727071617508422`, `uv_surface_occupancy_ratio=0.5145282745361328`, `uv_surface_final_visible_coverage_ratio=0.7548626376681581`, and `global_coverage_ratio=0.3883981704711914`.
- The only current native-chart quality blocker is `global_coverage_floor`.
- Additional dilation probes were previously recorded as plateauing around `0.393`, so post-fill alone is not the next best quality lever.
- `atlas_rect_coverage_ratio=0.9853703780534515`, so shelf packing is no longer the measured bottleneck; chart-internal fill remains low.

## Required Outcome

1. `make_native_chart_uvs` splits low-fill charts through deterministic bounded native logic.
2. Chart UV stats expose low-fill split policy and counts, separate from oversized-chart splitting.
3. Focused tests prove a low-fill chart improves without breaking coplanar, hard-crease, rotated-rectangle, and oversized-chart behavior.
4. The real Pixal3D native-chart export improves `chart_rect_fill_ratio` or `global_coverage_ratio` over the Phase 20 baseline.
5. Docs describe low-fill splitting as a native candidate improvement, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| LFCS-01 | Low-fill splitting is deterministic and bounded. | Focused tests assert split diagnostics and stable output for repeated inputs. |
| LFCS-02 | Existing chart behavior remains intact. | Existing coplanar, hard-crease, rotated-rectangle, and oversized-chart tests still pass. |
| LFCS-03 | Real chart quality advances. | Heavy fixture reports `chart_rect_fill_ratio > 0.5727071617508422` or `global_coverage_ratio > 0.3883981704711914`. |
| LFCS-04 | Runtime guardrails remain honest. | Heavy fixture keeps UV-bin guard passing and reports remaining blockers without threshold relaxation. |
| LFCS-05 | Default export behavior remains unchanged outside native chart quality. | Full package/root Pixal3D tests still pass. |
| LFCS-06 | Docs match the parity boundary. | Docs say low-fill splitting improves the native candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** C++ low-fill chart splitting, stats, focused tests, real fixture proof, docs, regression/build verification.
- **Deferred:** LSCM/ABF unwrap, overlap removal, xatlas chart parity, default UV backend switch, texture bake kernel changes, threshold relaxation.
- **Anti-goals:** changing decoded model outputs, changing Metal bake kernels, relaxing chart readiness floors, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native C++ and deterministic for identical inputs.
- Keep runtime bounded with fixed split limits and existing UV-bin guards.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Low-fill splitting may increase atlas occupancy but not enough to clear global coverage. Mitigation: require real fixture proof and keep blocker diagnostics truthful.
- **Runtime risk:** More charts can increase duplicated vertices, UV-bin face references, and bake overhead. Mitigation: bounded split thresholds, guard assertions, and diagnostics.
- **Regression risk:** UV layout changes can affect texture bake results. Mitigation: focused tests, real chart heavy fixture, package/root tests, and build checks.

## Blocking Questions Or Assumptions

Assumption: bounded low-fill splitting is the next best chart-quality lever because packing is efficient, occupancy barely clears its floor, and post-fill probes previously plateaued below the global coverage floor.
