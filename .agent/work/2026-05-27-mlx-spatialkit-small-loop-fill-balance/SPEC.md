# mlx-spatialkit Small Loop Fill Balance Spec

## Bounded Goal

Tune small boundary-loop repair to preserve more UV coverage while retaining measurable geometry-hole reduction.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and visually comparable Pixal3D exports before production readiness is claimed.

## Selected Lenses

- **engineering:** Correct the repair policy from the real fixture tradeoff, not from local topology metrics alone.
- **runtime:** Keep the repair native, deterministic, and bounded.
- **product:** Preserve visual/texture quality while still reducing small geometry holes.

## Current Evidence

- Verified cap-4 small-loop repair: `boundary_loop_count=1479`, `xatlas_utilization_ratio=0.680372125614308`, `visual_all=true`.
- `/tmp` cap-3 probe: `boundary_loop_count=1872`, `xatlas_utilization_ratio=0.6828063257125282`, `visual_all=true`.
- Pre-repair geometry-diagnostics baseline: `boundary_loop_count=2594`, `xatlas_utilization_ratio=0.6941716645020964`.

## Required Outcome

1. The native topology-aware repair fills only triangular closed boundary loops by default.
2. Focused tests prove triangular holes are filled and 4-edge holes are preserved.
3. Heavy Pixal3D reference-target export preserves measurable geometry improvement (`boundary_loop_count < 2594`) while improving UV utilization beyond the cap-4 policy (`xatlas_utilization_ratio > 0.680372125614308`).
4. Existing production readiness, visual comparison, and xatlas non-equivalence contracts remain intact.
5. Docs and roadmap describe the repair/UV balance choice and the remaining boundaries.
6. Heavy/generated artifacts stay under `/tmp`; no push, tag, publish, or release metadata work.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| SLFB-01 | Repair cap is balanced for UV quality. | `small_boundary_loop_fill_max_edges == 3` in native stats. |
| SLFB-02 | Focused behavior is explicit. | Tests fill a triangular closed loop and preserve a 4-edge loop. |
| SLFB-03 | Heavy topology still improves. | Reference-target `boundary_loop_count < 2594` and `nonmanifold_edges=0`. |
| SLFB-04 | Heavy UV tradeoff improves versus cap 4. | Reference-target `xatlas_utilization_ratio > 0.680372125614308`. |
| SLFB-05 | No false parity claim. | Xatlas chart parity/equivalence remain false. |
| SLFB-06 | Docs and hygiene hold. | Docs/roadmap updated; package/root/build checks pass. |

## Scope Coverage Decisions

- **Included:** Change default small-loop cap from 4 to 3, focused tests, heavy gate, docs/roadmap refresh.
- **Deferred:** Adaptive repair selection, open-boundary repair, larger-loop planar fill, UV xatlas parity.
- **Anti-goals:** Chasing local loop count at the cost of texture coverage, relaxing visual checks, claiming xatlas parity.

## Constraints

- Keep implementation native C++.
- Keep repair stats explicit.
- Preserve target-face budget and nonmanifold guards.

## Risks

- **Fewer holes filled:** Cap 3 leaves more closed loops than cap 4. Mitigation: this cycle explicitly optimizes balanced geometry plus UV quality.
- **Still below xatlas UV quality:** The cap-3 policy remains below the pre-repair UV utilization. Mitigation: keep parity false and track the gap.
