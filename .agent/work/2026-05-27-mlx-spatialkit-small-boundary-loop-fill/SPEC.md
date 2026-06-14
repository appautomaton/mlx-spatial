# mlx-spatialkit Small Boundary Loop Fill Spec

## Bounded Goal

Reduce visible geometry holes by filling small closed boundary loops in the native production simplification path when face budget allows.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and visually comparable Pixal3D exports before production readiness is claimed.

## Selected Lenses

- **engineering:** Act on the measured geometry-hole evidence instead of guessing from triangle count or UV coverage.
- **runtime:** Keep repair native C++, deterministic, bounded by loop size and target-face budget.
- **product:** Improve the actual visible mesh by closing small topology holes while keeping xatlas/UV parity boundaries explicit.

## Current Evidence

- Latest verified geometry diagnostics show the final reference-target export has `23822` boundary edges, `2594` closed boundary loops, `808` open boundary-chain components, and `nonmanifold_edges=0`.
- Final face count is `198618` against the reference-target budget `212542`, leaving enough face budget for a bounded small-loop fill.
- A normal-drift chart-growth candidate was probed during this cycle and rejected because it reduced real fixture utilization below the verified baseline.

## Required Outcome

1. The topology-aware production simplifier fills small closed boundary loops after simplification when doing so stays within the target-face budget.
2. Repair stats report considered loops, filled loops, rejected loops, face budget, and faces added.
3. Focused tests prove small interior holes are filled while larger/open boundaries remain unfilled.
4. Heavy Pixal3D reference-target export shows fewer final boundary loops than the verified `2594` baseline without introducing nonmanifold edges or exceeding target faces.
5. Existing UV/xatlas honesty contracts remain intact.
6. Docs and roadmap explain this as bounded small-hole repair, not full remesh or xatlas parity.
7. Heavy/generated artifacts stay under `/tmp`; no push, tag, publish, or release metadata work.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| SBLF-01 | Small closed loop fill is native and bounded. | `simplify_mesh(..., backend="topology-aware")` runs loop fill with max loop size and target-face budget stats. |
| SBLF-02 | Focused repair coverage exists. | Unit tests show an interior hole loop is filled and a large/open boundary is preserved. |
| SBLF-03 | Heavy final topology improves. | Reference-target export reports `boundary_loop_count < 2594`, `nonmanifold_edges=0`, and `final_faces <= 212542`. |
| SBLF-04 | Production and xatlas contracts stay honest. | Existing production readiness passes; xatlas parity/equivalence remain false. |
| SBLF-05 | Docs/roadmap are current. | Spatialkit/Pixal3D docs and ROADMAP describe bounded loop fill and remaining boundaries. |
| SBLF-06 | Repo/package hygiene holds. | Focused tests, heavy reference-target test, package tests, root Pixal3D tests, and `/tmp` build inspection pass. |

## Scope Coverage Decisions

- **Included:** Native small closed boundary-loop fill, stats, focused tests, heavy real-fixture topology gate, docs/roadmap refresh.
- **Deferred:** Full remeshing, open-boundary repair, UV/xatlas parity, semantic hole classification, Metal repair kernels.
- **Anti-goals:** Filling large/open boundaries, exceeding target-face budget, hiding repair failures behind production-ready status, adding external remesh dependencies.

## Constraints

- Run only in the topology-aware production simplifier path.
- Keep the fill deterministic and bounded by loop edge count and remaining target-face budget.
- Reject candidate fills that would create degenerate, duplicate, or nonmanifold faces.
- Keep heavy/generated outputs under `/tmp`.

## Risks

- **False fill risk:** Some small closed loops can be intentional openings. Mitigation: limit to small loops and preserve detailed repair stats.
- **Topology risk:** Fan fill can create bad triangles on nonplanar loops. Mitigation: reject degenerate/duplicate/nonmanifold candidate patches and verify final metrics.
- **Budget risk:** Added faces can exceed target. Mitigation: hard budget gate before accepting a loop fill.
