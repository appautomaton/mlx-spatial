# mlx-spatialkit Chart UV Local Projection Gate Spec

## Bounded Goal

Replace fixed global-axis native chart projection with deterministic per-chart local-frame projection and prove the real Pixal3D chart UV candidate improves beyond the shelf-packing baseline.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: C++ geometry algorithm hardening

## Selected Lenses

- **engineering:** Improve actual chart UV geometry instead of relaxing readiness thresholds.
- **runtime:** Keep the algorithm deterministic, native, bounded, and allocation-aware for many Pixal3D charts.
- **product:** Chart exports should visibly move toward production parity while diagnostics stay honest about remaining blockers.

## Current Evidence

- Phase 16 changed chart packing from equal-grid cells to aspect-aware shelf packing.
- The real fixture improved from `uv_surface_occupancy_ratio=0.23263072967529297` to `0.3310232162475586`, but still reports `quality_blocked`.
- The current implementation still projects every chart onto a global coordinate plane selected from the average normal, which wastes UV area for diagonal or arbitrarily oriented charts.
- Shelf packing diagnostics show high rectangle packing efficiency, so the next bottleneck is chart-internal UV fill, not atlas rectangle placement.

## Required Outcome

1. `make_native_chart_uvs` projects charts through a deterministic local tangent frame with a bounded PCA/rotation choice instead of fixed global axes.
2. Chart UV stats report projection diagnostics: projection backend, rotation candidate count, chart rectangle fill estimate, and existing shelf packing diagnostics.
3. Focused tests prove coplanar/hard-crease grouping remains intact and a rotated coplanar chart gets better rect fill from local projection.
4. The real Pixal3D native-chart export improves UV-surface occupancy over the Phase 16 shelf baseline of `0.3310232162475586`.
5. Docs describe local projection as another native chart candidate improvement, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| CLPG-01 | Native chart projection is deterministic local-frame/PCA projection. | Focused chart tests assert projection diagnostics and stable chart grouping. |
| CLPG-02 | Rotated coplanar chart fill improves. | Focused test asserts chart rect fill estimate is near full for a diagonal rectangular chart. |
| CLPG-03 | Real chart export occupancy improves. | Heavy chart fixture reports `uv_surface_occupancy_ratio > 0.3310232162475586`. |
| CLPG-04 | Default export behavior remains unchanged. | Full package/root Pixal3D tests still pass, including default face-atlas assertions. |
| CLPG-05 | Docs match the parity boundary. | Docs say local projection improves the candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** C++ local-frame projection, bounded rotation/PCA choice, stats, focused tests, real fixture proof, docs, regression/build verification.
- **Deferred:** full xatlas-quality cuts, true LSCM/ABF unwrap, chart seam optimization, default backend switch, readiness-threshold relaxation.
- **Anti-goals:** changing decoded model outputs, changing Metal bake behavior, changing chart quality floors, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native C++ and deterministic for identical inputs.
- Keep runtime bounded: no unbounded search, global solve, hull-heavy unwrap, or external dependency.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Local projection may improve occupancy but still not reach quality-ready thresholds. Mitigation: readiness diagnostics remain truthful.
- **Projection risk:** Curved charts may still distort under any simple planar projection. Mitigation: this phase only claims bounded native candidate improvement.
- **Regression risk:** UV layout changes can affect GLB texture bake results. Mitigation: run focused chart tests, real chart heavy fixture, package/root tests, and build checks.

## Blocking Questions Or Assumptions

Assumption: the next useful native increment is reducing chart-internal rectangle waste with local projection, because Phase 16 already made rectangle packing utilization high.
