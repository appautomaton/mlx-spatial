# SPEC: mlx-spatialkit Open-Boundary Diagnostics

## Bounded Goal

Make remaining open-boundary topology measurable by extending native `mesh_metrics` to classify open boundary components, edge counts, endpoint counts, branch counts, and small-open-component totals.

## Broader Intent

This moves `mlx-spatialkit` toward a dependable native export backend by turning the remaining visible-hole problem into a precise topology contract before attempting open-chain repair.

## Work Scale And Shape

- scale: small
- shape: native C++ diagnostics contract with focused tests, real-fixture assertions, and docs alignment
- selected lenses: engineering, runtime

## Constraints And Risks

- Do not change mesh repair, simplification, UV packing, texture baking, GLB writing, model inference, or readiness semantics.
- Keep the diagnostics native and deterministic.
- Heavy generated artifacts stay under `/tmp`.
- Risk: more metrics can become noise unless they directly distinguish closed loops, simple open chains, and branched open components.

## Required Outcome

- `mesh_metrics` reports open-boundary edge count, max open-boundary component size, endpoint count, branch-vertex count, simple open-chain count, branched open-component count, and small-open-component count/edge totals.
- Existing closed-loop metrics remain unchanged.
- The real Pixal3D reference-target export preserves the new metrics under `export_metrics.metrics`.
- Docs describe these as diagnostics for the next repair decision, not as a repair or parity claim.

## Acceptance Criteria

- Focused mesh-processing tests cover closed-loop metrics and branched open-boundary metrics.
- Heavy native-chart reference-target fixture records the new open-boundary fields and remains production-quality-ready with no quality warnings.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not repair open chains, change closed-loop fallback policy, implement remeshing, add xatlas, alter Metal texture baking, release, publish, push, or claim production equivalence.
