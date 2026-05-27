# SPEC: mlx-spatialkit Split-Position Search

## Bounded Goal

Improve native-chart UV utilization by expanding the bounded low-fill split-position search from three to five deterministic fractions, then keep the change only if the real Pixal3D fixture improves without readiness regressions.

## Broader Intent

This continues the native export backend quality work by reducing the measured xatlas-utilization gap through chart-shape policy, while keeping xatlas parity boundaries honest.

## Work Scale And Shape

- scale: small
- shape: native C++ UV chart tuning with real-fixture verification
- selected lenses: engineering, runtime

## Constraints And Risks

- Keep the change limited to low-fill split-position search and related diagnostics/tests/docs.
- Do not change tile padding, texture baking, mesh repair, default UV backend, or production-equivalence semantics.
- Heavy generated artifacts stay under `/tmp`.
- Risk: a denser search may add CPU work or chart splits without a meaningful occupancy gain.

## Required Outcome

- Native-chart diagnostics report `low_fill_split_position_candidates=5`.
- Real reference-target native-chart export remains artifact-ready and production-quality-ready.
- The xatlas-utilization ratio improves above the current `0.6973` baseline or the tuning is not kept.
- Docs describe this as measured native-chart progress, not xatlas parity.

## Acceptance Criteria

- Focused GLB writer tests pass and assert five split positions.
- The heavy reference-target native-chart fixture passes with a stronger xatlas-utilization threshold than Phase 42.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not implement xatlas, a new atlas packer, remeshing, zero-padding default changes, CUDA/cuMesh behavior, release tags, publishing, or pushing.
