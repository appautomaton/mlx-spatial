# SPEC: mlx-spatialkit Small-Chart UV Splitting

## Bounded Goal

Improve native-chart UV quality by allowing low-fill 4- and 5-face charts to split safely, then verify that the real Pixal3D fixture improves xatlas-utilization proximity without artifact/readiness regressions.

## Broader Intent

This advances `mlx-spatialkit` toward a dependable native export backend by attacking the largest remaining native-chart parity gap: chart-shape utilization, not padding or mesh repair.

## Work Scale And Shape

- scale: medium
- shape: native C++ UV chart policy with real-fixture verification
- selected lenses: engineering, runtime

## Constraints And Risks

- Keep the change inside native-chart splitting and its diagnostics/tests.
- Do not change texture baking, GLB writing, mesh repair, default UV backend, or production-equivalence semantics.
- Do not switch native-chart default tile padding to `0`; the `/tmp` padding probe showed only a small gain and leaves filtering-bleed risk.
- Heavy generated artifacts stay under `/tmp`.
- Risk: more chart splits can increase duplicated vertices and chart count without enough occupancy gain.

## Required Outcome

- Low-fill splitting can evaluate 4- and 5-face charts with 2-face minimum children.
- Diagnostics expose the new split thresholds.
- The real reference-target native-chart fixture remains artifact-ready and production-quality-ready.
- The measured xatlas-utilization ratio improves above the current cap-8 baseline of about `0.6826`.

## Acceptance Criteria

- Focused GLB writer tests pass and cover the small-chart split thresholds.
- The heavy reference-target native-chart fixture passes with tightened native-chart assertions.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not implement xatlas, full UV unwrapping parity, atlas packing replacement, zero-padding default switch, CUDA/cuMesh behavior, release tags, publishing, or pushing.
