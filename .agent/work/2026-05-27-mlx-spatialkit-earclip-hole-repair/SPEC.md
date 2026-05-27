# SPEC: mlx-spatialkit Ear-Clip Small-Hole Repair

## Bounded Goal

Improve native Pixal3D mesh repair by replacing fan-only small boundary-loop filling with projected ear-clipping and, if verified on the real fixture, widening the default fill cap from 4 to 8 edges.

## Broader Intent

This moves `mlx-spatialkit` toward dependable native export quality by reducing visible small holes in the native GLB path while preserving conservative topology guards and clear diagnostics.

## Work Scale And Shape

- scale: medium
- shape: native C++ geometry repair with Python contract/docs/test updates
- selected lenses: engineering, runtime

## Constraints And Risks

- Keep repair bounded to closed boundary loops at or below `small_boundary_loop_fill_max_edges`.
- Reject invalid patches that would create degenerate faces, duplicate faces, or nonmanifold edges.
- Do not claim full remesh, arbitrary N-gon filling, xatlas parity, or production equivalence.
- Heavy generated artifacts and probes stay under `/tmp`.
- Risk: filling larger loops can hide real topology defects if diagnostics overstate readiness.

## Required Outcome

- Native small-loop repair uses projected ear-clipping instead of a single fan anchor.
- Diagnostics identify the loop-fill algorithm and resolved cap.
- Default Pixal3D export cap becomes 8 only if real-fixture evidence shows reduced boundary holes without nonmanifold regressions.
- Docs explain the bounded 8-edge policy and remaining limitations.

## Acceptance Criteria

- Focused mesh-processing tests cover concave or larger closed boundary-loop filling, cap behavior, invalid cap validation, and no nonmanifold regressions.
- The real Pixal3D geometry probe or heavy native-chart fixture shows boundary-loop/edge reduction versus the prior cap-4 baseline without export blockers.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not implement full remeshing, global hole filling, xatlas integration, CUDA/cuMesh behavior, or new Python-side mesh hot paths.
- Do not change Metal texture baking, UV packing policy, model inference, release tags, publishing, pushing, or public artifact formats beyond diagnostics fields.
