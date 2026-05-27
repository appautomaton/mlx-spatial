# SPEC: mlx-spatialkit Alternative Triangulation Repair

## Bounded Goal

Add bounded topology-aware alternative triangulation for small boundary-loop patches so safe holes are not rejected only because the first ear-clipping diagonal conflicts with existing mesh topology.

## Broader Intent

This advances `mlx-spatialkit` toward dependable Pixal3D export quality by reducing remaining holes with native C++ geometry logic instead of loosening caps or hiding unsafe topology behind passing tests.

## Work Scale And Shape

- scale: medium
- shape: native C++ triangulation search with focused synthetic tests, real-fixture verification, and docs alignment
- selected lenses: engineering, runtime

## Source Evidence

- Baseline after repair-cap alignment: `10070` boundary edges, `1075` boundary loops, `216` branched open components, `4133` open-boundary edges, `403` branch vertices, `0` nonmanifold edges.
- Residual rejected patches are dominated by topology conflicts: `2421` nonmanifold rejections and `624` duplicate rejections.
- `/tmp` probe with alternative triangulation: `8882` boundary edges, `946` boundary loops, `185` branched open components, `3599` open-boundary edges, `329` branch vertices, `249` alternative-triangulation fills, `0` nonmanifold edges, no quality warnings, and `production_quality_ready=true`.

## Required Outcome

- When the primary projected ear-clipping patch is rejected for duplicate or nonmanifold topology, native code may enumerate bounded alternate ear-clipping triangulations for the same small loop.
- Alternate patches must pass the existing degenerate, duplicate, nonmanifold, and face-budget guards before being applied.
- Diagnostics expose alternative-triangulation attempts and fills.
- The real Pixal3D reference-target fixture improves boundary-loop and open-boundary topology versus the current baseline while keeping `nonmanifold_edges=0`, no quality warnings, and `production_quality_ready=true`.

## Acceptance Criteria

- Focused mesh tests prove a quad loop can be repaired through an alternate diagonal when the primary diagonal is topology-blocked.
- Focused mesh tests prove existing cap and fallback behavior still works.
- Heavy native-chart reference-target fixture shows lower boundary-loop/open-boundary counts than the current baseline without readiness regression.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not loosen repair caps, implement endpoint-chain repair, arbitrary branch graph closure, global remeshing, xatlas parity, 1M-face export parity, CUDA/cuMesh behavior, model inference changes, release tags, publishing, pushing, or new public artifact formats beyond diagnostics fields.
