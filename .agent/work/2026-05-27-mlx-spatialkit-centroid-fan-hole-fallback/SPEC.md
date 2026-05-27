# SPEC: mlx-spatialkit Centroid-Fan Hole Fallback

## Bounded Goal

Reduce remaining small closed-loop geometry holes by adding a conservative centroid-fan fallback after projected ear-clipping fails, with explicit native diagnostics for fill method and rejection reasons.

## Broader Intent

This continues moving `mlx-spatialkit` toward a dependable native Pixal3D export backend by improving local mesh repair while keeping topology safety, memory-aware native execution, and production-equivalence boundaries honest.

## Work Scale And Shape

- scale: medium
- shape: native C++ geometry repair with focused diagnostics, Python-contract tests, real-fixture verification, and docs alignment
- selected lenses: engineering, runtime

## Constraints And Risks

- Keep repair bounded to closed boundary loops at or below `small_boundary_loop_fill_max_edges`.
- Keep projected ear-clipping as the primary method; use centroid-fan fallback only when primary patching fails, capped internally at 6 edges to preserve the native-chart quality gate.
- Accept fallback patches only when they pass degenerate-face, duplicate-face, nonmanifold-edge, face-budget, and target-face guards.
- Do not repair open boundary chains, implement global remeshing, add xatlas, change UV packing, change texture baking policy, or claim production equivalence.
- Heavy generated artifacts and probes stay under `/tmp`.
- Risk: centroid fans can hide real topology defects if rejection/fill diagnostics are vague or readiness gates overstate parity.

## Required Outcome

- Native repair can fill additional small closed loops that projected ear-clipping currently rejects.
- Diagnostics expose primary/fallback fill counts and rejection reason counts.
- The real Pixal3D reference-target fixture improves export topology versus the cap-only baseline (`1089` boundary loops, `18348` boundary edges) without nonmanifold regressions or quality-readiness regressions.
- Docs explain that this is closed-loop local repair, not open-chain repair, full remesh, xatlas parity, or production equivalence.

## Acceptance Criteria

- Focused mesh-processing tests pass and assert the fallback diagnostics contract.
- A geometry-only real-fixture probe or heavy native-chart fixture shows lower boundary loop/edge counts than the cap-only baseline while preserving `nonmanifold_edges=0`, `target_reached=true`, and empty export blockers.
- The heavy native-chart reference-target fixture remains artifact-ready and production-quality-ready.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not implement open-boundary repair, arbitrary large N-gon filling, global remesh, xatlas integration, CUDA/cuMesh behavior, model inference changes, release tags, publishing, pushing, or public artifact format changes beyond diagnostics fields.
