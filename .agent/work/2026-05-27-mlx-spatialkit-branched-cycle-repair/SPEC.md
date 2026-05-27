# SPEC: mlx-spatialkit Branched-Cycle Repair

## Bounded Goal

Reduce branched boundary topology by extracting small simple cycles from branched open-boundary components and filling only the cycles that pass the existing native topology guards.

## Broader Intent

This advances `mlx-spatialkit` toward dependable native Pixal3D export quality by addressing the next measured geometry gap without falling into unsafe endpoint-closing or global remeshing.

## Work Scale And Shape

- scale: medium
- shape: native C++ geometry repair with focused synthetic tests, real-fixture topology verification, and docs alignment
- selected lenses: engineering, runtime

## Constraints And Risks

- Reuse the existing degenerate-face, duplicate-face, nonmanifold-edge, and target-face budget guards.
- Keep the repair bounded to small simple cycles discovered inside branched boundary components; do not repair arbitrary open chains.
- Keep this deterministic and native; no Python hot path, no new dependency, no xatlas, no remesh.
- Heavy generated artifacts stay under `/tmp`.
- Risk: filling too many branch cycles can hurt native-chart UV utilization or hide real topology defects, so the real fixture must prove topology improvement without readiness regression.

## Required Outcome

- Native simplification can fill small simple cycles found inside branched open-boundary components.
- Diagnostics expose branch-cycle candidate, fill, reject, and budget-limited counts.
- The real Pixal3D reference-target fixture reduces branched open-boundary topology versus the current baseline (`808` branched components, `12622` open-boundary edges) while keeping nonmanifold edges at `0`, no quality warnings, and `production_quality_ready=true`.
- Docs explain this as conservative branched-cycle repair, not open-chain repair, global remesh, xatlas parity, or production equivalence.

## Acceptance Criteria

- Focused mesh-processing tests cover a branched/pinched boundary fixture that is repaired by branch-cycle extraction.
- Heavy native-chart reference-target fixture shows lower open-boundary edge/component counts without readiness regressions.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not implement endpoint-chain repair, arbitrary branch graph closure, full remesh, xatlas integration, CUDA/cuMesh behavior, model inference changes, release tags, publishing, pushing, or public artifact format changes beyond diagnostics fields.
