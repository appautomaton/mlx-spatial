# SPEC: mlx-spatialkit Repair Cap Alignment

## Bounded Goal

Align native small-boundary repair caps so public Pixal3D repair policy is respected while using the measured safer limits for centroid fallback and branched-cycle repair.

## Broader Intent

This advances `mlx-spatialkit` toward dependable Pixal3D export quality by reducing remaining holes without hiding unsafe topology behind successful tests.

## Work Scale And Shape

- scale: medium
- shape: native C++ repair-policy fix with focused synthetic tests, real-fixture verification, and docs alignment
- selected lenses: engineering, runtime

## Source Evidence

- Baseline real fixture after branched-cycle repair: `12016` boundary edges, `353` branched open components, `6340` open-boundary edges, `715` branch vertices, `0` nonmanifold edges.
- Probe with branch cap `6` and fallback cap `8`: `10070` boundary edges, `216` branched open components, `4133` open-boundary edges, `403` branch vertices, `0` nonmanifold edges, no quality warnings.
- Contract bug found during probing: `small_boundary_loop_fill_max_edges=3` still fills 4-edge branched cycles because the branch-cycle cap is hard-coded independently of the public cap.

## Required Outcome

- Public `small_boundary_loop_fill_max_edges` bounds closed-loop repair, centroid fallback, and branched-cycle extraction.
- Centroid fallback supports the same 8-edge public repair cap.
- Branched-cycle repair remains more conservative than closed-loop repair, with a policy cap of 6 edges.
- Diagnostics expose both policy caps and effective caps clearly enough to explain disabled or low-cap runs.
- The real Pixal3D reference-target fixture improves open-boundary topology versus the current baseline while keeping `nonmanifold_edges=0`, no quality warnings, and `production_quality_ready=true`.

## Acceptance Criteria

- Focused mesh tests prove a low public cap prevents larger branched-cycle repair.
- Focused mesh tests cover the expanded 8-edge centroid fallback policy.
- Heavy native-chart reference-target fixture shows lower open-boundary component/edge counts than the current baseline without readiness regression.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build succeeds.

## Anti-Goals

- Do not implement endpoint-chain repair, arbitrary branch graph closure, global remeshing, xatlas parity, 1M-face export parity, CUDA/cuMesh behavior, model inference changes, release tags, publishing, pushing, or new public artifact formats beyond diagnostics fields.
