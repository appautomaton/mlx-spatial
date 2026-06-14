# SPEC: mlx-spatialkit Quad Loop Repair Default

## Bounded Goal

Make the native topology-aware export path fill bounded 4-edge boundary loops by default, with real-fixture diagnostics proving reduced geometry-hole topology without introducing nonmanifold export blockers.

## Broader Intent

This moves `mlx-spatialkit` closer to a dependable native backend for `mlx-spatial` by reducing visible small-hole risk in the C++ geometry path while keeping production-equivalence boundaries honest.

## Work Scale And Shape

- scale: small/medium
- shape: native geometry policy change with tests, real-fixture gate, and docs
- selected lenses: engineering, runtime

## Constraints And Risks

- Use the existing bounded C++ repair path; do not implement general remeshing, xatlas parity, or cap-8+ polygon repair in this change.
- Default repair cap becomes `4`; `0` must still disable the policy and explicit lower caps must be respected.
- The repair must stay inside the target-face budget and keep `nonmanifold_edges=0`.
- Heavy generated artifacts stay under `/tmp`.
- Risk: filling larger loops too aggressively can hide real topology defects, so this change is limited to triangle and quad loops by default.

## Required Outcome

- Public Python, export, and native-binding defaults agree on `small_boundary_loop_fill_max_edges=4`.
- Quad boundary holes are filled by default in topology-aware simplification.
- An explicit cap of `3` preserves the previous triangle-only behavior.
- Real Pixal3D native-chart reference-target diagnostics show fewer final boundary loops than the prior cap-3 baseline while keeping export blockers empty.
- Documentation describes the policy as bounded triangle/quad small-loop repair, not full remesh or xatlas parity.

## Acceptance Criteria

- Unit tests prove default quad-loop filling, explicit cap-3 preservation, disable behavior, and nonmanifold-free output.
- The heavy real Pixal3D native-chart reference-target test asserts cap `4` and improved boundary-loop thresholds.
- Docs and roadmap are updated without adding release/tag/push scope.
- Package tests and native build still pass.

## Anti-Goals

- Do not implement arbitrary N-gon hole filling, full remesh, xatlas charting, or CUDA/cuMesh behavior.
- Do not change MLX model inference or decoded NPZ formats.
- Do not push, tag, publish, or release.
