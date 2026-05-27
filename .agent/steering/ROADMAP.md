# Roadmap

## Phase 39: Quad Loop Repair Default

- status: done
- change: `2026-05-27-mlx-spatialkit-quad-loop-repair`
- objective: Make bounded triangle/quad loop repair the default topology-aware native export policy.
- why now: Real Pixal3D diagnostics still show many final boundary loops, and cap-4 probing reduced loops without nonmanifold export blockers.
- likely outputs: Cap-4 default across Python/native bindings, focused mesh tests, real Pixal3D heavy gate, docs, and package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-quad-loop-repair/SPEC.md`
- exit signal: Default exports report `small_boundary_loop_fill_max_edges=4`, quad holes fill in unit tests, and the real native-chart reference-target fixture keeps `nonmanifold_edges=0` with lower boundary-loop thresholds.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-quad-loop-repair`.
- Diagnostics now separate scalar reference-target quality from production equivalence.
- Native geometry repair now defaults to bounded triangle/quad loop repair while keeping arbitrary remesh and xatlas parity deferred.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Zero-padding default switch remains deferred until separately justified.
- Implementing xatlas parity, full remesh, arbitrary N-gon filling, or CUDA/cuMesh behavior is outside the current repair-default change.
