# Roadmap

## Phase 47: Repair Cap Alignment

- status: done
- change: `2026-05-27-mlx-spatialkit-repair-cap-alignment`
- objective: Align native small-boundary repair caps so public Pixal3D repair policy is respected while reducing remaining open-boundary topology.
- why now: The latest branch-cycle repair improved visible holes, but probing found both a measurable cap-6 topology improvement and a contract bug where low public caps still allowed larger branched-cycle repairs.
- likely outputs: C++ effective-cap policy, focused mesh tests, real-fixture topology evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-repair-cap-alignment/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture lowers open-boundary topology without nonmanifold or readiness regressions.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-repair-cap-alignment`.
- Native geometry repair uses bounded projected ear-clipping, 8-edge centroid-fan fallback, and guarded 6-edge branch-cycle repair with effective caps clamped by the public Pixal3D repair setting.
- The real fixture still has remaining holes/topology work, and xatlas/1M parity remains deferred.
- Native-chart export is scalar production-quality ready for the reference-target fixture, but production equivalence remains false because xatlas chart parity and upstream 1M-face export parity are still deferred.
- Prior completed phases are consolidated here instead of kept as roadmap history; detailed evidence remains in `.agent/work/<change>/`.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Implementing endpoint-chain repair, arbitrary branch graph closure, full remesh, xatlas parity, 1M-face export parity, or CUDA/cuMesh behavior remains outside the active phase.
