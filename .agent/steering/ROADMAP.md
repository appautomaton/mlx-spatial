# Roadmap

## Phase 48: Alternative Triangulation Repair

- status: done
- change: `2026-05-27-mlx-spatialkit-alternative-triangulation-repair`
- objective: Fill topology-blocked small boundary loops by trying bounded alternate ear-clipping triangulations under existing native guards.
- why now: The latest fixture still has many small boundary loops rejected for duplicate/nonmanifold topology, and probing showed alternate triangulation can reduce holes without nonmanifold or readiness regressions.
- likely outputs: C++ alternate triangulation search, focused mesh tests, real-fixture topology evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-alternative-triangulation-repair/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture lowers boundary-loop/open-boundary topology without nonmanifold or readiness regressions.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-alternative-triangulation-repair`.
- Native geometry repair uses bounded projected ear-clipping, guarded alternative triangulation for topology-blocked small loops, 8-edge centroid-fan fallback, and guarded 6-edge branch-cycle repair with effective caps clamped by the public Pixal3D repair setting.
- The real fixture still has remaining holes/topology work, and xatlas/1M parity remains deferred.
- Native-chart export is scalar production-quality ready for the reference-target fixture, but production equivalence remains false because xatlas chart parity and upstream 1M-face export parity are still deferred.
- Prior completed phases are consolidated here instead of kept as roadmap history; detailed evidence remains in `.agent/work/<change>/`.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Implementing endpoint-chain repair, arbitrary branch graph closure, full remesh, xatlas parity, 1M-face export parity, or CUDA/cuMesh behavior remains outside the active phase.
