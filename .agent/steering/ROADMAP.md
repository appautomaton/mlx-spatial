# Roadmap

## Phase 38: Readiness Contract Semantics

- status: done
- change: `2026-05-27-mlx-spatialkit-readiness-semantics`
- objective: Separate scalar export quality readiness from full Pixal3D production-equivalence readiness in diagnostics, tests, and docs.
- why now: The native export path can pass scalar reference-target checks while xatlas chart parity and upstream-setting parity remain open.
- likely outputs: `quality.production_equivalence`, `result.production_equivalence_ready`, focused tests, real-fixture gate, docs wording, and compact Automaton evidence.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-readiness-semantics/SPEC.md`
- exit signal: Current reference-target exports may keep `production_quality_ready=true`, but diagnostics and docs clearly keep production equivalence false until remaining parity boundaries are closed.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-readiness-semantics`.
- Native geometry, chart UV, Metal bake, gutter fill, GLB viewer compatibility, and memory diagnostics are materially in place for real Pixal3D decoded fixtures.
- The open quality-equivalence boundary is now technical: diagnostics separate scalar reference-target quality from production equivalence, while xatlas chart parity remains false.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Zero-padding default switch remains deferred until separately justified.
- Implementing xatlas parity, full remesh, or CUDA/cuMesh behavior is outside the current readiness-semantics change.
