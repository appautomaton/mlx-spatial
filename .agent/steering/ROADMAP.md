# Roadmap

## Phase 40: Metal GIL Release

- status: done
- change: `2026-05-27-mlx-spatialkit-metal-gil-release`
- objective: Release the Python GIL during Metal command-buffer waits and prove concurrent texture-bake API behavior.
- why now: The backend quality bar includes thread safety and memory visibility, and a GIL-held GPU wait can starve Python monitor threads during a hot stage.
- likely outputs: Narrow native GIL-release boundary, concurrent bake test, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-metal-gil-release/SPEC.md`
- exit signal: Texture bake tests and package suite pass, concurrent public API calls are deterministic, and docs describe the GIL boundary without overstating full export parallelism.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-metal-gil-release`.
- Diagnostics separate scalar reference-target quality from production equivalence.
- Native geometry repair defaults to bounded triangle/quad loop repair.
- Metal texture bake now releases the Python GIL during command-buffer waits so Python monitor threads can sample during GPU execution.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Zero-padding default switch remains deferred until separately justified.
- Implementing xatlas parity, full remesh, arbitrary N-gon filling, or CUDA/cuMesh behavior is outside the current Metal GIL-release change.
