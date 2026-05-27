# Roadmap

## Phase 40: Metal GIL Release

- status: done
- change: `2026-05-27-mlx-spatialkit-metal-gil-release`
- objective: Release the Python GIL during Metal command-buffer waits and prove concurrent texture-bake API behavior.
- why now: The backend quality bar includes thread safety and memory visibility, and a GIL-held GPU wait can starve Python monitor threads during a hot stage.
- likely outputs: Narrow native GIL-release boundary, concurrent bake test, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-metal-gil-release/SPEC.md`
- exit signal: Texture bake tests and package suite pass, concurrent public API calls are deterministic, and docs describe the GIL boundary without overstating full export parallelism.

## Phase 41: Ear-Clip Small-Hole Repair

- status: done
- change: `2026-05-27-mlx-spatialkit-earclip-hole-repair`
- objective: Replace fan-only small boundary-loop filling with projected ear-clipping and use the measured 8-edge default for Pixal3D exports.
- why now: The current native GLB is visually close, but remaining small geometry holes are a direct quality gap before treating `mlx-spatialkit` as a dependable export backend.
- likely outputs: C++ ear-clipping repair, cap-8 default, focused mesh tests, real-fixture topology evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-earclip-hole-repair/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and real-fixture diagnostics show fewer boundary loops/edges without nonmanifold regressions.

## Phase 42: Small-Chart UV Splitting

- status: done
- change: `2026-05-27-mlx-spatialkit-small-chart-splitting`
- objective: Let native-chart low-fill splitting improve 4- and 5-face chart islands while preserving strict xatlas-parity boundaries.
- why now: The current xatlas-utilization gap is dominated by chart-shape utilization; zero-padding only made a small measured difference and carries filtering-bleed risk.
- likely outputs: C++ low-fill split threshold change, focused GLB writer tests, real-fixture xatlas-utilization evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-small-chart-splitting/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture improves xatlas-utilization ratio without readiness regressions.

## Phase 43: Split-Position Search

- status: done
- change: `2026-05-27-mlx-spatialkit-split-position-search`
- objective: Expand native-chart low-fill splitting from three to five deterministic split positions when measuring chart-utilization benefit.
- why now: The post-Phase-42 diagnostics still showed many rejected low-fill split candidates, making split-position density a bounded next lever.
- likely outputs: C++ split-position search update, focused GLB writer tests, real-fixture xatlas-utilization evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-split-position-search/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture improves xatlas-utilization ratio without readiness regressions.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-split-position-search`.
- Diagnostics separate scalar reference-target quality from production equivalence.
- Native geometry repair now uses bounded projected ear-clipping with an 8-edge default for Pixal3D exports.
- Native-chart xatlas-utilization ratio improved through bounded small-chart splitting and denser split-position search, not by changing padding defaults.
- Metal texture bake now releases the Python GIL during command-buffer waits so Python monitor threads can sample during GPU execution.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Zero-padding default switch remains deferred until separately justified.
- Implementing xatlas parity, full remesh, arbitrary large N-gon filling, or CUDA/cuMesh behavior is outside the current split-position search change.
