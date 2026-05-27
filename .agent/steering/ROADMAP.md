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

## Phase 44: Centroid-Fan Hole Fallback

- status: done
- change: `2026-05-27-mlx-spatialkit-centroid-fan-hole-fallback`
- objective: Add a guarded centroid-fan fallback for small closed boundary loops that projected ear-clipping rejects.
- why now: A cap-only probe showed that raising the fill cap from 8 to 32 did not reduce the remaining `1089` closed boundary loops, so the next lever is rejection handling rather than more loop-size allowance.
- likely outputs: C++ fallback repair, method/rejection diagnostics, focused mesh tests, real-fixture topology evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-centroid-fan-hole-fallback/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture lowers boundary-loop/edge counts without readiness regressions.

## Phase 45: Open-Boundary Diagnostics

- status: done
- change: `2026-05-27-mlx-spatialkit-open-boundary-diagnostics`
- objective: Extend native mesh metrics so remaining open boundary components are classified before attempting repair.
- why now: The latest fixture has `808` open boundary components after closed-loop repair, but current metrics do not separate simple open chains from branched open components.
- likely outputs: C++ mesh metric fields, focused tests, real-fixture metric assertions, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-open-boundary-diagnostics/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture records actionable open-boundary diagnostics without readiness regressions.

## Phase 46: Branched-Cycle Repair

- status: done
- change: `2026-05-27-mlx-spatialkit-branched-cycle-repair`
- objective: Fill small simple cycles found inside branched open-boundary components using existing native topology guards.
- why now: Open-boundary diagnostics showed `808` branched components and no simple open chains, while a branch-cycle probe found bounded simple cycles that can be tested without endpoint-closing.
- likely outputs: C++ branch-cycle extraction/fill, focused mesh tests, real-fixture topology evidence, docs, package/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-branched-cycle-repair/SPEC.md`
- exit signal: Focused and package tests pass, native build succeeds, and the real fixture reduces branched/open-boundary topology without readiness regressions.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-branched-cycle-repair`.
- Diagnostics separate scalar reference-target quality from production equivalence.
- Native geometry repair now uses bounded projected ear-clipping with an 8-edge public cap, a conservative 6-edge centroid-fan fallback, and guarded 4-edge branch-cycle repair for small cycles inside branched boundary components.
- Native open-boundary diagnostics now distinguish simple open chains from branched open components; the current real fixture has branched open components, not simple open chains.
- Native-chart xatlas-utilization ratio improved through bounded small-chart splitting and denser split-position search, not by changing padding defaults.
- Metal texture bake now releases the Python GIL during command-buffer waits so Python monitor threads can sample during GPU execution.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Zero-padding default switch remains deferred until separately justified.
- Implementing open-boundary repair, xatlas parity, full remesh, arbitrary large N-gon filling, or CUDA/cuMesh behavior is outside the current diagnostics change.
