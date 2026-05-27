# Roadmap

## Phase 31: Xatlas Deficit Diagnostics Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-xatlas-deficit-diagnostics-gate`
- objective: Make the remaining native-chart versus xatlas utilization deficit explicit in diagnostics and tests.
- why now: Native-chart readiness now passes, but the utilization ratio is still only about `0.685` of the xatlas reference and can be hidden behind scalar readiness.
- likely outputs: Deficit fields, utilization-equivalence check, focused and heavy assertions, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-xatlas-deficit-diagnostics-gate/SPEC.md`
- exit signal: Native-chart diagnostics quantify the remaining deficit while readiness and non-parity contracts stay explicit.

## Phase 32: Native Chart Hole Reduction

- status: done
- change: `2026-05-27-mlx-spatialkit-native-chart-hole-reduction`
- objective: Identify and reduce native-chart UV holes that keep spatialkit below the Pixal3D xatlas reference.
- why now: Deficit diagnostics show scalar native-chart readiness passes while xatlas utilization remains about `0.685`; the next work must reduce that measured gap instead of adding micro-roadmap tweaks.
- likely outputs: Hole baseline, bounded native chart-generation improvement, focused tests, real-fixture quality gate, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-native-chart-hole-reduction/SPEC.md`
- exit signal: Real native-chart reference-target export improves UV-surface occupancy or xatlas-utilization ratio while parity remains honestly false.

## Phase 33: Geometry Hole Diagnostics

- status: done
- change: `2026-05-27-mlx-spatialkit-geometry-hole-diagnostics`
- objective: Make native geometry hole risk measurable with boundary-loop diagnostics before adding repair heuristics.
- why now: The latest reference-target export has acceptable shape/color and close face-count parity, but final export metrics still show `23822` boundary edges; we need loop-level topology evidence before blaming triangle count or UVs.
- likely outputs: Native boundary-loop metrics, focused tests, heavy Pixal3D export assertion, docs, roadmap current-state refresh.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-geometry-hole-diagnostics/SPEC.md`
- exit signal: Export diagnostics show remaining visible-hole risk as `2594` closed boundary loops and `808` open boundary-chain components, while xatlas and repair parity remain explicitly open.

## Phase 34: Small Boundary Loop Fill

- status: done
- change: `2026-05-27-mlx-spatialkit-small-boundary-loop-fill`
- objective: Fill small closed boundary loops in the native topology-aware simplifier when target-face budget allows.
- why now: Geometry diagnostics showed `2594` closed boundary loops and remaining face budget in the reference-target export; this directly addresses the small visible-hole class.
- likely outputs: Native small-loop repair, repair stats, focused tests, heavy topology gate, docs.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-small-boundary-loop-fill/SPEC.md`
- exit signal: Reference-target export closed boundary loops drop from `2594` to `1479` without new nonmanifold edges or false xatlas parity.

## Phase 35: Small Loop Fill Balance

- status: done
- change: `2026-05-27-mlx-spatialkit-small-loop-fill-balance`
- objective: Tune small-loop repair to triangular holes so UV utilization improves while geometry repair remains measurable.
- why now: Cap-4 repair reduced topology holes but lowered native-chart UV utilization; a cap-3 probe keeps topology improvement with better texture coverage.
- likely outputs: Cap-3 repair policy, focused triangular/quad-hole tests, heavy balance gate, docs.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-small-loop-fill-balance/SPEC.md`
- exit signal: Reference-target export keeps `boundary_loop_count=1872 < 2594` and improves xatlas-utilization ratio to `0.6828063257125282`, above the cap-4 repair baseline.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-small-loop-fill-balance`.
- Phases 1-30 are consolidated into the current verified native export line: Pixal3D decoded fixtures can export GLB through the spatialkit native path with geometry readiness, texture coverage, viewer compatibility, memory telemetry, browser visual proof, and native-chart UV/Metal bake gates.
- Remaining measured boundary: xatlas chart parity is still false. Latest reference-target native-chart xatlas-utilization ratio before geometry repair was `0.6941716645020964`, with UV-surface occupancy `0.5768346786499023`.
- Active geometry repair evidence: bounded triangular loop fill reduces final closed boundary loops from `2594` to `1872` with `nonmanifold_edges=0`; UV utilization remains below xatlas parity and is tracked separately.
- Explicit 1M/4096 native-chart exports are upstream-setting ready, with xatlas parity still the open quality-equivalence gap.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
