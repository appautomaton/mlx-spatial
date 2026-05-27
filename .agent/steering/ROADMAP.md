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

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-native-chart-minimum-padding-gate`.
- Phases 1-30 are consolidated into the current verified native export line: Pixal3D decoded fixtures can export GLB through the spatialkit native path with geometry readiness, texture coverage, viewer compatibility, memory telemetry, browser visual proof, and native-chart UV/Metal bake gates.
- Remaining measured boundary: xatlas chart parity is still false. Latest reference-target native-chart xatlas-utilization ratio is `0.685133792850289`, with UV-surface occupancy `0.5693244934082031`.
- Explicit 1M/4096 native-chart exports are upstream-setting ready, with xatlas parity still the open quality-equivalence gap.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
