# mlx-spatialkit Peak Memory Telemetry Gate Spec

## Bounded Goal

Add stage-level peak host-memory telemetry to Pixal3D `export_pixal3d_glb` diagnostics so heavy native exports report observed RSS peaks during each major stage, not only sparse boundary samples.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

The current export path has better quality gates, but memory diagnostics can still under-explain what a user sees in Activity Monitor because they sample before or after stages. This change makes the export trace more honest before deeper memory optimization work.

## Work Scale And Shape

- Scale: capability hardening
- Shape: runtime diagnostics and coverage

## Selected Lenses

- **runtime:** Track observed process RSS peaks while long native stages run.
- **engineering:** Keep telemetry dependency-light, thread-safe, and out of native hot paths.
- **product:** Diagnostics should explain memory behavior in a way a developer can compare with local system monitors.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating Pixal3D decoded NPZ artifacts into GLB output through the `mlx-spatialkit` companion backend.

## Current Evidence

- `export_pixal3d_glb` records `memory_samples` at labels such as `start`, `after_texture_bake`, and `after_write_glb`.
- `_memory_sample()` reports `current_rss_bytes` from `ps` and `max_rss_bytes` from `resource.getrusage`.
- Stage timings exist in `timings_sec`, but no per-stage memory peak exists.
- Heavy tests currently assert only that `after_write_glb` appears in `memory_samples`.

## Required Outcome

1. Export diagnostics include a `memory` summary with sample source, poll interval, sample count, observed process RSS peak, max RSS high-water, and per-stage peaks.
2. Each major `_timed_stage` records start/end/current/peak RSS telemetry for that stage.
3. Existing boundary samples remain available in `memory_samples` for backward-readable diagnostics.
4. The monitor is thread-safe, has bounded retained data, and stops cleanly before diagnostics are written.
5. Tests cover the monitor with deterministic fake samples and the heavy real fixture asserts the new memory summary exists.
6. Docs describe what the memory numbers do and do not prove.

## Memory Telemetry Targets

| ID | Target | Required evidence |
|---|---|---|
| MTG-01 | Stage peak capture | Unit test proves stage start/end and peak RSS are recorded from deterministic samples. |
| MTG-02 | Bounded telemetry | Summary keeps aggregate counts/peaks and per-stage aggregates, not an unbounded sample log. |
| MTG-03 | Real export integration | Heavy Pixal3D export diagnostics include `memory.stage_peaks` for long export stages. |
| MTG-04 | Boundary compatibility | Existing `memory_samples` labels remain present. |
| MTG-05 | Honest semantics | Docs state the source is process RSS/high-water telemetry, not full system pressure or Metal allocator accounting. |

## Constraints

- Keep heavy/generated artifacts under `/tmp` in tests and development runs.
- Do not add psutil or other new runtime dependencies.
- Do not change model inference, decoded NPZ generation, geometry quality thresholds, or visual parity thresholds.
- Do not add GPU allocator claims that the telemetry does not measure.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Sampling gap risk:** A polling monitor can still miss very short spikes. Mitigation: label values as observed peaks and keep `ru_maxrss` high-water.
- **Overhead risk:** Frequent `ps` sampling could add overhead. Mitigation: use a modest default interval and retain only aggregate telemetry.
- **False equivalence risk:** Activity Monitor includes broader process/system context. Mitigation: document exactly what the diagnostics measure.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| MTG-01 | Thread-safe memory monitor records deterministic stage peaks. | Focused package unit tests exercise fake samples and stage tracking. |
| MTG-02 | `export_pixal3d_glb` integrates the monitor across `_timed_stage` calls. | Heavy real fixture diagnostics include `memory.sample_count`, `peak_current_rss_bytes`, and stage peaks for `texture_bake` and `write_glb`. |
| MTG-03 | Existing memory samples stay readable. | Heavy tests continue asserting `after_write_glb` and validate sample source fields. |
| MTG-04 | Docs explain memory telemetry semantics. | README/Pixal3D docs mention observed process RSS peaks and limitations. |
| MTG-05 | Repo/package hygiene holds. | Package tests, heavy fixture test, root Pixal3D tests, build artifact inspection, and `git status --short` stay clean. |

## Scope Coverage Decisions

- **Included:** process RSS polling monitor, per-stage aggregate peak telemetry, diagnostics integration, deterministic unit tests, real-fixture assertions, docs, package/root/build verification.
- **Deferred:** memory optimization, Metal heap accounting, MLX allocator accounting, system-wide memory pressure, browser-rendered GLB proof, xatlas/4096/1M parity.
- **Anti-goals:** claiming exact Activity Monitor equivalence, adding heavy dependencies, storing unbounded samples, changing export quality thresholds.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: process RSS plus `ru_maxrss` is the right next measurement layer because it directly addresses the current diagnostic gap without adding package dependencies.
