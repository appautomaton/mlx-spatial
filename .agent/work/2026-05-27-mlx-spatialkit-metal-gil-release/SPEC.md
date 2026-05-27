# SPEC: mlx-spatialkit Metal GIL Release

## Bounded Goal

Release the Python GIL while the native Metal texture bake waits on GPU completion, and add concurrent public-API coverage for the native bake path.

## Broader Intent

This supports the larger `mlx-spatialkit` backend goal by improving thread behavior and memory-monitor fidelity around an Apple GPU hot path without changing model outputs or export quality thresholds.

## Work Scale And Shape

- scale: small
- shape: native runtime hardening with focused tests
- selected lenses: engineering, runtime

## Constraints And Risks

- Release the GIL only around code that does not call Python or nanobind APIs.
- Keep Objective-C objects inside the existing autorelease boundary.
- Do not change texture bake outputs, diagnostics semantics, or GLB export defaults.
- Heavy generated artifacts remain under `/tmp`.
- Risk: a too-wide GIL release around Python buffer or `nb::dict` work would be unsafe.

## Required Outcome

- The Metal command-buffer `commit`/`waitUntilCompleted` span runs without holding the Python GIL.
- Existing texture diagnostics and outputs remain deterministic.
- Tests cover concurrent `bake_pbr_texture` calls through the public Python API.
- Docs mention that the GPU wait releases the GIL for monitor/thread progress.

## Acceptance Criteria

- Focused texture bake tests pass, including a concurrent bake test.
- Full `mlx-spatialkit` non-heavy suite passes.
- Native package build still succeeds.
- No public API or artifact format changes.

## Anti-Goals

- Do not release the GIL around Python buffer loading, nanobind result construction, or Python exception construction.
- Do not change Metal kernels, chart packing policy, mesh repair policy, or production-equivalence semantics.
- Do not push, tag, publish, or release.
