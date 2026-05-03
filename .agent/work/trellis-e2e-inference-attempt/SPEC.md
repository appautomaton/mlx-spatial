# SPEC: TRELLIS.2 End-to-End Inference Attempt

## Bounded Goal

Create the first inference-only TRELLIS.2 image-to-3D attempt in `mlx_spatial` by tracing the working `vendors/trellis-mac` path, wiring the equivalent MLX pipeline stages around local real weights, and producing either a runnable partial result or a precise blocker ledger for missing MLX compute stages.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Inference only; do not add training paths, trainers, gradient flows, or optimizer behavior.
- `vendors/trellis-mac` is the primary runnable reference for stage order and operational behavior; `vendors/TRELLIS.2` is the original architecture/reference source.
- Runtime implementation must not import vendor modules; vendor code may be read during planning/execution and cited in local artifacts.
- Real weights are local under ignored `weights/trellis2/`; default tests must not require real weights, network access, Hugging Face credentials, PyTorch, Transformers, or vendor imports.
- Full image-to-GLB completion is an attempted outcome, not a guaranteed acceptance criterion for this first slice.
- Missing compute must fail explicitly with stage name, missing operation, reference location, and recommended next slice; do not hide missing stages behind fake outputs.
- Keep generated artifacts out of tracked source unless they are tiny deterministic test fixtures or textual execution evidence.

## Required Behavior

- Trace the `vendors/trellis-mac` image-to-3D inference flow and capture the stage order, model/config loading points, sampler path, decoder path, mesh/export path, and MLX replacement points.
- Add an inference-only pipeline entrypoint or object in `mlx_spatial` that models the traced TRELLIS.2 stage flow.
- Use existing TRELLIS.2 asset validation and weight-loading tools to verify local real weights before attempting compute.
- Implement any small missing helpers directly required to advance the first path when they are bounded and testable.
- For unimplemented stages, raise or return structured blockers that identify the exact missing MLX operation and reference source.
- Provide a dry-run or readiness mode that validates assets, configs, weight probes, and stage wiring without requiring full model compute.
- Provide an execution attempt path that accepts an image input and advances through implemented stages until completion or the first precise blocker.
- Preserve default test independence with fake fixtures and unit tests for stage ordering, readiness reporting, blocker structure, and no-vendor-import behavior.

## Acceptance Criteria

- A traced inference-flow artifact exists for the TRELLIS.2 image-to-3D path, grounded in `vendors/trellis-mac` and original TRELLIS.2 reference files.
- A public inference-only pipeline API or CLI exists in `mlx_spatial` and exposes readiness/dry-run and attempt modes.
- The pipeline validates local TRELLIS.2 assets and can load configured real-weight probes through existing MLX tooling.
- Fake-fixture tests verify stage order, config/asset readiness behavior, blocker shape, and explicit unimplemented-stage behavior.
- If any stage is implemented, it has targeted tests and does not rely on real weights in default tests.
- A real local execution attempt is run against `weights/trellis2/` and a sample image if available, or stops with a documented missing-input blocker.
- The real attempt outcome is recorded as either completed output paths or a blocker ledger with stage name, missing operation, reference location, and next-slice recommendation.
- Default `uv run pytest` passes without real weights, network access, Hugging Face credentials, PyTorch, Transformers, or vendor imports.
- No real weights or large generated image/mesh/GLB outputs are committed.

## Blocking Questions Or Assumptions

- Assumption: the first attempt may end at a blocker before producing a mesh/GLB; that is acceptable if the blocker ledger is precise and verified.
- Assumption: the smallest useful pipeline surface includes readiness/dry-run plus execution attempt, not only static documentation.
- Assumption: a sample image may already exist in the workspace or can be created as a tiny local fixture; if no suitable image exists, the real attempt records a missing-input blocker rather than downloading sample data.
- Assumption: PyTorch reference execution is optional for diagnosis only and must not become a default dependency or default test requirement.

## Anti-Goals

- Do not implement training.
- Do not claim full TRELLIS.2 parity unless measured.
- Do not fake model outputs to make the pipeline appear complete.
- Do not import from `vendors/` at runtime.
- Do not require real weights in default tests.
- Do not add PyTorch, Transformers, or Hugging Face Hub as base runtime dependencies.
- Do not commit real weights, large generated assets, or environment-specific output dumps.
- Do not optimize performance before the stage flow and blockers are known.
