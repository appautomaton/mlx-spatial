# SPEC: MLX Spatial Environment Bootstrap

## Bounded Goal

Create a root development environment for this repository that can install with `uv`, import the first-party package, import MLX, and run initial primitive-focused tests while keeping Torch/Transformers parity tooling optional.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- The base environment must be MLX-first and must not require Torch, Transformers, or vendored model setup to pass the initial success signal.
- The first success signal is `uv` setup plus import/tests, not end-to-end TRELLIS.2, Hunyuan, or SAM3D inference.
- Local framework checkouts exist at `/Users/ac/dev/ai/ai-frameworks`, including `mlx/`, `pytorch/`, and `transformers/`, but they should be treated as optional parity resources rather than mandatory base dependencies.
- The root repo currently has no first-party package manifest, source tree, tests, or root README, so this change must establish the package boundary before adding model-specific modules.
- Vendor projects under `vendors/` are reference material and should not be imported or modified as part of this bootstrap.

## Blocking Questions Or Assumptions

- Assumption: `uv` is the standard project workflow for the first environment milestone.
- Assumption: the base dependency should use an installable MLX dependency unless a later plan explicitly chooses an editable local MLX path.
- Assumption: optional parity tooling can be represented as dev extras, documentation, or test markers, but parity tests should not run by default.
- Assumption: the initial primitive tests can be minimal smoke/shape tests that prove package and MLX execution rather than numerical parity.

## Anti-Goals

- Do not implement TRELLIS.2, Hunyuan, SAM3D, or any other model inference path in this change.
- Do not port CUDA, PyTorch, MPS, or vendor kernels into the root package.
- Do not make local absolute paths mandatory for normal setup, tests, or import success.
- Do not alter vendored projects under `vendors/`.
- Do not introduce broad spatial APIs beyond what is needed to prove the environment and test loop work.

## Acceptance Criteria

- A contributor can create/sync the root environment with `uv`.
- The root package can be imported from tests.
- MLX can be imported from the environment in a test.
- The default test command passes without Torch, Transformers, or vendor setup.
- Optional parity resources are documented or isolated so they do not affect base setup.

## Recommended Next Skill

`auto-plan`

## Review: Product
- Verdict: approved_with_risks
- Strength: Establishes the missing root package and MLX test loop before spending effort on model-specific ports.
- Concern: Future model work will need checkpoint/download tooling such as Hugging Face access, but making it a base dependency now would blur the bootstrap success signal.
- Action: Proceed to `auto-plan`; keep Hugging Face tooling optional or deferred to the first model-download slice rather than required for base import/tests.
- De-scoped: End-to-end model inference, mandatory Torch/Transformers parity setup, mandatory local absolute framework paths, mandatory Hugging Face download flow.
