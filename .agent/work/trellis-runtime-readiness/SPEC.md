# SPEC: TRELLIS.2 MLX Runtime Readiness

## Bounded Goal

Build a TRELLIS.2 MLX runtime readiness layer that can execute a minimal weighted sparse convolution reference primitive, validate local model asset readiness, and document the bring-up path without downloading or loading real model weights by default.

## Selected Lenses

- product
- engineering
- runtime
- content

## Objective

Move `mlx_spatial` from isolated sparse primitives toward a usable spatial-model runtime foundation by combining the first weighted sparse operation with model asset readiness conventions for TRELLIS.2.

## Broader Intent

TRELLIS.2 is the first proving ground for a broader MLX spatial-model library. The runtime contracts and asset readiness patterns should remain reusable for later SAM3D and Hunyuan-family geometry integrations.

## Target User

Developers porting or experimenting with spatial model inference on Apple Silicon who need clear MLX sparse tensor contracts and a safe path for bringing local model weights into the repo without breaking default development workflows.

## Desired Outcome

After this change, the repo should answer three practical questions:

- How do sparse map rows, source features, kernel weights, and target features fit together for the first weighted sparse operation?
- Where should TRELLIS.2 model assets live locally, and how can a developer check whether expected files are present?
- What is the next logical TRELLIS bring-up path after sparse weighted compute and asset readiness?

## Audience

Engineers using this repo to build or port spatial model inference components; they know Python and MLX but need explicit contracts for sparse feature movement, weighted sparse compute, and local model asset handling.

## Thesis

The repo should become ready for TRELLIS.2 weights by defining executable MLX sparse compute and asset validation boundaries first, not by downloading checkpoints before there is code that can safely consume them.

## Voice Direction

Technical reference voice. Short direct sentences. State constraints and commands plainly. Avoid hype, broad claims about model quality, and vague roadmap promises.

## Content Anti-Goals

- No promotional language such as "groundbreaking", "powerful", or "seamless".
- No claims that TRELLIS.2 inference is supported unless a verified command proves it.
- No vague "future work" ending; name concrete next slices.
- No hidden dependency claims; every required optional tool must be named.

## Constraints

- Default install and tests must continue to require only PyPI `mlx` plus dev `pytest`; no base PyTorch, Transformers, Hugging Face, checkpoint, or vendor dependency.
- Weighted sparse convolution must consume the existing row contract `(target_index, source_index, kernel_index)` and build on existing gather/scatter semantics.
- Model asset readiness must validate local files and directory conventions only; it must not download, authenticate, or import Hugging Face tooling during default tests.
- Generated or downloaded weights must be protected from accidental commits with a clear ignore rule or documented out-of-repo cache convention.
- Documentation must distinguish implemented runtime readiness from unsupported full TRELLIS.2 inference.
- Optional parity or asset checks must be gated and skipped by default.
- Vendor projects under `vendors/` may be referenced for intent but must not be imported, modified, or required by default tests.

## Blocking Questions Or Assumptions

- Assumption: this change targets readiness for TRELLIS.2, not a complete TRELLIS.2 inference pipeline.
- Assumption: the weighted sparse convolution API should use synthetic features and weights in tests; real checkpoint tensor loading belongs to a later spec.
- Assumption: local model assets should be treated as external runtime inputs, not committed repository files.
- Assumption: the first asset manifest can describe expected TRELLIS.2 assets at a high level even if exact final checkpoint filenames need later refinement from vendor/Hugging Face inspection.
- Assumption: Hugging Face CLI instructions can be documented without adding `huggingface_hub` as a dependency.

## Scope Boundary

In scope:

- Minimal MLX weighted sparse convolution reference primitive over existing map rows.
- Explicit sparse feature/map/kernel/target contracts in docstrings, tests, and README.
- MLX-only numeric tests for weighted sparse convolution with deterministic small arrays.
- Optional PyTorch parity scaffold for weighted sparse convolution, skipped by default.
- A model asset manifest or config that defines TRELLIS.2 asset expectations without storing weights.
- A validation helper or lightweight CLI/module function that checks expected asset files exist locally and reports missing assets clearly.
- `.gitignore` or equivalent protection for local model weights if the convention uses an in-repo path.
- README documentation for local asset directory convention, Hugging Face CLI download pattern, manifest validation, and TRELLIS bring-up path.
- Optional tests for manifest parsing and missing-file validation using tiny fake files only.

## Anti-Goals

- Do not download model weights in default tests or during implementation.
- Do not require Hugging Face login, network access, or credentials.
- Do not load TRELLIS.2 checkpoints into MLX tensors.
- Do not implement full TRELLIS.2 model architecture, transformer blocks, decoders, mesh extraction, or GLB export.
- Do not import from or modify `vendors/`.
- Do not add PyTorch, Transformers, or Hugging Face packages to base runtime dependencies.
- Do not optimize sparse convolution performance beyond a clear correctness-first reference implementation.
- Do not create a generic asset manager beyond the needs of TRELLIS.2 runtime readiness.

## Acceptance Criteria

- A public MLX weighted sparse convolution reference helper exists and is documented.
- The helper consumes map rows, source features, and kernel weights with explicit shape contracts.
- The helper produces deterministic target features by multiplying source features by the row-selected kernel weight and summing into target slots.
- MLX-only tests cover exact numeric output, duplicate target accumulation, empty maps, invalid shapes, out-of-bounds indices, and weight/channel mismatches.
- Existing sparse map and gather/scatter tests continue to pass.
- A TRELLIS.2 asset manifest/config exists and contains enough metadata to validate a local asset directory without downloading weights.
- A validation helper or command reports present and missing TRELLIS.2 asset files deterministically.
- Asset validation tests use tiny temporary fake files and do not require network access or real model weights.
- Local weight artifacts are ignored or directed outside the repository so they are not committed accidentally.
- README documents the weighted sparse convolution contract, local asset convention, Hugging Face CLI command pattern, validation workflow, and clear unsupported boundaries.
- Optional PyTorch/vendor/asset parity checks remain gated and skipped by default.
- `uv run pytest` passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, vendor setup, or network access.

## Risks

- Weight layout may need adjustment after deeper TRELLIS.2 layer inspection; this spec should choose a simple documented layout and avoid pretending it is final model parity.
- Asset filenames may differ across TRELLIS.2 distributions; the manifest should be easy to update and tests should validate behavior rather than hard-code claims about unavailable remote repos.
- Combining sparse compute and asset readiness is larger than prior slices; planning should split execution into independently verifiable slices.
- Documentation could overstate readiness; verification must check that unsupported full inference boundaries are explicit.

## Recommended Next Skill

`auto-plan`
