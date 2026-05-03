# SPEC: TRELLIS.2 Sparse Convolution Map Primitives

## Bounded Goal

Implement MLX sparse convolution map primitives that derive deterministic kernel offsets and gather/scatter index pairs from active sparse voxel coordinates, with MLX-only default tests and optional local PyTorch parity checks.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- The API must build on existing `mlx_spatial.ovoxel` and `mlx_spatial.topology` conventions: `(z, y, x)` coordinates, row-major indexing, plain MLX array returns, and deterministic ordering.
- This slice should produce sparse convolution maps only: kernel offsets, source indices, target indices, and offset/kernel-slot identifiers; it must not implement convolution math, learnable weights, or neural network layers.
- Default tests must remain MLX-only and must not require PyTorch, Transformers, Hugging Face, checkpoints, local framework paths, or vendor setup.
- Optional PyTorch parity must be gated and, when enabled, use the local PyTorch checkout at `/Users/ac/dev/ai/ai-frameworks/pytorch`.
- Runtime code must not import from or modify `vendors/`.

## Blocking Questions Or Assumptions

- Assumption: the first map should target stride-1 same-grid sparse convolution neighborhoods for active input and output coordinates in the same coordinate set.
- Assumption: the initial kernel should support configurable odd 3D kernel sizes, with `(3, 3, 3)` as the primary tested case.
- Assumption: map rows should be deterministic: output/target coordinate input order, then kernel offset order, with each row identifying the active input/source index when present.
- Assumption: transposed convolution, strided convolution, dilation beyond explicit offsets, batching, and feature gathering are out of scope for this change.

## Anti-Goals

- Do not implement sparse convolution compute, matrix multiplication, weights, activations, or neural modules.
- Do not implement TRELLIS.2 model inference, VAE/DiT execution, mesh extraction, texture baking, checkpoint loading, or Hugging Face download flows.
- Do not add Torch, Transformers, Hugging Face, or local PyTorch as required base dependencies.
- Do not introduce model-specific containers that make the maps unusable for SAM3D or Hunyuan-family sparse geometry later.
- Do not modify files under `vendors/`.

## Acceptance Criteria

- `mlx_spatial` exposes sparse convolution map helpers in a clear module or namespace.
- The implementation returns deterministic kernel offsets for odd 3D kernels, including `(3, 3, 3)`.
- The implementation returns deterministic source/target/kernel-offset index triples or equivalent plain MLX arrays for active sparse coordinates.
- Invalid inputs are rejected, including non-3D coordinates, duplicate coordinates, non-positive shapes, out-of-bounds coordinates, and invalid kernel sizes.
- Default tests verify offset ordering, map row ordering, missing-neighbor handling, invalid inputs, and model-neutral usability using only MLX and pytest.
- Optional local PyTorch parity checks are clearly gated and skipped by default.
- `uv run pytest` passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, or vendor setup.

## Recommended Next Skill

`auto-plan`
