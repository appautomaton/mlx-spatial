# SPEC: TRELLIS.2 Sparse Voxel Topology Helpers

## Bounded Goal

Implement MLX sparse voxel topology helpers that derive 26-neighbor active voxel adjacency, neighbor offsets, and deterministic grid edge/cell relationship primitives from sparse coordinates, with MLX-only default tests and optional local PyTorch parity checks.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- The topology API must build on the existing `mlx_spatial.ovoxel` coordinate/index conventions, including row-major ordering and plain MLX array returns.
- The first adjacency mode is full 26-neighbor 3D connectivity, including face, edge, and corner neighbors; lower-order 6/18 modes may be internal conveniences but are not required user-facing outputs for this slice.
- Default tests must remain MLX-only and must not require PyTorch, Transformers, Hugging Face, checkpoints, local framework paths, or vendor setup.
- Optional PyTorch parity must be gated and, when enabled, use the local PyTorch checkout at `/Users/ac/dev/ai/ai-frameworks/pytorch`.
- Vendor projects may be read as references during planning or implementation, but runtime code must not import from or modify `vendors/`.

## Blocking Questions Or Assumptions

- Assumption: topology helpers should operate on sparse integer coordinates shaped `(n, 3)` and a positive 3D grid shape.
- Assumption: adjacency output can be represented as deterministic integer pair arrays and/or neighbor index tables, as long as row ordering is documented and tested.
- Assumption: edge/cell relationship primitives should stay geometric and deterministic, not mesh extraction or rendering.
- Assumption: this slice can add documentation and tests but should not add model weights, Hugging Face, sparse convolutions, neural layers, or end-to-end TRELLIS.2 inference.

## Anti-Goals

- Do not implement TRELLIS.2 image-to-3D inference, VAE/DiT model execution, texture baking, mesh extraction, or checkpoint loading.
- Do not add Torch, Transformers, Hugging Face, or local PyTorch as required base dependencies.
- Do not port CUDA/Metal kernels or sparse convolution operations.
- Do not introduce model-specific containers that make the topology helpers unusable for SAM3D or Hunyuan-family geometry later.
- Do not modify files under `vendors/`.

## Acceptance Criteria

- `mlx_spatial` exposes a topology helper module or namespace for 3D sparse voxel topology.
- The implementation returns deterministic 26-neighbor relationships for small sparse coordinate sets.
- The implementation exposes or documents the 26 neighbor offsets and their ordering.
- Grid edge/cell relationship helpers are deterministic and documented as topology primitives, not mesh extraction.
- Default tests verify adjacency, offset ordering, edge/cell behavior, invalid inputs, and future-model-neutral usability using only MLX and pytest.
- Optional local PyTorch parity checks are clearly gated and skipped by default.
- `uv run pytest` passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, or vendor setup.

## Recommended Next Skill

`auto-plan`
