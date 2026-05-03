# SPEC: TRELLIS.2 O-Voxel-Inspired MLX Primitives

## Bounded Goal

Implement the first TRELLIS.2 O-Voxel-inspired sparse coordinate/grid primitives in `mlx_spatial`, with default MLX-only tests and optional local PyTorch parity checks gated behind an explicit marker or environment flag.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- The default package and test path must remain MLX-first and must not require PyTorch, Transformers, Hugging Face, checkpoint downloads, or vendor setup.
- Optional PyTorch parity must use the local PyTorch checkout at `/Users/ac/dev/ai/ai-frameworks/pytorch` when enabled, because that checkout contains workstation-specific Metal-related changes.
- The implementation should target reusable sparse coordinate/grid primitives that are useful for TRELLIS.2/O-Voxel work without attempting full TRELLIS.2 inference.
- Vendor projects under `vendors/` may be read as references during planning or implementation, but must not be modified or imported by default runtime code.
- The active MLX dependency remains the standard installable `mlx` package for this repo unless a later spec explicitly changes the backend dependency strategy.

## Blocking Questions Or Assumptions

- Assumption: the first primitive set should cover coordinate/grid mechanics, such as dense-to-sparse coordinate generation, flatten/unflatten index mapping, and simple validity masks, rather than mesh extraction or neural network layers.
- Assumption: parity tests are optional developer checks and should be skipped by default unless an explicit marker or environment variable is set.
- Assumption: local PyTorch parity may be invoked through documentation or test configuration, but absolute local paths should not be required for normal `uv run pytest`.
- Assumption: Hugging Face/model weights are out of scope for this change because no model inference is being implemented.

## Anti-Goals

- Do not implement end-to-end TRELLIS.2 image-to-3D inference.
- Do not download model weights or add Hugging Face as a required dependency.
- Do not add Torch, Transformers, or local PyTorch as required base dependencies.
- Do not port CUDA/Metal kernels or sparse convolution layers in this slice.
- Do not modify files under `vendors/`.

## Acceptance Criteria

- `mlx_spatial` exposes a small O-Voxel/sparse-grid primitive module or namespace.
- Default tests verify the primitive behavior using only MLX and pytest.
- Optional PyTorch parity checks are clearly gated and skipped by default.
- Documentation explains how the optional local PyTorch parity path should be enabled later or now, without making it mandatory.
- `uv run pytest` passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, or vendor setup.

## Recommended Next Skill

`auto-plan`
