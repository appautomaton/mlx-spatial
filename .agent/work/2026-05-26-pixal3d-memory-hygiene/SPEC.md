# Pixal3D Memory Hygiene Spec

## Objective

Reduce avoidable Pixal3D MLX unified-memory pressure by releasing completed
stage objects and clearing unused MLX cache at safe stage boundaries.

## Constraints

- Do not change model math, sampler settings, token guards, export settings, or
  output semantics.
- Keep the existing successful `1024_cascade` path intact.
- Preserve intermediate artifacts and trace metadata.
- Do not hide memory pressure by removing explicit guards.

## Acceptance Criteria

- Large stage probe/conditioning objects are deleted after their durable outputs
  and downstream tensors are established.
- `clear_mlx_cache()` is called after major Pixal3D stages where no pending MLX
  value from that stage is needed.
- Trace metadata records explicit memory checkpoints after cleanup boundaries.
- Focused Pixal3D/shared-kernel tests and full test suite pass.
