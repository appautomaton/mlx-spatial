# ATTEMPT: TRELLIS.2 MLX DINOv3 Forward

## Local Real-Weight Attempt

Command:

```bash
uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/dinov3-mlx-forward-alpha.json
```

Result:

- completed stages: `input-image`, `asset-config-validation`, `checkpoint-probe-readiness`, `image-preprocessing-background`, `image-conditioning`
- output: `cond`, shape `(1, 1029, 1024)`, dtype `float32`
- blocker stage: `sparse-structure-sampling`
- blocker operation: `MLX sparse structure flow model construction`
- output artifact: `outputs/trellis2/dinov3-mlx-forward-alpha.json`

Evidence from the generated trace:

- local DINOv3 assets resolved at `weights/dinov3-vitl16-pretrain-lvd1689m`
- DINOv3 forward key map validated against the real `model.safetensors`
- patch embedding and token assembly ran on the alpha image tensor
- token shape reached `(1, 1029, 1024)`
- patch grid reached `(32, 32)`
- RoPE geometry validated and applied to patch-token q/k tensors
- all 24 MLX DINOv3 transformer layers evaluated
- final layer normalization applied
- conditioning output shape reached `(1, 1029, 1024)`

Current exact blocker:

```text
sparse flow checkpoint exposes required conditioning keys
(`blocks.0.cross_attn.to_kv.weight`, `blocks.0.norm2.weight`), but the
MLX sparse flow model and sampler are not implemented.
```

## Next Slice

Implement MLX sparse structure flow model construction and FlowEuler sampler dispatch.
