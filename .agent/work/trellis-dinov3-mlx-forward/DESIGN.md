# DESIGN: TRELLIS.2 MLX DINOv3 Forward

## Boundary

This change stays inside TRELLIS.2 image conditioning. It does not implement
sparse sampling, SLat sampling, decoders, export, RMBG/BiRefNet, or a
Transformers compatibility layer.

The runtime path remains:

```text
Trellis2InferencePipeline.attempt_forward_trace(...)
  -> preprocess image
  -> prepare_dinov3_image_tensor(...)
  -> assess_dinov3_mlx_conditioning(...)
  -> image-conditioning output or exact DINOv3 blocker
  -> existing sparse-structure boundary if conditioning output exists
```

## Current Real Checkpoint Layout

The local `facebook/dinov3-vitl16-pretrain-lvd1689m` checkpoint exposes:

```text
embeddings.cls_token                     (1, 1, 1024)
embeddings.register_tokens               (1, 4, 1024)
embeddings.patch_embeddings.weight       (1024, 3, 16, 16)
embeddings.patch_embeddings.bias         (1024,)
layer.0.attention.q_proj.weight          (1024, 1024)
layer.0.attention.k_proj.weight          (1024, 1024)
layer.0.attention.v_proj.weight          (1024, 1024)
layer.0.attention.o_proj.weight          (1024, 1024)
layer.0.layer_scale1.lambda1             (1024,)
layer.0.layer_scale2.lambda1             (1024,)
layer.0.mlp.up_proj.weight               (4096, 1024)
layer.0.mlp.down_proj.weight             (1024, 4096)
layer.0.norm1.weight/bias                (1024,)
layer.0.norm2.weight/bias                (1024,)
norm.weight/bias                         (1024,)
```

There are 24 transformer layers, hidden size 1024, 16 heads, patch size 16,
and 4 register tokens. The TRELLIS.2 runtime image-conditioning resolution is
512, so the patch-token grid is 32 x 32 before cls/register handling.

## Implementation Shape

Keep `trellis2_dinov3.py` as the public orchestration and blocker surface.
Add a small MLX-only forward implementation module when execution starts,
likely `trellis2_dinov3_forward.py`, to avoid mixing asset validation with
model math.

The forward implementation should be separable into probes:

1. checkpoint key map and selected tensor loader;
2. patch embedding and token assembly;
3. RoPE or position embedding construction/application;
4. single transformer block;
5. repeated block loop and final norm;
6. output metadata dispatch.

Each probe should accept tiny fake configs/checkpoints. The real checkpoint
path should load only the tensors needed for the current probe, then either
continue or return a named blocker. Full real-weight loading should not happen
until the relevant probe has passed fake tests.

## Tensor Layout

Use MLX arrays. Prefer explicit layout conversions at module boundaries:

- image input: BCHW from `prepare_dinov3_image_tensor(...)`;
- patch embedding output: token sequence `(B, N, D)`;
- attention internal shape: `(B, heads, N, head_dim)` or MLX fast attention's
  required equivalent;
- conditioning output: `(B, N, 1024)`.

If MLX `nn.Conv2d` or `mx.conv2d` requires NHWC, make that conversion local to
patch embedding and test it with fake weights.

## Blocker Policy

A blocker is acceptable only if it is more specific than the current
`MLX DINOv3 transformer block construction` placeholder. Examples:

- missing checkpoint key: `layer.0.attention.q_proj.weight`;
- unsupported RoPE layout or rotation dimension;
- shape mismatch for `(B, N, D)` versus attention projection;
- unavailable MLX op;
- memory/block-size limit at real 512-resolution ViT-L/16.

Real attempts must never emit fake conditioning metadata.

## Verification Strategy

Default tests remain fake-fixture only. Real local attempts are manual or
slice-level verification commands and must read ignored weights only when those
commands are run.

The closeout can be either:

- real conditioning metadata and sparse-boundary dispatch; or
- a precise DINOv3 forward blocker that names the first unsupported component.
