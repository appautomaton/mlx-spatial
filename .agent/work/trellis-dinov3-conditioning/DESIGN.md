# DESIGN: TRELLIS.2 DINOv3 Conditioning

## Boundary

This change extends the existing TRELLIS.2 forward trace at the `image-conditioning`
stage. It does not implement sparse sampling, decoders, export, RMBG/BiRefNet,
or a Torch/Transformers compatibility layer.

The runtime path remains:

1. `Trellis2InferencePipeline.attempt_forward_trace(...)`
2. TRELLIS.2 pipeline config discovery
3. image preprocessing/background stage
4. DINOv3 asset validation, config/checkpoint inventory, and MLX conditioning attempt
5. existing sparse-structure boundary dispatch if conditioning metadata is produced

## Asset Model

The TRELLIS.2 pipeline names `facebook/dinov3-vitl16-pretrain-lvd1689m`.
The local convention is:

```text
weights/dinov3-vitl16-pretrain-lvd1689m/
  config.json
  model.safetensors
```

Validation must be deterministic and offline. Runtime imports must not trigger
network access or Hugging Face resolution. Download help can name an explicit
developer command, but execution remains user-controlled because the model may
require authentication or terms acceptance.

## Runtime Modules

Keep generic asset manifests in `model_assets.py`.

Add a small DINOv3-specific inspection/attempt layer rather than expanding
`trellis2_forward.py` into a full model implementation file. The likely shape is:

- asset manifest and helper exports for local DINOv3 validation;
- config parser that validates the fields required by ViT-L/16 image features;
- safetensors inventory that can report required/missing checkpoint keys and
  tensor shapes without loading the full model into memory;
- MLX construction/forward probe that either emits real conditioning tensor
  metadata or returns the first unsupported field, key, shape, module, or op.

`trellis2_forward.py` should orchestrate the stage and preserve existing result
dataclasses. New dataclasses are allowed only when they make blockers or
inventory results easier to test.

## Fake Fixtures

Default tests use tiny fake DINOv3 configs/checkpoints. They should cover:

- missing assets;
- present but incompatible config/checkpoint data;
- deterministic inventory of keys and shapes;
- a fake-compatible conditioning path that produces width `cond_channels`;
- downstream sparse-boundary dispatch after fake conditioning.

Fake fixture success is not a claim that real DINOv3 inference works. Real local
attempts must use real local files and either emit real output metadata or a real
port blocker.

## Port Strategy

Implement the smallest MLX probe that can discover the next truthful boundary.
The first pass should map the DINOv3 config and checkpoint structure before
committing to a complete ViT implementation.

If full DINOv3 forward execution is feasible in this slice, it should produce
conditioning features whose last dimension matches TRELLIS.2 sparse-flow
`cond_channels` (currently 1024). If not feasible, the blocker must name the
first unmapped config field, checkpoint key, tensor shape, module, or MLX op.

## Documentation

Docs should separate:

- local TRELLIS.2 weights;
- local RMBG-2.0 weights and their current MLX op blocker;
- local DINOv3 weights and their validation/download-help surface;
- current forward-trace boundary.
