# SPEC: TRELLIS.2 Sparse Structure Sampling Boundary

## Bounded Goal

Implement the next TRELLIS.2 forward-trace boundary after DINOv3 conditioning: MLX sparse structure flow model construction and FlowEuler sampler dispatch, far enough to produce sparse-structure sample metadata or the first exact model/sampler/checkpoint/op/memory blocker.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- The starting point is a real `image-conditioning` output from the completed `trellis-dinov3-mlx-forward` change: `cond` shape `(1, 1029, 1024)`, dtype `float32`.
- The local sparse flow model is configured by `weights/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.json`: resolution `16`, input/output channels `8`, model channels `1536`, 30 transformer blocks, 12 heads, `pe_mode=rope`, `share_mod=true`, q/k RMS norm enabled, and bfloat16 torso intent.
- `weights/trellis2/pipeline.json` requests `FlowEulerGuidanceIntervalSampler` with `steps=12`, `guidance_strength=7.5`, `guidance_rescale=0.7`, `guidance_interval=[0.6, 1.0]`, and `rescale_t=5.0`.
- Default tests must not require real TRELLIS.2 weights, network, Torch, Transformers, or generated sample outputs.
- Real-weight attempts may use ignored `weights/`, `inputs/`, and `outputs/` paths, but must not commit checkpoints or generated binary artifacts.
- This change must keep the pipeline blocker-driven: do not fake sparse samples for the real path.

## Required Behavior

- Parse the sparse structure flow model config and sampler config from local TRELLIS.2 assets.
- Inspect and selectively load the sparse flow checkpoint keys needed by the first executable boundary.
- Build MLX-only sparse flow scaffolding for the input noise layout `(B, 8, 16, 16, 16)`, timestep embedding route, conditioning width validation, and checkpoint key mapping.
- Implement the FlowEuler timestep schedule and dispatch contract in MLX against fake fixtures.
- Integrate the new boundary into `attempt_forward_trace(...)` after real DINOv3 conditioning.
- When the full real sparse flow model is not yet executable, return a precise blocker naming the first unsupported transformer/module/op/key/shape/memory boundary.

## Acceptance Criteria

- Fake fixtures validate sparse flow config parsing, sampler parameter parsing, noise shape, timestep schedule, and conditioning width.
- Fake MLX sparse flow sampler dispatch advances past the previous `MLX sparse structure flow model construction` placeholder and records sparse-sampling metadata or a more precise blocker.
- Real alpha forward trace still completes DINOv3 conditioning and then reaches a blocker more specific than `MLX sparse structure flow model construction`.
- The blocker, if present, names the exact missing SparseStructureFlowModel component, checkpoint key group, FlowEuler CFG/guidance behavior, MLX op, shape, dtype, or memory issue.
- Tests cover missing/mismatched sparse flow checkpoint keys and do not load the full real checkpoint by default.
- `uv run pytest` passes.

## Blocking Questions Or Assumptions

- Assumption: this slice may stop before sparse decoder execution; producing decoded coordinates is not required unless sparse flow sampling naturally becomes executable within scope.
- Assumption: classifier-free guidance and guidance interval are part of the sampling contract, but can be staged with an exact blocker if the first model-forward boundary is not ready.
- Assumption: real bfloat16 checkpoint tensors can be loaded through existing safetensors/MLX fallback behavior; if not, the blocker should name the dtype/loading limitation.

## Anti-Goals

- Do not implement sparse structure decoder, SLat shape/texture sampling, decoders, mesh extraction, texture baking, GLB/OBJ export, or UI.
- Do not add Torch or Transformers as runtime dependencies.
- Do not alter the completed DINOv3 conditioning behavior except for passing its output into the sparse sampling boundary.
- Do not download weights automatically.
- Do not attempt training, dataset tooling, or parity with TRELLIS.2 training code.
