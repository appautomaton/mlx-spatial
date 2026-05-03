# SPEC: TRELLIS.2 DINOv3 Conditioning

## Bounded Goal

Make the TRELLIS.2 forward trace resolve local `facebook/dinov3-vitl16-pretrain-lvd1689m` assets and attempt MLX `DinoV3FeatureExtractor` construction/forward, producing real conditioning tensor metadata or the first exact DINOv3 config, key, shape, or operation blocker.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Use the existing `mlx-spatial` package, `Trellis2InferencePipeline.attempt_forward_trace(...)`, and the structured blocker/result shapes already added for forward tracing.
- Use `weights/dinov3-vitl16-pretrain-lvd1689m/` as the local DINOv3 asset root for `facebook/dinov3-vitl16-pretrain-lvd1689m`.
- Local asset acquisition must be explicit; imports, tests, validation, and attempt mode must not silently download DINOv3 assets.
- Runtime code must not import PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor modules.
- Default tests must use tiny fake DINOv3 fixtures and must not require real DINOv3 weights, real TRELLIS.2 weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- If the real DINOv3 checkpoint/config cannot be mapped to MLX in this slice, return a precise `image-conditioning` blocker naming the first unsupported config field, checkpoint key, tensor shape, module, or MLX operation.
- If real conditioning tensors are produced, preserve the existing downstream dispatch into the sparse-structure boundary and do not continue into full sampling.

## Required Behavior

- Add DINOv3 asset metadata and validation for the expected local files under `weights/dinov3-vitl16-pretrain-lvd1689m/`.
- Add a manual download/help surface naming `facebook/dinov3-vitl16-pretrain-lvd1689m`, the local root, expected files, and the fact that access may require explicit Hugging Face authentication or terms acceptance.
- Inspect local DINOv3 config and safetensors files with MLX-compatible loaders without importing Transformers.
- Map enough DINOv3 config/checkpoint structure to identify the patch embedding, hidden size, number of layers, attention heads, MLP shape, normalization keys, RoPE/position behavior, and expected output feature width.
- Attempt MLX module construction for the DINOv3 image-conditioning path used by TRELLIS.2:
  - image input tensor is already normalized by `prepare_dinov3_image_tensor(...)`;
  - output must be shaped as conditioning features with final width matching the TRELLIS.2 sparse-flow `cond_channels` expectation, currently 1024;
  - if full forward execution is incomplete, the blocker must name the first unimplemented module/op/key.
- Update `attempt_forward_trace(...)` so local DINOv3 assets move the blocker from generic asset validation to either real conditioning output metadata or a more precise DINOv3 port blocker.
- Add fake-fixture tests for asset validation, config/key inventory, module construction blockers, successful fake conditioning output, and preservation of downstream sparse-boundary dispatch.
- Add real local attempt evidence showing how far `inputs/trellis2/demo-alpha.webp` advances when DINOv3 assets are absent or present locally.

## Acceptance Criteria

- Public DINOv3 asset validation/helper APIs exist under `mlx_spatial` and validate present/missing local files deterministically.
- A manual DINOv3 download/help command or documented command surface exists and names `facebook/dinov3-vitl16-pretrain-lvd1689m` plus `weights/dinov3-vitl16-pretrain-lvd1689m`.
- Fake DINOv3 config/checkpoint fixtures can be inspected without real weights and return deterministic model/key inventory.
- Missing local DINOv3 assets continue to return a precise `image-conditioning` / `local DINOv3 asset validation` blocker.
- Present but incompatible fake DINOv3 assets return a precise blocker naming the config field, key, shape, module, or MLX op that prevents construction or forward execution.
- If a fake-compatible DINOv3 fixture is supplied, `attempt_forward_trace(...)` records an `image-conditioning` output and reaches the existing `sparse-structure-sampling` boundary.
- If real local DINOv3 assets are present, the attempt either emits real conditioning tensor metadata or reports the first exact real DINOv3 port blocker.
- Default `uv run pytest` passes without real DINOv3/TRELLIS.2/RMBG weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- Runtime dependency metadata still excludes PyTorch, TorchVision, Transformers, Hugging Face Hub, and ONNX Runtime.
- No real DINOv3 weights, TRELLIS.2 weights, RMBG weights, or generated outputs are tracked by git.
- README or TRELLIS docs describe the DINOv3 local-asset requirement, validation/download-help surface, and current forward-trace boundary.

## Blocking Questions Or Assumptions

- Assumption: `facebook/dinov3-vitl16-pretrain-lvd1689m` is the intended image-conditioning model because it is the value in local `weights/trellis2/pipeline.json`.
- Assumption: `weights/dinov3-vitl16-pretrain-lvd1689m/` is the local convention unless planning finds an existing repo convention that is clearly better.
- Assumption: DINOv3 asset acquisition may require Hugging Face authentication or terms acceptance, so any download must remain explicit and outside default tests.
- Assumption: an exact DINOv3 config/key/op blocker is an acceptable completion state if full MLX forward is not feasible in this slice.
- Assumption: downstream sparse-structure sampling remains out of scope except for the already implemented boundary dispatch.

## Anti-Goals

- Do not implement full sparse-structure sampling, SLat sampling, decoders, mesh extraction, texture baking, or GLB/OBJ export.
- Do not solve RMBG/BiRefNet `deform_conv2d`.
- Do not add PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor modules as runtime dependencies.
- Do not silently download DINOv3 assets during import, tests, validation, or attempt mode.
- Do not fake real DINOv3 output for real local attempts.
- Do not claim full TRELLIS.2 image-to-3D inference works.
