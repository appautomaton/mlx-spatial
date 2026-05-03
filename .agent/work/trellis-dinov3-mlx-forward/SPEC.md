# SPEC: TRELLIS.2 MLX DINOv3 Forward

## Bounded Goal

Implement the TRELLIS.2 DINOv3 image-conditioning forward path in MLX far enough to produce real conditioning tensor metadata from local `facebook/dinov3-vitl16-pretrain-lvd1689m` assets, or return the first exact MLX DINOv3 embedding, RoPE, attention, MLP, normalization, checkpoint-key, shape, or memory blocker.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Use the existing `mlx-spatial` package, `Trellis2InferencePipeline.attempt_forward_trace(...)`, and `src/mlx_spatial/trellis2_dinov3.py` inspection/blocker surface.
- Use the explicit local DINOv3 root `weights/dinov3-vitl16-pretrain-lvd1689m/`; validation and default tests must not silently download assets.
- Runtime code must not import PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor modules.
- Default tests must use tiny fake DINOv3 configs/checkpoints and must not require real DINOv3/TRELLIS/RMBG weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports.
- Real local attempts must not fake real DINOv3 output; if real forward is incomplete, return a precise blocker naming the first unsupported key, shape, module, MLX op, dtype, layout, or memory boundary.
- Scope is only image-conditioning. Do not continue into sparse-structure sampling except through the existing boundary dispatch after conditioning metadata exists.
- Memory use must be explicit: avoid loading unnecessary tensors, and if full real forward is too large for the local runtime, report a structured memory/block-size blocker rather than crashing.

## Required Behavior

- Add an MLX DINOv3 forward module/probe that maps the local DINOv3 config and checkpoint into executable pieces:
  - patch embedding from `embeddings.patch_embeddings.*`;
  - register/class token handling needed by the local checkpoint;
  - RoPE/position embedding behavior used by Hugging Face `DINOv3ViTModel`;
  - transformer layer attention projections, attention output, MLP, residuals, and layer norms;
  - final layer normalization matching the TRELLIS.2 `DinoV3FeatureExtractor.extract_features(...)` boundary.
- Load only the checkpoint tensors needed for the attempted path, with deterministic missing-key and shape blockers.
- Preserve `prepare_dinov3_image_tensor(...)` as the normalized BCHW image input boundary.
- Update `assess_dinov3_mlx_conditioning(...)` so real local DINOv3 assets attempt MLX forward execution instead of immediately blocking at transformer construction.
- Preserve fake-fixture behavior and add fake executable fixtures that prove embeddings, RoPE/attention/MLP/norm routing, and output metadata without real weights.
- Refresh the real alpha forward trace at `inputs/trellis2/demo-alpha.webp`; it must either record a real `image-conditioning` output and reach the existing sparse boundary, or report the first exact DINOv3 forward blocker.
- Document the current boundary in README and `.agent/work/trellis-dinov3-mlx-forward/ATTEMPT.md`.

## Acceptance Criteria

- Public or internal DINOv3 MLX forward helpers exist and are exercised by tests without importing forbidden runtime dependencies.
- Fake executable DINOv3 fixtures produce deterministic conditioning metadata with last dimension 1024 and reach the existing sparse-structure boundary.
- Missing and malformed fake checkpoint keys return precise blockers naming the key, tensor shape, or module that stopped construction or forward execution.
- Real local DINOv3 assets are inspected and then attempted through the MLX forward path; the previous generic `MLX DINOv3 transformer block construction` blocker is replaced by either real conditioning output metadata or a more specific blocker.
- If real conditioning metadata is produced, `attempt_forward_trace(...)` records an `image-conditioning` output and dispatches to the existing sparse-structure boundary without entering full sampling.
- If real forward is incomplete, the blocker names the first unsupported embedding/RoPE/attention/MLP/norm/checkpoint/memory requirement and includes the local checkpoint/config reference.
- Default `uv run pytest` passes without real weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor imports.
- Runtime dependency metadata still excludes PyTorch, TorchVision, Transformers, Hugging Face Hub, and ONNX Runtime.
- Real DINOv3 weights, TRELLIS.2 weights, RMBG weights, and generated outputs remain untracked/ignored by git.
- README or TRELLIS docs describe the new MLX DINOv3 forward boundary and the current blocker or output metadata.

## Blocking Questions Or Assumptions

- Assumption: completing a full real DINOv3 ViT-L/16 forward may be too large for one slice; a precise blocker is acceptable if it is more specific than the current transformer-construction placeholder.
- Assumption: the TRELLIS.2 conditioning resolution remains 512, while the local DINOv3 config file reports base `image_size=224`; the forward path must handle the runtime image tensor size used by TRELLIS.2.
- Assumption: the expected output feature width remains 1024 because local sparse-flow `cond_channels` is 1024.
- Assumption: exact numerical parity with Transformers is desirable but not required for this slice unless a tiny fake fixture can make parity-like assertions without adding forbidden runtime dependencies.
- Assumption: downstream sparse-structure sampling remains out of scope except for existing boundary dispatch.

## Anti-Goals

- Do not implement sparse-structure sampling, SLat sampling, decoders, mesh extraction, texture baking, or GLB/OBJ export.
- Do not solve RMBG/BiRefNet `deform_conv2d`.
- Do not add PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor modules as runtime dependencies.
- Do not silently download DINOv3 assets during import, tests, validation, or attempt mode.
- Do not fake real DINOv3 output for real local attempts.
- Do not claim full TRELLIS.2 image-to-3D inference works.
