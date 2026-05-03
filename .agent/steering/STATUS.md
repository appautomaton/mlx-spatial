---
active_change: trellis2-mlx-inference-pipeline
stage: verify
---

# Status

## Current Change

- active change: `trellis2-mlx-inference-pipeline`
- current stage: `verify`

## What Is True Now

- TRELLIS.2 weights are present locally under ignored weights/trellis2 and asset validation reports ready.
- The project can inspect and probe selected TRELLIS.2 checkpoints with MLX arrays.
- The inference pipeline currently supports readiness/dry-run and image-path attempt mode with structured blockers.
- The previous `trellis-e2e-inference-attempt` change is verified: uv run pytest reported 62 passed and 5 skipped.
- The previous `trellis-image-preprocessing-background` change is verified: full default verification reported `83 passed, 5 skipped`.
- The previous generic preprocessing blocker has been replaced by real alpha preprocessing and a precise RGB/RMBG port blocker.
- The previous `trellis-forward-trace-conditioning` change is verified: full default verification reported `94 passed, 5 skipped`.
- The previous `trellis-dinov3-conditioning` change is verified: full default verification reported `106 passed, 5 skipped`.
- The previous `trellis-dinov3-mlx-forward` change is verified and closed.
- The active change is `trellis2-mlx-inference-pipeline`, reframed from the previous sparse-only boundary into a full TRELLIS.2 MLX inference pipeline spec.
- Canonical spec now exists at `.agent/work/trellis2-mlx-inference-pipeline/SPEC.md`.
- Canonical design now exists at `.agent/work/trellis2-mlx-inference-pipeline/DESIGN.md`.
- Canonical plan now exists at `.agent/work/trellis2-mlx-inference-pipeline/PLAN.md`.
- The previous engineering review correction has been applied: Slice 3 and Slice 8 now use `attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --dino-root weights/dinov3-vitl16-pretrain-lvd1689m`.
- The previous SLat ordering correction has been applied: Slice 6 now samples texture SLat from `shape_slat`, and Slice 7 now covers combined shape/texture decode.
- Engineering review verdict is `approved_with_risks`: execution is safe to start, with expected real MLX op, dtype, checkpoint-layout, memory, decoder, and export blockers treated as first-class outcomes.
- Slice 1 is complete: full TRELLIS.2 pipeline contract discovery now parses model keys, sampler configs, normalization values, and default pipeline type from `pipeline.json`.
- Slice 2 is complete: sparse structure FlowEuler schedule, guidance interval, noise shape validation, fake sampling metadata, and sharper real sparse boundary reporting are implemented.
- Slice 3 is complete: sparse structure forward probing now loads selected sparse-flow tensors, executes the MLX input projection, validates first-block checkpoint keys, and stops at the exact unported transformer block boundary.
- Slice 4 is complete: sparse structure decoder config parsing, checkpoint key probing, coordinate extraction, and a standalone decoder boundary dispatcher are implemented.
- Slice 5 is complete: shape SLat config parsing, route selection, real checkpoint probing, sparse-coordinate validation, and first sparse-feature input projection are implemented.
- Slice 6 is complete: texture SLat route selection, upstream `shape_slat` blocking, shape-feature validation, concat feature projection, and real texture checkpoint probing are implemented.
- Slice 7 is complete: shape/texture latent decoder config parsing, checkpoint probing, SLat layout validation, `from_latent` projections, and combined decode blockers are implemented.
- Slice 8 is complete: export path validation, ignored-outputs artifact writing metadata, and export readiness/blocker assessment are implemented.
- The new spec covers real image input through DINOv3 conditioning, sparse structure sampling/decoding, SLat shape and texture sampling/decoding, and final mesh/export boundary, with exact blockers instead of synthetic downstream progress.
- Planning identifies `DinoV3FeatureExtractor` with model `facebook/dinov3-vitl16-pretrain-lvd1689m` as the image-conditioning dependency from `weights/trellis2/pipeline.json`.
- `trellis-dinov3-mlx-forward` completed forward key map/tensor loader, patch embedding/token assembly, RoPE probe, one MLX DINOv3 transformer block, full fake forward integration, real DINOv3 forward probing, docs, and final verification.
- Local `weights/dinov3-vitl16-pretrain-lvd1689m` now validates ready with `config.json` and `model.safetensors`.
- The local DINOv3 checkpoint is readable: ViT-L/16 config, hidden size 1024, 24 layers, 415 tensors, patch embedding shape `(1024, 3, 16, 16)`.
- DINOv3 image-conditioning now completes with local real weights: the live alpha forward trace produces `cond` with shape `(1, 1029, 1024)` and dtype `float32` after 24 transformer layers and final layer normalization.
- The current blocker is `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- The real sparse input projection executes to `(1, 4096, 1536)` before stopping at sparse 3D RoPE/shared-modulation block execution.
- The standalone sparse decoder boundary currently reports missing local sparse decoder assets at `weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json`; the main trace does not reach decoder while sparse flow is blocked.
- The standalone shape-SLat boundary maps the configured `1024_cascade` route; without coordinates it blocks on upstream sparse-coordinate availability, and with fake coordinates it executes real local shape-SLat input projection to `(2, 1536)` before `ModulatedSparseTransformerCrossBlock`.
- The standalone texture-SLat boundary maps the configured `1024_cascade` route; without `shape_slat` it blocks on upstream shape-SLat availability, and with fake `shape_slat` it executes real local texture-SLat concat projection to `(2, 1536)` before `ModulatedSparseTransformerCrossBlock`.
- The standalone decode boundary blocks on missing `shape_slat`/`texture_slat`; with fake latents it executes real local shape and texture decoder `from_latent` projections to `(2, 1024)` before the shape decoder sparse ConvNeXt/FlexiDualGrid boundary.
- Sparse flow config is local at `weights/trellis2/ckpts/ss_flow_img_dit_1_3B_64_bf16.json`: resolution 16, channels 8, model channels 1536, 30 transformer blocks, 12 heads, rope positions, shared modulation, q/k RMS norm, and bfloat16 torso intent.
- Pipeline sampler config requests `FlowEulerGuidanceIntervalSampler` with 12 steps, guidance strength 7.5, guidance rescale 0.7, guidance interval `[0.6, 1.0]`, and `rescale_t=5.0`.
- Current targeted Slice 1 verification passed: `uv run pytest tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `28 passed`.
- Current targeted Slice 2 verification passed: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py` reported `22 passed`.
- Current targeted Slice 3 verification passed: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py` reported `25 passed`.
- Current targeted Slice 4 verification passed: `uv run pytest tests/test_trellis2_sparse_structure.py tests/test_trellis2_forward.py` reported `34 passed`.
- Current targeted Slice 5 verification passed: `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py tests/test_trellis2_inference.py` reported `43 passed`.
- Current targeted Slice 6 verification passed: `uv run pytest tests/test_trellis2_slat.py tests/test_trellis2_forward.py` reported `41 passed`.
- Current targeted Slice 7 verification passed: `uv run pytest tests/test_trellis2_decode.py tests/test_trellis2_forward.py` reported `33 passed`.
- Current targeted Slice 8 verification passed: `uv run pytest tests/test_trellis2_export.py` reported `6 passed`.
- Current real alpha trace passes through DINOv3 and reports `sparse-structure-sampling` / `MLX sparse structure ModulatedTransformerCrossBlock forward`.
- Current full suite verification passed: `uv run pytest` reported `174 passed, 5 skipped`.
- Current execution evidence is recorded in `.agent/work/trellis2-mlx-inference-pipeline/PLAN.md`.
- Current final verification evidence is recorded in `.agent/work/trellis2-mlx-inference-pipeline/VERIFY.md`.
- Auto-verify passed for Slice 8: export path policy, blocked-export reporting, README, and VERIFY evidence all match the approved acceptance criteria.
- Previous DINOv3 execution evidence is recorded in `.agent/work/trellis-dinov3-mlx-forward/ATTEMPT.md`, `.agent/work/trellis-dinov3-mlx-forward/PLAN.md`, and `.agent/work/trellis-dinov3-mlx-forward/VERIFY.md`.
- Local `weights/rmbg2` assets are present and validate ready.
- The alpha sample `inputs/trellis2/demo-alpha.webp` now completes `image-preprocessing-background`; in forward-trace mode it also completes DINOv3 `image-conditioning` and blocks next at `sparse-structure-sampling`.
- The RGB sample `inputs/trellis2/demo-rgb-background.png` routes through local RMBG assessment and blocks on MLX op coverage: RMBG-2.0 uses `torchvision.ops.deform_conv2d` in `ASPPDeformable`, and this MLX runtime has no `mlx.nn.DeformConv2d`.

## Next Step

Frame the next implementation slice for sparse `ModulatedTransformerCrossBlock` forward.

## Open Risks

- Full image-to-3D inference, sparse/latent sampling, decoder execution, mesh extraction, and GLB export are not implemented yet.
- Export metadata and path policy are implemented, but real GLB/OBJ writing still depends on decoded mesh/texture payloads.
- `briaai/RMBG-2.0` is gated and non-commercial; local safetensor acquisition must stay explicit and out of default tests.
- TRELLIS.2 parity remains staged and blocker-driven rather than end-to-end runnable.
