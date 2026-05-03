# SPEC: TRELLIS.2 MLX Inference Pipeline

## Bounded Goal

Implement a blocker-driven, MLX-first TRELLIS.2 image-to-3D inference pipeline from real image input through DINOv3 conditioning, sparse structure sampling/decoding, SLat shape and texture sampling/decoding, and final mesh/export boundary, producing a real artifact when feasible or the first exact blocker at each remaining stage.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Start from the verified DINOv3 image-conditioning path from `trellis-dinov3-mlx-forward`: local real weights produce `cond` with shape `(1, 1029, 1024)` and dtype `float32`.
- Keep runtime execution MLX-first; do not add Torch or Transformers as runtime fallbacks for the TRELLIS.2 pipeline.
- Use local ignored assets under `weights/trellis2`, `weights/dinov3-vitl16-pretrain-lvd1689m`, `inputs/`, and `outputs/`; do not commit checkpoints or generated binary artifacts.
- Default tests must not require real weights, network access, Torch, Transformers, or generated mesh/image outputs.
- Real path execution must not advance through a stage with synthetic outputs. A stage either runs real computation from real inputs or returns a structured blocker.
- The scope is intentionally aggressive and must be planned as ordered checkpoint slices with verification after each stage.
- RGB background removal remains optional for this spec because local RMBG-2.0 is currently blocked by deformable convolution coverage; alpha input must remain a valid route through the rest of the pipeline.

## Required Behavior

- Preserve the completed real image preprocessing and DINOv3 conditioning behavior.
- Discover and document the full TRELLIS.2 image-to-3D pipeline contract from local config and vendored source, including required model keys, checkpoint layouts, sampler configs, normalization parameters, and stage order.
- Implement or precisely block MLX sparse structure flow sampling with FlowEuler guidance interval behavior, using noise layout `(B, 8, 16, 16, 16)` and DINO conditioning width validation.
- Implement or precisely block sparse structure decoder execution, producing sparse coordinates only when the decoder runs from real sampled sparse structure latents.
- Implement or precisely block SLat shape sampling for the configured 512/1024 or cascade path, including sparse coordinate conditioning and sampler behavior.
- Implement or precisely block shape SLat decoding into the first mesh or mesh-like intermediate the TRELLIS.2 pipeline expects.
- Implement or precisely block SLat texture sampling and texture decoding against the decoded shape/mesh boundary.
- Implement or precisely block final mesh/export handling into `outputs/`, keeping artifact metadata structured even when export is blocked.
- For every stage, add fake-fixture coverage for parser/shape/key-map behavior, targeted real-weight probes where appropriate, and structured trace output for completed stages and blockers.

## Acceptance Criteria

- The current sparse-only spec is superseded by this full-pipeline spec and the active Automaton state points at this file.
- A later `auto-plan` can split the work into ordered execution slices without reopening product scope.
- `attempt_forward_trace(...)` remains the user-visible runtime surface and reports completed stages plus the first exact blocker without fabricated downstream progress.
- Real alpha input continues to complete image preprocessing and DINOv3 conditioning before entering the newly planned TRELLIS.2 stages.
- Each implemented slice either advances the real trace to a deeper stage or records a blocker naming the exact missing model component, checkpoint key group, tensor shape, dtype, MLX op, memory issue, sampler behavior, decoder behavior, or export boundary.
- Default `uv run pytest` remains the verification gate for committed behavior.
- README/status/verify artifacts are updated at each execution milestone to reflect the real current boundary.

## Blocking Questions Or Assumptions

- Assumption: this spec may still finish with a precise blocker before final GLB/OBJ export; that is acceptable if the blocker is exact and the completed stage outputs are real.
- Assumption: the first execution slice should begin at sparse structure flow/sampler construction because DINOv3 conditioning is already verified.
- Assumption: sparse decoder, SLat flow models, SLat decoders, mesh extraction, or texture export may expose MLX op gaps that should be captured as blockers rather than bypassed.
- Assumption: RGB background removal can remain a parallel or later enhancement unless alpha-image inference is no longer sufficient for pipeline validation.

## Anti-Goals

- Do not implement training, dataset tooling, fine-tuning, or evaluation suites.
- Do not auto-download gated or large weights during tests or default commands.
- Do not introduce Torch/Transformers runtime fallback execution to make the path appear complete.
- Do not optimize performance, memory, quantization, or batching before correctness boundaries are proven.
- Do not build a UI, server API, or notebook workflow as part of this spec.
- Do not hide unsupported stages behind placeholder tensors, canned meshes, or demo artifacts.
