# DESIGN: TRELLIS.2 End-to-End Inference Attempt

## Pipeline Contract

- Add an inference-only pipeline surface under `mlx_spatial.trellis2_inference` or equivalent.
- Expose two modes:
  - readiness/dry-run: validate local assets, configs, weight probes, and stage wiring.
  - attempt: accept an image path and run implemented stages until output or first blocker.
- Do not fake stage outputs. A missing compute stage must return or raise a structured blocker.

## Blocker Model

- Blockers should include `stage`, `operation`, `reference`, `reason`, and `next_slice`.
- Blockers are valid first-slice outputs when full inference is not yet implementable.
- Blockers must be deterministic in tests and specific enough to drive the next spec.

## Stage Discovery

- `vendors/trellis-mac/generate.py` is the primary runnable reference for stage order.
- `vendors/TRELLIS.2/trellis2/pipelines/*` and `vendors/TRELLIS.2/trellis2/models/*` are the original architecture references for pipeline and model loading behavior.
- Discovery is read-only; runtime code must not import from `vendors/`.

## Expected Initial Stages

- input image validation/loading boundary
- asset/config validation
- checkpoint/probe readiness
- image preprocessing/background handling boundary
- image feature extraction boundary
- sparse structure sampling boundary
- shape SLat sampling boundary
- texture SLat sampling boundary
- shape decoder boundary
- texture decoder boundary
- mesh extraction/export boundary

## Verification Strategy

- Default tests use fake assets, fake image paths or tiny fixtures, and no real weights.
- Real local attempt may use `weights/trellis2/` and a generated tiny local sample image if no suitable input exists.
- The first attempt is allowed to end at a precise blocker.
- Full GLB output is not required for this change.
