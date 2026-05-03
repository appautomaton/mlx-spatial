# DESIGN: TRELLIS.2 MLX Inference Pipeline

## Runtime Shape

The user-facing runtime surface remains `Trellis2InferencePipeline.attempt_forward_trace(...)`.

The pipeline should stay blocker-driven:

1. Validate assets and probes.
2. Preprocess image.
3. Produce real DINOv3 conditioning.
4. Try the next TRELLIS.2 stage.
5. Append real outputs for completed stages.
6. Stop at the first exact blocker.

No real stage may advance by placeholder tensors.

## Module Boundaries

- `trellis2_forward.py`: orchestration dataclasses, pipeline config discovery, trace dispatch helpers.
- `trellis2_sparse_structure.py`: sparse structure flow config, checkpoint key map, FlowEuler sampler, sparse flow model boundary, sparse decoder boundary.
- `trellis2_slat.py`: shape/texture SLat config, checkpoint key maps, sampler contracts, SLat flow boundaries.
- `trellis2_decode.py`: shape decoder, texture decoder, mesh-like intermediate contracts.
- `trellis2_export.py`: final artifact metadata and mesh/export boundary.
- `trellis2_inference.py`: `attempt_forward_trace(...)` sequencing and blocker conversion.

These module names are planning targets. Execution may keep a slice inside fewer files when the local code shape makes that cleaner.

## Data Contracts

- DINOv3 conditioning: `cond`, shape `(B, T, 1024)`, dtype `float32`.
- Sparse structure latent noise: `(B, 8, 16, 16, 16)` for the local sparse flow config.
- Sparse structure output: real sampled latent plus decoded sparse coordinates only after the decoder runs.
- Shape SLat: sparse-coordinate conditioned latent state for configured `512`, `1024`, or `1024_cascade` pipeline type.
- Texture SLat: texture latent state conditioned on image features and normalized `shape_slat`, before shape or texture decoding.
- Decode latent: combined boundary that consumes real `shape_slat`, real `tex_slat`, and output resolution, then decodes shape and texture together.
- Export output: structured metadata under `outputs/` plus artifact path only when a real artifact is written.

## Error Contract

Every failed stage returns `Trellis2ForwardBlocker` with:

- `stage`
- `operation`
- `reference`
- `reason`
- `next_slice`

Blockers should name the concrete missing key group, shape, dtype, MLX op, memory limit, decoder boundary, or export dependency. Generic blockers are only acceptable before a stage has been inspected.

## Test Strategy

- Default tests use fake configs, fake safetensors, and tiny MLX arrays.
- Real-weight tests are explicit local probes and must not run by default.
- Each slice adds targeted tests for parser, key-map, shape, sampler, or blocker behavior before moving deeper.
- Full-suite verification remains `uv run pytest`.
