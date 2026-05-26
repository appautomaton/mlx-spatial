# Pixal3D Latent Decode Boundary Spec

## Bounded Goal

Advance the Pixal3D MLX runtime from completed texture SLat into shared shape and texture decoder execution, writing inspectable decoded shape-field and texture-PBR voxel artifacts while preserving mesh extraction/GLB export as the next blocker.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing Torch/CUDA runtime paths.

## Work Scale And Shape

- Scale: medium implementation cycle.
- Shape: inference pipeline progression plus decoded artifact writing, tests, and docs.
- Selected lenses: engineering, runtime.

## Constraints And Risks

- Preserve existing Pixal3D sparse, shape SLat, HR cascade, texture SLat, trace, and artifact contracts.
- Runtime code remains MLX-first and must not add Torch, CUDA, nvdiffrast, Kaolin, NAF Torch Hub, or vendor-source imports.
- Decode execution must reuse shared `trellis2_decode.py` helpers rather than a Pixal3D-specific decoder fork.
- Decoder token limits must remain explicit and Apple GPU memory-aware; use the Pixal3D `max_num_tokens` guard unless a narrower local check is needed.
- Normal CLI runs still cannot pass the NAF boundary without MLX NAF-equivalent features; this cycle must not pretend the full CLI path is complete.
- Mesh extraction, PBR baking, and textured GLB export remain out of scope for this cycle.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXLD-01 | After `texture_slat.npz`, Pixal3D runs the shared FlexiDualGrid shape decoder against HR shape SLat features and writes `shape_decoder_fields.npz`. |
| PXLD-02 | Pixal3D then runs the shared guided texture decoder against texture SLat features and shape decoder subdivisions, writing `texture_decoder_pbr.npz`. |
| PXLD-03 | Runtime metadata records decoder configs/checkpoints, decoded tensor shapes, subdivision guide shapes, token guard, and the next mesh/export blocker. |
| PXLD-04 | If either decoder cannot complete, the pipeline returns a structured decode blocker with the relevant asset path and reason. |
| PXLD-05 | Docs and script descriptions remain honest about current support and remaining Pixal3D blockers. |

## Acceptance Criteria

- Existing Pixal3D tests still pass with previous blockers and artifacts intact.
- A valid fake Pixal3D root with fake decode assets plus explicit LR/HR/texture NAF reaches shape decoder execution and writes `shape_decoder_fields.npz`.
- The same fake root reaches texture decoder execution, writes `texture_decoder_pbr.npz`, and blocks at mesh extraction/export.
- Targeted Pixal3D/SLat/decode tests, full test suite, import scan, lock check, diff check, build, artifact checker, and git hygiene pass.

## Anti-Goals

- Do not implement NAF itself in this cycle.
- Do not implement mesh extraction, PBR baking, or final GLB export for Pixal3D in this cycle.
- Do not download, vendor, or redistribute real Pixal3D weights.
- Do not change public package versioning during this decode-boundary cycle.
