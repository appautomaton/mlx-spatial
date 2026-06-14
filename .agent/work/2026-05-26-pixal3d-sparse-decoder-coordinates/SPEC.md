# Pixal3D Sparse Decoder Coordinates Spec

## Bounded Goal

Advance the Pixal3D MLX runtime from sparse-flow latent output through sparse-decoder coordinate extraction, write the sparse-structure coordinate artifact, and return a precise `shape-slat-sampling` blocker when shape SLat is the next missing stage.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing PyTorch/CUDA-only runtime paths.

## Work Scale And Shape

- Scale: small implementation cycle.
- Shape: inference pipeline progression plus artifact/test/docs updates.
- Selected lenses: engineering, runtime.

## Constraints And Risks

- Preserve the existing `Pixal3DInferencePipeline.generate` staged trace shape.
- Do not claim textured GLB generation, shape SLat, texture SLat, high-resolution NAF, or MoGe auto-camera support.
- Runtime code remains Torch-free and must not add CUDA, nvdiffrast, Kaolin, or vendor-source imports.
- Real `weights/pixal3d` may be absent locally; tests must prove behavior with MLX fake checkpoints that exercise the same config/checkpoint path.
- Artifact writing must keep arrays explicit and inspectable without hiding blocker state.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXCOORD-01 | When sparse decoder returns non-empty coordinates, Pixal3D writes a `sparse_structure.npz` artifact containing coordinates and metadata. |
| PXCOORD-02 | The Pixal3D result includes both sparse projection and sparse structure artifacts and records `artifact:sparse_structure` in completed stages. |
| PXCOORD-03 | The next blocker after successful sparse coordinate extraction is `shape-slat-sampling`, with metadata that names the coordinate shape and next target. |
| PXCOORD-04 | Existing sparse-flow and sparse-projection blockers remain structured and non-regressive. |
| PXCOORD-05 | Docs and scripts describe the current expected outputs and remaining Pixal3D blocker accurately. |

## Acceptance Criteria

- A valid fake Pixal3D sparse-flow plus sparse-decoder root reaches `sparse-structure-decoding`, writes `sparse_structure.npz`, then blocks at `shape-slat-sampling`.
- The `sparse_structure.npz` file contains coordinate data, shape metadata, target resolution, and pipeline context that can be inspected with NumPy.
- Existing Pixal3D tests still pass, including invalid fake roots that stop before sparse decoder coordinates.
- Targeted Pixal3D/export tests, full test suite, lock check, diff check, and release artifact hygiene pass.

## Anti-Goals

- Do not implement shape SLat sampling or mesh/GLB export in this cycle.
- Do not download or vendor real Pixal3D weights.
- Do not replace shared TRELLIS.2 sparse-structure helpers.
- Do not change package versioning or publish artifacts.
