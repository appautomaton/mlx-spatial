# Pixal3D Shape SLat LR Boundary Spec

## Bounded Goal

Advance the Pixal3D MLX runtime from sparse decoder coordinates into the 512-resolution shape SLat probe, write a shape-SLat intermediate artifact when that probe succeeds, and return a precise cascade/decoder blocker for the next missing stage.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing PyTorch/CUDA-only runtime paths.

## Work Scale And Shape

- Scale: medium implementation cycle.
- Shape: inference pipeline progression plus projection-indexing helper, artifact writer, tests, and docs.
- Selected lenses: engineering, runtime.

## Constraints And Risks

- Preserve the existing Pixal3D staged trace contract and current sparse-coordinate artifacts.
- Runtime code remains Torch-free and must not add CUDA, nvdiffrast, Kaolin, NAF Torch Hub, or vendor-source imports.
- Real NAF high-resolution feature extraction remains unimplemented; this cycle may expose a sharper NAF blocker for normal runs and use explicit test-supplied NAF feature maps to prove the downstream SLat path.
- Shape SLat execution must use the shared MLX `trellis2_slat.py` probe rather than a Pixal3D-specific duplicate.
- Full shape decoder, HR cascade upsample, texture SLat, PBR decode, and GLB export remain out of scope for this cycle.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXSLAT-01 | Pixal3D can build coordinate-indexed shape projection conditioning from full projected features and sparse decoder coordinates. |
| PXSLAT-02 | When NAF features are unavailable, the runtime reports a structured shape projection blocker instead of pretending shape SLat was attempted. |
| PXSLAT-03 | With compatible fake shape SLat assets and explicit NAF features, the runtime runs the shared 512 shape SLat probe and writes `shape_slat_lr.npz`. |
| PXSLAT-04 | The result records shape SLat metadata and returns the next real blocker: HR cascade/shape decoder handoff. |
| PXSLAT-05 | Existing sparse projection, sparse flow, sparse decoder, and Pixal3D docs remain coherent and non-regressive. |

## Acceptance Criteria

- A helper test proves projected feature selection from sparse coordinates is deterministic and validates coordinate shape/bounds.
- Existing sparse-coordinate fake root reaches a structured shape projection blocker when NAF features are absent.
- A valid fake Pixal3D root with fake shape SLat assets plus explicit NAF features reaches 512 shape SLat, writes `shape_slat_lr.npz`, and blocks at the next cascade/decoder boundary.
- Targeted Pixal3D/SLat tests, full test suite, lock check, diff check, build, artifact checker, and git hygiene pass.

## Anti-Goals

- Do not implement NAF itself in this cycle.
- Do not implement HR shape SLat cascade, shape decoder, texture SLat, texture/PBR decoder, or final GLB export.
- Do not download, vendor, or redistribute real Pixal3D weights.
- Do not change package versioning or publish artifacts.
