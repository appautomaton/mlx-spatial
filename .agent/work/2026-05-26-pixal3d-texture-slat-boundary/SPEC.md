# Pixal3D Texture SLat Boundary Spec

## Bounded Goal

Advance the Pixal3D MLX runtime from completed HR shape SLat into texture projection and texture SLat probing, writing an inspectable texture SLat artifact when explicit texture NAF features are supplied and preserving the next decode/export blocker.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing Torch/CUDA runtime paths.

## Work Scale And Shape

- Scale: medium implementation cycle.
- Shape: inference pipeline progression plus texture artifact writing, tests, and docs.
- Selected lenses: engineering, runtime.

## Constraints And Risks

- Preserve existing sparse, 512 shape SLat, HR coordinate, and HR shape SLat trace/artifact contracts.
- Runtime code remains MLX-first and must not add Torch, CUDA, nvdiffrast, Kaolin, NAF Torch Hub, or vendor-source imports.
- Texture SLat execution must use the shared `trellis2_slat.py` texture probe rather than a Pixal3D-specific duplicate.
- Match upstream Pixal3D texture sampling semantics by normalizing HR shape SLat features before using them as texture concat conditioning, then applying texture SLat normalization to sampled texture features.
- Real NAF high-resolution feature extraction remains unimplemented; this cycle may accept explicit test-supplied texture NAF feature maps and must report a structured texture projection blocker when they are absent.
- Full shape/texture decode, PBR baking, mesh extraction, and GLB export remain out of scope for this cycle.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXTX-01 | Pixal3D can build coordinate-indexed texture projection conditioning from HR shape SLat coordinates. |
| PXTX-02 | Without texture NAF features, the runtime returns a structured texture projection blocker instead of pretending texture SLat was attempted. |
| PXTX-03 | With explicit texture NAF features and compatible fake texture flow assets, the runtime runs the shared 1024 texture SLat probe and writes `texture_slat.npz`. |
| PXTX-04 | Runtime metadata records texture projection selection, normalized shape concat feature shape, sampled texture feature shape, and the next decode blocker. |
| PXTX-05 | Docs and script descriptions remain honest about current support and remaining Pixal3D blockers. |

## Acceptance Criteria

- Existing Pixal3D tests still pass with the previous NAF blockers and shape artifacts intact.
- A valid fake Pixal3D root with fake texture flow assets plus explicit LR/HR shape NAF reaches texture projection conditioning and blocks there when texture NAF is absent.
- The same fake root plus explicit texture NAF reaches texture SLat, writes `texture_slat.npz`, and blocks at latent decode/export.
- Targeted Pixal3D/SLat/decode tests, full test suite, lock check, diff check, build, artifact checker, and git hygiene pass.

## Anti-Goals

- Do not implement NAF itself in this cycle.
- Do not implement full shape/texture decode, PBR baking, mesh extraction, or final GLB export.
- Do not download, vendor, or redistribute real Pixal3D weights.
- Do not change package versioning or publish artifacts.
