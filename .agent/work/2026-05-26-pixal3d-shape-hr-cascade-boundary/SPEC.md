# Pixal3D Shape HR Cascade Boundary Spec

## Bounded Goal

Advance the Pixal3D MLX runtime from the completed 512 shape SLat artifact into the guarded HR shape cascade boundary, writing inspectable HR coordinate and optional HR shape SLat artifacts while preserving the next real blocker.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing Torch/CUDA runtime paths.

## Work Scale And Shape

- Scale: medium implementation cycle.
- Shape: inference pipeline progression plus cascade coordinate planning, artifact writing, tests, and docs.
- Selected lenses: engineering, runtime.

## Constraints And Risks

- Preserve the existing sparse projection, sparse structure, and 512 shape SLat trace/artifact contracts.
- Runtime code remains MLX-first and must not add Torch, CUDA, nvdiffrast, Kaolin, NAF Torch Hub, or vendor-source imports.
- Use the shared `trellis2_decode.py` shape decoder upsample helper rather than duplicating Pixal3D-specific sparse decoder logic.
- Apply Pixal3D's HR token guard before HR shape SLat sampling and record the selected HR resolution/grid.
- Real NAF high-resolution feature extraction remains unimplemented; this cycle may accept explicit test-supplied NAF feature maps and must report a structured blocker when they are absent.
- Full texture SLat, shape/texture decode, PBR bake, mesh extraction, and GLB export remain out of scope for this cycle.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXHR-01 | Pixal3D can upsample 512 shape SLat coordinates through the shared MLX shape decoder helper when compatible assets are present. |
| PXHR-02 | The runtime quantizes/deduplicates HR coordinates using Pixal3D's token guard and records actual HR resolution, grid resolution, and token count. |
| PXHR-03 | The runtime writes an inspectable HR coordinate artifact after successful LR-to-HR cascade coordinate planning. |
| PXHR-04 | With explicit HR NAF features and compatible fake HR flow assets, the runtime runs the shared 1024 shape SLat probe and writes `shape_slat_hr.npz`. |
| PXHR-05 | Without HR NAF features, the runtime returns a structured HR projection blocker instead of pretending HR shape SLat was attempted. |
| PXHR-06 | Docs and script descriptions remain honest about current support and remaining Pixal3D blockers. |

## Acceptance Criteria

- Existing Pixal3D tests still pass with the previous NAF blocker and 512 shape SLat behavior intact.
- A valid fake Pixal3D root with fake shape decoder assets plus explicit LR NAF reaches HR coordinate planning, writes an HR coordinate NPZ, and blocks at HR projection conditioning when HR NAF is absent.
- A valid fake Pixal3D root with fake shape decoder, fake 1024 shape SLat assets, explicit LR NAF, and explicit HR NAF reaches HR shape SLat, writes `shape_slat_hr.npz`, and blocks at the next texture/NAF boundary.
- Targeted Pixal3D/SLat/decode tests, full test suite, lock check, diff check, build, artifact checker, and git hygiene pass.

## Anti-Goals

- Do not implement NAF itself in this cycle.
- Do not implement texture SLat, shape decoder mesh extraction, texture/PBR decoding, or final GLB export.
- Do not download, vendor, or redistribute real Pixal3D weights.
- Do not change package versioning or publish artifacts.
