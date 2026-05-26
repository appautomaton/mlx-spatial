# Pixal3D Mesh Export Boundary Spec

## Bounded Goal

Advance the Pixal3D MLX runtime from decoded shape/texture tensors into shared FlexiDualGrid mesh extraction, texture baking, and textured GLB writing when lower-level explicit NAF inputs allow the pipeline to reach decoded artifacts.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing Torch/CUDA runtime paths.

## Work Scale And Shape

- Scale: medium implementation cycle.
- Shape: inference pipeline export progression plus GLB artifact writing, tests, and docs.
- Selected lenses: engineering, runtime.

## Constraints And Risks

- Preserve existing Pixal3D sparse, SLat, decode, trace, and artifact contracts.
- Runtime code remains MLX-first and must not add Torch, CUDA, nvdiffrast, Kaolin, NAF Torch Hub, or vendor-source imports.
- Mesh extraction and texture baking must reuse shared `ovoxel.py` and `trellis2_export.py` helpers rather than introducing a Pixal3D-specific exporter stack.
- GLB metadata should identify Pixal3D instead of incorrectly labeling the final file as TRELLIS.2.
- Export options must keep Apple GPU and CPU memory guards explicit: texture size, face target, xatlas face guard, xatlas chunking, and texture-bake backend.
- Normal CLI runs still cannot pass the NAF boundary without MLX NAF-equivalent features; this cycle must not claim full user-facing Pixal3D generation is complete.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXME-01 | After `texture_decoder_pbr.npz`, Pixal3D extracts a FlexiDualGrid mesh from decoded shape fields using shared mesh helpers. |
| PXME-02 | Pixal3D bakes decoded PBR voxels onto the mesh using the shared Mac-native texture baking helper. |
| PXME-03 | Pixal3D writes the requested `.glb` output and returns a ready result when export succeeds. |
| PXME-04 | Export failures return structured `mesh-export` or `glb-export` blockers while preserving already written intermediate artifacts. |
| PXME-05 | Docs, CLI help, and script defaults describe the new GLB path and the remaining NAF blocker honestly. |

## Acceptance Criteria

- Existing Pixal3D tests still pass with previous blockers and artifacts intact.
- A fake Pixal3D root with explicit LR/HR/texture NAF can reach export through monkeypatched mesh/bake/write helpers and returns a ready GLB result.
- Export failure paths preserve `shape_decoder_fields.npz` and `texture_decoder_pbr.npz` and return structured blockers.
- Targeted Pixal3D/SLat/decode/export tests, full test suite, import scan, lock check, diff check, build, artifact checker, and git hygiene pass.

## Anti-Goals

- Do not implement NAF itself in this cycle.
- Do not implement MoGe auto-camera for Pixal3D in this cycle.
- Do not download, vendor, or redistribute real Pixal3D weights.
- Do not change public package versioning during this mesh-export cycle.
