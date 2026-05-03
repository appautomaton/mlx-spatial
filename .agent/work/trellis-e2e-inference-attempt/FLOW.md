# FLOW: TRELLIS.2 Image-to-3D Inference

## Reference Basis

- Primary runnable path: `vendors/trellis-mac/generate.py`.
- Original pipeline architecture: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py`.
- Original pipeline/model loading: `vendors/TRELLIS.2/trellis2/pipelines/base.py`, `vendors/TRELLIS.2/trellis2/pipelines/__init__.py`, and `vendors/TRELLIS.2/trellis2/models/__init__.py`.
- Runtime MLX code must not import from `vendors/`.

## Runnable Entrypoint Flow

1. Backend setup and environment selection.
   - Reference: `vendors/trellis-mac/generate.py:8-34`.
   - Behavior: sets PyTorch/MPS fallback, attention backend, sparse attention backend, sparse conv backend, and import paths.
   - MLX replacement point: avoid PyTorch/MPS environment setup; define MLX-native backend capabilities instead.

2. CLI/input validation.
   - Reference: `vendors/trellis-mac/generate.py:43-70`.
   - Behavior: parses image path, seed, output stem, pipeline type, texture settings, and step override; exits if image path is missing.
   - MLX replacement point: validate image path and execution options without importing PIL in default tests unless a bounded image loader is needed.

3. Pipeline loading.
   - Reference: `vendors/trellis-mac/generate.py:76-86`; `vendors/TRELLIS.2/trellis2/pipelines/base.py:21-50`; `vendors/TRELLIS.2/trellis2/pipelines/__init__.py:26-45`.
   - Behavior: loads `Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")`, resolves `pipeline.json`, loads configured model checkpoints, and moves the pipeline to MPS.
   - MLX replacement point: validate local `weights/trellis2/pipeline.json`, `texturing_pipeline.json`, and checkpoint files; load selected safetensors into MLX arrays; do not construct PyTorch modules.

4. Image load.
   - Reference: `vendors/trellis-mac/generate.py:88-90`.
   - Behavior: opens the input image with PIL and reports dimensions.
   - MLX replacement point: first attempt only validates file existence and records image path; full image decode/preprocess belongs to a later slice unless small and isolated.

5. Pipeline run dispatch.
   - Reference: `vendors/trellis-mac/generate.py:119-127`; `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:488-595`.
   - Behavior: calls `pipeline.run(...)` with seed, pipeline type, and sampler parameter overrides.
   - MLX replacement point: stage dispatcher should advance until first unimplemented compute stage and return a structured blocker.

## Original Pipeline Stage Order

1. Pipeline type validation.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:517-534`.
   - Models checked: 512/1024 shape SLat flow, texture SLat flow, and cascade variants.
   - Initial MLX blocker risk: config-to-model mapping is not yet represented as MLX modules.

2. Image preprocessing/background removal.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127-162` and run call at `536-537`.
   - Behavior: handles RGBA alpha, resizes to max 1024, invokes rembg when needed, crops alpha bounding box, composites RGB over alpha.
   - Initial MLX blocker: background removal model (`rembg_model`) is not implemented in MLX.

3. Image conditioning.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:164-186` and run calls at `539-540`.
   - Behavior: image feature extractor generates conditioning at 512 and optionally 1024 resolution, plus negative conditioning.
   - Initial MLX blocker: DINOv3/image feature extractor is not implemented in MLX.

4. Sparse structure sampling.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:188-235` and run calls at `541-545`.
   - Behavior: creates dense random noise, runs sparse structure flow sampler, decodes sparse structure, pools if needed, and extracts active coordinates.
   - Initial MLX blocker: flow sampler and sparse structure decoder are not implemented in MLX.

5. Shape SLat sampling.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:237-275`, cascade path at `277-364`, and run branches at `546-573`.
   - Behavior: builds sparse latent noise from coordinates, runs shape SLat flow sampler, applies normalization, and may upsample/cascade.
   - Initial MLX blocker: SparseTensor flow sampler, SLat flow blocks, and cascade upsample are not implemented in MLX.

6. Texture SLat sampling.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:391-432` and run calls at `551-588`.
   - Behavior: normalizes shape SLat, concatenates condition, runs texture SLat flow sampler, then denormalizes texture SLat.
   - Initial MLX blocker: texture SLat flow sampler and concat-conditioned SparseTensor path are not implemented in MLX.

7. Shape latent decoding.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:366-389`.
   - Behavior: sets decoder resolution and returns meshes plus substructures.
   - Initial MLX blocker: shape SLat decoder is not implemented in MLX.

8. Texture latent decoding.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:434-453`.
   - Behavior: decodes texture voxels guided by shape substructures.
   - Initial MLX blocker: texture SLat decoder is not implemented in MLX.

9. Mesh-with-voxel assembly.
   - Reference: `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:455-486`.
   - Behavior: fills holes, combines mesh vertices/faces with texture voxels and PBR attribute layout.
   - Initial MLX blocker: mesh representation, hole filling, and voxel attribute assembly are not implemented in MLX.

10. GLB/OBJ export and texture baking.
    - Reference: `vendors/trellis-mac/generate.py:146-299`.
    - Behavior: validates non-empty mesh, optionally bakes PBR textures through Metal `o_voxel.postprocess.to_glb`, falls back to KDTree texture baker, or exports vertex-color GLB and OBJ.
    - Initial MLX blocker: mesh extraction/export and texture baking are not implemented in `mlx_spatial`.

## First Attempt Boundary

- The first MLX attempt should validate local assets and configured checkpoint probes, validate the image path, then stop at the first compute stage that is not implemented.
- The expected first blocker is image preprocessing/background removal or image conditioning, depending on whether preprocessing is treated as file-boundary validation or real image processing.
- A blocker is acceptable only if it names the stage, missing operation, reference location, reason, and recommended next slice.

## Initial Next-Slice Candidates

- Image preprocessing without learned rembg for RGBA inputs.
- DINOv3/image feature extraction strategy for MLX or optional external reference path.
- Sparse structure flow sampler and decoder mapping.
- Shape SLat flow block parity.
- Shape decoder implementation and mesh extraction path.
