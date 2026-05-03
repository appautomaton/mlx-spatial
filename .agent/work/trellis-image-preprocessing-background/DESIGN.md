# DESIGN: TRELLIS.2 Image Preprocessing Background

## Pipeline Boundary

- Keep the stage name `image-preprocessing-background`.
- Add a concrete preprocessing result object that carries the input path, output image metadata, whether alpha was supplied or generated, and an optional blocker.
- `Trellis2InferencePipeline.attempt(...)` should call the preprocessor after asset/probe readiness and before `image-conditioning`.
- When preprocessing succeeds, the next blocker should be `image-conditioning`.
- When preprocessing fails, the blocker remains at `image-preprocessing-background` but must name the concrete missing boundary.

## Deterministic Preprocessing

- Use Pillow for image decode and image fixture generation if the implementation confirms the dependency is needed.
- Mirror the reference algorithm from `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127-162`:
  - useful alpha means RGBA with at least one alpha value below 255;
  - max side is clamped to 1024 with Lanczos resize;
  - foreground bbox uses alpha values greater than `0.8 * 255`;
  - crop to a square bbox around the alpha foreground;
  - output is RGB multiplied by alpha.
- Empty alpha foreground must return a structured blocker, not crash on bbox reduction.

## RMBG Asset Boundary

- Keep TRELLIS.2 assets under `weights/trellis2`.
- Add a separate RMBG asset root convention, preferably `weights/rmbg2`, unless implementation finds an existing local convention.
- Asset validation should check local files only. Expected files are at least `model.safetensors`, `config.json`, `BiRefNet_config.py`, and `birefnet.py` if the implementation needs architecture metadata.
- Add a manual command/help surface for `briaai/RMBG-2.0`; do not call Hugging Face from package import, validation, tests, or attempt mode.

## MLX BiRefNet Boundary

- Start with safetensors inspection/loading helpers and fake checkpoint tests.
- Port the smallest architecture surface required to instantiate and map the real checkpoint.
- If a full forward pass is blocked, return a `Trellis2InferenceBlocker` with:
  - `stage`: `image-preprocessing-background`
  - `operation`: `MLX BiRefNet <specific module/op/key>`
  - `reference`: local source or vendor/reference file
  - `reason`: concrete mismatch or unsupported operation
  - `next_slice`: the next implementation target
- Do not introduce PyTorch, TorchVision, Transformers, ONNX Runtime, or vendor imports as fallback execution paths.

## Verification Strategy

- Unit tests use tiny generated RGBA/RGB images and fake safetensors.
- Default tests must not require real `weights/trellis2` or `weights/rmbg2`.
- Real local verification should include:
  - an RGBA alpha attempt against `weights/trellis2` proving the stage advances to `image-conditioning`;
  - an RGB attempt proving either RMBG execution or a precise RMBG blocker.
