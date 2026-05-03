# SPEC: TRELLIS.2 Image Preprocessing Background

## Bounded Goal

Implement the TRELLIS.2 `image-preprocessing-background` stage in `mlx_spatial` so real image inputs are decoded, normalized, alpha-cropped/composited, and, when local gated RMBG weights are present, processed through an MLX-native BiRefNet background-removal path before the pipeline advances to `image-conditioning` or a precise lower-level blocker.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Keep this as part of the existing `mlx-spatial` Python package and `mlx_spatial.trellis2` / `mlx_spatial.trellis2_inference` surfaces; do not create a separate `mlx-spatial-trellis2` package.
- Runtime implementation must not import from `vendors/`, PyTorch, TorchVision, Transformers, or Hugging Face Hub.
- Default tests must not require real TRELLIS.2 weights, RMBG weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, or vendor imports.
- `briaai/RMBG-2.0` is gated and non-commercial; acquisition must be explicit and local, and no default command should silently download it.
- If image I/O requires a new dependency such as Pillow, make it a deliberate project dependency change and keep tests deterministic with tiny generated fixtures.
- MLX-native RMBG support must load local safetensors and use explicit weight/key mapping; do not add a PyTorch or ONNX fallback as the production path in this change.
- If BiRefNet cannot be fully ported in this slice because of an unsupported MLX operation or unresolved architecture/key mapping, the pipeline must return a precise blocker naming the module/op/key and next slice.
- Full TRELLIS.2 inference parity is still staged; this change only owns the image preprocessing/background stage.

## Required Behavior

- Add a real image decode/preprocess boundary for TRELLIS.2 attempt mode that accepts a local image path and loads an actual image, not placeholder text.
- Implement deterministic TRELLIS.2 preprocessing behavior from the reference pipeline:
  - detect useful RGBA alpha when present;
  - resize images so the maximum side is no larger than 1024 while preserving aspect ratio;
  - crop around the alpha foreground bounding box using the reference threshold semantics;
  - composite RGB over alpha and return an RGB image suitable for the next stage.
- For images that already contain non-opaque alpha, complete preprocessing without requiring RMBG weights.
- For RGB or fully opaque images, route through a local MLX RMBG/BiRefNet path when explicitly configured weights are present.
- Add RMBG asset tooling that can validate and describe the expected local `briaai/RMBG-2.0` safetensors/config/code assets without downloading them in default tests.
- Add an explicit manual download/help surface for RMBG assets analogous to the TRELLIS.2 asset tooling, with license/gated-access wording.
- Add an MLX BiRefNet port attempt that:
  - mirrors the reference preprocessing normalization for RMBG input;
  - loads local RMBG safetensors into MLX arrays;
  - maps checkpoint keys to MLX module parameters;
  - produces an alpha matte for compatible checkpoints, or returns a structured blocker for the first unsupported architecture/key/op boundary.
- Update `Trellis2InferencePipeline.attempt(...)` so `image-preprocessing-background` is no longer a blanket unimplemented blocker.
- Preserve the existing blocker model shape: `stage`, `operation`, `reference`, `reason`, and `next_slice`.
- The real local attempt against `weights/trellis2/` must advance to `image-conditioning` for an RGBA-alpha input, and must either run RMBG or report a precise RMBG-specific blocker for an RGB input.

## Acceptance Criteria

- A public image preprocessing API exists under `mlx_spatial` and is used by TRELLIS.2 attempt mode.
- Tiny generated RGBA fixtures verify decode, max-side resize, foreground alpha bounding-box crop, RGB-over-alpha composite, and output mode/shape without real model weights.
- Tiny generated RGB fixtures verify the conditional RMBG path selection and deterministic blocker behavior when RMBG assets are absent.
- RMBG asset validation reports present/missing local assets deterministically and does not require network access.
- A manual RMBG download/help command or documented command surface exists and makes gated/non-commercial status explicit.
- If local RMBG safetensors are present, the implementation can inspect/load selected tensors as MLX arrays with tests using fake safetensors fixtures.
- If the BiRefNet MLX port reaches runnable parity, RGB input preprocessing completes and returns a TRELLIS-ready RGB image.
- If the BiRefNet MLX port is incomplete, the attempt returns a blocker at `image-preprocessing-background` with an operation such as `MLX BiRefNet <module/op/key>` rather than the previous generic preprocessing blocker.
- For an RGBA-alpha input with valid TRELLIS.2 assets, the real attempt completes `image-preprocessing-background` and stops next at `image-conditioning`.
- Default `uv run pytest` passes without real weights, network access, Hugging Face credentials, PyTorch, TorchVision, Transformers, or vendor imports.
- No real RMBG weights, TRELLIS.2 weights, large generated images, or generated model outputs are committed.
- `README.md` or the TRELLIS.2 CLI help documents the preprocessing behavior, RMBG local-asset requirement, and remaining `image-conditioning` boundary.

## Blocking Questions Or Assumptions

- Assumption: the combined spec is coherent because deterministic preprocessing and RMBG are both part of the single observable TRELLIS.2 `image-preprocessing-background` stage.
- Assumption: `weights/trellis2` remains the TRELLIS.2 asset root; RMBG assets may live in a separate ignored local path unless implementation finds a cleaner local convention.
- Assumption: downloading `briaai/RMBG-2.0` requires explicit user action or credentials and is not part of default verification.
- Assumption: an incomplete MLX BiRefNet port is acceptable only if the blocker is concrete enough to plan the next implementation slice.
- Assumption: image-conditioning remains out of scope and should become the next blocker after successful preprocessing.

## Anti-Goals

- Do not implement DINOv3/image conditioning.
- Do not implement sparse structure sampling, SLat sampling, decoders, mesh extraction, texture baking, or GLB/OBJ export.
- Do not add training paths, trainers, optimizer behavior, or differentiable-rendering parity.
- Do not import runtime code from `vendors/`.
- Do not add PyTorch, TorchVision, Transformers, Hugging Face Hub, or ONNX Runtime as required production dependencies for this path.
- Do not silently download gated RMBG assets during tests, package import, CLI validation, or attempt mode.
- Do not fake background removal with placeholder alpha masks when the input does not already contain alpha.
- Do not claim full TRELLIS.2 image-to-3D inference works because this stage advances.
