---
license: other
license_name: apple-machine-learning-research-model-license-agreement
license_link: https://github.com/apple/ml-lito/blob/main/LICENSE_MODEL
library_name: mlx
pipeline_tag: image-to-3d
tags:
  - mlx
  - safetensors
  - image-to-3d
  - gaussian-splatting
  - 3dgs
  - fp32
  - runtime-only
  - research-only
  - non-commercial
base_model:
  - apple/ml-lito
---

# LiTo Runtime FP32 for `mlx-spatial`

Run Apple's LiTo image-to-3D Gaussian Splat model on Apple Silicon through `mlx-spatial`, using MLX-ready safetensors instead of local `.ckpt` conversion.

This runtime-pruned bundle is for researchers who want a practical Mac-native
LiTo inference path: download one repository and generate a 3D Gaussian Splat
PLY from an input image. No CUDA is required.

## Quick Start: Image to 3DGS on Apple Silicon

Install `mlx-spatial`:

```bash
pip install \
  "mlx-spatial @ git+https://github.com/appautomaton/mlx-spatial.git@afa6b6512567dc9964294a1cec8601e1b505802e"
```

This model card requires the bundle-local LiTo dependency layout introduced by
the immutable runtime commit above.

Download this model bundle:

```bash
hf download appautomaton/lito-research-mlx \
  --local-dir weights/lito-research-mlx
```

Validate the local layout:

```bash
mlx-spatial-lito validate weights/lito-research-mlx
mlx-spatial-lito inspect weights/lito-research-mlx --limit 10
```

Generate a Gaussian-splat PLY:

```bash
mlx-spatial-lito generate inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --memory-profile balanced \
  --print-metrics
```

The output is a 3D Gaussian Splat PLY, not a mesh. Use a 3DGS-aware viewer such as KIRI Engine's 3DGS Render Blender add-on. Blender's native PLY importer can read the container but does not render LiTo Gaussian splat fields correctly.

## What This Model Bundle Provides

This Hugging Face repository is a self-contained LiTo runtime bundle for
`mlx-spatial`:

```text
tokenizer/lito_new.safetensors
image_to_3d/lito_dit_rgba.safetensors
dependencies/trellis/LICENSE
dependencies/trellis/SOURCE.json
dependencies/trellis/ckpts/ss_dec_conv3d_16l8_fp16.json
dependencies/trellis/ckpts/ss_dec_conv3d_16l8_fp16.safetensors
```

The embedded decoder is the exact 147,591,972-byte checkpoint from
`microsoft/TRELLIS-image-large` revision
`25e0d31ffbebe4b5a97464dd851910efc3002d96`, with SHA-256
`1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a`.
It converts LiTo voxel latents into Gaussian initialization coordinates.
`mlx-spatial-lito validate` requires this bundle-local decoder and does not
search a separate TRELLIS checkout.

| File | Logical tensors | Bytes |
| --- | ---: | ---: |
| `image_to_3d/lito_dit_rgba.safetensors` | 1,016 | 3,713,563,945 |
| `tokenizer/lito_new.safetensors` | 467 | 521,201,761 |
| `dependencies/trellis/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 74 | 147,591,972 |
| **Runtime weights** | **1,557** | **4,382,357,678** |

## Best For

- Apple Silicon MLX inference experiments.
- Image-to-3D Gaussian Splat generation with `mlx-spatial`.
- Research workflows that need LiTo weights in safetensors format.
- Local 3DGS inspection in KIRI, Gaussian-splat-aware Blender add-ons, or compatible 3DGS viewers.

## Current Limitations

- Research-only, non-commercial license boundary from Apple.
- This is an unofficial converted derivative bundle, not an Apple-hosted official MLX package.
- Current `mlx-spatial` LiTo support targets image-to-3D Gaussian Splat inference; it does not provide LiTo training, fine-tuning, mesh extraction, multi-image conditioning, or video conditioning.
- Visual quality depends strongly on input matting and alpha quality. Inputs with broad or noisy alpha masks can produce weaker holes, handles, and fine structures.
- CUDA is not required and is not used by `mlx-spatial` LiTo inference.

## Conversion Details

The LiTo files were converted from Apple's original `.ckpt` checkpoints to
safetensors and pruned to the modules read by the main inference path. The
bundle retains the EMA velocity estimator, DINO/RGBA image conditioner,
Gaussian decoder, and voxel decoder. It removes the non-EMA training copy,
duplicate tokenizer, mesh/fpoint/LPIPS modules, and training-only decoders.
Retained tensor values, shapes, and dtypes are unchanged; no retained LiTo
tensor is quantized. The embedded TRELLIS decoder is redistributed unchanged
from the immutable Microsoft source revision recorded above.

## Verification

- Both checkpoints and the embedded decoder pass `mlx-spatial-lito validate`.
- Architecture inspection recovers 28 DiT blocks, 6 Gaussian Perceiver blocks,
  and 4 voxel decoder blocks.
- The bundle exposes 1,483 LiTo logical inference tensors.
- A bundle-local 20-step runtime validation produced 8,192 finite Gaussians
  after explicitly capping occupied cells for packaging verification. This is
  not an uncapped quality benchmark.

## Project Links

- Runtime package: `mlx-spatial`
- `mlx-spatial` PyPI package: https://pypi.org/project/mlx-spatial/
- `mlx-spatial` source: https://github.com/appautomaton/mlx-spatial
- This model repo: https://huggingface.co/appautomaton/lito-research-mlx

## Apple LiTo Source and License

This bundle is based on Apple's LiTo research release:

- Apple LiTo project: https://apple.github.io/ml-lito/
- Apple LiTo source code: https://github.com/apple/ml-lito
- Apple model license: https://github.com/apple/ml-lito/blob/main/LICENSE_MODEL

Apple's LiTo model weights are released under the Apple Machine Learning Research Model License Agreement. Use is limited to non-commercial scientific research and academic development activities. Commercial product use is not permitted.

License and source access last checked: 2026-08-06.

This repository is not an Apple release and is not endorsed by Apple. Redistribution of this converted bundle must keep Apple's license terms, attribution notice, and modification disclosure.

Required attribution notice:

> Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

The embedded TRELLIS dependency is licensed under the MIT License. Its license
copy and immutable source metadata are included under `dependencies/trellis/`.
