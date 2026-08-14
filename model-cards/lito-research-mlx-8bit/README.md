---
license: other
license_name: apple-machine-learning-research-model-license-agreement
license_link: https://github.com/apple/ml-lito/blob/main/LICENSE_MODEL
library_name: mlx
pipeline_tag: image-to-3d
base_model:
  - appautomaton/lito-research-mlx
tags:
  - mlx
  - apple-silicon
  - safetensors
  - image-to-3d
  - gaussian-splatting
  - 3dgs
  - 8-bit
  - affine-quantization
  - runtime-only
  - research-only
  - non-commercial
---

# LiTo Runtime 8-bit Affine for `mlx-spatial`

A self-contained LiTo inference bundle with selective affine INT8 weights for
Apple Silicon. Accuracy-sensitive boundaries remain in FP32. This is an
unofficial derivative for non-commercial research use, not an Apple release.

## Use

```bash
pip install \
  "mlx-spatial @ git+https://github.com/appautomaton/mlx-spatial.git@afa6b6512567dc9964294a1cec8601e1b505802e"

hf download appautomaton/lito-research-mlx-8bit \
  --local-dir weights/lito-research-mlx-8bit

mlx-spatial-lito validate weights/lito-research-mlx-8bit

mlx-spatial-lito generate inputs/lito/object-rgba.png \
  --weights-root weights/lito-research-mlx-8bit \
  --output outputs/lito/object-8bit.ply \
  --format ply \
  --num-steps 20 \
  --cfg-scale 3.0 \
  --print-metrics
```

The output is a 3D Gaussian Splat PLY, not a triangle mesh. A clean RGBA
foreground matte is strongly recommended.

## Bundle

| File | Logical tensors | Quantized matrices | Bytes |
| --- | ---: | ---: | ---: |
| `image_to_3d/lito_dit_rgba.safetensors` | 1,016 | 224 | 2,004,347,727 |
| `tokenizer/lito_new.safetensors` | 467 | 124 | 168,537,103 |
| `dependencies/trellis/ckpts/ss_dec_conv3d_16l8_fp16.safetensors` | 74 | 0 | 147,591,972 |
| **Runtime weights** | **1,557** | **348** | **2,320,476,802** |

The bundle also includes the decoder config, Microsoft MIT license, and an
immutable source manifest under `dependencies/trellis/`.

The LiTo quantization scheme is affine 8-bit with group size 64. Packed weights
are stored as `uint32` with FP32 scales and biases and execute directly through
`mx.quantized_matmul`. Internal attention and MLP matrices in the EMA DiT,
Gaussian decoder, and voxel decoder are quantized. The image conditioner,
convolutions, embeddings, normalizations, boundary projections, and output
heads remain FP32.

## Embedded TRELLIS Decoder

LiTo uses the TRELLIS sparse-structure decoder to convert voxel latents into
Gaussian initialization coordinates. This bundle embeds the exact checkpoint
from `microsoft/TRELLIS-image-large` revision
`25e0d31ffbebe4b5a97464dd851910efc3002d96`:

```text
dependencies/trellis/LICENSE
dependencies/trellis/SOURCE.json
dependencies/trellis/ckpts/ss_dec_conv3d_16l8_fp16.json
dependencies/trellis/ckpts/ss_dec_conv3d_16l8_fp16.safetensors
```

The decoder safetensors SHA-256 is
`1c76d4a40519aa2d711cc263a8404105231ac26db31d946bed48b84fee79009a`.
The runtime does not search a separate TRELLIS checkout.

## Verification

- Both checkpoints pass `mlx-spatial-lito validate`.
- Architecture inspection recovers 28 DiT blocks, 6 Gaussian Perceiver blocks,
  and 4 voxel decoder blocks.
- Real-weight Linear probes measured `0.51%–0.61%` relative RMSE and cosine
  similarity above `0.99998` against FP32.
- An uncapped 20-step run produced 557,568 finite Gaussians in 2 minutes
  39.76 seconds, with 11.60 GiB peak active MLX memory.
- A separate bundle-local 20-step validation produced 8,192 finite Gaussians
  after explicitly capping occupied cells for packaging verification.

These figures are one local Apple Silicon observation, not a general benchmark
or an official quality-equivalence claim.

## Limitations and Licenses

- Inference only; training and mesh-specific modules are intentionally absent.
- Quantization can change generation details relative to FP32.
- Single-view reconstruction cannot determine unseen geometry with certainty.
- Commercial use is not permitted by Apple's model license.

LiTo weights are covered by the bundled `LICENSE_MODEL`. The embedded TRELLIS
decoder is covered by the MIT License under `dependencies/trellis/LICENSE`.

> Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

## Links

- [FP32 runtime variant](https://huggingface.co/appautomaton/lito-research-mlx)
- [`appautomaton/mlx-spatial`](https://github.com/appautomaton/mlx-spatial)
- [Apple LiTo project](https://apple.github.io/ml-lito/)
- [Apple LiTo source](https://github.com/apple/ml-lito)
