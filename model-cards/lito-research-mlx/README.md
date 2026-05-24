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
  - research-only
  - non-commercial
base_model:
  - apple/ml-lito
---

# LiTo Research MLX for mlx-spatial

Run Apple's LiTo image-to-3D Gaussian Splat model on Apple Silicon through `mlx-spatial`, using MLX-ready safetensors instead of local `.ckpt` conversion.

This bundle is for researchers who want a practical Mac-native LiTo inference path: download the weights, point `mlx-spatial-lito` at them, and generate a 3D Gaussian Splat PLY from an input image. No CUDA is required.

## Quick Start: Image to 3DGS on Apple Silicon

Install `mlx-spatial`:

```bash
pip install mlx-spatial
```

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
  --memory-profile safe \
  --print-metrics
```

The output is a 3D Gaussian Splat PLY, not a mesh. Use a 3DGS-aware viewer such as KIRI Engine's 3DGS Render Blender add-on. Blender's native PLY importer can read the container but does not render LiTo Gaussian splat fields correctly.

## What This Model Bundle Provides

This Hugging Face repository contains the LiTo-specific safetensors expected by `mlx-spatial`:

```text
tokenizer/lito_new.safetensors
image_to_3d/lito_dit_rgba.safetensors
```

It also includes lightweight conversion metadata:

```text
tokenizer/conversion_metadata/lito_new.yaml
image_to_3d/conversion_metadata/lito_dit_rgba.yaml
```

End-to-end LiTo generation in `mlx-spatial` also needs the TRELLIS sparse-structure decoder weights from the separate TRELLIS.2 setup. Those TRELLIS.2 weights are not included in this LiTo bundle.

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

The files in this repository were converted from Apple's original `.ckpt` checkpoints to safetensors for local MLX loading. The conversion changes storage format and local layout only.

No training, fine-tuning, quantization, pruning, or tensor-value modification was applied.

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

This repository is not an Apple release and is not endorsed by Apple. Redistribution of this converted bundle must keep Apple's license terms, attribution notice, and modification disclosure.

Required attribution notice:

> Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.
