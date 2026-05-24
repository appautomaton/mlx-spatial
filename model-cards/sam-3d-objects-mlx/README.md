---
library_name: mlx
pipeline_tag: image-to-3d
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE
tags:
  - mlx
  - apple-silicon
  - sam-3d
  - sam-3d-objects
  - image-to-3d
  - 3d-reconstruction
  - gaussian-splatting
  - mesh
  - glb
  - safetensors
base_model:
  - facebook/sam-3d-objects
---

# SAM 3D Objects MLX for mlx-spatial

Run SAM 3D Objects on Apple Silicon through `mlx-spatial`, using an MLX-ready safetensors bundle instead of local PyTorch checkpoint conversion.

This bundle is for users who want masked single-image object reconstruction on a Mac: download the model, provide an image plus object mask, and generate SAM3D Gaussian or mesh artifacts with `mlx-spatial-sam3d`. No CUDA is required.

## Quick Start: Masked Image to 3D Object

Install `mlx-spatial`:

```bash
pip install mlx-spatial
```

Download this model bundle:

```bash
hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
```

Validate the local layout:

```bash
mlx-spatial-sam3d validate weights/sam-3d-objects-mlx
mlx-spatial-sam3d inspect weights/sam-3d-objects-mlx
```

Generate a Gaussian-splat PLY:

```bash
mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx image.png \
  --mask mask.png \
  --output outputs/sam3d/object/gaussians.ply \
  --trace-output outputs/sam3d/object/trace.json
```

Generate a GLB mesh as well:

```bash
mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx image.png \
  --mask mask.png \
  --output outputs/sam3d/object/gaussians.ply \
  --glb-output outputs/sam3d/object/object.glb \
  --trace-output outputs/sam3d/object/trace.json
```

The trace records quality diagnostics such as sparse-structure occupancy, geometry range, opacity, selected mask, and output paths.

## What This Model Bundle Provides

This Hugging Face repository contains the converted SAM 3D Objects checkpoint bundle expected by `mlx-spatial`:

```text
checkpoints/pipeline.yaml
checkpoints/ss_generator.safetensors
checkpoints/slat_generator.safetensors
checkpoints/ss_decoder.safetensors
checkpoints/slat_decoder_gs.safetensors
checkpoints/slat_decoder_gs_4.safetensors
checkpoints/slat_decoder_mesh.safetensors
checkpoints/conversion_metadata/
conversion_manifest.json
weight-audit-source-vs-mlx.json
```

It also includes the converted MoGe ViT-L pointmap dependency used by the default SAM3D preprocessing path:

```text
moge/model.safetensors
moge/conversion_metadata/model.yaml
```

The bundled MoGe checkpoint lets the normal `mlx-spatial-sam3d reconstruct` command run from one model repository. Advanced users can still pass a different MoGe root or provide an external pointmap.

## Best For

- Apple Silicon MLX inference experiments.
- Masked single-image object reconstruction.
- SAM3D Gaussian Splat PLY generation with `mlx-spatial`.
- SAM3D mesh or GLB export workflows.
- Researchers and developers who need SAM 3D Objects weights in safetensors format.

## Current Limitations

- This is an unofficial converted derivative bundle, not an official Meta or MoGe release.
- The upstream `facebook/sam-3d-objects` Hugging Face repository is gated. Users should have access to the upstream model and accept the upstream terms before using this conversion.
- Reconstruction requires an input image and a useful binary object mask.
- Standard 3D Gaussian viewers may use different coordinate conventions than SAM 3D Objects' native output convention.
- This is not an int8, 4-bit, or otherwise quantized model.
- CUDA is not required and is not used by `mlx-spatial` SAM3D inference.

## Conversion Fidelity

The converted checkpoint bundle was audited against the original SAM 3D Objects checkpoint files.

| Role | Tensors | Missing | Extra | Shape mismatches | Nonzero numeric diffs | Max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ss_generator` | 1,741 | 0 | 0 | 0 | 0 | 0.0 |
| `slat_generator` | 1,225 | 0 | 0 | 0 | 0 | 0.0 |
| `ss_decoder` | 74 | 0 | 0 | 0 | 0 | 0.0 |
| `slat_decoder_gs` | 101 | 0 | 0 | 0 | 0 | 0.0 |
| `slat_decoder_mesh` | 120 | 0 | 0 | 0 | 0 | 0.0 |
| `slat_decoder_gs_4` | 101 | 0 | 0 | 0 | 0 | 0.0 |

Total compared SAM3D tensors: 3,362.

Some decoder tensors are stored as `float32` in this safetensors bundle even when the source checkpoint tensor was `float16`. This is lossless for value preservation. The numeric audit compares values after `float32` materialization and found zero difference.

See `weight-audit-source-vs-mlx.json` for the audit summary.

## Conversion Details

This bundle was produced from the original SAM 3D Objects checkpoint layout with:

```bash
mlx-spatial-sam3d convert weights/sam-3d-objects \
  --output-root weights/sam-3d-objects-mlx \
  --moge-root weights/moge-vitl \
  --moge-output-root weights/sam-3d-objects-mlx/moge \
  --max-archive-gb 16
```

The conversion rewrites checkpoint references in `pipeline.yaml` from PyTorch checkpoint files to `.safetensors` files. It does not quantize the model or change the architecture.

## Project Links

- Runtime package: `mlx-spatial`
- `mlx-spatial` PyPI package: https://pypi.org/project/mlx-spatial/
- `mlx-spatial` source: https://github.com/appautomaton/mlx-spatial
- This model repo: https://huggingface.co/appautomaton/sam-3d-objects-mlx

## Upstream Source and License

This bundle is based on Meta's SAM 3D Objects release and includes a converted MoGe dependency:

- Upstream SAM 3D Objects model: https://huggingface.co/facebook/sam-3d-objects
- Upstream SAM 3D Objects code: https://github.com/facebookresearch/sam-3d-objects
- SAM License: https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE
- MoGe dependency: included as converted `moge/model.safetensors`

The original SAM 3D Objects checkpoints and code are licensed by Meta under the SAM License. The included MoGe dependency follows its own upstream license and terms.

This repository is not an official Meta or MoGe release. Users are responsible for complying with the upstream SAM 3D Objects and MoGe license, access, and use requirements.

If you use this conversion, cite the original SAM 3D Objects work and link to the upstream model and code.
