---
license: other
license_name: trellis2-dinov3-rmbg-multiple-licenses
license_link: https://huggingface.co/appautomaton/trellis2-mlx-8bit/blob/main/LICENSES.md
library_name: mlx
pipeline_tag: image-to-3d
base_model:
  - microsoft/TRELLIS.2-4B
  - facebook/dinov3-vitl16-pretrain-lvd1689m
  - briaai/RMBG-2.0
base_model_relation: quantized
language:
  - en
tags:
  - mlx
  - apple-silicon
  - safetensors
  - trellis2
  - o-voxel
  - image-to-3d
  - mesh-generation
  - pbr
  - glb
  - 8-bit
  - affine-quantization
  - mixed-precision
  - non-commercial
---

# TRELLIS.2 MLX 8-bit Affine for `mlx-spatial`

<p align="center">
  <a href="https://appautomaton.github.io"><img alt="App Automaton project" src="https://img.shields.io/badge/App_Automaton-Project-5B5BD6?style=for-the-badge"></a>
  <a href="https://github.com/appautomaton/mlx-spatial"><img alt="GitHub — appautomaton/mlx-spatial" src="https://img.shields.io/badge/GitHub-mlx--spatial-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white"></a>
  <a href="https://appautomaton.renocrypt.com/mlx-spatial/"><img alt="mlx-spatial documentation" src="https://img.shields.io/badge/Documentation-mlx--spatial-0A7BBB?style=for-the-badge&amp;logo=readthedocs&amp;logoColor=white"></a>
  <a href="https://pypi.org/project/mlx-spatial/"><img alt="mlx-spatial on PyPI" src="https://img.shields.io/pypi/v/mlx-spatial?style=for-the-badge&amp;logo=pypi&amp;logoColor=white&amp;label=PyPI"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/appautomaton/trellis2-mlx-8bit"><img alt="TRELLIS.2 MLX 8-bit" src="https://img.shields.io/badge/Hugging_Face-MLX_8--bit-FF9D00?style=for-the-badge&amp;logo=huggingface&amp;logoColor=000"></a>
</p>

A self-contained TRELLIS.2 image-to-3D bundle with selective 8-bit affine
weights for direct execution by
[`mlx-spatial`](https://github.com/appautomaton/mlx-spatial) on Apple Silicon.
The bundle includes the TRELLIS.2 checkpoints, DINOv3 image conditioner, and
RMBG-2.0 foreground extractor required by the local inference pipeline.

Large transformer and block-internal MLP matrices execute through MLX's packed
quantized matrix multiplication. Convolutions and accuracy-sensitive boundary
tensors remain in their source precision.

> This is an unofficial quantized derivative. It is not a Microsoft, Meta, or
> BRIA release. Because the bundle contains RMBG-2.0, it is restricted to
> non-commercial use unless the user obtains separate commercial rights from
> BRIA.

## Variants

| Variant | Precision | Complete bundle size | Model |
| --- | --- | ---: | --- |
| Source dependency set | BF16, FP16, and FP32 | 18.483 GB | TRELLIS.2-4B + DINOv3 ViT-L/16 + RMBG-2.0 |
| MLX 8-bit bundle | Selective affine INT8 with retained BF16, FP16, and FP32 | 11.248 GB | **This model** |

The complete 8-bit bundle is 39.1% smaller and does not require duplicate
full-precision checkpoints. Both forms expose the same 5,587 logical tensors
with identical names, shapes, and declared source dtypes.

## Compatibility

This format requires an `mlx-spatial` build that includes TRELLIS.2 affine
checkpoint support. Until that support is available in a tagged PyPI release,
install the current project revision:

```bash
pip install \
  "mlx-spatial @ git+https://github.com/appautomaton/mlx-spatial.git@main"
```

The runtime targets Apple Silicon, Python 3.13, and MLX `0.32.x`. It does not
use Torch, CUDA, or a dequantized full-precision checkpoint.

## Use

Download the complete bundle:

```bash
hf download appautomaton/trellis2-mlx-8bit \
  --local-dir weights/trellis2-mlx-8bit
```

Validate its three runtime roots:

```bash
mlx-spatial-trellis2 validate \
  --root weights/trellis2-mlx-8bit

mlx-spatial-trellis2 dinov3-validate \
  --root weights/trellis2-mlx-8bit/dinov3

mlx-spatial-trellis2 rmbg-validate \
  --root weights/trellis2-mlx-8bit/rmbg
```

Generate a textured GLB:

```bash
mlx-spatial-trellis2 generate-textured \
  weights/trellis2-mlx-8bit \
  inputs/trellis2/object.png \
  --dino-root weights/trellis2-mlx-8bit/dinov3 \
  --rmbg-root weights/trellis2-mlx-8bit/rmbg \
  --output outputs/trellis2/object-8bit/model.glb \
  --pipeline-type 1024_cascade \
  --seed 42
```

`1024_cascade` is the recommended quality tier; use `512` when lower memory
use or faster iteration matters more.

Do not pass `--slat-steps` for a quality run; the model configuration uses 12
steps. `--slat-steps 1` is intended only for a quick runtime smoke test.

RGBA inputs use their alpha channel directly. RGB inputs are passed through
the bundled RMBG-2.0 model. A clean single-object foreground and an uncropped
silhouette generally produce the most useful reconstruction.

## Bundle

| Component | Logical tensors | INT8 matrices | Safetensors bytes |
| --- | ---: | ---: | ---: |
| TRELLIS.2 flow, VAE, and decoder checkpoints | 4,418 | 1,306 | 10,560,255,296 |
| `dinov3/model.safetensors` | 415 | 144 | 344,386,453 |
| `rmbg/model.safetensors` | 754 | 96 | 343,122,500 |
| **Total** | **5,587** | **1,546** | **11,247,764,249** |

Configuration, model card, license, and RMBG support files account for the
small difference between checkpoint bytes and the complete directory size.

The repository preserves the TRELLIS.2 checkpoint names and layout used by the
source model. DINOv3 and RMBG-2.0 are bundled under `dinov3/` and `rmbg/` so a
single download contains every runtime weight.

## Quantization

The quantization scheme is affine 8-bit with group size 64. Packed weights are
stored as `uint32` with FP32 scales and biases, then executed directly through
`mx.quantized_matmul`.

The following block-internal two-dimensional weights are quantized:

- self-attention QKV and output projections in all five TRELLIS.2 flow DiTs;
- cross-attention query, key/value, and output projections in those DiTs;
- DiT MLP input and output matrices;
- internal ConvNeXt MLP matrices in the shape and texture encoders and
  decoders;
- DINOv3 attention Q/K/V/output and MLP up/down projections;
- RMBG-2.0 Swin attention QKV/output and MLP FC matrices.

The following tensors remain in their source precision:

- all dense and sparse convolutions, including the sparse-structure decoder;
- input, output, patch, latent, subdivision, and final prediction boundaries;
- timestep and condition projections, embeddings, learned tokens, and
  positional tensors;
- normalization parameters, RMSNorm parameters, modulation tensors, and
  biases;
- RMBG stage-transition reductions;
- matrices whose input dimension is incompatible with group size 64.

Physical packed arrays use internal qweight, scale, and bias suffixes. The
runtime reconstructs the original logical names from safetensors metadata, so
the existing TRELLIS.2 pipeline and configuration continue to use the same
checkpoint contract. Format details are embedded under
`mlx_spatial.trellis2.quantization`.

## Reproducing the Bundle

Starting from the three source weight roots:

```bash
mlx-spatial-trellis2-quantize \
  weights/trellis2 \
  weights/trellis2-mlx-8bit \
  --dinov3-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --rmbg-root weights/rmbg2 \
  --bits 8 \
  --group-size 64
```

The command creates a complete runtime root. Do not add full-precision copies
of the quantized checkpoints to the 8-bit repository.

## Verification

- Logical checkpoint inspection matched all 5,587 source tensor names,
  shapes, and dtypes exactly.
- The quantization inventory contains 1,546 packed matrices: 1,050 DiT
  matrices, 256 encoder/decoder MLP matrices, 144 DINOv3 matrices, and 96
  RMBG-2.0 matrices.
- The complete `mlx-spatial` repository suite passed 1,175 tests; 10 tests
  were skipped, 50 were deselected, and 3 expected failures remained.
- A 512-resolution, 12-step run completed background handling, DINOv3
  conditioning, sparse-structure sampling, shape and texture SLat sampling,
  both decoders, and textured GLB export with no blocker.
- That run produced 4,022 shape tokens, 1,041,541 decoded shape/texture points,
  and a 12,083,344-byte GLB. The rendered asset retained coherent geometry and
  recognizable appearance.
- The same run completed in 152.23 seconds, observed 3.516 GB peak MLX
  allocator use, and recorded zero swap growth.
- A separate `1024_cascade`, 12-step run completed the same end-to-end path
  and produced a Blender-readable 12,670,448-byte GLB with 199,884 faces and
  embedded 1024 x 1024 PBR textures.

The 512 runtime and memory figures are one local Apple Silicon observation,
not a general benchmark. The `1024_cascade` run overlapped another MLX workload,
so it establishes compatibility and artifact health rather than performance.
Neither run is a formal claim of visual equivalence to the source weights.

## Limitations

- Quantization changes the sampling trajectory. Geometry, pose, topology,
  texture placement, material values, and unseen surfaces can differ from the
  source-precision model even with the same seed.
- End-to-end validation covers `512` and `1024_cascade`. The standalone `1024`
  and `1536_cascade` routes have not received an equivalent quality evaluation.
- Single-view reconstruction cannot determine unseen geometry with certainty.
- Fine detail depends on foreground extraction, cropping, occlusion,
  reflections, transparency, and thin structures.
- Generated meshes can contain small holes, non-manifold regions, or other
  topology requiring downstream cleanup.
- The packed checkpoint format requires `mlx-spatial`; generic safetensors
  readers expose the physical packed arrays rather than the logical matrices.
- This bundle supports inference, not training or fine-tuning.
- Commercial use of the bundled RMBG-2.0 weights requires separate rights
  from BRIA.

## License and Attribution

This is a combined derivative; no single permissive license covers every
included component. Read [`LICENSES.md`](LICENSES.md) and all referenced terms
before downloading, using, or redistributing the bundle.

- Microsoft TRELLIS.2 files are provided under the MIT License. A copy is
  included as [`LICENSE_TRELLIS2`](LICENSE_TRELLIS2).
- Meta DINOv3 files and derivatives are governed by the DINOv3 License. The
  complete agreement is included as
  [`LICENSE_DINOV3.md`](LICENSE_DINOV3.md), as required for redistribution.
- BRIA RMBG-2.0 is provided for non-commercial use under CC BY-NC 4.0. Its
  attribution and change notice are included in
  [`LICENSE_RMBG2.md`](LICENSE_RMBG2.md). Commercial use requires separate
  authorization from BRIA.

The combined bundle must therefore be treated as non-commercial unless the
user has obtained all additional rights that their use requires. This
repository is not affiliated with or endorsed by Microsoft, Meta, BRIA, or the
original authors.

## Links

- [App Automaton](https://appautomaton.github.io)
- [AppAutomaton models on Hugging Face](https://huggingface.co/appautomaton)
- [`appautomaton/mlx-spatial`](https://github.com/appautomaton/mlx-spatial) — MLX-native 3D and spatial inference for Apple Silicon.
- [`mlx-spatial` documentation](https://appautomaton.renocrypt.com/mlx-spatial/)
- [`mlx-spatial` on PyPI](https://pypi.org/project/mlx-spatial/)
- [TRELLIS.2 guide](https://github.com/appautomaton/mlx-spatial/blob/main/docs/trellis2.md)
- [TRELLIS.2 MLX 8-bit model](https://huggingface.co/appautomaton/trellis2-mlx-8bit)
- [Microsoft TRELLIS.2-4B model](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [Microsoft TRELLIS.2 source](https://github.com/microsoft/TRELLIS.2)
- [TRELLIS.2 project page](https://microsoft.github.io/TRELLIS.2/)
- [Meta DINOv3 ViT-L/16 model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [Meta DINOv3 source](https://github.com/facebookresearch/dinov3)
- [BRIA RMBG-2.0 model](https://huggingface.co/briaai/RMBG-2.0)
- [BRIA RMBG-2.0 source](https://github.com/Bria-AI/RMBG-2.0)
- [MLX](https://github.com/ml-explore/mlx)
