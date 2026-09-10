---
license: other
license_name: sam-license
license_link: https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE
library_name: mlx
pipeline_tag: image-to-3d
base_model:
  - appautomaton/sam-3d-objects-mlx
base_model_relation: quantized
tags:
  - mlx
  - apple-silicon
  - safetensors
  - sam-3d
  - sam-3d-objects
  - image-to-3d
  - 3d-reconstruction
  - gaussian-splatting
  - mesh
  - glb
  - 8-bit
  - affine-quantization
  - mixed-precision
---

# SAM 3D Objects MLX 8-bit Affine for `mlx-spatial`

<p align="center">
  <a href="https://appautomaton.com"><img alt="App Automaton project" src="https://img.shields.io/badge/App_Automaton-Project-5B5BD6?style=for-the-badge"></a>
  <a href="https://github.com/appautomaton/mlx-spatial"><img alt="GitHub — appautomaton/mlx-spatial" src="https://img.shields.io/badge/GitHub-mlx--spatial-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white"></a>
  <a href="https://appautomaton.com/mlx-spatial/"><img alt="mlx-spatial documentation" src="https://img.shields.io/badge/Documentation-mlx--spatial-0A7BBB?style=for-the-badge&amp;logo=readthedocs&amp;logoColor=white"></a>
  <a href="https://pypi.org/project/mlx-spatial/"><img alt="mlx-spatial on PyPI" src="https://img.shields.io/pypi/v/mlx-spatial?style=for-the-badge&amp;logo=pypi&amp;logoColor=white&amp;label=PyPI"></a>
</p>

<p align="center">
  <a href="https://huggingface.co/appautomaton/sam-3d-objects-mlx"><img alt="SAM 3D Objects MLX full-precision variant" src="https://img.shields.io/badge/Hugging_Face-MLX_FP-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=000"></a>
  <a href="https://huggingface.co/appautomaton/sam-3d-objects-mlx-8bit"><img alt="SAM 3D Objects MLX 8-bit variant" src="https://img.shields.io/badge/Hugging_Face-MLX_8--bit-FF9D00?style=for-the-badge&amp;logo=huggingface&amp;logoColor=000"></a>
</p>

A self-contained SAM 3D Objects inference bundle with selective 8-bit affine
weights, built for direct execution by
[`mlx-spatial`](https://github.com/appautomaton/mlx-spatial) on Apple Silicon.
Large transformer matrices run through MLX's packed quantized matrix
multiplication, while convolutional and accuracy-sensitive boundary tensors
remain in their original precision.

> This is an unofficial quantized derivative. It is not a Meta or MoGe
> release.

## Variants

| Variant | Precision | Checkpoint size | Model |
| --- | --- | ---: | --- |
| MLX converted | Source FP16/FP32 | 13.705 GB | [`sam-3d-objects-mlx`](https://huggingface.co/appautomaton/sam-3d-objects-mlx) |
| MLX 8-bit | Selective affine INT8 with retained FP16/FP32 | 4.625 GB | **This model** |

The 8-bit checkpoint payload is 66.3% smaller. Both variants expose the same
3,783 logical tensors with identical names, shapes, and declared source
dtypes. The full-precision repository is not required to run this variant.

## Compatibility

This format requires an `mlx-spatial` build that includes SAM3D affine
checkpoint support. Until that support is available in a tagged PyPI release,
install the current project revision:

```bash
pip install \
  "mlx-spatial @ git+https://github.com/appautomaton/mlx-spatial.git@main"
```

The runtime targets Apple Silicon and MLX `0.32.x`. It does not use Torch,
CUDA, or a dequantized full-precision checkpoint.

## Use

Download the complete bundle:

```bash
hf download appautomaton/sam-3d-objects-mlx-8bit \
  --local-dir weights/sam-3d-objects-mlx-8bit
```

Validate and inspect it without loading all weights:

```bash
mlx-spatial-sam3d validate weights/sam-3d-objects-mlx-8bit
mlx-spatial-sam3d inspect weights/sam-3d-objects-mlx-8bit
```

Generate a Gaussian Splat PLY:

```bash
mlx-spatial-sam3d reconstruct \
  weights/sam-3d-objects-mlx-8bit \
  inputs/sam3d/object-rmbg.png \
  --mask inputs/sam3d/object-mask.png \
  --moge-root weights/sam-3d-objects-mlx-8bit/moge \
  --output outputs/sam3d/object-8bit/gaussians.ply \
  --trace-output outputs/sam3d/object-8bit/trace.json \
  --memory-profile balanced
```

Add `--glb-output outputs/sam3d/object-8bit/object.glb` to decode and export a
mesh-backed GLB in the same run.

SAM3D requires a binary object mask aligned with the input image. A clean
foreground extraction is strongly recommended: transparent background pixels
should have alpha zero, and the supplied mask should cover the intended object
without including the surrounding scene.

## Bundle

The repository preserves the directory and checkpoint names used by the
full-precision MLX bundle:

| Checkpoint | Logical tensors | INT8 matrices | Bytes |
| --- | ---: | ---: | ---: |
| `checkpoints/ss_generator.safetensors` | 1,741 | 556 | 2,197,959,476 |
| `checkpoints/ss_decoder.safetensors` | 74 | 0 | 147,592,136 |
| `checkpoints/slat_generator.safetensors` | 1,225 | 384 | 1,578,129,504 |
| `checkpoints/slat_decoder_gs.safetensors` | 101 | 48 | 97,148,339 |
| `checkpoints/slat_decoder_gs_4.safetensors` | 101 | 48 | 95,942,210 |
| `checkpoints/slat_decoder_mesh.safetensors` | 120 | 48 | 119,546,584 |
| `moge/model.safetensors` | 421 | 96 | 388,551,606 |
| **Total** | **3,783** | **1,180** | **4,624,869,855** |

The remaining YAML configuration, conversion metadata, audit, and license
files are small and are included alongside the checkpoints.

## Quantization

The quantization scheme is affine 8-bit with group size 64. Packed weights are
stored as `uint32` with FP32 scales and biases and are executed directly with
`mx.quantized_matmul`.

The following block-internal two-dimensional weights are quantized:

- DINO condition-encoder attention and MLP matrices;
- the point-condition transformer block;
- sparse-structure and structured-latent generator transformer blocks;
- Gaussian and mesh decoder transformer torso blocks;
- the bundled MoGe ViT backbone attention and MLP matrices.

The following tensors remain in their original precision:

- all dense, sparse, and transposed convolutions;
- the complete sparse-structure decoder;
- patch embeddings, input mappings, condition projections, and latent
  mappings;
- normalization parameters, biases, learned tokens, and positional tensors;
- final Gaussian, mesh, geometry, pose, and MoGe output heads;
- matrices whose input dimension is incompatible with group size 64.

Physical packed arrays use internal qweight, scale, and bias suffixes. The
runtime reconstructs the original logical tensor names from metadata, so model
code and configuration continue to use the same checkpoint contract. Format
details are embedded under `mlx_spatial.sam3d.quantization` in each quantized
safetensors file.

## Reproducing the Bundle

Starting from the full-precision MLX conversion:

```bash
mlx-spatial-sam3d-quantize \
  weights/sam-3d-objects-mlx \
  weights/sam-3d-objects-mlx-8bit \
  --bits 8 \
  --group-size 64
```

The quantized output is a complete runtime root. Do not place copies of the
full-precision checkpoints in the 8-bit repository.

## Verification

- Logical checkpoint inspection matched all 3,783 source tensor names, shapes,
  and dtypes exactly.
- The quantization inventory contains 1,180 packed matrices across six
  quantized checkpoints; the convolutional sparse-structure decoder remains
  unchanged.
- A sampled static weight audit measured median 8-bit reconstruction SQNR of
  approximately 45 dB across checkpoint groups, with a worst observed matrix
  at 41 dB.
- The SAM3D test suite passed 163 tests, with one test deselected.
- A background-removed 900 × 900 input with an aligned mask completed all nine
  inference stages with no blocker and produced 495,008 finite Gaussians.
- The generated geometry passed the runtime's nominal axis-range check.

The end-to-end run establishes runtime compatibility and artifact health. It
is not a formal claim of visual equivalence to the full-precision model, and
the observed run is not presented as a general performance benchmark.

## Limitations

- Quantization can change sparse occupancy, geometry, appearance, and other
  generation details relative to the full-precision variant.
- A useful object-aligned mask is required. Background leakage, broad masks,
  soft edges, reflections, and thin structures can reduce reconstruction
  quality.
- Single-view reconstruction cannot determine unseen geometry with certainty.
- The packed checkpoint format requires `mlx-spatial`; generic safetensors
  loaders will see the physical packed arrays rather than the logical weights.
- This bundle supports inference, not training or fine-tuning.
- Standard PLY viewers may not render Gaussian Splat fields correctly. Use a
  3DGS-aware viewer for the Gaussian artifact, or request GLB export.

## License and Attribution

This bundle is derived from Meta's SAM 3D Objects release and is distributed
under the SAM License. Read the bundled `LICENSE` and the
[upstream SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE)
before use. Redistribution of SAM Materials and derivative works remains
subject to that agreement.

The bundle also contains a converted and quantized
[`Ruicheng/moge-vitl`](https://huggingface.co/Ruicheng/moge-vitl) checkpoint
used for pointmap estimation. That checkpoint is published under Apache 2.0.
Users are responsible for complying with both sets of terms.

This repository is not affiliated with or endorsed by Meta or the MoGe
authors. Publications using these weights should acknowledge the original SAM
3D Objects and MoGe work.

## Links

- [App Automaton](https://appautomaton.com)
- [AppAutomaton models on Hugging Face](https://huggingface.co/appautomaton)
- [`appautomaton/mlx-spatial`](https://github.com/appautomaton/mlx-spatial) — MLX-native 3D and spatial inference for Apple Silicon.
- [`mlx-spatial` documentation](https://appautomaton.com/mlx-spatial/)
- [`mlx-spatial` on PyPI](https://pypi.org/project/mlx-spatial/)
- [SAM3D guide](https://github.com/appautomaton/mlx-spatial/blob/main/docs/sam3d.md)
- [Full-precision MLX variant](https://huggingface.co/appautomaton/sam-3d-objects-mlx)
- [Upstream SAM 3D Objects model](https://huggingface.co/facebook/sam-3d-objects)
- [Upstream SAM 3D Objects source](https://github.com/facebookresearch/sam-3d-objects)
- [MoGe ViT-L checkpoint](https://huggingface.co/Ruicheng/moge-vitl)
- [MoGe source](https://github.com/microsoft/moge)
- [MLX](https://github.com/ml-explore/mlx)
