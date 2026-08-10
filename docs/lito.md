# LiTo

LiTo is the Apple image-to-3DGS research path in `mlx-spatial`. It takes one object-centric RGB/RGBA image and writes a Gaussian Splat artifact, not a mesh.

Use SAM3D or TRELLIS.2 when you need object mesh or GLB workflows. Use HY-WorldMirror for scene reconstruction.

## Status

- Runtime target: MLX on Apple Silicon.
- Output: Gaussian Splat PLY by default.
- Default path: checkpoint-backed local safetensors inference.
- Smoke path: available only with `--source-contract-smoke`; those outputs are synthetic contract probes, not Apple LiTo results.
- License: Apple's model license is research-only and non-commercial.

The Python package does not include LiTo weights. Keep weights and generated artifacts out of git.

## Assets

Recommended runtime bundle:

```bash
uv run hf download appautomaton/lito-research-mlx \
  --local-dir weights/lito-research-mlx
uv run hf download microsoft/TRELLIS-image-large \
  ckpts/ss_dec_conv3d_16l8_fp16.json \
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  --local-dir weights/trellis2/microsoft/TRELLIS-image-large
uv run mlx-spatial-lito validate weights/lito-research-mlx
uv run mlx-spatial-lito inspect weights/lito-research-mlx --limit 10
```

Expected converted layout:

```text
weights/lito-research-mlx/tokenizer/lito_new.safetensors
weights/lito-research-mlx/image_to_3d/lito_dit_rgba.safetensors
weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json
weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors
```

The first two files are the LiTo bundle. The final two are the sparse-structure
decoder used to convert LiTo voxel latents into Gaussian initialization
coordinates. They come from `microsoft/TRELLIS-image-large`, not from the
`microsoft/TRELLIS.2-4B` bundle used by the separate TRELLIS.2 pipeline.
`mlx-spatial-lito validate` checks all four runtime files; `inspect` reads only
the two LiTo safetensors.

Maintainers can print Apple CDN download commands and convert local `.ckpt` files:

```bash
uv run mlx-spatial-lito download-command
uv run python -m mlx_spatial.lito_assets convert weights/lito-raw weights/lito-research-mlx
uv run mlx-spatial-lito-prune \
  weights/lito-research-mlx \
  weights/lito-research-mlx \
  --overwrite
```

The conversion step preserves the official research checkpoint exactly. The
prune step then atomically turns it into the full-precision runtime bundle by
removing modules that the main inference path never reads. Converted weights
are an unofficial derivative and must preserve Apple's research license
boundary when published separately.

## Quantization

Create a separate 8-bit affine bundle from the full-precision MLX weights:

```bash
uv run mlx-spatial-lito-quantize \
  weights/lito-research-mlx \
  weights/lito-research-mlx-8bit \
  --bits 8 \
  --group-size 64
```

The quantizer first removes checkpoint modules that the inference runtime never
loads: the non-EMA velocity estimator, the duplicate embedded tokenizer, and
the mesh/fpoint/LPIPS/training decoders. From the remaining runtime tensors, it
packs only internal DiT and decoder attention/MLP matrices. It keeps the
DINO/RGBA conditioner, convolution weights, embeddings, normalization
parameters, timestep and condition projections, latent boundary projections,
and final Gaussian/voxel/velocity heads at their source precision. Each
checkpoint records the exact policy, logical tensor shapes, bit width, group
size, and affine mode in safetensors metadata. The normal LiTo loader detects
that metadata and executes packed matrices directly with MLX quantized matrix
multiplication; no Torch or intermediate dequantized checkpoint is involved.

Pass the new root to inference exactly as you would the full-precision root:

```bash
uv run mlx-spatial-lito generate inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx-8bit \
  --output outputs/lito/sample-8bit.ply \
  --format ply \
  --print-metrics
```

## Inputs

Use an object-centric RGB/RGBA image:

```text
inputs/lito/sample.png
```

RGBA inputs use the alpha channel. RGB inputs are preprocessed through the local LiTo image-conditioning path. Do not commit Apple generated or modified sample images unless a separate redistributable license allows it.

## Run

Package CLI:

```bash
uv run mlx-spatial-lito generate inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --format ply \
  --memory-profile balanced \
  --print-metrics
```

Repository script with the same user-facing defaults:

```bash
uv run python scripts/lito/generate.py inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --memory-profile balanced \
  --print-metrics
```

Use the synthetic smoke path only when validating framework plumbing:

```bash
uv run mlx-spatial-lito generate inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample-smoke.ply \
  --memory-profile safe \
  --source-contract-smoke
```

## Outputs

Default checkpoint-backed output:

```text
outputs/lito/<name>.ply
```

The default PLY storage is `binary_little_endian`. Use `--ply-storage ascii` only for debugging or text diffs. LiTo PLY files contain Gaussian splat fields; Blender's native Stanford PLY importer can read the container but will not render the splats correctly. Use a 3DGS-aware viewer such as KIRI's Blender 3DGS add-on.

## Runtime Options

Common package CLI generation flags. The repository script exposes the
user-facing subset shown by `uv run python scripts/lito/generate.py --help`.

| Flag | Use |
| --- | --- |
| `--format {ply,splat,safetensors}` | Select output artifact format. Use `ply` for checkpoint-backed viewer output; checkpoint-backed `splat` export is not implemented. |
| `--ply-storage {binary_little_endian,ascii}` | Select PLY storage. Use `binary_little_endian` for normal runs and `ascii` only for debugging or text diffs. |
| `--memory-profile {safe,balanced,large}` | Select source-contract smoke execution settings. Checkpoint-backed generation always preserves complete occupied-cell coverage. |
| `--max-init-coords-per-batch N` | Package CLI only. Explicitly truncate occupied cells for debugging; omitted by default. |
| `--num-steps N` | Sampling steps; default follows `LITO_RECOMMENDED_NUM_STEPS`. |
| `--cfg-scale X` | Classifier-free guidance scale. |
| `--seed N` | Make local sampling reproducible. |
| `--resolution N` | Square preprocessing resolution; the default follows `LITO_RECOMMENDED_RESOLUTION`. |
| `--render-size N` | Source-contract smoke render size; checkpoint-backed PLY export ignores it. |
| `--print-metrics` | Print per-stage timing and MLX memory metrics. |
| `--source-contract-smoke` | Use synthetic contract output instead of checkpoint-backed inference. |

Complete LiTo output can produce large PLY files because each occupied init cell expands to 64 Gaussian splats. An explicit `--max-init-coords-per-batch N` trades away surface completeness and should not be used for quality output.

## Memory

Profiles:

| Profile | Default behavior |
| --- | --- |
| `safe` | Small source-contract smoke workload. |
| `balanced` | Default source-contract smoke workload. |
| `large` | Larger source-contract smoke workload. |

LiTo reports stage metrics when `--print-metrics` is set. If a run blocks because required assets are missing or memory safety limits are exceeded, keep the blocker message with the trace or issue report.

## API

```python
from mlx_spatial.lito import LitoInferencePipeline
from mlx_spatial.lito_inference import LITO_RECOMMENDED_NUM_STEPS

pipe = LitoInferencePipeline(weights_root="weights/lito-research-mlx", memory_profile="balanced")
result = pipe.generate(
    "inputs/lito/sample.png",
    output_path="outputs/lito/sample.ply",
    num_steps=LITO_RECOMMENDED_NUM_STEPS,
    seed=42,
)
print(result.output_path)
```

For synthetic smoke only:

```python
pipe = LitoInferencePipeline(
    weights_root="weights/lito-research-mlx",
    memory_profile="safe",
    source_contract_smoke=True,
)
```

## Development Notes

- Runtime code must work without `vendors/`.
- Do not add Torch, CUDA, xformers, flash-attention, or gsplat CUDA to the runtime path.
- Source-contract fixtures under `tests/fixtures/lito/` lock local tensor schemas and are not vendor numerical captures.
- Optional parity probes must stay dev-only and non-blocking.
- Keep deferred runtime work out of this stable user page.
