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
uv run mlx-spatial-lito validate weights/lito-research-mlx
uv run mlx-spatial-lito inspect weights/lito-research-mlx --limit 10
```

Expected converted layout:

```text
weights/lito-research-mlx/tokenizer/lito_new.safetensors
weights/lito-research-mlx/image_to_3d/lito_dit_rgba.safetensors
```

Maintainers can print Apple CDN download commands and convert local `.ckpt` files:

```bash
uv run mlx-spatial-lito download-command
uv run python -m mlx_spatial.lito_assets convert weights/lito-raw weights/lito-research-mlx
```

Converted weights are an unofficial derivative and must preserve Apple's research license boundary when published separately.

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
python scripts/lito/generate.py inputs/lito/sample.png \
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
user-facing subset shown by `python scripts/lito/generate.py --help`.

| Flag | Use |
| --- | --- |
| `--format {ply,splat,safetensors}` | Select output artifact format. Use `ply` for checkpoint-backed viewer output; checkpoint-backed `splat` export is not implemented. |
| `--ply-storage {binary_little_endian,ascii}` | Select PLY storage. Use `binary_little_endian` for normal runs and `ascii` only for debugging or text diffs. |
| `--memory-profile {safe,balanced,large}` | Select memory and init-coordinate defaults. |
| `--max-init-coords-per-batch {profile,none,N}` | Package CLI only. Use profile cap, upstream-style full occupied cells, or an explicit cap. |
| `--num-steps N` | Sampling steps; default follows `LITO_RECOMMENDED_NUM_STEPS`. |
| `--cfg-scale X` | Classifier-free guidance scale. |
| `--seed N` | Make local sampling reproducible. |
| `--resolution N` | Square preprocessing resolution; the default follows `LITO_RECOMMENDED_RESOLUTION`. |
| `--render-size N` | Source-contract smoke render size; checkpoint-backed PLY export ignores it. |
| `--print-metrics` | Print per-stage timing and MLX memory metrics. |
| `--source-contract-smoke` | Use synthetic contract output instead of checkpoint-backed inference. |

`--max-init-coords-per-batch none` can produce very large PLY files because each occupied init cell expands to many Gaussian splats. Keep those outputs under `outputs/lito/`.

## Memory

Profiles:

| Profile | Default behavior |
| --- | --- |
| `safe` | Conservative init-coordinate cap for lower-memory smoke and debugging runs. |
| `balanced` | Practical default for high-memory Apple Silicon development systems. |
| `large` | Higher-cap run path; use only when memory headroom is clear. |

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
