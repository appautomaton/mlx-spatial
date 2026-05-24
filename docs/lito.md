# LiTo

`mlx-spatial` now has an Apple LiTo image-to-3DGS pipeline for Apple Silicon bring-up. The runtime target is MLX on M-series Macs. The output is a Gaussian splat artifact, not a mesh.

This phase validates the local MLX-compatible pipeline shape, CLI, metrics, memory limits, asset routing, and export surface. It does not claim full Apple checkpoint numerical parity yet; upstream PyTorch, CUDA, and gsplat paths remain static architecture references while the MLX implementation is filled in behind the same interfaces.

The default `generate` path is checkpoint-backed: it validates converted checkpoint files, runs the local safetensors-to-MLX path, and writes a checkpoint-backed 3DGS artifact. Synthetic PLY generation is available only with `--source-contract-smoke` and should not be treated as an actual Apple LiTo result.

LiTo support follows the same package shape as SAM3D, TRELLIS.2, and HY-World 2.0: local assets under `weights/`, ignored user inputs under `inputs/lito/`, generated artifacts under `outputs/lito/`, a package CLI, and a readable repository sample script.

## License Summary

Apple's model license is research-only and non-commercial. Local checkpoint conversion is used here only for that research integration path. Converted weights are not redistributed by the `mlx-spatial` source repository. If converted weights are published in a separate model repository, the model card must preserve Apple's research-only license boundary, include Apple's model license, identify the files as an unofficial converted derivative, and disclose the storage/layout conversion.

Apple's generated sample license is CC BY-NC-ND 4.0. Apple generated or modified sample fixtures are not redistributed here. Contributors should download sample inputs locally and keep generated or adapted samples out of git unless a separate redistributable source image is used.

The practical setup rule is:

- download official checkpoints locally;
- convert them locally to MLX-readable safetensors;
- keep `weights/lito-raw/`, `weights/lito-research-mlx/`, `inputs/lito/`, and `outputs/lito/` untracked;
- treat `vendors/ml-lito/` as a development reference only.

## Weight Acquisition

Slice 0 found no official or `mlx-community` LiTo safetensors repository on Hugging Face. The source-of-truth conversion path remains Apple CDN `.ckpt` download followed by local conversion. A converted AppAutomaton bundle is published for research use at:

```text
appautomaton/lito-research-mlx
```

Download the AppAutomaton bundle:

```bash
hf download appautomaton/lito-research-mlx \
  --local-dir weights/lito-research-mlx
```

Print the current download commands:

```bash
uv run python -m mlx_spatial.lito_assets download-command weights/lito-raw
```

Equivalent direct commands:

```bash
mkdir -p weights/lito-raw
curl -L https://ml-site.cdn-apple.com/models/lito/lito_new.ckpt \
  -o weights/lito-raw/lito_new.ckpt
curl -L https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt \
  -o weights/lito-raw/lito_dit_rgba.ckpt
```

Or convert the official checkpoints locally:

```bash
uv run python -m mlx_spatial.lito_assets convert weights/lito-raw weights/lito-research-mlx
uv run python -m mlx_spatial.lito_assets validate weights/lito-research-mlx
uv run python -m mlx_spatial.lito_assets inspect weights/lito-research-mlx --limit 10
```

Expected converted layout:

```text
weights/lito-research-mlx/tokenizer/lito_new.safetensors
weights/lito-research-mlx/image_to_3d/lito_dit_rgba.safetensors
```

The checkpoint-backed boundary can inspect these safetensors headers without loading full tensors. Current real-weight inventory is:

```text
image_to_3d/lito_dit_rgba.safetensors: 2793 tensors; DiT 28 blocks, 8192 latent tokens, dim 32, hidden 1152, condition dim 2048
tokenizer/lito_new.safetensors: 1108 tensors; Gaussian decoder 6 blocks, expansion ratio 64, SH degree 3; voxel decoder init grid 16x16x16
```

It can also selectively load and remap real patch-encoder / DiT / Gaussian decoder tensors from those safetensors into local MLX-compatible names. The DINOv2 ViT-L/14-reg branch plus LiTo's RGBA learnable branch run from checkpoint weights for RGBA inputs, the DiT sampler runs on local MLX, the voxel/TRELLIS path produces init coordinates, and the Gaussian Perceiver + `decode_gs` path exports checkpoint-backed 3DGS tensors.

## Sample Inputs

Place local LiTo inputs under:

```text
inputs/lito/
```

Use your own RGB/RGBA object image, or download Apple demo images locally if the upstream repo or CDN publishes them. Do not commit Apple generated or modified sample images.

## CLI Usage

Validate and inspect assets:

```bash
uv run mlx-spatial-lito validate weights/lito-research-mlx
uv run mlx-spatial-lito inspect weights/lito-research-mlx --limit 10
uv run mlx-spatial-lito download-command
```

Checkpoint-backed generation is the default. With local LiTo and TRELLIS weights present, this writes a checkpoint-backed PLY:

```bash
uv run mlx-spatial-lito generate inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --format ply \
  --memory-profile balanced \
  --print-metrics
```

Checkpoint-backed PLY export defaults to standard `binary_little_endian` storage, matching the format expected by normal PLY readers while keeping large uncapped outputs much smaller than ASCII. Use `--ply-storage ascii` only for debugging or text diffs. This does not change the viewer requirement: use a Gaussian-splat-aware viewer such as KIRI's Blender 3DGS add-on. Blender's native Stanford PLY importer can read PLY containers, but it does not render LiTo Gaussian splat fields correctly.

A current quality-check run uses all occupied init cells from the sampled latent:

```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/teacup-quality-crop-uncapped.ply \
  --memory-profile safe \
  --max-init-coords-per-batch none \
  --num-steps 20 \
  --seed 42 \
  --print-metrics
```

That run writes a checkpoint-backed PLY with `1108288` vertices from `17317` occupied init cells. It stayed within the memory safety envelope on the M4 Max development system: peak active memory was about `15.28 GB` and peak cache memory was about `21.87 GB`.

A second real-object quality-check run is:

```bash
uv run mlx-spatial-lito generate inputs/trellis2/beer-mug.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/beer-mug-quality-uncapped.ply \
  --memory-profile safe \
  --max-init-coords-per-batch none \
  --num-steps 20 \
  --seed 42 \
  --print-metrics
```

That run writes a checkpoint-backed PLY with `925952` vertices from `14468` occupied init cells and also stayed around `15.28 GB` peak active memory.

The old uncapped ASCII PLY files from bring-up were large (`~765 MB` for teacup and `~619 MB` for beer mug). Binary PLY reduces the storage overhead, but these are still large Gaussian-splat artifacts and should stay under `outputs/lito/`; they are ignored and should not be committed. The `inputs/lito/smoke.png` file is only a color-blob framework probe and should not be used for qualitative LiTo assessment.

Run the synthetic source-contract smoke path only when that is what you mean to test:

```bash
uv run mlx-spatial-lito generate inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample-smoke.ply \
  --format ply \
  --memory-profile safe \
  --source-contract-smoke
```

Any existing `outputs/lito/smoke*.ply` files from the bring-up run are source-contract smoke artifacts, not real Apple checkpoint-backed LiTo results. Their PLY headers say `comment mlx-spatial LiTo source-contract smoke 3DGS export`.

The `generate` command supports `--format {ply,splat,safetensors}`, `--ply-storage {binary_little_endian|ascii}`, `--memory-profile {safe,balanced,large}`, `--max-init-coords-per-batch {profile|none|N}`, `--num-steps`, `--cfg-scale`, `--seed`, `--print-metrics`, and `--source-contract-smoke`. The default output format is PLY with binary-little-endian storage for checkpoint-backed results.

Current checkpoint-backed backend progress covers the inference path: real safetensors-to-MLX weight loading, LiTo DINO/RGBA image conditioning to `(B, 1374, 2048)` condition tokens for 518px inputs, LiTo DiT velocity/sampling, LiTo voxel decoder low-res `ss_latent`, TRELLIS sparse-structure occupancy decode, occupancy-to-init-coordinate extraction, Gaussian point-query encoding, all Gaussian Perceiver blocks with localized-voxel self-attention, weighted output heads, Gaussian decode, and checkpoint-backed PLY export.

## Sample Script

The repository script mirrors the SAM3D sample-script pattern and delegates to the package CLI:

```bash
python scripts/lito/generate.py inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --memory-profile balanced \
  --print-metrics
```

Use the script when you want the documented recommended defaults without remembering the full CLI surface. Pass `--source-contract-smoke` explicitly for the synthetic smoke path.

## Recommended Defaults

Slice 0 recorded upstream defaults from `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` and related LiTo sources. Keep `docs/lito.md` in sync with the `LITO_RECOMMENDED_*` constants in `mlx_spatial.lito_inference`.

| Setting | Value | Source |
|---|---:|---|
| `LITO_RECOMMENDED_NUM_STEPS` | `20` | Demo `/generate-stream` default and warmup path |
| `LITO_RECOMMENDED_CFG_SCALE` | `3.0` | Demo `/generate-stream` default and `inference_sample_latent_mlx` default |
| `LITO_RECOMMENDED_RESOLUTION` | `518` | Demo preprocessing and `--img_resolution` default |
| `LITO_RECOMMENDED_SAMPLER` | `heun` | Upstream MLX rectified-flow ODE sampling path |
| `LITO_RECOMMENDED_MLX_COMPUTE_DTYPE` | `float16` | Demo passes `mlx_compute_dtype="float16"` |
| `LITO_RECOMMENDED_DECODE_STEPS_FOR_SAMPLE_XYZ` | `50` | Demo Gaussian decode path |

The public demo exposes no user seed control. The MLX sampling path creates runtime noise with `mx.random.normal(...)`. For reproducible local tests, pass an explicit `--seed`.

Preprocessing follows the upstream demo: EXIF transpose, optional background removal, crop/pad with `fill_ratio=0.8`, `keep_optical_axis=True`, alpha threshold `0.8`, square resize to `518`, straight RGBA plus premultiplied RGB in `[0, 1]`, and ImageNet RGB normalization for DINO conditioning.

## Memory Profiles

LiTo uses memory profiles:

- `safe`: caps real init cells at 512 before 64x Gaussian expansion when `--max-init-coords-per-batch profile` is used.
- `balanced`: caps real init cells at 2048 when `--max-init-coords-per-batch profile` is used; default profile for M4 Max-style development systems.
- `large`: caps real init cells at 8192 when `--max-init-coords-per-batch profile` is used and may approach the soft memory threshold.

For quality inspection on a high-memory Apple Silicon system, pass `--max-init-coords-per-batch none` to match upstream's full occupied-cell behavior. This can produce large artifacts because each occupied cell expands to 64 Gaussian splats. On the current teacup run, `17317` occupied cells expanded to `1108288` PLY vertices.

The current development ceiling is a 128 GB unified-memory system. LiTo runtime code must warn when peak active MLX memory crosses 90 GB and must raise `LitoMemoryLimitExceeded` when it crosses 100 GB. M2/M3 tile or streaming profiles are deferred to Phase 4+.

Every stage that allocates large tensors should measure memory after an `mx.eval(...)` or `mx.synchronize()` boundary. Mid-stage measurements without an eval boundary are not load-bearing.

## Metrics

The explicit source-contract smoke path returns a `LitoGenerationResult` with the same shape that checkpoint-backed generation will use:

```python
metrics: dict[str, dict[str, float]]
```

Expected stages:

```text
preprocess
condition
tokenize
dit
decode
render
export
```

Each stage records `wall_time_s`, `peak_active_memory_gb`, and `peak_cache_memory_gb` when the runtime surface is available. Use `--print-metrics` on the CLI to print the same values during generation.

## Source-Contract Fixtures

LiTo fixtures live under:

```text
tests/fixtures/lito/
```

Regenerate and validate them with:

```bash
uv run python scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite
uv run python scripts/lito/validate_fixtures.py tests/fixtures/lito --verbose
```

These are deterministic local source-contract fixtures. They are not vendor numerical captures. They do not import Apple LiTo runtime code, Torch, CUDA, xformers, flash-attention, gsplat CUDA, or MLX during fixture generation. Their purpose is to lock tensor shapes, dtypes, token ordering, schema, and local MLX-compatible operation contracts so each LiTo module can be developed without a CUDA reference run.

CUDA is not allowed for LiTo acceptance. Upstream CUDA, PyTorch, and gsplat code is static source reference only. Optional Torch/MPS parity may be added only when a path runs without CUDA-only packages; it is never a required acceptance gate.

Do not add Torch to the runtime path or project dependency groups for LiTo generation. A transient `uv --with torch --with torchvision ...` probe can work on macOS/MPS, but adding Torch to `uv.lock` pulled NVIDIA/CUDA packages during resolution, and the upstream hybrid backend would still violate the runtime rule against vendor imports. Keep Torch external for optional parity/probe work only.

Tolerance summary:

| Area | Fixture role | Local contract tolerance |
|---|---|---|
| Image conditioner | shape, dtype, token order, normalization | `atol=2e-3`, `rtol=2e-3`; optional CPU/MPS Torch parity `rtol=2e-2` |
| Tokenizer | `8192 x 32` latent source contract | `atol=2e-3`, `rtol=2e-3` |
| DiT | sampled microtrajectory milestones | local fixtures `atol=2e-3`, `rtol=2e-3`; upstream MLX trajectory drift budget `atol=5e-2`, `rtol=5e-2` |
| Render | Gaussian schema, camera convention, image/alpha output | exact shape, alpha in `[0, 1]`, local MAE `<=1e-5`; optional non-CUDA image comparison PSNR `>=28 dB` or MAE `<=0.02` |

## Programmatic API

The orchestration layer is available through the package API:

```python
from mlx_spatial.lito import LitoInferencePipeline
from mlx_spatial.lito_inference import (
    LITO_RECOMMENDED_NUM_STEPS,
)

pipe = LitoInferencePipeline(weights_root="weights/lito-research-mlx", memory_profile="balanced")
result = pipe.generate(
    "inputs/lito/sample.png",
    output_path="outputs/lito/sample.ply",
    num_steps=LITO_RECOMMENDED_NUM_STEPS,
    seed=42,
)
print(result.output_path)
```

The direct backend raises `LitoBackendUnavailable` if required converted LiTo or TRELLIS assets are missing, or if the sampled latent produces no occupied TRELLIS cells at the upstream threshold. For synthetic smoke only:

```python
pipe = LitoInferencePipeline(
    weights_root="weights/lito-research-mlx",
    memory_profile="safe",
    source_contract_smoke=True,
)
```

## Vendor Reference

`vendors/ml-lito/` is a shallow local clone of `apple/ml-lito` used for architecture, license, and source-contract review. Runtime package code must work when `vendors/` is absent and must not import vendor modules.

Upstream backend status from Slice 0 and the later verify-gap correction:

- DiT and Gaussian decode have upstream MLX paths.
- Tokenizer encoder and render remain PyTorch/gsplat source references; DINO/RGBA conditioning and voxel/TRELLIS init-coordinate generation now have local MLX inference ports.
- LF-conditioned rendering is adapter-feasible with the existing `gs_rasterize.py` surface.
- A vendor-backed optional backend is not compliant for package runtime. The remaining real path must be local code: safetensors headers and tensors, local DINO conditioning, MLX DiT sampling, tokenizer/init-coordinate logic, MLX Gaussian decoding, and local export.

## Phase 4+ Candidates

Deferred LiTo work is tracked in `.agent/steering/ROADMAP.md`: MLX training or fine-tuning, redistributable model packaging, mesh extraction from 3DGS+LF outputs, multi-image and video conditioning, M2/M3 memory profiles, and cross-pipeline ablations.
