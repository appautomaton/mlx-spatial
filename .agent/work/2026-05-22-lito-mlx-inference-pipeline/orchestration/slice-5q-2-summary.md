# Slice 5Q-2 Summary

## Route

Subagent worker: `019e564c-4591-73c2-ad50-1d08fcdc7cb7`.

## Selected Fix Target

Decode/init-coordinate coverage. Slice 5Q-1 found the safe-profile path capped init cells at `512`, while upstream keeps all occupied cells after thresholding. Baseline `32768` vertices equals `512 * 64`.

## Changes

- Added `--max-init-coords-per-batch {profile|none|N}` to `mlx-spatial-lito generate`.
- `profile` preserves current memory-profile cap behavior.
- `none` disables top-k capping and passes `max_cells_per_batch=None`.
- Positive integer `N` applies an explicit cap.
- Plumbed the override through `LitoInferencePipeline` into `LitoRealBackendConfig` and `DirectMlxLitoBackend.decode_sampled_latents_to_gaussians(...)`.

## Verification

```bash
uv run mlx-spatial-lito generate --help
uv run pytest tests/test_lito_real_backend.py tests/test_lito_inference.py tests/test_lito_cli.py tests/test_lito_quality.py -q
bash -lc '! rg -n "import torch|from torch|cuda\\.|xformers|flash_attn|gsplat|vendors/ml-lito|from lito|import lito" src/mlx_spatial/lito_real_backend.py src/mlx_spatial/lito_inference.py src/mlx_spatial/lito.py'
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-fix.ply --memory-profile safe --max-init-coords-per-batch 1024 --render-size 12 --num-steps 20 --seed 42 --print-metrics
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-fix.ply --compare /tmp/lito-teacup-quality-baseline.json --json /tmp/lito-teacup-quality-fix.json
```

Results:

- CLI help includes `--max-init-coords-per-batch {profile|none|N}`.
- Targeted tests: `69 passed`.
- Forbidden runtime import scan: passed.
- 1024-cap generation produced `outputs/lito/teacup-quality-fix.ply`.
- Inspector: checkpoint-backed, `65536` vertices, `62` properties, no flags, `failure_classification=stats_sane_visual_review_required`.
- Comparison vs. baseline: `vertex_count_delta=32768`, bbox span grew by `[0.0265, 0.0349, 0.0146]`, scale median increased by `0.0000379`, opacity median decreased by `0.0167`.

## Correction

The original Slice 5Q-2 verification command did not include the new cap override because the exact fix target was selected after Slice 5Q-1. For this pass, the meaningful quality command is the same teacup generation with `--max-init-coords-per-batch 1024`. A previous attempted `2048` run was interrupted by context transition and produced no artifact.

## Outcome

The implementation fixes the cap-control mismatch and narrows the quality failure from "safe profile hard cap only" to "needs visual inspection of larger-coverage output, then possibly preprocessing/crop fix if surfaces remain broken."
