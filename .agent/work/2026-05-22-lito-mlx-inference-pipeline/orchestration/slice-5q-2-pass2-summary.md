# Slice 5Q-2 Pass 2 Summary

Date: 2026-05-23

## Trigger

Human visual inspection rejected `outputs/lito/teacup-quality-fix.ply`; the render remained sparse and fragmented. AC-07 remains open.

## Fix Target

Conditioning/preprocessing parity, specifically upstream LiTo optical-axis crop behavior for useful-alpha RGBA inputs. The current teacup and beer-mug inputs already contain useful transparency, so this pass did not add a background-removal dependency.

## Implementation

- `src/mlx_spatial/lito_inference.py`
  - Replaced foreground-recentering crop behavior with upstream-style `keep_optical_axis=True` crop/pad math.
  - Uses alpha threshold `0.8`, `fill_ratio=0.8`, original image center as optical axis, pad ratios `0.5`, and transparent zero padding outside image bounds.
  - Keeps resize after crop/pad and keeps runtime dependencies limited to PIL/numpy/MLX.
- `tests/test_lito_inference.py`
  - Added an off-axis alpha crop test that fails under foreground-recentering behavior.
  - Added preprocessing output shape, dtype, and range coverage.

## Verification

```bash
uv run pytest tests/test_lito_inference.py tests/test_lito_quality.py -q
```

Result: `14 passed`.

```bash
bash -lc '! rg -n "import torch|from torch|cuda\\.|xformers|flash_attn|gsplat|vendors/ml-lito|from lito|import lito|rembg" src/mlx_spatial/lito_inference.py'
```

Result: passed; no forbidden runtime imports.

Spec review: `APPROVED`.

Code-quality review: `APPROVED`.

## Regeneration Evidence

The user asked whether a new run may address the visual issue. Two higher-coverage teacup candidates were generated:

```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-fix-4096.ply --memory-profile safe --max-init-coords-per-batch 4096 --render-size 12 --num-steps 20 --seed 42 --print-metrics
```

```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-crop-4096.ply --memory-profile safe --max-init-coords-per-batch 4096 --render-size 12 --num-steps 20 --seed 42 --print-metrics
```

Latest candidate: `outputs/lito/teacup-quality-crop-4096.ply`.

Inspector:

```bash
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-crop-4096.ply --compare /tmp/lito-teacup-quality-fix-4096.json --json /tmp/lito-teacup-quality-crop-4096.json
```

Evidence:

- Checkpoint-backed header: yes
- Source-contract smoke header: no
- Vertex count: `262144`
- Property count: `62`
- Inspector flags: none
- Failure classification: `stats_sane_visual_review_required`
- Opacity probability median: `0.138184`
- Scale exp median: `0.002623`
- Quaternion norm median: `1.000000`
- Compared to the pre-crop 4096 run, opacity median increased by `0.01136`; vertex count was unchanged.

## Gate State

The new visual candidate is:

- `outputs/lito/teacup-quality-crop-4096.ply`

This still requires human visual inspection before AC-07 can close. If rejected, the next fix should not repeat the same crop/cap pass; likely remaining targets are deeper conditioning numerical parity or Gaussian decode/export convention.
