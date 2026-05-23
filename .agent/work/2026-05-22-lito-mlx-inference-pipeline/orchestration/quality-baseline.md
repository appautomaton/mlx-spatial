# Slice 5Q-0 Quality Baseline

## Command

```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-baseline.ply --memory-profile safe --render-size 12 --num-steps 20 --seed 42 --print-metrics
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-baseline.ply --json /tmp/lito-teacup-quality-baseline.json
uv run pytest tests/test_lito_quality.py -q
```

## Result

- Output: `outputs/lito/teacup-quality-baseline.ply`
- Sidecar: `outputs/lito/teacup-quality-baseline.safetensors`
- Input: `inputs/trellis2/teacup.png`
- Smoke image excluded: `inputs/lito/smoke.png`
- Header: `comment mlx-spatial LiTo checkpoint-backed 3DGS export`
- Vertex count: `32768`
- Property count: `62`
- Failure classification: `stats_sane_visual_review_required`
- Inspector flags: none
- Quality failure source: user visual inspection reports broken surfaces, so AC-07 remains open despite schema/stat sanity.

## Generation Metrics

| Stage | wall_time_s | peak_active_memory_gb | peak_cache_memory_gb |
|---|---:|---:|---:|
| preprocess | 0.1025 | 0.0000 | 0.0000 |
| condition | 0.2664 | 0.2925 | 0.3035 |
| dit | 157.2781 | 15.2769 | 16.3873 |
| decode | 0.3654 | 1.9974 | 18.8862 |
| export | 0.3683 | 0.0174 | 18.8862 |

## Inspector Summary

| Field | Value |
|---|---:|
| bbox min | `[-0.6850, -0.9722, -0.8679]` |
| bbox max | `[0.8769, 0.9498, 0.9835]` |
| bbox span | `[1.5619, 1.9221, 1.8514]` |
| opacity probability median | `0.0443` |
| opacity probability p99 | `0.9989` |
| scale exp median | `0.001832` |
| scale exp p99 | `0.010040` |
| quaternion norm median | `1.00000003` |
| SH f_dc p01 / p99 | `-6.3404 / 12.4513` |
| SH f_rest p01 / p99 | `-0.8813 / 0.7232` |

## Slice Evidence

- `scripts/lito/inspect_quality.py` added for PLY schema/stat inspection.
- `tests/test_lito_quality.py` added and passed: `3 passed`.
- Slice 5Q-0 acceptance is met; proceed to Slice 5Q-1 parallel read-only audits.
