# Slice 5Q-3 Uncapped Coverage Handoff

Date: 2026-05-23

## Trigger

Human visual inspection of `outputs/lito/teacup-quality-crop-4096.ply` showed clear improvement but still missing surface regions. The next suspected issue was remaining init-coordinate under-coverage.

## Occupancy Diagnostic

A memory-conscious diagnostic ran preprocessing, conditioning, DiT sampling, and init-coordinate generation only. It stopped before Gaussian decode/export.

Evidence:

- Input: `inputs/trellis2/teacup.png`
- Seed: `42`
- Steps: `20`
- CFG scale: `3.0`
- Preprocessed shape: `[518, 518, 4]`
- Condition shape: `[1, 1374, 2048]`
- Latent shape: `[1, 8192, 32]`
- Uncapped occupied cells: `17317`
- Previous current candidate cap: `4096`
- Coverage ratio of 4096-cap candidate: about `23.7%`
- Peak active memory during diagnostic: about `16.42 GB`

Conclusion: the 4096-cell candidate was still dropping most occupied init cells, so a full uncapped decode was justified and stayed well below the 100 GB hard memory ceiling.

## Uncapped Candidate

Command:

```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-crop-uncapped.ply --memory-profile safe --max-init-coords-per-batch none --render-size 12 --num-steps 20 --seed 42 --print-metrics
```

Runtime evidence:

- Output: `outputs/lito/teacup-quality-crop-uncapped.ply`
- Safetensors: `outputs/lito/teacup-quality-crop-uncapped.safetensors`
- Decoded cells: `17317`
- Exported Gaussians: `1108288`
- Peak active memory: `15.2769 GB` in DiT, `1.9974 GB` in decode, `0.2138 GB` in export
- Peak cache memory: `21.8705 GB`
- Wall time: `185.0727 s` DiT, `0.8844 s` decode, `10.6865 s` export
- File sizes: PLY about `765 MB`, safetensors about `249 MB`

Inspector:

```bash
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-crop-uncapped.ply --compare /tmp/lito-teacup-quality-crop-4096.json --json /tmp/lito-teacup-quality-crop-uncapped.json
```

Inspector evidence:

- Checkpoint-backed header: yes
- Source-contract smoke header: no
- Vertex count: `1108288`
- Property count: `62`
- Inspector flags: none
- Failure classification: `stats_sane_visual_review_required`
- Opacity probability median: `0.056885`
- Scale exp median: `0.004650`
- Quaternion norm median: `1.000000`
- Compared with the 4096 candidate, vertex count increased by `846144`.

Verification:

```bash
uv run pytest tests/test_lito_inference.py tests/test_lito_real_backend.py tests/test_lito_quality.py -q
```

Result: `61 passed`.

```bash
bash -lc '! rg -n "import torch|from torch|cuda\\.|xformers|flash_attn|gsplat|vendors/ml-lito|from lito|import lito|rembg" src/mlx_spatial/lito_inference.py src/mlx_spatial/lito.py src/mlx_spatial/lito_real_backend.py'
```

Result: passed; no forbidden runtime imports.

## Gate State

The current visual candidate is:

- `outputs/lito/teacup-quality-crop-uncapped.ply`

This output uses all occupied init cells for the measured latent and is the best current test of the init-coverage hypothesis. It is large but stayed within the memory safety envelope.

If this is still visually rejected, the next fix target should shift away from init coverage and crop behavior. Likely remaining targets are:

- tensor-level conditioning parity against upstream Torch/MPS where executable,
- Gaussian decode/export convention checks, especially quaternion/scale/opacity/SH ordering,
- source-level render/viewer convention mismatch if the visual checker is not a 3DGS-aware renderer.
