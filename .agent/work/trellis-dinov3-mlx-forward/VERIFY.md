# VERIFY: TRELLIS.2 MLX DINOv3 Forward

## Result

PASS.

The MLX DINOv3 image-conditioning boundary now runs with local real
`facebook/dinov3-vitl16-pretrain-lvd1689m` assets and advances the TRELLIS.2
forward trace to the sparse-structure sampling boundary.

## Evidence

Real local attempt:

```bash
uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/dinov3-mlx-forward-alpha.json
```

Observed result:

- completed stages: `input-image`, `asset-config-validation`, `checkpoint-probe-readiness`, `image-preprocessing-background`, `image-conditioning`
- output: `cond`
- output shape: `(1, 1029, 1024)`
- output dtype: `float32`
- DINOv3 detail: 24 transformer layers, patch grid `(32, 32)`
- current blocker: `sparse-structure-sampling` / `MLX sparse structure flow model construction`

Targeted verification:

```bash
uv run pytest tests/test_trellis2_dinov3_forward.py tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py
```

Observed result: `33 passed`.

Full verification:

```bash
uv run pytest
```

Observed result: `118 passed, 5 skipped`.

## Dependency Check

Runtime dependencies remain MLX-first and do not add Torch or Transformers:

- `mlx`
- `numpy`
- `pillow`
- `safetensors`

Hugging Face tooling remains in the dev dependency group only.

## Current Boundary

DINOv3 image conditioning is no longer the active blocker. The next unimplemented
TRELLIS.2 stage is sparse-structure sampling: MLX sparse structure flow model
construction and FlowEuler sampler dispatch.
