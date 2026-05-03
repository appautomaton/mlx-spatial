# VERIFY: TRELLIS.2 DINOv3 Conditioning

## Result

PASS.

The change now resolves local DINOv3 assets, inspects the real config and
checkpoint offline, refreshes the real alpha forward trace, and records the
first real MLX DINOv3 blocker.

## Evidence

### Asset Validation

```bash
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
```

```text
ready=True
present=2
missing=0
```

### Real DINOv3 Inventory

```text
model_type: dinov3_vit
hidden_size: 1024
num_hidden_layers: 24
num_attention_heads: 16
tensor_count: 415
patch_embedding_shape: (1024, 3, 16, 16)
```

### Forward Trace

```bash
uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/demo-alpha-forward-trace.json
```

```text
completed=('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')
outputs=()
blocker_stage=image-conditioning
operation=MLX DINOv3 transformer block construction
```

The ignored JSON evidence is `outputs/trellis2/demo-alpha-forward-trace.json`.

## Acceptance Criteria

- DINOv3 asset validation/helper APIs exist and validate the local files: PASS.
- Manual DINOv3 download/help surface exists and names the repo plus local root: PASS.
- Fake DINOv3 config/checkpoint fixtures inspect deterministically without real weights: PASS.
- Missing local DINOv3 assets return a precise local asset blocker: PASS.
- Incompatible fake assets return precise config/key/shape blockers: PASS.
- Fake-compatible DINOv3 fixtures record conditioning metadata and reach sparse boundary: PASS.
- Real local DINOv3 assets report the first exact real port blocker: PASS.
- Default tests pass without requiring network access or forbidden runtime deps: PASS.
- Runtime dependencies exclude PyTorch, TorchVision, Transformers, Hugging Face Hub, and ONNX Runtime: PASS.
- Real weights and generated outputs are ignored by git: PASS.
- README documents DINOv3 assets, validation/download help, and current boundary: PASS.

## Commands

Fresh verification was run during `auto-verify`; the outputs below are from the
verification pass, not copied from execution notes.

```bash
uv run pytest tests/test_trellis2_tools.py tests/test_trellis2_forward.py tests/test_trellis2_inference.py
```

```text
36 passed
```

```bash
uv run pytest tests/test_trellis2_dinov3.py tests/test_trellis2_forward.py
```

```text
21 passed
```

```bash
uv run pytest
```

```text
106 passed, 5 skipped
```

```bash
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
```

```text
ready=True
present=2
missing=0
```

```bash
uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/demo-alpha-forward-trace.json
```

```text
completed=('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')
outputs=()
blocker_stage=image-conditioning
operation=MLX DINOv3 transformer block construction
```

```bash
uv run python -c "import tomllib; c=tomllib.loads(open('pyproject.toml','rb').read().decode()); deps='\n'.join(c['project']['dependencies']).lower(); print('torch' in deps, 'torchvision' in deps, 'transformers' in deps, 'huggingface-hub' in deps, 'onnx' in deps); print(deps)"
```

```text
False False False False False
mlx
numpy
pillow>=12.2.0
safetensors
```

```bash
git status --short --ignored weights/dinov3-vitl16-pretrain-lvd1689m outputs/trellis2/demo-alpha-forward-trace.json inputs outputs weights
```

```text
!! inputs/
!! outputs/
!! weights/
```

## Content Checks

PASS.

- Audience: PASS. README text targets local developers/operators by naming exact commands, paths, and current blocker.
- Thesis: PASS. The README section states that DINOv3 assets are explicit local assets and the current boundary is MLX DINOv3 transformer/RoPE construction.
- Content anti-goals: PASS. README does not claim full TRELLIS.2 inference, sparse sampling, decoders, mesh export, RMBG deformable convolution, silent downloads, or real DINOv3 output.
- Channel: PASS. The content is README documentation and uses command/path-oriented developer format.
- Source policy: PASS. Claims are backed by local command output and local files; no external citations or quotes were introduced.
- Factual risk: PASS. Technical claims are verified by fresh local commands in this file.
- Format: PASS. README keeps a concise command and status block under the existing TRELLIS.2 local runtime section.
- Anti-slop scan: PASS. No promotional language, vague attribution, or generic conclusion was added.

## Current Blocker

```text
stage: image-conditioning
operation: MLX DINOv3 transformer block construction
reference: weights/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
reason: DINOv3 config and checkpoint inventory are readable, but MLX forward execution is not implemented for the transformer layers with RoPE position embeddings
next_slice: implement MLX DINOv3 embeddings, RoPE, attention, MLP, and final layer_norm forward pass
```
