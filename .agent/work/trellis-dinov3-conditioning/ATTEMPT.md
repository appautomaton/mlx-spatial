# TRELLIS.2 DINOv3 Conditioning Checkpoint

## Local Asset Root

```text
weights/dinov3-vitl16-pretrain-lvd1689m
```

## Validation Command

```bash
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
```

## Validation Result

```text
ready=True
present=2
missing=0
```

## Download Command

```bash
uv run hf download facebook/dinov3-vitl16-pretrain-lvd1689m config.json model.safetensors --local-dir weights/dinov3-vitl16-pretrain-lvd1689m
```

## Local File Inventory

```text
config.json: 745B
model.safetensors: 1.1G
```

## DINOv3 Inventory

```text
model_type: dinov3_vit
image_size: 224
patch_size: 16
hidden_size: 1024
num_hidden_layers: 24
num_attention_heads: 16
intermediate_size: 4096
num_register_tokens: 4
tensor_count: 415
patch_embedding_key: embeddings.patch_embeddings.weight
patch_embedding_shape: (1024, 3, 16, 16)
observed_layer_count: 24
```

## Slice 5 Probe

```bash
uv run python -c "from mlx_spatial import assess_dinov3_mlx_conditioning; r=assess_dinov3_mlx_conditioning('weights/dinov3-vitl16-pretrain-lvd1689m', expected_feature_width=1024); print(r.blocker)"
```

## Slice 5 Result

```text
stage: image-conditioning
operation: MLX DINOv3 transformer block construction
reference: weights/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
reason: DINOv3 config and checkpoint inventory are readable, but MLX forward execution is not implemented for the transformer layers with RoPE position embeddings
next_slice: implement MLX DINOv3 embeddings, RoPE, attention, MLP, and final layer_norm forward pass
```

## Slice 6 Forward Trace

```bash
uv run mlx-spatial-trellis2 attempt-forward-trace weights/trellis2 inputs/trellis2/demo-alpha.webp --output outputs/trellis2/demo-alpha-forward-trace.json
```

The JSON output is ignored under `outputs/`.

```text
completed=('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')
outputs=()
blocker_stage=image-conditioning
operation=MLX DINOv3 transformer block construction
```

The recorded JSON blocker is:

```text
stage: image-conditioning
operation: MLX DINOv3 transformer block construction
reference: weights/dinov3-vitl16-pretrain-lvd1689m/model.safetensors
reason: facebook/dinov3-vitl16-pretrain-lvd1689m assets at weights/dinov3-vitl16-pretrain-lvd1689m: DINOv3 config and checkpoint inventory are readable, but MLX forward execution is not implemented for the transformer layers with RoPE position embeddings
next_slice: implement MLX DINOv3 embeddings, RoPE, attention, MLP, and final layer_norm forward pass
```

## Interpretation

Slices 1-6 are implemented and verified through the real local DINOv3 asset
boundary. The next blocker is no longer asset availability. It is the MLX
DINOv3 transformer/RoPE forward construction path.
