# TRELLIS.2 Conditioning Assessment

## Status

Blocked at the first local DINOv3 asset boundary.

## Config Discovery

Command:

```bash
uv run python -c "from mlx_spatial import discover_trellis2_conditioning_config, default_dinov3_root; d=discover_trellis2_conditioning_config('weights/trellis2'); c=d.config; print('ready=', d.ready); print('family=', c.image_model_family); print('model=', c.image_model_name); print('resolution=', c.conditioning_resolution); print('width=', c.expected_feature_width); print('sparse_config=', c.sparse_flow_config_path); print('sparse_checkpoint=', c.sparse_flow_checkpoint_path); print('dino_root=', default_dinov3_root('weights/trellis2', c.image_model_name))"
```

Output:

```text
ready= True
family= DinoV3FeatureExtractor
model= facebook/dinov3-vitl16-pretrain-lvd1689m
resolution= 512
width= 1024
sparse_config= ckpts/ss_flow_img_dit_1_3B_64_bf16.json
sparse_checkpoint= ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
dino_root= weights/dinov3-vitl16-pretrain-lvd1689m
```

## Live Forward Trace

Command:

```bash
uv run python -c "from mlx_spatial import Trellis2InferencePipeline; r=Trellis2InferencePipeline('weights/trellis2').attempt_forward_trace('inputs/trellis2/demo-alpha.webp'); print('completed=', r.completed_stages); print('outputs=', r.outputs); print('blocker_stage=', r.blocker.stage if r.blocker else None); print('operation=', r.blocker.operation if r.blocker else None); print('reason=', r.blocker.reason if r.blocker else None); print('next=', r.blocker.next_slice if r.blocker else None)"
```

Output:

```text
completed= ('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')
outputs= ()
blocker_stage= image-conditioning
operation= local DINOv3 asset validation
reason= facebook/dinov3-vitl16-pretrain-lvd1689m assets are not present at weights/dinov3-vitl16-pretrain-lvd1689m; missing local files: ['config.json', 'model.safetensors']
next= place facebook/dinov3-vitl16-pretrain-lvd1689m MLX-compatible assets under weights/dinov3-vitl16-pretrain-lvd1689m
```

## Interpretation

The forward trace now enters the image-conditioning assessment path and no longer returns the old generic `MLX image feature extraction / conditioning` blocker. The next concrete requirement is local MLX-compatible DINOv3 assets for `facebook/dinov3-vitl16-pretrain-lvd1689m`.

This checkpoint does not download DINOv3 assets and does not use PyTorch, TorchVision, Transformers, Hugging Face Hub, ONNX Runtime, or vendor runtime imports.
