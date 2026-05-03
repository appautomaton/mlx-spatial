# TRELLIS.2 Forward Trace Attempt

## Local Input

- image: `inputs/trellis2/demo-alpha.webp`
- TRELLIS.2 root: `weights/trellis2`
- generated evidence: `outputs/trellis2/forward-trace/demo-alpha-forward-trace.json`

The evidence JSON is under ignored `outputs/`.

## Conditioning Config

```text
family= DinoV3FeatureExtractor
model= facebook/dinov3-vitl16-pretrain-lvd1689m
resolution= 512
expected_feature_width= 1024
sparse_config= ckpts/ss_flow_img_dit_1_3B_64_bf16.json
sparse_checkpoint= ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
dino_root= weights/dinov3-vitl16-pretrain-lvd1689m
```

## Command

```bash
uv run python -c "from mlx_spatial import Trellis2InferencePipeline; r=Trellis2InferencePipeline('weights/trellis2').attempt_forward_trace('inputs/trellis2/demo-alpha.webp'); print(r.completed_stages); print(r.blocker.stage if r.blocker else None); print(r.blocker.operation if r.blocker else None)"
```

## Result

```text
completed= ('input-image', 'asset-config-validation', 'checkpoint-probe-readiness', 'image-preprocessing-background')
outputs= ()
blocker_stage= image-conditioning
operation= local DINOv3 asset validation
reason= facebook/dinov3-vitl16-pretrain-lvd1689m assets are not present at weights/dinov3-vitl16-pretrain-lvd1689m; missing local files: ['config.json', 'model.safetensors']
```

## Interpretation

The real local alpha attempt no longer reports the old generic `MLX image feature extraction / conditioning` blocker. It now enters the forward-trace conditioning layer, parses the local TRELLIS.2 config, prepares the image tensor path, and stops at a concrete DINOv3 local asset requirement.

The next real compute blocker is local DINOv3 asset availability for `facebook/dinov3-vitl16-pretrain-lvd1689m`.
