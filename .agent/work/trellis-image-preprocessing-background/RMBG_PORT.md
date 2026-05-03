# RMBG Port Attempt

## Status

Blocked at the first required MLX operation for the downloaded RMBG-2.0 architecture.

## Command Tried

```bash
uv run hf download briaai/RMBG-2.0 --local-dir weights/rmbg2 model.safetensors config.json BiRefNet_config.py birefnet.py
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
uv run python -c "from mlx_spatial import assess_rmbg2_mlx_port; a=assess_rmbg2_mlx_port('weights/rmbg2'); print(a)"
```

## Result

```text
ready=True
present=4
missing=0

ready= False
tensor_count= 754
prefixes= ('bb', 'decoder', 'squeeze_module')
blocker_stage= image-preprocessing-background
operation= MLX BiRefNet deformable convolution
reason= RMBG-2.0 BiRefNet imports torchvision.ops.deform_conv2d, but mlx.nn has no DeformConv2d implementation
```

## Blocker

- stage: `image-preprocessing-background`
- operation: `MLX BiRefNet deformable convolution`
- reference: `weights/rmbg2/birefnet.py:1230-1295`
- reason: `RMBG-2.0` uses `torchvision.ops.deform_conv2d` in the `ASPPDeformable` decoder path, while this MLX runtime has no `mlx.nn.DeformConv2d` implementation.
- next_slice: implement or replace deformable convolution for the RMBG-2.0 `ASPPDeformable` decoder path.

## Required Local Files

```text
weights/rmbg2/model.safetensors
weights/rmbg2/config.json
weights/rmbg2/BiRefNet_config.py
weights/rmbg2/birefnet.py
```

## Local Asset Inventory

- `weights/rmbg2/model.safetensors`: 844 MB
- tensor count: 754
- top-level checkpoint prefixes: `bb`, `decoder`, `squeeze_module`
