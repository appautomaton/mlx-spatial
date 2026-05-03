# TRELLIS.2 Preprocessing Attempt

## Local Inputs

- `inputs/trellis2/demo-alpha.webp`: official TRELLIS.2 demo image copied from `vendors/TRELLIS.2/assets/example_image/`; mode `RGBA`, size `(774, 774)`, alpha extrema `(0, 255)`.
- `inputs/trellis2/demo-rgb-background.png`: local RGB background-control image generated from the demo foreground over a synthetic textured background; mode `RGB`, size `(774, 774)`.

Both paths live under ignored `inputs/` and are local attempt assets, not test fixtures.

## Local RMBG Assets

`weights/rmbg2` is present and validated:

```text
ready=True
present=4
missing=0
```

Inventory:

```text
model.safetensors: 844 MB
tensor_count: 754
top_level_prefixes: ('bb', 'decoder', 'squeeze_module')
```

## Commands

```bash
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
uv run python -c "from mlx_spatial import assess_rmbg2_mlx_port; a=assess_rmbg2_mlx_port('weights/rmbg2'); print(a.ready, a.blocker)"
uv run python -c "from mlx_spatial.trellis2_inference import Trellis2InferencePipeline; print(Trellis2InferencePipeline('weights/trellis2').dry_run(load_probes=False).blocker.stage)"
```

The output snapshots generated for inspection are ignored:

```text
outputs/trellis2/preprocessing/demo-alpha-preprocessed.png
outputs/trellis2/attempts/demo-alpha-attempt.json
outputs/trellis2/attempts/demo-rgb-background-attempt.json
```

## Alpha Input Result

`inputs/trellis2/demo-alpha.webp` completed the preprocessing boundary:

```text
completed_stages:
- input-image
- asset-config-validation
- checkpoint-probe-readiness
- image-preprocessing-background

blocker.stage: image-conditioning
blocker.operation: MLX image feature extraction / conditioning
```

This proves the deterministic local path can decode, resize/crop/composite by alpha, and move the attempt boundary from `image-preprocessing-background` to `image-conditioning`.

## RGB Input Result

`inputs/trellis2/demo-rgb-background.png` routed through local RMBG assessment and stopped at the first concrete MLX port blocker:

```text
completed_stages:
- input-image
- asset-config-validation
- checkpoint-probe-readiness

blocker.stage: image-preprocessing-background
blocker.operation: MLX BiRefNet deformable convolution
blocker.reference: weights/rmbg2/birefnet.py:1230-1295
blocker.reason: RMBG-2.0 BiRefNet imports torchvision.ops.deform_conv2d, but mlx.nn has no DeformConv2d implementation
blocker.next_slice: implement or replace deformable convolution for the RMBG-2.0 ASPPDeformable decoder path
```

This is the expected closeout for the current slice: RGB/opaque input no longer receives a generic placeholder blocker, and the next compute gap is named precisely.
