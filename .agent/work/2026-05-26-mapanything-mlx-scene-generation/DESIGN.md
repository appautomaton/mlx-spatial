# Design: MapAnything MLX Scene Generation

## Runtime Path

```text
inputs/map-anything/desk/*.jpg
  -> preprocess_mapanything_images
  -> MLX DINOv2 giant encoder, 24 blocks
  -> image-only geometry fusion boundary
  -> MLX multi-view alternating attention, 16 blocks
  -> DPT dense head + pose head + scale head
  -> forward scene representation
  -> infer postprocess
  -> scene predictions bundle + lightweight export
```

## Vendored Reference Contract

The reference path is `MapAnything.infer(...)` over image-only views:

```text
load_images(folder)
  -> validate/preprocess views
  -> _configure_geometric_input_config(no geometric inputs)
  -> forward(views)
  -> postprocess_model_outputs_for_inference(...)
```

`forward(views)` performs:

```text
_encode_n_views
  -> _encode_and_fuse_optional_geometric_inputs
  -> scale_token
  -> MultiViewTransformerInput(
       features=encoder_features,
       additional_input_tokens_per_view=encoder_registers,
       additional_input_tokens=scale_token,
     )
  -> info_sharing, with intermediate features at layers 7 and 11
  -> downstream_head(dense, pose, scale)
  -> per-view raw outputs
```

For image-only Desk input, the geometric input configuration becomes a no-op input mode; the model still needs the fusion boundary to behave like the reference, but it should not require ray/depth/pose/calibration tensors.

## Main Data Shapes

```text
images                  [V, 3, H, W]
encoder features/view   [B, 1536, H/14, W/14]
scale token             [B, 1536, 1]
info-sharing output     list[V] of [B, 1536, H/14, W/14]
dense head input        4 tensors: encoder, info[7], info[11], info[final]
dense output            [B*V, C, H, W] where C covers ray dirs, depth, confidence, mask
pose output             [B*V, 7] -> translation + quaternion
scale output            [B]
scene output/view       depth, intrinsics, camera pose, confidence, mask, world points
```

## Dependency Boundary

Runtime code may use `mlx`, `numpy`, `Pillow`, existing local checkpoint helpers, and lightweight standard-library IO. Torch, TorchVision, UniCeption, OpenCV, and `vendors/map-anything` stay in dev-only reference tools/tests.

## Export Boundary

The required generation artifact is a `.npz` scene prediction bundle with typed arrays and metadata. Add `.ply` or `.glb` only if the implementation can do so without pulling a heavy runtime dependency into the package. Export is downstream of correct scene tensors, not a substitute for them.
