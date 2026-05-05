# Texture Pipeline Contract

## Source Of Truth

Upstream TRELLIS.2 image-to-3D texturing is not a separate placeholder paint step. The route is:

```text
shape_slat -> texture SLat sampler with concat_cond=normalized shape_slat
  -> texture decoder guided by shape decoder subdivisions
  -> MeshWithVoxel(mesh vertices/faces + texture voxel coords/attrs)
  -> UV unwrap, texture bake, PBR GLB export
```

Primary references:

- `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:31-40`: model keys loaded by the image-to-3D pipeline.
- `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:391-432`: texture SLat normalizes shape SLat, appends random texture noise, samples with `concat_cond=shape_slat`, then denormalizes texture SLat.
- `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:434-453`: texture decoder call is `tex_slat_decoder(slat, guide_subs=subs) * 0.5 + 0.5`.
- `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:470-486`: decode output couples mesh and texture voxels through `MeshWithVoxel`.
- `vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:517-591`: pipeline type branches.
- `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:195-323`: official CUDA UV/bake/PBR export path.
- `vendors/trellis-mac/backends/texture_baker.py:17-37`, `:115-234`, `:237-261`: Mac fallback shape: xatlas UV unwrap, NumPy rasterization, scipy KDTree sampling, trimesh PBR GLB export.

## Texture Route Metadata

Local discovery must read the model map from `pipeline.json`, not just the static asset manifest. The static `TRELLIS2_ASSETS.required_paths` currently requires the 512 texture SLat and texture decoder paths, while the real pipeline config also requires the 1024 texture SLat route used by all non-512 texture branches.

| pipeline type | conditioning | sparse res | texture SLat model key | texture SLat config/checkpoint | texture decoder config/checkpoint | decode/export res |
| --- | --- | ---: | --- | --- | --- | ---: |
| `512` | `cond_512` | 32 | `tex_slat_flow_model_512` | `ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.{json,safetensors}` | `ckpts/tex_dec_next_dc_f16c32_fp16.{json,safetensors}` | 512 |
| `1024` | `cond_1024` | 64 | `tex_slat_flow_model_1024` | `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.{json,safetensors}` | `ckpts/tex_dec_next_dc_f16c32_fp16.{json,safetensors}` | 1024 |
| `1024_cascade` | shape LR `cond_512`, HR/texture `cond_1024` | 32 | `tex_slat_flow_model_1024` | `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.{json,safetensors}` | `ckpts/tex_dec_next_dc_f16c32_fp16.{json,safetensors}` | 1024 |
| `1536_cascade` | shape LR `cond_512`, HR/texture `cond_1024` | 32 | `tex_slat_flow_model_1024` | `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.{json,safetensors}` | `ckpts/tex_dec_next_dc_f16c32_fp16.{json,safetensors}` | 1536 |

There is no separate 1536 texture SLat checkpoint. The 1536 cascade branch uses the 1024 texture SLat model and texture decoder, then couples the result with the shape decode at 1536 output resolution.

## Shape Outputs Needed By Texture

Texture decoding needs these outputs from the shape path:

- Sparse structure coordinates: reused as shape SLat coordinates and, after cascade when applicable, as the layout for shape SLat features.
- Shape SLat features: normalized with `shape_slat_normalization`, passed to the texture SLat sampler as `concat_cond`, and used to define texture noise width (`texture_flow.in_channels - shape_feature_width`).
- Shape decoder mesh: vertices/faces from FlexiDualGrid are the geometry base.
- Shape decoder subdivisions (`subs`): passed to the texture decoder as `guide_subs`.
- Texture decoder voxels: `coords[:, 1:]`, decoded attributes, `origin=[-0.5, -0.5, -0.5]`, `voxel_size=1/resolution`, and PBR layout `base_color`, `metallic`, `roughness`, `alpha` are assembled into `MeshWithVoxel`.

## Smallest Viable GLB/UV Strategy

The first repo-local GLB path should be fixture-first and Mac-native:

1. Unwrap fixture and live meshes with `xatlas` if dependency review accepts it.
2. Bake texture decoder voxel attributes into a UV atlas with NumPy plus scipy `cKDTree`, following the trellis-mac fallback shape.
3. Export GLB through `trimesh` PBR material support or a minimal internal GLB writer if dependency review rejects `trimesh`.
4. Verify every exported GLB in Blender headless before claiming textured export.

This slice does not approve or add those dependencies. It establishes the contract that later slices must either implement or block on explicitly.

## Local Test Contract

Tests should fail when:

- `pipeline.json` omits `tex_slat_flow_model_512`, `tex_slat_flow_model_1024`, or `tex_slat_decoder`.
- Route selection for `1536_cascade` points to a nonexistent 1536 texture checkpoint instead of the 1024 texture SLat model.
- Texture decoder config/checkpoint paths are not discovered alongside both texture SLat checkpoints.
