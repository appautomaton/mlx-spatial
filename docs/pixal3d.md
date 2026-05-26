# Pixal3D

Pixal3D is TencentARC's projection-conditioned image-to-3D pipeline. In
`mlx-spatial`, Pixal3D is currently an implementation track. The lower-level
runtime can write textured GLB after explicit NAF inputs allow it to reach
decoded shape and texture tensors; the normal CLI still needs the MLX NAF path.

Implemented now:

- local asset validation and checkpoint inspection for `TencentARC/Pixal3D`
- Pixal3D manual-FOV camera math matching the upstream inference script
- sparse-stage DINOv3 hidden-state extraction through the shared MLX DINOv3
  helper
- projection conditioning from DINOv3 hidden states: global tokens plus
  view-aligned sparse-structure grid features
- Pixal3D `image_attn_mode="proj"` block math in the shared sparse-structure
  and SLat flow boundaries
- sparse-structure FlowEuler probing through the shared MLX sparse flow helper
- sparse decoder coordinate extraction when a compatible sparse decoder
  checkpoint/config is available
- coordinate-indexed 512 shape SLat probing when explicit NAF-upsampled
  features are supplied to the lower-level runtime
- shared shape decoder LR-to-HR coordinate upsample, Pixal3D HR token
  quantization, and HR coordinate artifact writing
- coordinate-indexed 1024 shape SLat probing when explicit HR NAF-upsampled
  features are supplied to the lower-level runtime
- coordinate-indexed 1024 texture SLat probing when explicit texture
  NAF-upsampled features are supplied to the lower-level runtime
- shared FlexiDualGrid shape decoder execution after HR shape SLat, writing
  decoded 7-channel shape fields
- shared guided texture decoder execution after texture SLat, writing decoded
  6-channel PBR voxel attributes
- shared FlexiDualGrid mesh extraction, Mac-native texture baking, and
  Pixal3D-labeled textured GLB writing after decoded tensors are available
- cascade stage planning for `1024_cascade` and `1536_cascade`
- trace output, `sparse_projection.npz`, `sparse_structure.npz`, and
  shape/texture/decode intermediate artifacts as each MLX boundary completes

Still blocked:

- normal CLI runs still need an MLX NAF-equivalent feature path before shape
  SLat can run without explicit lower-level test features
- MoGe auto-camera is not wired for Pixal3D; use `--manual-fov`

## Assets

Download the upstream Pixal3D weights manually:

```bash
uv run mlx-spatial-pixal3d download-command weights/pixal3d
```

That prints:

```bash
uv run hf download TencentARC/Pixal3D --local-dir weights/pixal3d
```

Then validate:

```bash
uv run mlx-spatial-pixal3d validate weights/pixal3d
uv run mlx-spatial-pixal3d inspect weights/pixal3d --limit 5
```

The upstream Hugging Face metadata currently identifies the Pixal3D model repo
as MIT-licensed. Respect any Hugging Face access gates and upstream model-card
terms when downloading or redistributing outputs.

Pixal3D image conditioning also needs local DINOv3 ViT-L/16 assets:

```bash
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
uv run hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  config.json model.safetensors \
  --local-dir weights/dinov3-vitl16-pretrain-lvd1689m
```

## Recommended Run

Use the vendored sample image from the shallow upstream checkout:

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade \
  --manual-fov 0.2
```

When Pixal3D and DINOv3 assets are present, the current expected output is:

```text
outputs/pixal3d/sample/
  trace.json
  sparse_projection.npz
  sparse_structure.npz          # written after sparse decoder coordinates are available
  shape_slat_lr.npz             # written only after LR NAF features are supplied
  shape_slat_hr_coordinates.npz # written after compatible shape decoder upsample
  shape_slat_hr.npz             # written only after HR NAF features are supplied
  texture_slat.npz              # written only after texture NAF features are supplied
  shape_decoder_fields.npz      # written after shared FlexiDualGrid shape decode
  texture_decoder_pbr.npz       # written after guided texture PBR voxel decode
  model.glb                     # written after mesh extraction and texture baking
```

If the DINOv3 assets are missing, the CLI returns an `image-conditioning`
blocker with the exact root and download command. If DINOv3 conditioning
completes and sparse-flow assets are mapped, the runtime can execute the sparse
FlowEuler boundary. If the sparse decoder also produces coordinates, the runtime
writes `sparse_structure.npz`. Normal CLI runs then stop at
`shape-projection-conditioning` until MLX NAF features are available. With
explicit lower-level NAF features, the runtime can probe 512 shape SLat, write
`shape_slat_lr.npz`, upsample guarded HR coordinates with the shared shape
decoder helper, write `shape_slat_hr_coordinates.npz`, optionally probe 1024
shape SLat and write `shape_slat_hr.npz`, optionally probe 1024 texture SLat
and write `texture_slat.npz`, run the shared shape and texture decoders, write
`shape_decoder_fields.npz` and `texture_decoder_pbr.npz`, then write a
Pixal3D-labeled textured GLB with the shared mesh extraction and texture baking
helpers.

## Settings

- pipeline type: use `1024_cascade` on Apple Silicon by default
- high-memory mode: `1536_cascade`
- seed: `42`
- max tokens: `49152`, matching the upstream cascade guard
- texture size: `1024`
- GLB face target: `50000`
- xatlas face guard: `auto`
- texture bake backend: `kdtree`
- manual FOV: radians, for example `0.2`
- DINOv3 root: `weights/dinov3-vitl16-pretrain-lvd1689m`
- sample image: `vendors/Pixal3D/assets/images/0_img.png`

The cascade planner starts at 1024 or 1536 output resolution, quantizes HR
coordinates onto `resolution / 16`, and steps the HR resolution down by 128
until the sparse token count is below `max_num_tokens` or the 1024 floor is
reached.

## Runtime Boundary

Runtime modules are Torch-free:

- `pixal3d_assets.py`: asset manifest, validation, config parsing, and probes
- `pixal3d_camera.py`: manual-FOV camera and cascade planning
- `pixal3d_projection.py`: projection grid, FOV projection, feature sampling,
  coordinate-indexed feature selection, and NAF blocker
- `pixal3d_export.py`: intermediate projection, sparse-coordinate, HR
  coordinate, shape SLat, texture SLat, shape decoder, and texture decoder NPZ
  artifact writing plus Pixal3D-labeled GLB writing
- `pixal3d_inference.py`: staged orchestration, trace metadata, export settings,
  and blockers
- `trellis2_dinov3.py`, `trellis2_dinov3_forward.py`: shared MLX DINOv3
  hidden-state extraction
- `trellis2_sparse_structure.py`: shared sparse FlowEuler probing, sparse
  decoder boundary checks, and config-gated Pixal3D projection attention
- `trellis2_decode.py`: shared shape decoder coordinate upsample plus
  full shape/texture decoder execution
- `trellis2_export.py`: shared mesh postprocess, texture baking, and GLB payload
  helpers
- `trellis2_slat.py`: shared SLat flow boundary with config-gated Pixal3D
  projection attention

Dev-only PyTorch reference capture is guarded by `PIXAL3D_TORCH_REF=1` and
belongs in `tools/`, not runtime imports.
