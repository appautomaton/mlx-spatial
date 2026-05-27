# Pixal3D

Pixal3D is TencentARC's projection-conditioned image-to-3D pipeline. In
`mlx-spatial`, Pixal3D is currently an implementation track. The runtime can
use converted local NAF safetensors to build high-resolution projected DINOv3
features without Torch, and can write textured GLB after decoded shape and
texture tensors are available.

Implemented now:

- local asset validation and checkpoint inspection for `TencentARC/Pixal3D`
- MoGe-derived auto-camera through the existing converted MLX MoGe
  pointmap/intrinsics runtime
- Pixal3D manual-FOV camera math matching the upstream inference script as an
  explicit override
- sparse-stage DINOv3 hidden-state extraction through the shared MLX DINOv3
  helper
- projection conditioning from DINOv3 hidden states: global tokens plus
  view-aligned sparse-structure grid features
- Pixal3D `image_attn_mode="proj"` block math in the shared sparse-structure
  and SLat flow boundaries
- sparse-structure FlowEuler probing through the shared MLX sparse flow helper
- sparse decoder coordinate extraction when a compatible sparse decoder
  checkpoint/config is available
- coordinate-indexed 512 shape SLat probing with either explicit NAF-upsampled
  features or MLX NAF-projected features from local converted weights
- shared shape decoder LR-to-HR coordinate upsample, Pixal3D HR token
  quantization, and HR coordinate artifact writing
- coordinate-indexed 1024 shape SLat and texture SLat probing through the same
  explicit-override or MLX NAF-projection path
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

- missing converted NAF weights block NAF-projected stages until
  `weights/naf/naf_release.safetensors` is created locally
- missing converted MoGe weights block auto-camera until
  `weights/sam-3d-objects-mlx/moge/model.safetensors` is present; pass
  `--manual-fov` to use the deterministic override

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

Pixal3D's upstream projection stages use Valeo NAF. Convert the NAF release
checkpoint locally for the Torch-free runtime:

```bash
uv run --group torch-ref python scripts/pixal3d/convert_naf.py \
  --output weights/naf/naf_release.safetensors
```

Pixal3D auto-camera reuses the package's existing converted MLX MoGe root. The
default is `weights/sam-3d-objects-mlx/moge`, normally supplied by the public
`appautomaton/sam-3d-objects-mlx` bundle:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
```

This is a MoGe-derived MLX auto-camera path, not an exact claim of upstream
`Ruicheng/moge-2-vitl` parity.

## Recommended Run

Use the vendored sample image from the shallow upstream checkout:

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade
```

When Pixal3D, MoGe, DINOv3, and NAF assets are present, the current expected
output is:

```text
outputs/pixal3d/sample/
  trace.json
  sparse_projection.npz
  sparse_structure.npz          # written after sparse decoder coordinates are available
  shape_slat_lr.npz             # written after LR NAF projection succeeds
  shape_slat_hr_coordinates.npz # written after compatible shape decoder upsample
  shape_slat_hr.npz             # written after HR NAF projection succeeds
  texture_slat.npz              # written after texture NAF projection succeeds
  shape_decoder_fields.npz      # written after shared FlexiDualGrid shape decode
  texture_decoder_pbr.npz       # written after guided texture PBR voxel decode
  model.glb                     # written after mesh extraction and texture baking
```

If converted MoGe weights are missing, omitted `--manual-fov` returns a
structured `camera-setup` blocker with the MoGe root and memory profile. If the
DINOv3 assets are missing, the CLI returns an `image-conditioning` blocker with
the exact root and download command. If converted NAF weights are missing,
NAF-projected stages return a structured `naf-assets` blocker with the expected
safetensors path and conversion command. When MoGe, DINOv3, NAF, sparse-flow,
and sparse-decoder assets are present, the runtime can write
`sparse_structure.npz`, build coordinate-sampled NAF projections, probe 512
shape SLat, write `shape_slat_lr.npz`, upsample guarded HR coordinates, write
`shape_slat_hr_coordinates.npz`, probe 1024 shape and texture SLat stages as
downstream assets permit, run the shared shape and texture decoders, then write
a Pixal3D-labeled textured GLB.

## Settings

- pipeline type: use `1024_cascade` on Apple Silicon by default
- high-memory mode: `1536_cascade`
- seed: `42`
- max tokens: `49152`, matching the upstream cascade guard
- texture size: `1024`
- GLB face target: `50000`
- xatlas face guard: `auto`
- texture bake backend: `kdtree`
- MoGe root: `weights/sam-3d-objects-mlx/moge`
- MoGe memory profile: `balanced`; alternatives are `safe` and `large`
- manual FOV override: radians, for example `--manual-fov 0.2`
- DINOv3 root: `weights/dinov3-vitl16-pretrain-lvd1689m`
- NAF root: `weights/naf`
- NAF coordinate chunk size: `8192`
- sample image: `vendors/Pixal3D/assets/images/0_img.png`

The cascade planner starts at 1024 or 1536 output resolution, quantizes HR
coordinates onto `resolution / 16`, and steps the HR resolution down by 128
until the sparse token count is below `max_num_tokens` or the 1024 floor is
reached.

## Runtime Boundary

Runtime modules are Torch-free:

- `pixal3d_assets.py`: asset manifest, validation, config parsing, and probes
- `pixal3d_camera.py`: MoGe intrinsics camera conversion, manual-FOV camera
  override, and cascade planning
- `pixal3d_projection.py`: projection grid, FOV projection, feature sampling,
  coordinate-indexed feature selection, and explicit NAF map override support
- `naf.py`: Torch-free converted NAF safetensors loading, image encoder, RoPE,
  and coordinate-sampled neighborhood attention
- `pixal3d_export.py`: intermediate projection, sparse-coordinate, HR
  coordinate, shape SLat, texture SLat, shape decoder, and texture decoder NPZ
  artifact writing plus Pixal3D-labeled GLB writing
- `pixal3d_inference.py`: staged orchestration, MLX MoGe auto-camera handoff,
  trace metadata, export settings, and blockers
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

Dev-only PyTorch reference capture and NAF checkpoint conversion are setup
workflows. Runtime imports remain Torch-free.
