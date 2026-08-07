# Pixal3D

Pixal3D is TencentARC's projection-conditioned image-to-3D pipeline. The
`mlx-spatial` implementation is checkpoint-backed and can run through staged
MLX inference to a textured GLB. Its public API and production-equivalence gate
are still under development.

## Current Status

Implemented:

- Pixal3D asset validation and checkpoint inspection
- MLX MoGe auto-camera with an explicit manual-FOV override
- MLX DINOv3 conditioning and projection features
- converted NAF loading and coordinate-sampled projection
- sparse-structure, 512/1024 shape SLat, texture SLat, shape decoder, and
  texture decoder stages
- model-neutral decoded O-Voxel NPZ artifacts and trace output
- one native `mlx_spatial.spatialkit` path from decoded artifacts to GLB

The integrated SpatialKit module contains native and MLX/Metal
QEM edge-collapse paths, a narrow-band UDF double-cover behavior control,
single-layer nonmanifold repair, real global/clustered xatlas, a separate
`xatlas-equivalent-native` UV implementation, trilinear source projection,
native Telea postprocessing, PBR texture baking, and strict-viewer GLB checks.
Pixal3D now uses this path exclusively and fails clearly if SpatialKit cannot
load; it does not silently run an obsolete exporter.

The remaining release boundary is integration and evidence:

- full reference-scale `1M faces / 4096 texture` readiness is not established
- `production_equivalence_ready` remains the authoritative strict gate
- fine structures such as the violin-bow fixture need an explicit preservation
  metric at production settings
- Pixal3D remains an in-development pipeline even when it writes a valid GLB

## Assets

Download and inspect the upstream Pixal3D bundle:

```bash
uv run mlx-spatial-pixal3d download-command weights/pixal3d
uv run hf download TencentARC/Pixal3D --local-dir weights/pixal3d
uv run mlx-spatial-pixal3d validate weights/pixal3d
uv run mlx-spatial-pixal3d inspect weights/pixal3d --limit 5
```

Download the shared DINOv3 assets:

```bash
uv run mlx-spatial-trellis2 dinov3-download-command \
  weights/dinov3-vitl16-pretrain-lvd1689m
uv run hf download facebook/dinov3-vitl16-pretrain-lvd1689m \
  config.json model.safetensors \
  --local-dir weights/dinov3-vitl16-pretrain-lvd1689m
```

Convert the upstream NAF checkpoint for the Torch-free runtime:

```bash
uv run --group torch-ref python scripts/pixal3d/convert_naf.py \
  --output weights/naf/naf_release.safetensors
```

The default auto-camera uses the converted MoGe dependency in the public SAM3D
bundle:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
```

Runtime asset roots:

```text
weights/pixal3d/
weights/dinov3-vitl16-pretrain-lvd1689m/
weights/sam-3d-objects-mlx/moge/
weights/naf/naf_release.safetensors
```

## Recommended Run

Use an object-centric RGB/RGBA image. The example below uses a file from an
ignored local upstream checkout; replace it with your own image when that
checkout is absent.

```bash
uv run python scripts/pixal3d/generate.py \
  vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade
```

When all checkpoint stages complete, the output directory contains:

```text
trace.json
sparse_projection.npz
sparse_structure.npz
shape_slat_lr.npz
shape_slat_hr_coordinates.npz
shape_slat_hr.npz
texture_slat.npz
decoded/
  shape_decoder_fields.npz
  texture_decoder_pbr.npz
model.glb
```

Missing MoGe, DINOv3, NAF, or decoder assets produce a structured blocker in
`trace.json`. `--manual-fov 0.2` bypasses MoGe only when an explicit camera
override is intended.

## Defaults

- pipeline type: `1024_cascade`
- seed: `42`
- max tokens: `49152`
- shape upsample token limit: `1000000`
- shape decoder token limit: `1100000`
- texture decoder token limit: `1100000`
- texture size: `1024`
- GLB face target: `200000`
- GLB exporter: integrated SpatialKit
- MoGe memory profile: `balanced`
- NAF coordinate chunk size: `8192`

Use `uv run python scripts/pixal3d/generate.py --help` for the complete current
surface. The in-development API may change before Pixal3D is promoted to a
stable pipeline.

## SpatialKit Export Boundary

SpatialKit ships in the root distribution and is the only textured GLB export
path used by Pixal3D:

```bash
uv run python scripts/pixal3d/generate.py \
  inputs/pixal3d/object.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade
```

The pipeline writes `decoded/shape_decoder_fields.npz` and
`decoded/texture_decoder_pbr.npz`, releases inference tensors and the MLX
cache, then passes those files to SpatialKit. SpatialKit writes `model.glb` and
`diagnostics.json`. If its native
extension cannot load, the pipeline records a structured blocker before model
inference rather than falling back after expensive work.

Current cached comparisons already reject one tempting optimization:
`xatlas-parallel-spatial` preserves rendered appearance but is slower than
`xatlas-clustered` on both the 48k-face violin and 212k-face main assets, while
introducing explicit spatial partition cuts. It remains available for
Trellis2/SAM3D compatibility and controlled experiments, not as a Pixal3D
default candidate.

For direct decoded-NPZ experiments, `export_decoded_ovoxel_glb` exposes these
lower-level controls:

- `simplify_backend="single-layer-qem"` with `remesh=False`
- `simplify_backend="single-layer-mlx-qem"` with `remesh=False`
- `uv_backend="xatlas-global"` or `uv_backend="xatlas-clustered"`
- `uv_backend="xatlas-parallel-spatial"` with an explicit
  `xatlas_parallel_chunks` value greater than one for measured experiments
- `uv_backend="xatlas-equivalent-native"` for the separate native behavior port
- `texture_postprocess="telea"`
- `quality_preset="reference-target"`

The older plain `qem` path requires UDF remesh plus nonmanifold repair, but the
UDF result is a double cover and is not the current single-surface candidate.
The single-layer paths deliberately reject remesh. A valid artifact does not
imply reference-scale production equivalence; inspect
`quality.production_equivalence` and `result.production_equivalence_ready` in
`diagnostics.json`.

## Development Boundary

Runtime imports remain Torch-free. PyTorch reference capture, NAF conversion,
oracle generation, browser rendering, and full-scale parity runs are maintainer
workflows. Their inputs, outputs, logs, caches, and browser dependencies belong
under one task-specific system temporary directory as described in
`development.md`, not under the repository or in stable runtime documentation.
