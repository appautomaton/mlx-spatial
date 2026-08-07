# TRELLIS.2

TRELLIS.2 support is image-to-3D inference from a single RGB/RGBA object image. RGB inputs use RMBG-2.0 to produce foreground alpha; RGBA inputs use the alpha channel directly.

## Assets

TRELLIS.2 does not require a SAM3D-style MLX conversion step. The runtime reads the downloaded safetensors and JSON config layout directly:

```text
weights/trellis2/
weights/trellis2/pipeline.json
weights/trellis2/texturing_pipeline.json
weights/trellis2/ckpts/*.safetensors
```

The auxiliary models stay in separate roots:

```text
weights/rmbg2/
weights/dinov3-vitl16-pretrain-lvd1689m/
```

This means there is no separate `weights/trellis2-mlx/` bundle. The runtime boundary is safetensors plus expected TRELLIS.2 JSON configs; arbitrary PyTorch `.bin` checkpoints are not a supported input.

Print download commands:

```bash
uv run mlx-spatial-trellis2 download-command --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-download-command --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
```

## Validation

After downloading the assets, validate each root before running generation:

```bash
uv run mlx-spatial-trellis2 validate --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
```

## Inputs

Use one object-centric RGB/RGBA image:

```text
inputs/trellis2/cup-of-tea.jpg
```

RGBA inputs use the alpha channel directly. RGB inputs use the RMBG root to
estimate the foreground alpha, so validate `weights/rmbg2/` before RGB runs.

## Run

Recommended textured GLB script:

```bash
uv run python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

Shape-only OBJ script:

```bash
uv run python scripts/trellis2/generate_shape.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-shape-script
```

The script defaults are quality-oriented for Apple Silicon: 512 pipeline,
model-config SLat sampler steps, 1024 texture, 200k GLB face target, clustered
xatlas, MLX QEM simplification, and Metal PBR texture baking. Do not pass
`--slat-steps` for quality runs. Low step counts are only for explicit smoke
tests.

The textured pipeline writes decoded O-Voxel shape/PBR NPZ artifacts, releases
inference tensors, and calls the same integrated SpatialKit exporter as
Pixal3D. The current production policy uses the narrow-band remesh, MLX QEM,
clustered xatlas, Telea postprocessing, and Metal texture stages. The remesh is
an Apple-native behavior approximation, not a claim of numerical parity with
upstream cuMesh.

## Outputs

Textured runs write:

```text
outputs/trellis2/<run>/
  model.glb
  diagnostics.json
  artifact-manifest.json
  trace.json
  decoded/
    shape_decoder_fields.npz
    texture_decoder_pbr.npz
```

Shape-only runs write:

```text
outputs/trellis2/<run>/
  model.obj
  trace.json
```

User-facing scripts default to ignored `outputs/` paths. The library accepts an
explicit `.glb` destination, which lets tests and embedding applications use a
task-specific system temporary directory.

## Trace

`trace.json` records the selected route, completed stages, outputs, and any
blocker stage, operation, reference, and reason. The script also prints the
effective settings before generation, including pipeline type, sampler steps,
token limits, texture size, face target, and xatlas chunk count. Stage timings
are independent durations rather than cumulative timestamps.

## Export Caveat

Current GLB export is Apple-native and never executes CUDA code. The upstream
CUDA implementation remains a static algorithm reference for watertightness
and stage semantics; measured performance claims apply only to MLX, Metal, and
native CPU code on Apple Silicon. A valid TRELLIS.2 GLB reports
`artifact_ready`; production-equivalence fields remain false with
`model_reference_profile_unavailable` until a TRELLIS.2-specific reference
profile exists.
