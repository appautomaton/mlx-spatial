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
python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

Shape-only OBJ script:

```bash
python scripts/trellis2/generate_shape.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-shape-script
```

The script defaults are quality-oriented for Apple Silicon: 512 pipeline,
model-config SLat sampler steps, 1024 texture for GLB, 200k GLB face target,
global xatlas unwrap, and kdtree texture baking. Do not pass `--slat-steps` for
quality runs. Low step counts are only for explicit smoke tests.

## Outputs

Textured runs write:

```text
outputs/trellis2/<run>/
  model.glb
  trace.json
```

Shape-only runs write:

```text
outputs/trellis2/<run>/
  model.obj
  trace.json
```

Keep generated assets under `outputs/`; the export helpers reject arbitrary
output paths outside the ignored output tree.

## Trace

`trace.json` records the selected route, completed stages, outputs, and any
blocker stage, operation, reference, and reason. The script also prints the
effective settings before generation, including pipeline type, sampler steps,
token limits, texture size, face target, unwrap mode, and texture bake backend.

## Export Caveat

Current GLB export is Mac-native and does not use the official CUDA `cumesh` remeshing path. Good object-centric inputs work, but export quality still depends on mesh target, xatlas unwrap, and texture bake behavior.
