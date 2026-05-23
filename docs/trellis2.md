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

This means there is no separate `weights/trellis2-mlx/` bundle today. The boundary is still safetensors plus expected TRELLIS.2 JSON configs; arbitrary PyTorch `.bin` checkpoints are not a supported runtime input.

## Commands

Print download commands:

```bash
uv run mlx-spatial-trellis2 download-command --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-download-command --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
```

Run textured GLB generation:

```bash
python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

Do not pass `--slat-steps` for quality runs. Low step counts are only for explicit smoke tests.

## Export Caveat

Current GLB export is Mac-native and does not use the official CUDA `cumesh` remeshing path. Good object-centric inputs work, but export quality still depends on mesh target, xatlas unwrap, and texture bake behavior.
