# Pixal3D

Pixal3D is TencentARC's projection-conditioned image-to-3D pipeline. In
`mlx-spatial`, Pixal3D is currently an implementation track, not a full
production GLB path yet.

Implemented now:

- local asset validation and checkpoint inspection for `TencentARC/Pixal3D`
- Pixal3D manual-FOV camera math matching the upstream inference script
- projection conditioning from DINOv3 hidden states: global tokens plus
  view-aligned sparse-structure grid features
- Pixal3D `image_attn_mode="proj"` block math in the shared sparse-structure
  and SLat flow boundaries
- cascade stage planning for `1024_cascade` and `1536_cascade`
- trace output and `sparse_projection.npz` intermediate artifacts when the
  sparse projection boundary completes

Still blocked:

- Pixal3D DINOv3 image hidden-state extraction is not wired into the runtime
- full sparse-structure sampling, sparse decoder handoff, shape/texture SLat
  execution, and textured GLB export are not release-ready
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

## Recommended Run

Use the vendored sample image from the shallow upstream checkout:

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade \
  --manual-fov 0.2
```

Current expected output:

```text
outputs/pixal3d/sample/
  trace.json
```

When a caller supplies DINOv3 hidden states directly to the Python runtime, the
pipeline can also write:

```text
outputs/pixal3d/sample/
  sparse_projection.npz
```

The CLI intentionally returns a structured blocker until image hidden-state
extraction and downstream checkpoint execution are wired.

## Settings

- pipeline type: use `1024_cascade` on Apple Silicon by default
- high-memory mode: `1536_cascade`
- seed: `42`
- max tokens: `49152`, matching the upstream cascade guard
- manual FOV: radians, for example `0.2`
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
  and NAF blocker
- `pixal3d_export.py`: intermediate NPZ artifact writing
- `pixal3d_inference.py`: staged orchestration, trace metadata, and blockers
- `trellis2_sparse_structure.py`, `trellis2_slat.py`: shared flow boundaries
  with config-gated Pixal3D projection attention

Dev-only PyTorch reference capture is guarded by `PIXAL3D_TORCH_REF=1` and
belongs in `tools/`, not runtime imports.
