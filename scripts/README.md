# Scripts

Repository scripts are readable wrappers around the package CLIs. They encode
recommended defaults for users and keep maintainer-only conversion or audit
workflows separate from normal inference.

## Conventions

- Run repository scripts with `uv run python` after `uv sync`.
- Model weights stay under ignored `weights/` paths.
- User inference inputs and outputs may stay under ignored `inputs/` and
  `outputs/` paths.
- Tests and audits use one task-specific system temporary directory for all
  inputs, outputs, artifacts, logs, caches, and browser dependencies.
- Inference scripts write a trace when the runtime supports it.

## User-Facing Generation Scripts

### SAM3D

Download the public converted bundle, then reconstruct an image and exact
object mask:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
uv run python scripts/sam3d/reconstruct.py \
  inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-script
```

Defaults:

- SAM3D root: `weights/sam-3d-objects-mlx`
- MoGe root: `weights/sam-3d-objects-mlx/moge`
- memory profile: `balanced`
- output: `gaussians.ply` and `trace.json`

Inspect a trace with:

```bash
uv run python scripts/sam3d/inspect_trace.py \
  outputs/sam3d/living-room-script/trace.json
```

### TRELLIS.2

Generate a textured GLB:

```bash
uv run python scripts/trellis2/generate_textured.py \
  inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

Generate a shape-only OBJ:

```bash
uv run python scripts/trellis2/generate_shape.py \
  inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-shape-script
```

Defaults:

- pipeline type: `512`
- seed: `42`
- SLat steps: selected model config
- max tokens: `49152`
- decoder token limit: `1000000`
- textured GLB texture size: `1024`
- textured GLB face target: `200000`
- xatlas parallel chunks: `1`
- texture bake backend: `kdtree`

Low-step commands are smoke checks, not representative quality runs. See
`docs/trellis2.md` for the asset roots and validation commands.

### HY-WorldMirror 2.0

```bash
uv run python scripts/hyworld2/generate_scene.py \
  inputs/sam3d/kidsroom/image.png \
  --output-dir outputs/hyworld2/kidsroom-scene-script
```

Defaults:

- root: `weights/hy-world-2`
- memory profile: `large`
- heads: `camera,depth,normal,points`
- output: camera JSON, depth, normals, point-cloud PLY, and `trace.json`

Use `balanced` when a multi-frame run exceeds the `large` attention guard. The
optional Gaussian head is not part of the release path.

### MapAnything

```bash
uv run python scripts/mapanything/generate_scene.py \
  inputs/map-anything/desk \
  --output-dir outputs/mapanything/desk-script
```

Defaults:

- root: `weights/map-anything`
- resize mode: `fixed_mapping`
- stride: `1`
- patch size: model config, normally `14`
- postprocess: `apply_mask` and `mask_edges`
- output: `scene.npz` and `trace.json`

`scene.npz` is a scene tensor bundle, not a mesh or Gaussian splat.

### Pixal3D

Pixal3D is still in development. With Pixal3D, DINOv3, converted MoGe, and
converted NAF assets present, the MLX path can run through staged NPZ artifacts
to `model.glb`:

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

Defaults:

- pipeline type: `1024_cascade`
- seed: `42`
- max tokens: `49152`
- shape upsample token limit: `1000000`
- shape/texture decoder token limits: `1100000`
- texture size: `1024`
- GLB face target: `50000`
- texture bake backend: `kdtree`
- GLB export backend: `internal`
- MoGe memory profile: `balanced`

The integrated `mlx_spatial.spatialkit` backend already provides opt-in native
and MLX/Metal single-layer QEM, a narrow-band UDF double-cover control, real and
native-equivalent xatlas paths, Telea, Metal texture, and GLB stages. They are
not yet the default production contract. Full 1M/4096 readiness and
fine-structure preservation remain open; use `production_equivalence_ready`
instead of inferring readiness from the existence of a GLB. See
`docs/pixal3d.md` and `docs/spatialkit.md`.

### LiTo

LiTo needs both the converted LiTo bundle and the TRELLIS sparse-structure
decoder used for initialization coordinates:

```bash
uv run hf download appautomaton/lito-research-mlx \
  --local-dir weights/lito-research-mlx
uv run hf download microsoft/TRELLIS-image-large \
  ckpts/ss_dec_conv3d_16l8_fp16.json \
  ckpts/ss_dec_conv3d_16l8_fp16.safetensors \
  --local-dir weights/trellis2/microsoft/TRELLIS-image-large
uv run mlx-spatial-lito validate weights/lito-research-mlx
uv run python scripts/lito/generate.py inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --memory-profile balanced \
  --print-metrics
```

Checkpoint-backed `ply` is the normal output. Checkpoint-backed `splat` is not
implemented; that format is available only in the synthetic
`--source-contract-smoke` path.

## Maintainer Tools

- `scripts/pixal3d/convert_naf.py`: dev-only NAF conversion with the
  `torch-ref` dependency group.
- `scripts/sam3d/inspect_trace.py`: inspect SAM3D trace JSON.
- `scripts/lito/inspect_quality.py`: inspect LiTo Gaussian PLY quality signals.
- `scripts/lito/validate_fixtures.py`: validate small committed LiTo fixtures.
- `scripts/packaging/check_release_artifacts.py`: inspect release archives and
  Git worktree hygiene.

Release artifact checks:

```bash
release_tmp="$(mktemp -d /tmp/mlx-spatial-release.XXXXXX)"
uv build --out-dir "$release_tmp/dist"
uv run python scripts/packaging/check_release_artifacts.py \
  "$release_tmp"/dist/mlx_spatial-*.tar.gz \
  "$release_tmp"/dist/mlx_spatial-*-py3-none-any.whl
uv run python scripts/packaging/check_release_artifacts.py --git-hygiene
```

Heavy parity capture, oracle generation, browser rendering, and full-scale
quality runs are development workflows. Follow `docs/development.md` and keep
all of their temporary data under one task-specific system temporary root.
