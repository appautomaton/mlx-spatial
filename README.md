# mlx-spatial

MLX-native 3D and spatial inference tooling for Apple Silicon.

`mlx-spatial` is a practical runtime package for running a small set of modern
3D reconstruction pipelines locally with MLX. The first release is intentionally
focused: keep weights outside the package, validate the assets you downloaded,
then run clear command-line paths that produce inspectable outputs.

This is not a training framework, and it does not bundle model weights.

## What Works Now

The current package covers three model families:

| Pipeline | Input | Output | Weight setup |
| --- | --- | --- | --- |
| SAM 3D Objects | image + object mask | Gaussian PLY | AppAutomaton MLX bundle |
| TRELLIS.2 | object-centric RGB/RGBA image | shape OBJ or textured GLB | downloaded safetensors directly |
| HY-WorldMirror 2.0 | scene image or image frames | camera, depth, normals, point-cloud PLY | downloaded safetensors directly |

Honest status:

- SAM3D is the best object reconstruction path today. It uses the public
  `appautomaton/sam-3d-objects-mlx` bundle.
- TRELLIS.2 generation works, including textured GLB export. The export path is
  usable, but still an area we keep improving for texture and mesh quality.
- HY-WorldMirror works for scene reconstruction with `camera,depth,normal,points`.
  The optional Gaussian head is not part of the release-ready path yet.

## Install

For local development from this repo:

```bash
uv sync
uv run pytest -q
```

For package consumers after the PyPI release:

```bash
uv add mlx-spatial
```

Requirements:

- Python 3.11+
- Apple Silicon recommended
- MLX installed through the package dependencies
- model weights downloaded separately under `weights/`

## Command Line Tools

The package installs three CLIs:

```bash
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
```

The repository also includes readable script wrappers under `scripts/`. These
are the easiest starting point because they encode the settings we currently
recommend.

## Model Assets

Weights are intentionally not committed and not shipped in the wheel. Keep them
under ignored local folders:

```text
weights/sam-3d-objects-mlx/
weights/trellis2/
weights/rmbg2/
weights/dinov3-vitl16-pretrain-lvd1689m/
weights/hy-world-2/
```

SAM3D uses the converted AppAutomaton runtime bundle:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
uv run mlx-spatial-sam3d validate weights/sam-3d-objects-mlx
```

TRELLIS.2 and HY-WorldMirror do not need SAM3D-style conversion. They load the
downloaded safetensors and JSON configs directly:

```bash
uv run mlx-spatial-trellis2 download-command --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-download-command --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
uv run mlx-spatial-hyworld2 download-command weights/hy-world-2
```

Run the printed `hf download ...` commands, then validate:

```bash
uv run mlx-spatial-trellis2 validate --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
uv run mlx-spatial-hyworld2 validate weights/hy-world-2
```

Respect the licenses and access terms of the upstream model providers. The
Python package only provides runtime code.

## First Runs

### SAM3D Object Reconstruction

Use an image and the exact object mask you want reconstructed:

```bash
python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-script
```

Expected output:

```text
outputs/sam3d/living-room-script/
  gaussians.ply
  trace.json
```

Inspect the trace:

```bash
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-script/trace.json
```

### TRELLIS.2 Textured GLB

Use an object-centric image. RGBA images use their alpha channel directly; RGB
images use RMBG to estimate the foreground:

```bash
python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

Expected output:

```text
outputs/trellis2/cup-of-tea-script/
  model.glb
  trace.json
```

The default settings are quality-oriented for Apple Silicon: 512 pipeline,
model-config sampler steps, 1024 texture, 200k GLB face target, global xatlas
unwrap, and kdtree texture baking. Low-step runs are useful for smoke tests,
but they are not representative of output quality.

### HY-WorldMirror Scene Reconstruction

Use a scene image or a directory of scene frames. This pipeline does not take an
object mask:

```bash
python scripts/hyworld2/generate_scene.py inputs/sam3d/kidsroom/image.png \
  --output-dir outputs/hyworld2/kidsroom-scene-script
```

Expected output:

```text
outputs/hyworld2/kidsroom-scene-script/
  camera_params.json
  depth/
  normal/
  points/points.ply
  trace.json
```

The script uses the verified release path: real Tencent safetensors, `large`
memory profile, and `camera,depth,normal,points` heads.

## Repository Layout

```text
src/mlx_spatial/     package code
scripts/             readable user and maintainer wrappers
docs/                deeper setup, release, and architecture notes
tests/               unit and parity-oriented coverage
weights/             ignored local model assets
inputs/              ignored local sample inputs
outputs/             ignored generated results
vendors/             ignored upstream checkouts
```

## Documentation

- [scripts/README.md](scripts/README.md): recommended inference scripts and their defaults.
- [docs/sam3d.md](docs/sam3d.md): SAM3D setup, inference, quality gates, PLY expectations, and coordinate notes.
- [docs/trellis2.md](docs/trellis2.md): TRELLIS.2 asset layout, no-conversion note, scripts, and export caveats.
- [docs/architecture.md](docs/architecture.md): module map and pipeline boundaries.
- [docs/development.md](docs/development.md): tests, local asset rules, and contribution constraints.
- [docs/model-publishing.md](docs/model-publishing.md): AppAutomaton-first model bundles and model-card rules.
- [docs/release.md](docs/release.md): `0.0.1` release checklist.

## Release Hygiene

Before publishing, build and inspect the artifacts:

```bash
uv run pytest -q
uv build
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-0.0.1.tar.gz \
  dist/mlx_spatial-0.0.1-py3-none-any.whl
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

The build must not include local weights, generated outputs, inputs, vendor
checkouts, caches, or agent state.

Publishing is handled by the trusted-publishing workflow in
`.github/workflows/workflow.yaml`. Do not publish from local shell credentials.
