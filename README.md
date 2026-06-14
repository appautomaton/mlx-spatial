# mlx-spatial

[![PyPI](https://img.shields.io/pypi/v/mlx-spatial.svg)](https://pypi.org/project/mlx-spatial/)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://pypi.org/project/mlx-spatial/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/appautomaton/mlx-spatial/actions/workflows/test.yaml/badge.svg)](https://github.com/appautomaton/mlx-spatial/actions/workflows/test.yaml)

**MLX-native 3D and spatial inference for Apple Silicon.** Run modern 3D reconstruction and image-to-3D pipelines locally on [MLX](https://github.com/ml-explore/mlx).

`mlx-spatial` keeps model weights out of the wheel, validates the assets you download, and exposes one clear command path per pipeline that produces inspectable outputs.

> Not a training framework. Does not bundle model weights.

mlx-spatial is an [App Automaton](https://appautomaton.github.io) project. The `appautomaton` org hosts the [code on GitHub](https://github.com/appautomaton/mlx-spatial) and the converted [weights on Hugging Face](https://huggingface.co/appautomaton).

## Capabilities

| Pipeline | Task | Input | Output | Status |
| --- | --- | --- | --- | --- |
| **SAM3D** | object reconstruction | image + object mask | Gaussian PLY (+ optional GLB) | ✅ Stable |
| **TRELLIS.2** | image → textured mesh | object-centric RGB/RGBA | shape OBJ or textured GLB | ✅ Stable |
| **HY-WorldMirror 2.0** | scene reconstruction | scene image or frames | camera, depth, normals, point-cloud PLY | ✅ Stable |
| **LiTo** | image → 3D Gaussian splat | object-centric RGB/RGBA | 3DGS PLY | ✅ Stable |
| **MapAnything** | multi-view scene bundle | related scene views | scene `.npz` (depth, cameras, world points) | ✅ Stable |
| **Pixal3D** | projection-conditioned image → 3D | object-centric RGB/RGBA | trace + NPZ artifacts, textured GLB | 🚧 In development |

**Status:** ✅ Stable = checkpoint-backed, release-ready path · 🚧 In development = partially wired; API and outputs may change.

Pipeline notes:

- **SAM3D** — the strongest object-reconstruction path here; uses the public `appautomaton/sam-3d-objects-mlx` bundle.
- **TRELLIS.2** — textured GLB export works; texture and mesh quality are actively improving.
- **HY-WorldMirror** — release path covers `camera,depth,normal,points`. The optional Gaussian head is not release-ready.
- **LiTo** — outputs a Gaussian-splat PLY (not a mesh); open it in a 3DGS-aware viewer.
- **MapAnything** — outputs a scene `.npz` tensor bundle (not a mesh or splat); uses public `facebook/map-anything` weights.
- **Pixal3D** — projection-conditioned path being wired into MLX; see [docs/pixal3d.md](docs/pixal3d.md) for the current boundary.

## Requirements

- Python 3.13
- Apple Silicon (recommended)
- MLX — installed as a package dependency
- Model weights — downloaded separately into `weights/` (see [Model weights](#model-weights))

## Install

Package consumers:

```bash
uv add mlx-spatial   # or: pip install mlx-spatial
```

Local development from this repo:

```bash
uv sync
uv run pytest -q
```

## Command-line tools

Every pipeline ships a CLI:

```bash
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
uv run mlx-spatial-lito --help
uv run mlx-spatial-mapanything --help
uv run mlx-spatial-pixal3d --help
```

The `scripts/` wrappers are the easiest starting point — they encode recommended settings. See [scripts/README.md](scripts/README.md).

## Model weights

Weights are never committed and never shipped in the wheel. Download them into these ignored local folders:

```text
weights/sam-3d-objects-mlx/
weights/lito-research-mlx/
weights/trellis2/
weights/rmbg2/
weights/dinov3-vitl16-pretrain-lvd1689m/
weights/hy-world-2/
weights/map-anything/
weights/pixal3d/
weights/naf/
```

**Converted MLX bundles** (SAM3D, LiTo) — download, then validate:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx --local-dir weights/sam-3d-objects-mlx
uv run mlx-spatial-sam3d validate weights/sam-3d-objects-mlx

uv run hf download appautomaton/lito-research-mlx --local-dir weights/lito-research-mlx
uv run mlx-spatial-lito validate weights/lito-research-mlx
```

**Direct safetensors** (TRELLIS.2, HY-WorldMirror, MapAnything, Pixal3D) — print the `hf download` command, run it, then validate:

```bash
# 1. print the download commands
uv run mlx-spatial-trellis2 download-command --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-download-command --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
uv run mlx-spatial-hyworld2 download-command weights/hy-world-2
uv run mlx-spatial-mapanything download-command weights/map-anything
uv run mlx-spatial-pixal3d download-command weights/pixal3d

# 2. run the printed commands, then validate
uv run mlx-spatial-trellis2 validate --root weights/trellis2
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
uv run mlx-spatial-hyworld2 validate weights/hy-world-2
uv run mlx-spatial-mapanything validate weights/map-anything
uv run mlx-spatial-pixal3d validate weights/pixal3d
```

Respect the licenses and access terms of the upstream model providers.

## Running the pipelines

The examples below use the `scripts/` wrappers. Most pipelines also emit a `trace.json` describing the run.

### SAM3D — object reconstruction

Provide an image and the exact object mask you want reconstructed:

```bash
python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-script
```

Output: `gaussians.ply`, `trace.json`. Inspect the trace with:

```bash
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-script/trace.json
```

### TRELLIS.2 — textured GLB

Use an object-centric image. RGBA images use their alpha channel directly; RGB images use RMBG to estimate the foreground.

```bash
python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

Output: `model.glb`, `trace.json`.

Defaults are quality-oriented for Apple Silicon: 512 pipeline, model-config sampler steps, 1024 texture, 200k GLB face target, global xatlas unwrap, and kdtree texture baking. Low-step runs are useful for smoke tests but are not representative of output quality.

### HY-WorldMirror — scene reconstruction

Provide a scene image or a directory of scene frames. This pipeline does not take an object mask.

```bash
python scripts/hyworld2/generate_scene.py inputs/sam3d/kidsroom/image.png \
  --output-dir outputs/hyworld2/kidsroom-scene-script
```

Output: `camera_params.json`, `depth/`, `normal/`, `points/points.ply`, `trace.json`.

Uses the verified release path (real Tencent safetensors, `large` memory profile, `camera,depth,normal,points` heads). For frame directories, use `--memory-profile balanced` when `large` hits the attention guard.

### LiTo — image → 3D Gaussian splat

Use an object-centric image, ideally with an alpha mask.

```bash
python scripts/lito/generate.py inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --memory-profile balanced \
  --print-metrics
```

Output: `sample.ply`, `sample.safetensors`.

LiTo writes a Gaussian-splat PLY, not a mesh. Use a 3DGS-aware viewer such as KIRI's Blender 3DGS add-on — Blender's native PLY importer reads the container but does not render the 3DGS fields correctly.

### MapAnything — scene bundle

Provide a directory of related scene views. The Desk example is a two-image scene.

```bash
python scripts/mapanything/generate_scene.py inputs/map-anything/desk \
  --output-dir outputs/mapanything/desk-script
```

Output: `scene.npz`, `trace.json`.

Uses the upstream image-only inference settings: `fixed_mapping` preprocessing, stride `1`, checkpoint-derived patch size, DINOv2 normalization, and mask/edge-mask postprocessing. `scene.npz` mirrors the original Torch scene layout (images, depth, confidence, masks, intrinsics, camera poses, world points) with clean top-level keys, and also records `extrinsics`.

### Pixal3D — in development

By default Pixal3D derives camera parameters from the converted MLX MoGe root; pass `--manual-fov 0.2` only when you want the explicit override.

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade
```

Output begins with `trace.json` and `sparse_projection.npz`, then adds staged NPZ artifacts (`sparse_structure.npz`, shape/texture SLat bundles, decoder fields) and finally `model.glb` as each stage's checkpoint assets are mapped. See [docs/pixal3d.md](docs/pixal3d.md) for the current stage boundary and the `--shape-upsample-token-limit` / `--shape-decoder-token-limit` / `--texture-decoder-token-limit` flags.

## Repository layout

```text
src/mlx_spatial/   package code
scripts/           user and maintainer wrappers
docs/              setup, release, and architecture notes
tests/             unit and parity coverage
weights/           ignored local model assets
inputs/            ignored local sample inputs
outputs/           ignored generated results
vendors/           ignored upstream checkouts
```

## Documentation

| Doc | Contents |
| --- | --- |
| [docs/README.md](docs/README.md) | documentation map and reader contract |
| [scripts/README.md](scripts/README.md) | inference scripts and their defaults |
| [docs/sam3d.md](docs/sam3d.md) | SAM3D setup, inference, quality gates, PLY and coordinate notes |
| [docs/trellis2.md](docs/trellis2.md) | TRELLIS.2 asset layout, scripts, export caveats |
| [docs/hyworld2.md](docs/hyworld2.md) | HY-WorldMirror asset layout, scene inputs, memory profiles |
| [docs/lito.md](docs/lito.md) | LiTo setup, image-to-3DGS CLI, memory profiles, PLY viewing |
| [docs/mapanything.md](docs/mapanything.md) | MapAnything `.npz` schema, parity notes, viewer boundary |
| [docs/pixal3d.md](docs/pixal3d.md) | Pixal3D MLX boundary, recommended settings, blockers |
| [docs/architecture.md](docs/architecture.md) | module map and pipeline boundaries |
| [docs/development.md](docs/development.md) | tests, local asset rules, contribution constraints |
| [docs/model-publishing.md](docs/model-publishing.md) | model bundles and model-card rules |
| [docs/release.md](docs/release.md) | release checklist |

## Releasing (maintainers)

Build and inspect the artifacts before publishing:

```bash
uv run pytest -q
rm -rf dist
uv build
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-*.tar.gz \
  dist/mlx_spatial-*-py3-none-any.whl
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

The build must exclude local weights, generated outputs, inputs, vendor checkouts, caches, and agent state. Publishing is handled by the trusted-publishing workflow in `.github/workflows/workflow.yaml` — do not publish from local shell credentials.

## License

MIT — see [LICENSE](LICENSE).

Built and maintained by [App Automaton](https://appautomaton.github.io). Explore more MLX-native tooling for Apple Silicon — including [mlx-speech](https://github.com/appautomaton/mlx-speech) — on [GitHub](https://github.com/appautomaton) and [Hugging Face](https://huggingface.co/appautomaton).
