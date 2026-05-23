# Scripts

Scripts in this directory are stable examples and maintenance tools. They are
kept small so users can read the exact command defaults before running an
inference job.

## Conventions

- Outputs go under `outputs/`.
- Model weights stay under `weights/` and are never committed.
- Inference scripts write a trace file when the runtime supports it.
- Scripts use recommended quality settings by default and do not expose
  quality-gate bypasses.

## SAM3D

Run SAM 3D Objects reconstruction with MLX safetensors:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
```

```bash
python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-script
```

The script expects the public AppAutomaton runtime bundle at `weights/sam-3d-objects-mlx`. Local conversion from Meta's gated source repo is a maintainer/audit workflow, not the default user path.

Defaults:

- SAM3D root: `weights/sam-3d-objects-mlx`
- MoGe root: `weights/sam-3d-objects-mlx/moge`
- memory profile: `balanced`
- quality diagnostics: recorded in `trace.json` when the runtime reports non-nominal output metrics
- mask selection: the exact `--mask` path is used

Inspect a trace:

```bash
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-script/trace.json
```

## TRELLIS.2

Run TRELLIS.2 textured GLB generation with MLX safetensors:

```bash
python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-script
```

This is the production-like Apple Silicon path: 512 pipeline, model-config SLat steps, 1024 texture, 200k GLB face target, global xatlas unwrap, and kdtree texture bake. It can take several minutes. For smoke tests, make the quality tradeoff explicit:

```bash
python scripts/trellis2/generate_textured.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-smoke \
  --slat-steps 1 \
  --texture-size 512 \
  --glb-target-faces 20000 \
  --xatlas-parallel-chunks 0
```

Run shape-only OBJ generation:

```bash
python scripts/trellis2/generate_shape.py inputs/trellis2/cup-of-tea.jpg \
  --output-dir outputs/trellis2/cup-of-tea-shape-script
```

TRELLIS.2 image-to-3D inputs:

- input image: RGB or RGBA. RGBA uses the alpha channel directly; RGB uses RMBG to produce foreground alpha.
- TRELLIS.2 root: `weights/trellis2`
- RMBG root: `weights/rmbg2`
- DINOv3 root: `weights/dinov3-vitl16-pretrain-lvd1689m`
- output: `.glb` for `generate_textured.py`, `.obj` for `generate_shape.py`

No TRELLIS.2 conversion step is required. These scripts expect the downloaded TRELLIS.2 safetensors plus JSON configs under `weights/trellis2/`; they do not require a separate `*-mlx` converted model bundle.

Script defaults:

- pipeline type: `512`
- seed: `42`
- SLat steps: model config default, currently `12`
- max tokens: `49152`
- decoder token limit: `1000000`
- textured GLB texture size: `1024`
- textured GLB face target: `200000`
- xatlas parallel chunks: `1`
- texture bake backend: `kdtree`
- trace output: `trace.json` next to the generated asset

## HY-WorldMirror 2.0

Run HY-WorldMirror scene reconstruction with MLX safetensors:

```bash
python scripts/hyworld2/generate_scene.py inputs/sam3d/kidsroom/image.png \
  --output-dir outputs/hyworld2/kidsroom-scene-script
```

HY-WorldMirror inputs:

- input: a single RGB/RGBA scene image, or a directory of image frames
- masks: not used by this pipeline
- root: `weights/hy-world-2`
- output: camera JSON, depth maps, normal maps, point-cloud PLY, and `trace.json`

No HY-WorldMirror conversion step is required. The script expects Tencent's downloaded safetensors at `weights/hy-world-2/HY-WorldMirror-2.0/model.safetensors` plus `config.json`.

Script defaults:

- memory profile: `large`
- target size: official 952px path through the runtime memory profile
- heads: `camera,depth,normal,points`
- fixture tensors: disabled
- optional GS head: intentionally not exposed by this script because Gaussian preview/export is not release-ready

For frame directories, `large` preserves the official 952px path but can exceed
the attention guard as frame count grows. Use `--memory-profile balanced` for a
more reliable multi-frame run.

## Packaging

Check release artifacts for blocked local paths:

```bash
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-0.0.1.tar.gz \
  dist/mlx_spatial-0.0.1-py3-none-any.whl
```

Check generated/local files in git status:

```bash
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

## Deferred Scripts

- Full source-vs-converted SAM3D weight audit: useful, but it requires the
  original gated checkpoints and PyTorch. Keep the current audit output with the
  model bundle; do not make it a casual user command.
- Multi-output quality summarization: defer until the trace schema stabilizes
  across SAM3D, TRELLIS.2, and HY-World.
