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

## User-Facing Generation Scripts

### SAM3D

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

The script expects the public `appautomaton/sam-3d-objects-mlx` runtime bundle at `weights/sam-3d-objects-mlx`. Local conversion from Meta's gated source repo is a maintainer/audit workflow, not the default user path.

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

### TRELLIS.2

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

### HY-WorldMirror 2.0

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

### MapAnything

Run Meta MapAnything image-only multi-view scene generation with MLX
safetensors:

```bash
uv run hf download facebook/map-anything \
  --local-dir weights/map-anything
```

```bash
python scripts/mapanything/generate_scene.py inputs/map-anything/desk \
  --output-dir outputs/mapanything/desk-script
```

MapAnything inputs:

- input: a single image or a directory of related scene-view images
- masks: not provided by the user; the model predicts final masks
- root: `weights/map-anything`
- output: `scene.npz` and `trace.json`

No conversion step is required. The script expects Meta's downloaded
`config.json` and `model.safetensors` under `weights/map-anything/`.

Script defaults:

- resize mode: `fixed_mapping`, matching the upstream image-only inference path
- stride: `1`
- patch size: read from `weights/map-anything/config.json`, normally `14`
- normalization: DINOv2 image mean/std
- postprocess: `apply_mask` and `mask_edges`
- trace output: `trace.json` next to `scene.npz`
- runtime Torch dependency: none

The scene bundle contains `images`, `depth`, `confidence`, `masks`,
`intrinsics`, `camera_poses`, `extrinsics`, and `world_points`. It is not a mesh
or Gaussian Splat PLY. A colored point cloud or inline viewer can be derived
from the bundle for inspection, but that is a downstream visualization step.

### Pixal3D

Run TencentARC Pixal3D generation with local safetensors:

```bash
uv run hf download TencentARC/Pixal3D \
  --local-dir weights/pixal3d
```

```bash
uv run --group torch-ref python scripts/pixal3d/convert_naf.py \
  --output weights/naf/naf_release.safetensors
```

```bash
python scripts/pixal3d/generate.py vendors/Pixal3D/assets/images/0_img.png \
  --root weights/pixal3d \
  --dino-root weights/dinov3-vitl16-pretrain-lvd1689m \
  --moge-root weights/sam-3d-objects-mlx/moge \
  --naf-root weights/naf \
  --output-dir outputs/pixal3d/sample \
  --pipeline-type 1024_cascade
```

Pixal3D inputs:

- input: a single object-centric RGB/RGBA image
- root: `weights/pixal3d`
- DINOv3 root: `weights/dinov3-vitl16-pretrain-lvd1689m`
- MoGe root: `weights/sam-3d-objects-mlx/moge`
- NAF root: `weights/naf`
- sample image: `vendors/Pixal3D/assets/images/0_img.png`
- output: `trace.json`; completed MLX intermediate boundaries write
  `sparse_projection.npz`, after sparse decoding `sparse_structure.npz`, and
  after MLX NAF projection boundaries `shape_slat_lr.npz`,
  `shape_slat_hr_coordinates.npz`, `shape_slat_hr.npz`, and
  `texture_slat.npz`; compatible decoder assets then write
  `shape_decoder_fields.npz` and `texture_decoder_pbr.npz`; when decoded
  tensors are available the runtime writes `model.glb`

Script defaults:

- pipeline type: `1024_cascade`, the recommended Apple Silicon default
- seed: `42`
- max tokens: `49152`
- shape upsample token limit: `1000000`
- shape decoder token limit: `1100000`
- texture decoder token limit: `1100000`
- texture size: `1024`
- GLB face target: `50000`
- xatlas face guard: `auto`
- xatlas parallel chunks: `0`
- texture bake backend: `kdtree`
- GLB export backend: `internal`; use `--glb-export-backend spatialkit` only
  when the optional `mlx_spatialkit` package is installed or available from the
  local companion package
- MoGe memory profile: `balanced`
- NAF coordinate chunk size: `8192`
- manual FOV: optional `--manual-fov 0.2` overrides MoGe auto-camera and does
  not run MoGe
- missing converted MoGe weights produce a structured `camera-setup` blocker
  when `--manual-fov` is omitted; missing converted NAF weights produce a
  structured `naf-assets` blocker; with complete Pixal3D, DINOv3, MoGe, and NAF
  assets, the MLX path runs through 512/1024 shape SLat, texture SLat, shape and
  texture decode, and `model.glb` export
- spatialkit export writes `model.glb` and a `diagnostics.json` sidecar from
  decoded `shape_decoder_fields.npz` and `texture_decoder_pbr.npz`; if the
  optional package is missing, the Pixal3D script records the fallback and uses
  the internal writer. Native Metal texture fallback/fill and paired-triangle
  face-atlas packing are enabled. Default preview exports use `spatial-cluster`
  with `quality_tier=geometry_aware_preview`; decoded NPZ validation can call
  the companion API with `quality_preset="reference-target"` to select the
  native `topology-aware` simplifier and record production threshold pass/fail
  details against the checked-in Pixal3D reference trace. The current
  reference-target heavy fixture passes the measured spatialkit production gate,
  and the 4096 texture-size gate passes after atlas-size-aware Metal fallback
  and native dilation budgets are applied. Explicit 1M/4096 upstream-style
  exports report a separate `quality.upstream_export_settings` gate; passing
  that gate removes only the 1M-face setting deferral. The opt-in native-chart
  backend also has a 1M/4096 heavy gate where upstream-setting readiness and
  native-chart quality readiness pass, while 1024-reference visual comparison
  stays visibly mismatched on face count and texture size. This is not full
  upstream xatlas charting, xatlas chart equivalence, or CUDA/cuMesh remesh
  parity.
  Arbitrary non-atlas UV bakes now use `metal-uv-binned-nearest`, a bounded
  UV-space face-bin Metal lookup that reports bin grid and candidate diagnostics
  instead of scanning every face per texel. The current Pixal3D export still
  uses the paired-triangle face-atlas fast path. `mlx-spatialkit` exposes
  opt-in native chart-candidate UV generation for focused testing, and direct
  decoded-NPZ conversion can pass `uv_backend="native-chart"` to
  `mlx_spatialkit.export_pixal3d_glb`. The script default remains face-atlas,
  and xatlas chart parity remains a later parity boundary. Native chart UVs use
  deterministic oversized-chart splitting, local-frame/PCA projection, a
  bounded 19-candidate rotation search at 5-degree steps, bounded low-fill
  chart splitting, and aspect-aware shelves with a tighter backend default
  padding of `0.005`. They report oversized and low-fill split counts, projection
  candidate count and step, chart rect fill, padding source, and packing
  efficiency. When the Pixal3D xatlas reference trace is available, diagnostics
  also report `quality.xatlas_chart_parity` with reference chart count and
  utilization, native chart occupancy, ratios, and `parity_ready=false`; this
  measures the gap without adding xatlas to the spatialkit package runtime.
  The bounded low-fill splitter uses a higher fill target and one extra split
  depth to improve native chart fill toward those xatlas metrics without
  claiming xatlas equivalence. Eligible low-fill charts evaluate both local
  centroid split axes plus three fixed split positions, and diagnostics report
  the bounded partition-search count.
  Non-atlas UV bakes also use bounded native UV-surface fill and
  report raw exact coverage, surface-filled texels, and final visible coverage
  separately. Chart diagnostics separate `artifact_ready` from `quality_ready`;
  the current real fixture clears scalar native-chart coverage while xatlas
  chart parity remains deferred. With `quality_preset="reference-target"` and
  `uv_backend="native-chart"`, the real fixture also passes production and
  deterministic visual-comparison gates; this does not remove xatlas or
  1M/4096 upstream-setting parity boundaries. With explicit
  `target_faces=1000000` and `texture_size=4096`, the same backend passes its
  upstream-setting gate and leaves only xatlas chart parity deferred.
  Native spatialkit GLBs include generated normals and split large meshes into
  chunk-local uint16-indexed primitives. The diagnostics sidecar records
  `quality.glb_viewer_compatibility` for parseability, PBR texture presence,
  normals, uint16 indices, local index bounds, and large-mesh chunking. This is
  strict-viewer hardening for tools such as macOS Preview/Quick Look, not a
  decoded-output or xatlas parity change.
  When the reference GLB is available, reference-target export also writes a
  `visual_parity/` sidecar with machine-readable GLB/texture comparison metrics
  plus extracted candidate/reference base-color PNG previews. The checked-in
  reference GLB is 1024, so a 4096 candidate is expected to report texture-size
  mismatch while still passing coverage. Default deferred visual parity
  boundaries are limited to xatlas chart parity and 1M-face export-setting
  parity; the 1M boundary is removed for explicit 1M/4096 exports only after
  upstream-setting readiness passes, including explicit native-chart 1M/4096
  exports.
  The `diagnostics.json` file also includes observed process RSS peaks per
  export stage from `ps` and `resource.getrusage`; this is host-process
  telemetry, not full system pressure or Metal allocator accounting.
  For dev-only browser visual proof, install Playwright/Three under
  `/tmp/mlx-spatialkit-render-deps` and run
  `scripts/spatialkit/render_glb_visual_parity.cjs` against the generated GLB
  and checked-in reference GLB. It writes screenshot/JSON/HTML artifacts under
  `visual_parity/browser_render/` and does not add package runtime deps.

### LiTo

Run checkpoint-backed Apple LiTo image-to-3DGS generation:

```bash
python scripts/lito/generate.py inputs/lito/sample.png \
  --weights-root weights/lito-research-mlx \
  --output outputs/lito/sample.ply \
  --memory-profile balanced \
  --print-metrics
```

The script uses the upstream-recorded LiTo defaults from `LITO_RECOMMENDED_*`
and writes a 3D Gaussian Splat PLY plus a safetensors sidecar when `--format ply`
is selected. Checkpoint-backed `--format splat` is not implemented; keep `ply`
for real LiTo runs. Use `--memory-profile safe` only for lower-memory smoke/debug
runs, and pair it with `--source-contract-smoke` when testing the synthetic
framework probe path.

## Quality-Inspection And Fixture Tools

- `scripts/sam3d/inspect_trace.py`: inspect SAM3D trace JSON after a run.
- `scripts/lito/inspect_quality.py`: inspect LiTo Gaussian PLY quality signals.
- `scripts/lito/validate_fixtures.py`: validate LiTo fixture files used by tests.
- `scripts/lito/write_contract_fixtures.py`: write synthetic LiTo contract
  fixtures for maintainer checks.

## Packaging

Check release artifacts for blocked local paths:

```bash
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-*.tar.gz \
  dist/mlx_spatial-*-py3-none-any.whl
```

Check generated/local files in git status:

```bash
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

## Deferred Maintainer Scripts

- Full source-vs-converted SAM3D weight audit: useful, but it requires the
  original gated checkpoints and PyTorch. Keep the current audit output with the
  model bundle; do not make it a casual user command.
- Multi-output quality summarization: defer until the trace schema stabilizes
  across SAM3D, TRELLIS.2, and HY-World.
