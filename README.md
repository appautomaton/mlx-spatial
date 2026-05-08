# mlx-spatial

MLX-first primitives for 3D and spatial model inference on Apple Silicon.

This repository is currently in environment bootstrap. The first milestone is a root Python package that installs with `uv`, imports `mlx_spatial`, imports `mlx.core`, and runs primitive-focused tests.

## Setup

```bash
uv sync
uv run pytest
```

## Current Scope

- Root package: `mlx_spatial`
- Base runtime dependency: `mlx`
- Test runner: `pytest`
- First smoke primitive: model-neutral shape/grid behavior

## Sparse Grid Primitives

`mlx_spatial.ovoxel` contains the first O-Voxel-inspired coordinate helpers:

- `dense_coordinates(shape)` creates `int32` MLX coordinates with shape `(*shape, ndim)`.
- `flatten_coordinates(coordinates, shape)` maps `(..., ndim)` coordinates to row-major linear indices.
- `unflatten_indices(indices, shape)` maps row-major linear indices back to `(..., ndim)` coordinates.
- `in_bounds_mask(coordinates, shape)` returns a boolean mask for coordinates inside the grid.

These are coordinate/grid helpers, not full TRELLIS.2 inference. They are intentionally model-neutral so they can remain useful for future TRELLIS.2, SAM3D, and Hunyuan-family integrations.

## Sparse Voxel Topology

`mlx_spatial.topology` contains deterministic 3D sparse voxel topology helpers:

- `neighbor_offsets_26()` returns the 26 `(dz, dy, dx)` offsets as `int32`, ordered lexicographically from `[-1, -1, -1]` to `[1, 1, 1]`, excluding `[0, 0, 0]`.
- `adjacency_pairs_26(coordinates, shape)` returns active `(source_index, target_index)` pairs for sparse `(n, 3)` coordinates inside `(depth, height, width)`. Rows are ordered by source input order, then neighbor offset order.
- `grid_edges(shape)` returns dense axis-aligned edge endpoint indices using row-major dense indices. Edge groups are ordered by `z`, then `y`, then `x` axis; starts within each group are row-major.
- `grid_cells(shape)` returns 8-corner dense cell relationships using row-major dense indices. Cell starts are row-major and corners are lexicographic local offsets `(dz, dy, dx)`.

These are topology primitives, not mesh extraction and not TRELLIS.2 inference. They keep outputs as plain MLX arrays so later TRELLIS.2, SAM3D, and Hunyuan-family integrations can share the same coordinate contracts.

## Sparse Convolution Maps

`mlx_spatial.sparse_conv` contains deterministic map-construction helpers for sparse convolution-style operations:

- `kernel_offsets(kernel_size)` returns `int32` `(dz, dy, dx)` offsets for positive odd 3D kernels. Offsets are lexicographic and include the center offset.
- `sparse_conv_map(coordinates, shape, kernel_size=(3, 3, 3))` returns `int32` rows shaped `(m, 3)` with `(target_index, source_index, kernel_index)`.
- `gather_sparse_features(source_features, map_rows)` returns source feature rows ordered exactly like `map_rows`, using each row's `source_index` column.
- `scatter_sparse_features(row_features, map_rows, target_count)` sums per-map-row feature vectors into `target_index` slots and leaves targets with no rows as zero.
- `weighted_sparse_conv(source_features, map_rows, kernel_weights, target_count)` applies a correctness-first weighted sparse convolution reference operation.

Map rows are ordered by target coordinate input order, then kernel offset order. `kernel_index` is the row index into `kernel_offsets(kernel_size)`. Source coordinates use the convention `source = target + kernel_offsets(kernel_size)[kernel_index]`; missing source neighbors are omitted.

Gather/scatter helpers consume the same row contract without interpreting coordinates or kernel offsets. Empty maps return an empty gathered feature matrix with preserved channel count and a zero-filled scatter result with shape `(target_count, channels)`.

`weighted_sparse_conv` uses these shapes:

- `source_features`: `(source_count, in_channels)`
- `map_rows`: `(m, 3)` with `(target_index, source_index, kernel_index)`
- `kernel_weights`: `(kernel_count, in_channels, out_channels)`
- output: `(target_count, out_channels)`

For each map row, the operation computes `source_features[source_index] @ kernel_weights[kernel_index]` and sums the result into `target_index`. Duplicate targets accumulate deterministically. Targets with no rows remain zero.

These maps and feature movement helpers are primitives, not full TRELLIS.2 inference. They do not run transformer blocks, load checkpoints, decode meshes, export GLB files, or download model weights.

## TRELLIS.2 Runtime Readiness

`mlx_spatial.model_assets` contains dependency-free local asset readiness helpers:

- `TRELLIS2_ASSETS` describes the expected local TRELLIS.2 asset layout without storing weights.
- `validate_model_assets(root, manifest=TRELLIS2_ASSETS)` checks a local directory and reports deterministic `present` and `missing` file lists.

The recommended local path is `weights/trellis2/`. The root `weights/` directory is ignored so downloaded checkpoints are not committed.

Validate local assets with:

```bash
uv run python -c "from mlx_spatial import TRELLIS2_ASSETS, validate_model_assets; r = validate_model_assets(TRELLIS2_ASSETS.root_hint); print('ready=', r.ready); print('missing=', list(r.missing))"
```

When a concrete TRELLIS.2 distribution is selected, use the Hugging Face CLI with the same local directory convention:

```bash
huggingface-cli download <trellis2-repo-id> --local-dir weights/trellis2 --local-dir-use-symlinks False
```

The exact repository ID and any include/exclude patterns belong to the checkpoint-specific slice. This package does not require Hugging Face CLI, network access, login, or model weights for default tests.

## TRELLIS.2 Checkpoint Inspection

`mlx_spatial.checkpoint` contains local checkpoint helpers for safetensors files:

- `inspect_checkpoint(path, names=None, prefixes=None)` reports deterministic tensor metadata: name, shape, dtype, and source file.
- `load_checkpoint_tensors(path, names=None, prefixes=None)` loads selected tensors as `mlx.core.array` values.

Place local TRELLIS.2 files under the ignored `weights/trellis2/` directory. For example, a checkpoint path from the manifest is:

```text
weights/trellis2/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
```

Inspect metadata locally with:

```bash
uv run python -c "from mlx_spatial import inspect_checkpoint; infos = inspect_checkpoint('weights/trellis2/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors', prefixes=['model.']); print('\n'.join(f'{i.name} {i.shape} {i.dtype}' for i in infos[:20]))"
```

Load selected tensors into MLX with:

```bash
uv run python -c "from mlx_spatial import load_checkpoint_tensors; tensors = load_checkpoint_tensors('weights/trellis2/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors', names=['model.some_tensor']); print({k: v.shape for k, v in tensors.items()})"
```

Loading requires `names` or `prefixes` so a caller does not accidentally load a full checkpoint. The current checkpoint loader supports `.safetensors` only. PyTorch `.pt`/`.pth` checkpoints, full TRELLIS.2 inference, block parity, decoder execution, mesh extraction, GLB export, and automatic downloads are outside this slice.

## TRELLIS.2 Real-Weight Tools

`mlx_spatial.trellis2` adds TRELLIS.2-specific tooling over the generic asset and checkpoint helpers:

- `validate_trellis2_assets(root="weights/trellis2")` validates the local asset layout.
- `inspect_trellis2_checkpoints(root, checkpoint_paths=None)` inspects configured `.safetensors` checkpoint metadata.
- `inspect_trellis2_probe(root, group)` inspects tensors matched by a named probe group.
- `load_trellis2_probe(root, group)` loads tensors matched by a named probe group into MLX arrays.
- `trellis2_download_command(root="weights/trellis2")` returns the dev-environment Hugging Face CLI command.

The named probe groups are conservative code-level selections for real-weight inspection and loading:

- `sparse-structure-flow`
- `shape-slat-flow`
- `texture-slat-flow`
- `shape-decoder`
- `texture-decoder`

They are informed by `vendors/trellis-mac` using `microsoft/TRELLIS.2-4B` for generation and by the original TRELLIS.2 checkpoint convention of paired `.json` and `.safetensors` files. They are not a model architecture mapping and may need refinement after inspecting real local weights.

Hugging Face CLI is dev tooling, not a runtime dependency. Install dev dependencies with `uv sync --dev`, authenticate if needed, then download manually into the ignored local root:

```bash
uv run mlx-spatial-trellis2 download-command
uv run hf download microsoft/TRELLIS.2-4B --local-dir weights/trellis2
```

Validate and inspect local weights with:

```bash
uv run mlx-spatial-trellis2 validate --root weights/trellis2
uv run mlx-spatial-trellis2 inspect --root weights/trellis2 --checkpoint ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
uv run mlx-spatial-trellis2 probe --root weights/trellis2 shape-decoder --load
```

These tools load safetensors into RAM as `mlx.core.array` values for selected probes. The checkpoint/probe commands do not construct TRELLIS.2 modules, run sparse blocks, run decoders, extract meshes, export GLB files, create a committed real-weight tensor report, load `.pt`/`.pth` files, or download anything automatically during import, tests, or runtime helper calls.

## TRELLIS.2 RMBG Assets

TRELLIS.2 references `briaai/RMBG-2.0` for background removal on RGB or fully opaque input images. This model is a separate gated, non-commercial Hugging Face asset and is not included under `weights/trellis2/`.

The local convention for RMBG assets is:

```text
weights/rmbg2/
```

Validate the local RMBG layout without network access:

```bash
uv run mlx-spatial-trellis2 rmbg-validate --root weights/rmbg2
```

Print the manual download command:

```bash
uv run mlx-spatial-trellis2 rmbg-download-command --root weights/rmbg2
```

Review and accept the model terms before downloading. The package does not download gated RMBG assets during import, tests, validation, or inference attempts.

When `weights/rmbg2` is present, RGB and fully opaque TRELLIS.2 image attempts route through the local MLX RMBG port. The downloaded `briaai/RMBG-2.0` checkpoint validates with 754 tensors under the expected top-level prefixes `bb`, `decoder`, and `squeeze_module`. The local runtime in `mlx_spatial.trellis2_rmbg_forward` runs the Swin-L BiRefNet encoder/decoder, including the ASPPDeformable decoder blocks, directly in MLX and returns the RGBA alpha matte expected by upstream TRELLIS preprocessing.

The RGB preprocessing path can be checked independently:

```bash
uv run python -c "from mlx_spatial import preprocess_trellis2_image; r = preprocess_trellis2_image('inputs/trellis2/image.png', rmbg_root='weights/rmbg2'); print(r.ready, r.blocker, r.image.generated_alpha if r.image else None)"
```

## TRELLIS.2 Inference Attempt

`mlx_spatial.trellis2_inference` defines the first inference-only pipeline surface:

- `Trellis2InferencePipeline(root="weights/trellis2").dry_run(load_probes=False)` validates assets, checkpoint probes, and stage wiring without running model compute.
- `Trellis2InferencePipeline(root).attempt(image_path, load_probes=False)` validates an image path, runs readiness checks, and stops at the first missing MLX compute stage with a structured blocker.
- `Trellis2InferencePipeline(root).attempt_forward_trace(image_path)` runs the alpha-preprocessed path into the MLX-first image-conditioning assessment and, when conditioning metadata is available, into the first sparse-structure boundary.

Blockers include the stage, missing operation, reference location, reason, and recommended next slice. This is intentional: the current pipeline attempt does not fake outputs for unimplemented TRELLIS.2 stages.

Example:

```bash
uv run python -c "from mlx_spatial import Trellis2InferencePipeline; r = Trellis2InferencePipeline('weights/trellis2').dry_run(load_probes=True); print(r.ready); print(r.blocker)"
```

Image attempts now decode real image files. RGBA inputs with useful alpha are resized, foreground-cropped by alpha, and composited to RGB before the pipeline advances. RGB or fully opaque inputs require local RMBG assets; with `--rmbg-root weights/rmbg2`, the MLX BiRefNet port generates the alpha matte before DINO conditioning.

Local sample attempts use ignored paths:

```text
inputs/trellis2/demo-alpha.webp
inputs/trellis2/demo-rgb-background.png
outputs/trellis2/preprocessing/demo-alpha-preprocessed.png
outputs/trellis2/attempts/demo-alpha-attempt.json
outputs/trellis2/attempts/demo-rgb-background-attempt.json
outputs/trellis2/demo-alpha-sparse-preview.obj
```

The current shape local attempt validates `weights/trellis2/`, completes `image-preprocessing-background`, completes MLX DINOv3 `image-conditioning`, completes sparse-structure FlowEuler sampling, runs sparse-structure decoder coordinate extraction, runs shape-SLat sampling, runs the shape decoder to 7-channel FlexiDualGrid fields, and writes a shape OBJ. The textured path is a separate command: `generate-textured` runs texture SLat, the guided texture decoder, Mac-native xatlas/trilinear texture baking, and writes a textured GLB.

The forward-trace path resolves `weights/trellis2/pipeline.json` to:

```text
image_cond_model: DinoV3FeatureExtractor
model_name: facebook/dinov3-vitl16-pretrain-lvd1689m
conditioning_resolution: 512
expected_feature_width: 1024
sparse_flow_checkpoint: ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors
model contract:
  sparse_structure_flow_model
  sparse_structure_decoder
  shape_slat_flow_model_512
  shape_slat_flow_model_1024
  shape_slat_decoder
  tex_slat_flow_model_512
  tex_slat_flow_model_1024
  tex_slat_decoder
samplers:
  sparse_structure_sampler: FlowEulerGuidanceIntervalSampler steps=12 guidance=7.5 rescale_t=5.0 interval=[0.6, 1.0]
  shape_slat_sampler: FlowEulerGuidanceIntervalSampler steps=12 guidance=7.5 rescale_t=3.0 interval=[0.6, 1.0]
  tex_slat_sampler: FlowEulerGuidanceIntervalSampler steps=12 guidance=1.0 rescale_t=3.0 interval=[0.6, 0.9]
normalization:
  shape_slat_normalization: 32 mean values and 32 std values
  tex_slat_normalization: 32 mean values and 32 std values
```

DINOv3 assets are local and explicit:

```bash
uv run mlx-spatial-trellis2 dinov3-validate weights/dinov3-vitl16-pretrain-lvd1689m
uv run mlx-spatial-trellis2 dinov3-download-command weights/dinov3-vitl16-pretrain-lvd1689m
```

The expected local asset root is:

```text
weights/dinov3-vitl16-pretrain-lvd1689m/
  config.json
  model.safetensors
```

The downloaded local checkpoint currently validates as ViT-L/16: hidden size 1024, 24 layers, 415 tensors, and patch embedding shape `(1024, 3, 16, 16)`.

`attempt-forward-trace` remains a diagnostics surface for inspecting the staged model route and blocker metadata without committing to a full export run. The user-facing generation surfaces are now `generate-shape` for shape OBJ and `generate-textured` for textured GLB. Both keep exact-mode blockers and token guards instead of silently writing approximate artifacts.

Fixture and integration tests verify sparse sampling, sparse decoder handoff, 512 and cascade shape-SLat routing, texture-SLat sampling, bounded decoder level iteration, shape subdivision prediction, texture guide-subdivision consumption, output-layer projection, FlexiDualGrid mesh extraction, Mac-native mesh cleanup, xatlas unwrap, sparse-grid trilinear texture baking, and GLB writing.

The sparse structure decoder asset is expected at:

```text
weights/trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json
```

The shape-SLat path loads the configured route, projects sparse token features, applies coordinate-based 3D RoPE, adaptive modulation, exact full or guarded chunked attention, image cross-attention, MLP residuals, output projection, FlowEuler guidance, and shape-SLat normalization. Cascade routes run low-res shape SLat, upsample/deduplicate high-res coordinates, respect `--max-num-tokens`, and decode with the resolved mesh resolution.

The texture-SLat path uses the same MLX sampler pattern. For `1024_cascade`, it selects the local 1024 texture flow checkpoint, evolves `(N, 32)` texture-noise features while concatenating `(N, 32)` shape-SLat features for model input, applies texture-SLat normalization, then hands off to the guided texture decoder and GLB export path.

The combined decode/export path loads local shape and texture decoder checkpoints, runs `from_latent` projections, decoder levels, sparse convolution, layer norm, SiLU MLP, C2S up-blocks, and output-layer projection. The texture decoder consumes shape-predicted subdivision guides. Weighted sparse convolution performs per-kernel matmul and indexed accumulation in MLX. `--decoder-token-limit` remains the resource guard for expanded-token decoder runs.

Export handling is intentionally gated behind decoded mesh/texture availability. `generate-shape` accepts `.obj` targets under ignored `outputs/`; `.glb` targets are rejected with guidance to use `generate-textured`, which requires a `.glb` output and the Mac-native export dependencies.

A coarse debug preview can be generated today from the real sparse-structure coordinates:

```text
outputs/trellis2/demo-alpha-sparse-preview.obj
```

That file is an occupancy OBJ preview from the sparse-structure decoder coordinates. It is useful for checking the image-to-sparse-shape path visually, but it is not the final TRELLIS.2 FlexiDualGrid mesh and has no texture.

Next TRELLIS slices should be concrete and separately verified:

- live parity comparison against official TRELLIS.2/trellis-mac outputs for the same seed and image;
- higher-resolution cascade quality tuning under explicit memory/token guards;
- mesh topology/remeshing parity after texture bake metrics are stable;
- Blender-import and visual metric baselines for regenerated 512 and cascade artifacts.

## SAM 3D Objects Runtime Readiness

`mlx_spatial.sam3d` adds the first SAM 3D Objects surface for native object-centric reconstruction. The official source of truth is the vendored `vendors/sam-3d-objects` quickstart: `image + mask -> output["gs"].save_ply(...)`.

The local SAM3D convention is:

```text
weights/sam-3d-objects/
weights/sam-3d-objects-mlx/
weights/moge-vitl/
weights/moge-vitl-mlx/
```

The checkpoint bundle is gated on Hugging Face. Print the explicit download command with:

```bash
uv run mlx-spatial-sam3d download-command weights/sam-3d-objects
```

Validate and inspect the local bundle without importing PyTorch/CUDA:

```bash
uv run mlx-spatial-sam3d validate weights/sam-3d-objects
uv run mlx-spatial-sam3d inspect weights/sam-3d-objects
```

The inspector supports direct `pipeline.yaml`, `checkpoints/pipeline.yaml`, and `checkpoints/hf/pipeline.yaml` layouts. It parses the official pipeline YAML, resolves referenced config/checkpoint files, and inventories `.safetensors` metadata for the sparse-structure, SLat, and decoder checkpoints.

The official SAM3D release currently ships most model weights as PyTorch zip `.ckpt`/`.pt` files. Keep those out of the inference path by creating a local safetensors mirror:

```bash
uv run mlx-spatial-sam3d convert weights/sam-3d-objects \
  --output-root weights/sam-3d-objects-mlx \
  --moge-root weights/moge-vitl \
  --moge-output-root weights/moge-vitl-mlx \
  --max-archive-gb 16
```

The converter uses `pt-safe-loader` from the dev environment when possible, falls back to a restricted torch-zip state-dict extractor for the official SAM3D/MoGe checkpoint shapes, does not import PyTorch, rewrites the mirrored `pipeline.yaml` to reference `.safetensors`, and writes a `conversion_manifest.json` under the converted root.

The native output target for this milestone is an official-field Gaussian PLY:

```text
x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3
```

The current `reconstruct` command is exact-mode staged. It validates SAM3D and MoGe safetensors assets, runs MLX MoGe pointmap inference, applies official image/mask/pointmap preprocessing, runs SS condition fusion, SS ShortCut flow, SS decoding, SLat condition fusion, SLat FlowMatching, and the SLat gaussian decoder, then writes an official-field binary `gaussians.ply`. If `--glb-output` is requested, it runs the SLat mesh decoder, extracts a FlexiCubes mesh, cleans/simplifies the preview mesh by default, unwraps it with xatlas, bakes Gaussian DC colors into an embedded base-color PNG, and writes a Blender-readable textured `.glb` with normals and UVs. This GLB is still a practical preview mesh; the native high-quality SAM3D artifact remains `gaussians.ply`.

```bash
uv run mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx path/to/image.png \
  --mask path/to/mask.png \
  --moge-root weights/moge-vitl-mlx \
  --output outputs/sam3d/object/gaussians.ply \
  --glb-output outputs/sam3d/object/mesh.glb \
  --stage1-steps 2 \
  --stage2-steps 12 \
  --memory-profile balanced \
  --glb-postprocess cleaned \
  --glb-target-faces 300000 \
  --glb-texture gaussian \
  --glb-texture-size 1024 \
  --glb-xatlas-face-guard 400000 \
  --trace-output outputs/sam3d/object/trace.json
```

Use `--glb-texture none` to write the older vertex-color preview GLB. Textured GLB export requires `--glb-postprocess cleaned`; the xatlas face guard blocks accidental raw-million-face unwraps unless you explicitly raise it. Use `--no-glb-simplify` or `--glb-target-faces 0` only when you need the unsimplified cleaned preview mesh for debugging.

## HY-World PyTorch Parity

The shipped HY-World path stays MLX/NumPy-only. PyTorch, CUDA, `flash-attn`, and `gsplat` are not runtime dependencies. To prevent inaccurate parity claims, HY-World traces include `metadata.parity.numeric_parity_verified=false` until a dev-only reference bundle has been compared.

`mlx-spatial-hyworld2 reconstruct` defaults to `--memory-profile large`, which resizes frames to the official WorldMirror 952px target. Use `safe` or `balanced` only when you are intentionally trading source parity for lower memory use.

Use the optional reference dumper only when you intentionally want to run the vendored PyTorch code as an oracle:

```bash
HYWORLD2_TORCH_REF=1 uv run python tools/hyworld2_dump_torch_reference.py \
  weights/hy-world-2 inputs/hyworld2 \
  --vendor-root vendors/HY-World-2.0 \
  --output outputs/hyworld2/reference-torch.npz
```

The dumper installs local import shims for `flash-attn` and `gsplat` so macOS can capture tensors up to official native splat attributes without running CUDA rasterization. Write an MLX bundle during reconstruction and compare bundles with:

```bash
uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/hyworld2 \
  --output outputs/hyworld2/mlx-run \
  --heads camera,depth,normal,points,gs \
  --parity-output outputs/hyworld2/reference-mlx.npz

uv run mlx-spatial-hyworld2 parity-compare \
  outputs/hyworld2/reference-torch.npz \
  outputs/hyworld2/reference-mlx.npz
```

## Optional Local Resources

This workstation has local framework checkouts under `/Users/ac/dev/ai/ai-frameworks`, including:

- `/Users/ac/dev/ai/ai-frameworks/mlx`
- `/Users/ac/dev/ai/ai-frameworks/pytorch`
- `/Users/ac/dev/ai/ai-frameworks/transformers`

Those paths are optional reference and parity resources for later work. They are not required for base setup or default tests.

Torch, Transformers, Hugging Face download tooling, and vendored model setup are intentionally outside the bootstrap dependency path. They should be added as optional parity or model-download tooling when the first model-specific slice needs them.

Optional PyTorch parity checks are skipped by default. To run parity scaffolding against the local PyTorch checkout, use:

```bash
MLX_SPATIAL_RUN_TORCH_PARITY=1 uv run pytest -m torch_parity
```

The parity path expects the local PyTorch checkout at `/Users/ac/dev/ai/ai-frameworks/pytorch` and is not required for normal development.

## Vendor References

The `vendors/` directory contains reference projects for future MLX inference work. The bootstrap package does not import from or modify vendor projects.
