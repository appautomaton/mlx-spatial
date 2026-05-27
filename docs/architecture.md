# Architecture

`mlx-spatial` is organized as pipeline-specific command surfaces over shared spatial primitives.

## Shared Primitives

- `checkpoint.py`: safetensors inspection and selected tensor loading.
- `model_assets.py`: dependency-light asset manifests and local readiness checks.
- `grid.py`, `ovoxel.py`, `topology.py`, `sparse_conv.py`, `spatial_interp.py`: coordinate, topology, sparse feature movement, and interpolation primitives.
- `gs_rasterize.py` and `metal/gs_rasterize.metal`: Gaussian rasterization CPU reference and Metal-backed runtime path.
- `mesh_to_fdg.py`, `export_utils.py`: mesh and export helpers used by generated geometry paths.

These modules should remain model-neutral unless a pipeline contract requires otherwise.

## SAM3D Boundary

Entry point: `mlx_spatial.sam3d:main`, exposed as `mlx-spatial-sam3d`.

Main modules:

- `sam3d_assets.py`: download command, validation, inspection, and conversion to safetensors.
- `sam3d_preprocess.py`: image, mask, crop, pointmap, and official tensor preprocessing.
- `sam3d_moge.py`: MoGe pointmap loading and inference boundary.
- `sam3d_condition.py`, `sam3d_transformer.py`, `sam3d_flow.py`, `sam3d_ss_flow.py`: conditioning and flow model support.
- `sam3d_ss.py`, `sam3d_slat.py`: sparse-structure and SLat stages.
- `sam3d_decoder.py`, `sam3d_gaussian.py`, `sam3d_mesh.py`: decoder and representation outputs.
- `sam3d_export.py`: PLY and GLB export helpers.
- `sam3d_inference.py`: staged orchestration, trace metadata, blockers, quality gates, and final artifacts.

Keep asset conversion, model execution, and export behavior separate. That separation makes source-vs-converted weight audits and reference parity checks tractable.

## TRELLIS.2 Boundary

Entry point: `mlx_spatial.trellis2:main`, exposed as `mlx-spatial-trellis2`.

Main modules:

- `trellis2.py`: CLI command routing and asset helpers.
- `trellis2_preprocess.py`: image preprocessing and optional RMBG alpha generation boundary.
- `trellis2_rmbg.py`, `trellis2_rmbg_forward.py`: local RMBG asset inspection and MLX forward path.
- `trellis2_dinov3.py`, `trellis2_dinov3_forward.py`: DINOv3 asset inspection and conditioning path.
- `trellis2_sparse_structure.py`, `trellis2_slat.py`, `trellis2_forward.py`, `trellis2_decode.py`: staged generation and decode path.
- `trellis2_export.py`, `trellis2_texturing.py`: OBJ/GLB export and texture baking.
- `trellis2_inference.py`: forward traces, shape generation, textured GLB generation, structured blockers, and runtime options.

TRELLIS.2 uses separate roots for model assets, RMBG, and DINOv3. Keep those dependencies explicit in docs and commands.

## HY-World-2.0 Boundary

Entry point: `mlx_spatial.hyworld2:main`, exposed as `mlx-spatial-hyworld2`.

Main modules:

- `hyworld2_assets.py`: WorldMirror asset validation, config resolution, and checkpoint inspection.
- `hyworld2_preprocess.py`: input preparation.
- `hyworld2_camera.py`, `hyworld2_geometry.py`, `hyworld2_grid.py`: camera, geometry, and positional utilities.
- `hyworld2_layers.py`, `hyworld2_transformer.py`, `hyworld2_vit.py`, `hyworld2_heads.py`: MLX model components.
- `hyworld2_inference.py`: staged reconstruction and trace output.
- `hyworld2_export.py`, `hyworld2_sh.py`: export and spherical harmonics helpers.
- `hyworld2_parity.py`: dev-only tensor bundle comparison against reference outputs.

HY-World parity tooling is a development aid. Do not make reference bundles or vendor code part of package artifacts.

## LiTo Boundary

Entry point: `mlx_spatial.lito:main`, exposed as `mlx-spatial-lito`.

LiTo is a checkpoint-backed image-to-3DGS pipeline for Apple Silicon. The default path validates converted safetensors, preprocesses the input image, runs local MLX image conditioning, DiT sampling, voxel/TRELLIS init-coordinate decode, Gaussian Perceiver decode, and exports a Gaussian Splat PLY. Synthetic source-contract smoke generation remains available behind an explicit CLI flag for framework probes. Mesh extraction is not part of the release runtime contract.

Main modules:

- `lito_assets.py`: Apple CDN checkpoint download command, local `.ckpt` to safetensors conversion, validation, and inspection.
- `lito_condition.py`: source-contract image-conditioning adapter used by synthetic fixture tests.
- `lito_tokenizer.py`: source-contract point-cloud and ray feature tokenizer producing `8192 x 32` latent tokens for fixture tests.
- `lito_dit.py`: MLX flow-matching DiT contract, recommended step count, and LiTo memory-profile definitions.
- `lito_render.py`: LF-conditioned Gaussian adapter around `gs_rasterize.py`.
- `lito_inference.py`: end-to-end orchestration, per-stage metrics, memory thresholds, and export surface.
- `lito_real_backend.py`: checkpoint-backed backend boundary for the direct safetensors-to-MLX path. It has no Torch, CUDA, or vendor runtime dependency. It loads/remaps patch-encoder, DiT, voxel, and Gaussian decoder tensors, runs local conditioning and sampling, decodes Gaussian fields, and writes checkpoint-backed PLY only after a valid Gaussian dict returns.

LiTo reuses `gs_rasterize.py`, `metal/gs_rasterize.metal`, `hyworld2_sh.py`, and camera/export helpers where the contracts are model-neutral. CUDA is not a runtime option. Upstream CUDA, PyTorch, and gsplat paths are static architecture references only; optional Torch/MPS parity stays dev-only and non-blocking.

Local source-contract fixtures under `tests/fixtures/lito/` are deterministic synthetic fixtures, not vendor numerical captures. They lock tensor contracts while keeping vendor checkouts, generated Apple samples, and converted weights out of package artifacts.

## MapAnything Boundary

Entry point: `mlx_spatial.mapanything:main`, exposed as `mlx-spatial-mapanything`.

MapAnything is a checkpoint-backed multi-view scene geometry pipeline. The
supported MLX runtime validates the local `facebook/map-anything` assets,
preprocesses image-only scene views with fixed-mapping DINOv2 normalization,
runs the full DINOv2 encoder, fusion normalization, alternating-attention
info-sharing blocks, dense/pose/scale heads, and writes a scene `.npz` tensor
bundle. Mesh and Gaussian-splat export are downstream visualization/export work,
not part of the runtime contract.

Main modules:

- `mapanything.py`: CLI command routing for download, validate, inspect, and generation.
- `mapanything_assets.py`: local HF asset validation, config parsing, checkpoint inspection, and component routing.
- `mapanything_preprocess.py`: image discovery, fixed-mapping resize/crop, and DINOv2 normalization.
- `mapanything_model.py`: DINOv2 encoder and info-sharing MLX forward path.
- `mapanything_heads.py`: fusion norm plus dense, pose, and scale prediction heads.
- `mapanything_geometry.py`: depth, confidence, masks, intrinsics, camera pose, and world-point postprocessing.
- `mapanything_scene.py`: end-to-end scene orchestration, trace metadata, blockers, and `.npz` writing.
- `mapanything_parity.py`: dev-only tensor bundle comparison against Torch reference captures.

The MLX scene output intentionally uses clean top-level keys:
`images`, `depth`, `confidence`, `masks`, `intrinsics`, `camera_poses`,
`extrinsics`, and `world_points`. Torch reference captures from the vendored
pipeline use `scene.*` prefixes for the same semantic tensors. Keep Torch,
TorchVision, UniCeption, OpenCV, and vendor Python imports out of the runtime
dependencies; they belong only to explicit `torch-ref` parity workflows.

## Pixal3D Boundary

Entry point: `mlx_spatial.pixal3d:main`, exposed as `mlx-spatial-pixal3d`.

Pixal3D is a projection-conditioned image-to-3D implementation track. The
runtime currently validates TencentARC Pixal3D assets, derives camera params
from the existing converted MLX MoGe pointmap/intrinsics runtime when manual
FOV is omitted, preserves manual-FOV camera params as an override, runs
sparse-stage DINOv3 hidden-state extraction through the shared MLX DINOv3
helper, builds view-aligned projection conditioning, supports
`image_attn_mode="proj"` in the shared sparse-structure and SLat flow
boundaries, can execute the sparse FlowEuler probe when assets are mapped,
extracts sparse decoder coordinates when compatible sparse decoder assets are
available, builds coordinate-sampled MLX NAF projections from converted local
NAF weights, runs the 512 and 1024 shape SLat probes, upsamples guarded HR
coordinates through the shared shape decoder helper, runs the 1024 texture SLat
probe, runs shared shape/texture decoder execution, reuses shared mesh
extraction and texture baking, records cascade stage plans, and writes
trace/NPZ intermediate artifacts plus a Pixal3D-labeled textured GLB after
decoded tensors are available. The auto-camera path is MoGe-derived through the
available converted runtime and does not claim exact upstream MoGe v2 parity.

Main modules:

- `pixal3d.py`: CLI command routing for download, validate, inspect, probe, and generation.
- `pixal3d_assets.py`: upstream asset manifest, pipeline config parsing, checkpoint probes, and license/access note.
- `pixal3d_camera.py`: upstream-compatible MoGe-intrinsics and manual-FOV camera math, cascade HR token planning, and HR coordinate selection.
- `pixal3d_projection.py`: projection grid, front-view transform, FOV projection, feature sampling, coordinate-indexed feature selection, and explicit NAF map override support.
- `naf.py`: converted NAF safetensors loading, image encoder/RoPE, and coordinate-sampled neighborhood attention without Torch or NATTEN runtime imports.
- `pixal3d_export.py`: intermediate projection, sparse-coordinate, HR-coordinate, shape-SLat, texture-SLat, shape-decoder, texture-decoder, and textured-GLB artifact writers.
- `pixal3d_inference.py`: staged orchestration, trace metadata, memory snapshots, export settings, and blockers.
- `pixal3d_parity.py`: dev-only reference bundle helpers gated away from runtime imports.

Pixal3D reuses `trellis2_sparse_structure.py`, `trellis2_slat.py`,
`trellis2_decode.py`, and `trellis2_export.py` for shared flow, decoder, and
export math, but only through config-gated or caller-labeled paths so existing
TRELLIS.2 checkpoints and GLB metadata stay on their original behavior.

## CLI And Script Split

Package CLIs under `[project.scripts]` are the supported runtime surfaces. Repository scripts under `scripts/` are readable wrappers and maintenance tools that encode recommended settings for users and maintainers.

Add a script only when it is reusable, has argparse help, writes under `outputs/` by default, and does not depend on hidden local session state.

## Local Asset Rule

The package can inspect, validate, convert, and run local model assets, but the source distribution and wheel must not include:

- `.agent/`, `.codex/`, `.claude/`
- `weights/`, `inputs/`, `outputs/`, `vendors/`
- `.venv/`, caches, build outputs, generated probes

The artifact checker in `scripts/packaging/check_release_artifacts.py` enforces this release boundary.
