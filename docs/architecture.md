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

LiTo is currently a source-contract image-to-3DGS bring-up pipeline. It validates the MLX-compatible stage boundaries, CLI, metrics, memory limits, and Gaussian export surface while the full checkpoint-backed parity path is still future work. It produces Gaussian splat outputs such as PLY; mesh extraction is not part of the current runtime contract.

Main modules:

- `lito_assets.py`: Apple CDN checkpoint download command, local `.ckpt` to safetensors conversion, validation, and inspection.
- `lito_condition.py`: image-conditioning source-contract adapter for the upstream `SpatialDinov2` boundary.
- `lito_tokenizer.py`: point-cloud and ray feature tokenizer contract producing `8192 x 32` latent tokens.
- `lito_dit.py`: MLX flow-matching DiT contract, recommended step count, and LiTo memory-profile definitions.
- `lito_render.py`: LF-conditioned Gaussian adapter around `gs_rasterize.py`; the default Risk F decision is adapter-only.
- `lito_inference.py`: end-to-end orchestration, per-stage metrics, memory thresholds, and export surface.
- `lito_real_backend.py`: checkpoint-backed backend boundary for the direct safetensors-to-MLX path. It has no Torch, CUDA, or vendor runtime dependency, records header-only architecture inventory for the real converted weights, selectively loads/remaps real DiT and Gaussian decoder tensors, runs the real Gaussian coordinate/Fourier point-query stem and shape/color output heads for caller-supplied init coordinates or decoder query latents, ports LiTo Gaussian `decode_gs` equations, normalizes LiTo Gaussian dictionaries, and writes checkpoint-backed gsplat-style PLY only after a valid real Gaussian dict returns.

LiTo reuses `gs_rasterize.py`, `metal/gs_rasterize.metal`, `hyworld2_sh.py`, and camera/export helpers where the contracts are model-neutral. CUDA is not a runtime option. Upstream CUDA, PyTorch, and gsplat paths are static architecture references only; optional Torch/MPS parity stays dev-only and non-blocking.

Local source-contract fixtures under `tests/fixtures/lito/` are deterministic synthetic fixtures, not vendor numerical captures. They exist to lock tensor contracts while keeping vendor checkouts, generated Apple samples, and converted weights out of package artifacts.

## CLI And Script Split

Package CLIs under `[project.scripts]` are the supported runtime surfaces. Repository scripts under `scripts/` are readable wrappers and maintenance tools that encode recommended settings for users and maintainers.

Add a script only when it is reusable, has argparse help, writes under `outputs/` by default, and does not depend on hidden local session state.

## Local Asset Rule

The package can inspect, validate, convert, and run local model assets, but the source distribution and wheel must not include:

- `.agent/`, `.codex/`, `.claude/`
- `weights/`, `inputs/`, `outputs/`, `vendors/`
- `.venv/`, caches, build outputs, generated probes

The artifact checker in `scripts/packaging/check_release_artifacts.py` enforces this release boundary.
