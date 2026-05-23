# Slice 5 Verify-Gap Correction

## Trigger

User review found that the LiTo PLY files under `outputs/lito/` were not actual Apple checkpoint-backed outputs. Inspection confirmed their headers contain:

```text
comment mlx-spatial LiTo source-contract smoke 3DGS export
```

Those files are synthetic source-contract smoke artifacts and must not satisfy AC-07.

## Subagent Findings

- Galileo (`019e55a7-6b16-7ac3-851c-0bfa8b029da7`): a vendor-backed optional backend using upstream `load_model` is not spec-compliant. Package runtime must not import `vendors/`, `lito`, Torch, CUDA, xformers, flash-attention, gsplat, or `plibs`.
- Parfit (`019e55a7-2bc3-72c0-b7e1-3e264efe2231`): real DiT architecture is recoverable from safetensors headers plus fixed vendor config constants: EMA prefix, 28 blocks, `8192 x 32` latent, hidden dim `1152`, condition dim `2048`, 16 heads, SwiGLU, RMSNorm, and fixed remaps.
- Fermat (`019e55a7-5153-7ea0-971b-e0c088da9793`): Gaussian decoder architecture is recoverable from safetensors headers plus fixed vendor config constants: 6 blocks, 2 self-attn layers, query dim `512`, latent dim `32`, expansion `64`, SH degree `3`, and a separate voxel decoder/init-coordinate port is still required.
- Dewey (`019e55d0-ba39-7202-a785-7b3b3934de46`): safe minimal next Gaussian decoder boundary is block-0 Perceiver cross-attention through `ca_mlp` only. This subpath must include `kv_linear`, `ca_layer`, residual add, `ca_ln`, and `ca_mlp`; it must not claim full decoder parity because localized-voxel self-attention still needs voxel metadata.
- Sagan (`019e55dd-0177-7931-b1a3-096771381088`): localized-voxel metadata is portable without CUDA/vendor runtime by reimplementing `PackedPoint.get_bijk_info`: batch-aware voxel keys from init/query coordinates, stable sort/inverse indices, per-cell `cu_seq_lens`, `max_seq_lens`, and `chunk_start_idxs`; odd self-attention layers use a half-cell shift.
- Jason (`019e55dd-46c4-7570-abff-0f5a556accab`): the real DiT sampler should be a separate module that consumes precomputed `(B, 1374, 2048)` condition tokens and returns `(B, 8192, 32)` latent tokens; no extra DiT key remaps are needed beyond the current loader.
- Bohr (`019e55e2-43f0-7e33-828b-d73293b77e0d`): real image conditioning needs a LiTo-specific DINOv2 ViT-L/14 register-token port plus learned RGBA conv side branch; existing HY-World/TRELLIS DINO code is only partially reusable conceptually, not directly.
- Socrates (`019e55e6-3aa4-7993-9648-3fd14be8f2d6`): upstream init coordinates come from `inference_init_coords_for_decoder`: LiTo voxel decoder produces `(B, 8, 16, 16, 16)` sparse-structure latent, a Trellis sparse-structure decoder expands it to `(B, 1, 64, 64, 64)` occupancy probabilities, cells with probability `>= 0.5` are converted to centers `(ijk + 0.5) * (2 / 64) - 1`, and packed coordinates plus `q_seq_lens` feed the MLX Gaussian decoder. The safe next increment is a caller-supplied latent-token voxel decoder path; full generation still also needs Trellis occupancy decode, DiT sampling, and DINO conditioning.
- Carver (`019e55ee-7c4e-73b1-9ebd-ef6a6e1fc537`): confirmed the active voxel path is `SSLatentDecoder -> VectorDecoder`, not the simplified decoder. The low-res path is `latent_tokens (B,8192,32) -> input_linear 32->512 -> learned 16^3 z/y/x query + FourierZYX(include_input=True) -> 4 global Perceiver blocks -> FinalLayer -> ss_latent (B,8,16,16,16)`.
- Cicero (`019e55ee-b8dd-76e0-878c-4e4d6e7946cf`): confirmed LiTo's downstream occupancy decoder is TRELLIS `ss_dec_conv3d_16l8_fp16`, mapping `ss_latent (B,8,16,16,16)` to logits `(B,1,64,64,64)`. Local `trellis2_sparse_structure.py` already has the no-CUDA MLX decoder primitives and checkpoint validation.
- Lagrange (`019e55f9-35b5-7ea1-a1a0-b778b7da28a7`): confirmed the real DiT path is `velocity_estimator_ema.module.*`, with `(B,8192,32)` latents, `(B,M,2048)` condition tokens, 28 blocks, hidden dim 1152, 16 heads, SwiGLU hidden 3072, Fourier timestep embedding, PixArt AdaLN block modulation, final AdaLN, and `heun`/`euler` ODE sampling from `t_eps=1e-4` to `1.0`.
- Bacon (`019e55f9-4dae-7d13-96ac-960db3f3d2f7`): confirmed real LiTo conditioning should produce `(B,1374,2048)` for one 518px RGBA image: DINOv2 ViT-L/14 register tokens `(CLS + 4 regs + 37*37 patches)` concatenated with a learned RGBA conv branch. The converted DiT safetensors contain `patch_encoder.dinov2_model.*` and `patch_encoder.learnable_model.*`, so the remaining gap is code, not missing LiTo conditioner weights.

## Coordinator Fixes

- `src/mlx_spatial/lito_real_backend.py` no longer treats Torch as a runtime backend requirement.
- Added `inspect_lito_real_architecture(...)`, a header-only safetensors inventory for the real LiTo converted weights.
- Renamed the backend boundary to the direct MLX/safetensors shape and kept generation fail-closed until local DINO conditioning, MLX DiT sampling, init-coordinate generation, MLX Gaussian decoding, and export are implemented.
- Fixed checkpoint-backed PLY writer ordering so SH validation uses a defined Gaussian count.
- Added tests for import hygiene, header inventory, real-weight inventory when local assets are present, and checkpoint-backed PLY schema.

## Verification

- `head -n 8 outputs/lito/smoke.ply outputs/lito/smoke-final.ply outputs/lito/smoke-final-script.ply` -> each file has `comment mlx-spatial LiTo source-contract smoke 3DGS export`.
- `uv run pytest tests/test_lito_real_backend.py -q` -> `7 passed`.
- `uv run pytest tests/test_lito_inference.py tests/test_lito_cli.py tests/test_lito_real_backend.py -q` -> `26 passed`.
- `uv run pytest tests/test_lito_*.py -q` -> `70 passed`.
- `uv run pytest -q` -> `717 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-not-ready-proof-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with `LitoBackendUnavailable`; no PLY is written.
- `inspect_lito_real_architecture('weights/lito-mlx')` -> DiT `28 / 32 / 1152 / 2048`, Gaussian decoder `6 / 64 / SH degree 3`, voxel decoder `4 / (16, 16, 16, 512)`.

## Remaining Gap

Real AC-07 is still open. The next implementation work is the direct MLX port from converted safetensors: DINO image conditioner, DiT sampler/weight loader, tokenizer/init-coordinate path including voxel decoder, Gaussian decoder, and export.

## Loader And Decode Increment

Follow-up sidecar agents confirmed two minimal next steps:

- DiT: add a separate real LiTo DiT module/sampler later; the fixture `LitoDiT` has the wrong condition width for real weights.
- Gaussian decoder: implement explicit-init-coordinate raw decoder output decoding and real safetensor remapping before attempting full image generation.

Coordinator progress:

- `load_lito_dit_weight_arrays(...)` loads selected EMA DiT tensors and applies local remaps for `t_proj`, `t0_proj`, and final-layer AdaLN names.
- `load_lito_gaussian_decoder_weight_arrays(...)` loads selected `gs_decoder.*` tensors, remaps sequential MLP names, and splits fused `w12` weights into local `w1` / `w2` arrays.
- `decode_lito_gaussian_outputs(...)` ports the upstream LiTo `decode_gs` equations for raw shape/color outputs plus explicit init coordinates.
- `DirectMlxLitoBackend` exposes those helpers while keeping `generate_gaussians(...)` fail-closed for full image-to-3D.
- `run_lito_gaussian_output_heads(...)` now runs the real shape/color output MLP heads for caller-supplied decoder query latents using remapped checkpoint weights.
- `decode_lito_gaussian_query_latents(...)` combines the weighted output heads with explicit init-coordinate Gaussian decoding.
- `encode_lito_gaussian_query_points(...)` now runs LiTo's real coordinate/Fourier point-query stem (`xyz`, `xyz_encoded`, `point_linear`, `point_mlp`) for caller-supplied init coordinates.
- `decode_lito_gaussian_query_points(...)` combines the point-query stem, output heads, and explicit-coordinate Gaussian decode. It still stops before Perceiver attention over LiTo latents.
- `run_lito_gaussian_perceiver_block0_cross_only(...)` now runs the real Gaussian Perceiver block-0 cross-attention and CA-MLP subpath (`kv_linear`, `ca_layer`, residual, `ca_ln`, `ca_mlp`) for caller-supplied query latents and LiTo latent tokens.
- The cross-only helper raises if asked to include localized-voxel self-attention, so full Gaussian decoder parity remains fail-closed until voxel window metadata and self-attention are ported.
- `build_lito_local_voxel_info(...)` now builds LiTo localized-voxel self-attention metadata locally with NumPy/MLX integer grouping and no Torch/vendor runtime.
- `run_lito_gaussian_perceiver_block0_with_local_voxel_self_attention(...)` now runs block 0 through cross-attention, CA MLP, both localized-voxel self-attention layers, and both self MLPs for caller-supplied init coordinates and latent tokens.
- `run_lito_gaussian_perceiver_all_blocks_with_local_voxel_self_attention(...)` now runs all discovered Gaussian Perceiver blocks with the same localized-voxel metadata schedule as upstream.
- `decode_lito_gaussian_perceiver_all_blocks(...)` combines the point-query stem, all-block Gaussian Perceiver, output heads, and explicit-coordinate Gaussian decode for caller-supplied LiTo latent tokens and init coordinates.
- `load_lito_voxel_decoder_weight_arrays(...)` now loads real `voxel_decoder.*` tensors from `tokenizer/lito_new.safetensors`.
- `run_lito_voxel_decoder_lowres_latent(...)` runs the active LiTo `SSLatentDecoder -> VectorDecoder` low-res sparse-structure latent path for caller-supplied LiTo latent tokens.
- `decode_lito_trellis_sparse_structure_logits(...)` reuses the local TRELLIS.2 MLX sparse-structure decoder to map LiTo `ss_latent` to occupancy logits.
- `occ_grid_to_lito_init_coord(...)` ports the upstream occupancy-to-packed-coordinate conversion, including the `(z,y,x) -> (x,y,z)` nonzero order and `[-1,1]` 64^3 cell-center convention.
- `decode_lito_init_coords_from_latents(...)` chains caller-supplied LiTo latent tokens through LiTo voxel decode, TRELLIS occupancy decode, and packed Gaussian init-coordinate extraction.
- `run_lito_dit_velocity(...)` now runs the real LiTo DiT forward subpath from caller-supplied condition tokens and latent tokens: timestep Fourier embedding, condition token MLP, z/pos projection, selected DiT blocks, and final AdaLN projection.
- `sample_lito_dit_latents(...)` adds a local `euler`/`heun` ODE sampling wrapper for caller-supplied condition tokens and optional initial latents.

Verification:

- `uv run pytest tests/test_lito_real_backend.py -q` -> `13 passed`.
- `uv run pytest tests/test_lito_*.py -q` -> `76 passed`.
- `uv run pytest -q` -> `723 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`.
- Real subset load probe returned expected remapped shapes: DiT `(1152, 32)`, `(1152, 64)`, `(2304,)`; Gaussian decoder `(512, 195)`, `(512, 512)`, `(2048, 512)`, `(3136, 512)`.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-loaders-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after output-head forward support -> `15 passed`.
- `uv run pytest tests/test_lito_*.py -q` after output-head forward support -> `78 passed`.
- `uv run pytest -q` after output-head forward support -> `725 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after output-head forward support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-output-heads-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after point-query stem support -> `17 passed`.
- `uv run pytest tests/test_lito_*.py -q` after point-query stem support -> `80 passed`.
- `uv run pytest -q` after point-query stem support -> `727 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after point-query stem support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-query-stem-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after block-0 cross-only support -> `21 passed`.
- `uv run pytest tests/test_lito_*.py -q` after block-0 cross-only support -> `84 passed`.
- `uv run pytest -q` after block-0 cross-only support -> `731 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after block-0 cross-only support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`; first sandboxed attempt failed DNS fetching `hatchling`, then network-approved rerun passed.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-cross-only-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after block-0 localized-voxel support -> `24 passed`.
- `uv run pytest tests/test_lito_*.py -q` after block-0 localized-voxel support -> `87 passed`.
- `uv run pytest -q` after block-0 localized-voxel support -> `734 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after block-0 localized-voxel support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`; first sandboxed attempt failed DNS fetching `hatchling`, then network-approved rerun passed.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-local-voxel-block0-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after all-block Gaussian Perceiver support -> `26 passed`.
- `uv run pytest tests/test_lito_*.py -q` after all-block Gaussian Perceiver support -> `89 passed`.
- `uv run pytest -q` after all-block Gaussian Perceiver support -> `736 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after all-block Gaussian Perceiver support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-all-blocks-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after voxel/TRELLIS init-coordinate support -> `32 passed`.
- `uv run pytest tests/test_lito_*.py -q` after voxel/TRELLIS init-coordinate support -> `95 passed`.
- `uv run pytest -q` after voxel/TRELLIS init-coordinate support -> `742 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after voxel/TRELLIS init-coordinate support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`; first sandboxed attempt failed DNS fetching `hatchling`, then network-approved rerun passed.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-init-coords-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard.
- `uv run pytest tests/test_lito_real_backend.py -q` after real DiT caller-supplied-condition support -> `36 passed`.
- `uv run pytest tests/test_lito_*.py -q` after real DiT caller-supplied-condition support -> `99 passed`.
- `uv run pytest tests/test_lito_real_backend.py tests/test_lito_inference.py tests/test_lito_cli.py -q` after fail-closed message update -> `55 passed`.
- `uv run pytest -q` after real DiT caller-supplied-condition support -> `746 passed, 5 skipped, 2 warnings`.
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` after real DiT caller-supplied-condition support -> built `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`.
- `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-dit-message-20260523.ply --memory-profile safe --render-size 12` -> exits 1 with no output, preserving the no-fake-output guard. The error now identifies DINO/RGBA image conditioning as the remaining runtime gap.
