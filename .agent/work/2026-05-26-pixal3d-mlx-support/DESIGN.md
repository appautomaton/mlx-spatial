# Pixal3D MLX Support Design

## Runtime Shape

Pixal3D support should follow the existing model-family split:

```text
src/mlx_spatial/pixal3d.py              # CLI routing
src/mlx_spatial/pixal3d_assets.py       # HF asset manifest, validation, checkpoint inspection
src/mlx_spatial/pixal3d_camera.py       # manual FOV and MoGe-based camera params
src/mlx_spatial/pixal3d_projection.py   # DINOv3 projection-grid conditioning
src/mlx_spatial/pixal3d_inference.py    # pipeline orchestration, blockers, memory profile
src/mlx_spatial/pixal3d_export.py       # GLB/export boundary or structured blocker
```

Reuse existing TRELLIS.2 and shared primitives where they are already MLX-native:

- DINOv3 loading/forward helpers from `trellis2_dinov3*`.
- sparse-structure/SLat flow math from `trellis2_sparse_structure.py` and `trellis2_slat.py`.
- decoder/export primitives from `trellis2_decode.py`, `trellis2_export.py`, `ovoxel.py`, and `export_utils.py`.
- memory helpers from `mlx_memory.py`.
- MoGe conversion/runtime pieces from `sam3d_moge.py` if auto-camera needs MoGe.

Do not import `vendors/Pixal3D` from runtime code. Dev-only reference capture can import it under `tools/` with an explicit environment guard.

## Source Model Boundary

HF `TencentARC/Pixal3D` revision observed during framing:

- `pipeline.json`: model routing and sampler defaults.
- `ckpts/ss_flow_img_dit_1_3B_64_bf16.{json,safetensors}`
- `ckpts/ss_dec_conv3d_16l8_fp16.{json,safetensors}`
- `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.{json,safetensors}`
- `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.{json,safetensors}`
- `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.{json,safetensors}`
- `ckpts/shape_dec_next_dc_f16c32_fp16.{json,safetensors}`
- `ckpts/tex_dec_next_dc_f16c32_fp16.{json,safetensors}`

Default sampler settings:

- sparse structure: 12 steps, guidance 7.5, rescale 0.7, interval `[0.6, 1.0]`, `rescale_t=5.0`
- shape SLat: 12 steps, guidance 7.5, rescale 0.5, interval `[0.6, 1.0]`, `rescale_t=3.0`
- texture SLat: 12 steps, guidance 1.0, rescale 0.0, interval `[0.6, 0.9]`, `rescale_t=3.0`
- default pipeline: `1536_cascade`; low-memory default should prefer `1024_cascade`.

## Main Delta From TRELLIS.2

Pixal3D is not just a different checkpoint set. The required deltas are:

1. Projection conditioning:
   - DINOv3 global tokens remain `cond.global`.
   - 3D grid points are projected through camera FOV/distance/mesh scale into image feature maps.
   - sparse structure uses a projection grid of 16 for the released 64 sparse-structure flow.
   - shape 512 uses grid 32; shape 1024 and texture 1024 use grid 64.
2. Projection attention:
   - flow blocks use `image_attn_mode="proj"`.
   - dense attention adds `proj_linear(proj_context)` to global cross-attention output.
   - sparse attention does the same on `SparseTensor.feats`.
   - shape/texture SLat checkpoints expect `proj_in_channels=2048`, normally DINOv3 low-res plus NAF-upsampled features.
3. Camera setup:
   - manual FOV must work without MoGe.
   - auto-camera should reuse MoGe where practical and stay optional.
4. Export:
   - upstream uses `o_voxel.postprocess.to_glb`, CUDA demo wheels, and remeshing.
   - MLX runtime should attempt a Mac-native textured GLB path using existing primitives; if not feasible in the first execution cycle, return a structured export blocker after successful MLX predictions.

## Memory Strategy

- Default user-facing Pixal3D script should start with `1024_cascade`.
- Keep `1536_cascade` available behind an explicit `--pipeline-type 1536_cascade`.
- Load and evaluate stages sequentially; release checkpoint tensors and call `clear_mlx_cache()` between major stages when safe.
- Keep `max_num_tokens` guard from Pixal3D/TRELLIS.2. For 1536, reduce actual HR resolution in 128px steps until token count fits, matching upstream behavior.
- Real-weight smoke should record stage timings and MLX active/peak memory counters when available.

## Dev Reference Boundary

Dev-only parity should use:

```text
PIXAL3D_TORCH_REF=1 uv run --group torch-ref ...
```

Reference capture may import `vendors/Pixal3D`, Torch, and CUDA/MPS-capable libraries. Runtime package code may not.
