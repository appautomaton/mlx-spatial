# HY-World-2.0 WorldMirror MLX Design

## Parity Target

The shipped path targets the official `HY-World-2.0` `WorldMirrorPipeline` behavior for scene/world reconstruction. The relevant upstream structure is:

- `WorldMirrorPipeline.from_pretrained(...)` resolves `HY-WorldMirror-2.0/model.safetensors` plus `config.yaml` or `config.json`.
- `WorldMirrorPipeline.__call__(...)` prepares image/video-frame inputs, computes adaptive target size, runs `WorldMirror`, then saves depth, normals, camera outputs, points, and optionally Gaussian splats.
- `WorldMirror` is composed of `VisualGeometryTransformer`, `CameraHead`, dense `DPTHead` variants for points/depth/normals/GS, and a CUDA-bound Gaussian renderer.

This change ports the inference architecture, not distributed PyTorch runtime, FSDP, CUDA rendering, or training.

## Proposed Module Layout

- `src/mlx_spatial/hyworld2.py`
  - CLI entrypoint: `validate`, `inspect`, `reconstruct`, `download-command`.
- `src/mlx_spatial/hyworld2_assets.py`
  - WorldMirror asset validation, checkpoint/config resolution, safetensors key inspection.
- `src/mlx_spatial/hyworld2_preprocess.py`
  - Input discovery and official-style image preprocessing into `[B, S, 3, H, W]`.
- `src/mlx_spatial/hyworld2_inference.py`
  - Pipeline orchestration, staged head selection, blockers, trace metadata, memory profiles.
- `src/mlx_spatial/hyworld2_worldmirror.py`
  - MLX `WorldMirror` and `VisualGeometryTransformer` execution.
- `src/mlx_spatial/hyworld2_heads.py`
  - MLX `CameraHead` and DPT-style depth/normal/points/GS head components.
- `src/mlx_spatial/hyworld2_export.py`
  - Depth/normal image writing, camera JSON/NPY metadata, and PLY export.

The exact file split may be compressed during execution if a smaller local pattern is cleaner, but these responsibilities should remain separated.

## Public Command Contract

Expected entrypoint:

```bash
uv run mlx-spatial-hyworld2 validate weights/hy-world-2
uv run mlx-spatial-hyworld2 inspect weights/hy-world-2
uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/hyworld2/demo \
  --output outputs/hyworld2/demo \
  --heads depth,normal,points \
  --memory-profile balanced
```

Default model root interpretation:

- `weights/hy-world-2/HY-WorldMirror-2.0/model.safetensors`
- `weights/hy-world-2/HY-WorldMirror-2.0/config.yaml` or `config.json`

The CLI may also accept a direct `HY-WorldMirror-2.0` directory as long as validation reports the resolved path.

## Trace And Blockers

Each reconstruction attempt returns/writes trace metadata with:

- `completed_stages`
- `blocker` with `stage`, `operation`, `reason`
- resolved model directory and checkpoint/config status
- requested and enabled heads
- input frame count, target size, patch grid, token counts
- memory profile and guard decisions
- output artifact metadata

Expected stages:

1. `asset-validation`
2. `input-discovery`
3. `image-preprocessing`
4. `checkpoint-inspection`
5. `model-construction`
6. `visual-transformer`
7. `camera-head`
8. `depth-head`
9. `normal-head`
10. `points-head`
11. `gaussian-head`
12. `export`

If a stage is not implemented or unsafe, the command blocks there. It must not synthesize downstream artifacts.

## Memory Model

The initial CLI supports memory profiles:

- `safe`: smallest default frame count and strict activation guards.
- `balanced`: first live target, defaulting to 518 target size and a small frame count.
- `large`: opt-in higher frame count/target size when the user explicitly chooses it.

Execution rules:

- Keep learned compute in MLX arrays.
- Use `safetensors`/MLX loading for model tensors.
- Use explicit `mx.eval` boundaries after large model stages and chunked head execution.
- Guard global attention by estimated query/key activation size.
- Prefer exact query-block attention over approximate/windowed attention.
- Use frame chunking for DPT heads, matching the official intent of `frames_chunk_size`.
- Convert to NumPy only for file/export assembly.

## First Live Milestone

The first live milestone is not Gaussian rendering. It is:

```text
small image set -> MLX WorldMirror -> depth + normals + camera metadata + points.ply
```

Gaussian attribute export is staged after the above path is real. Real-time 3DGS rendering remains out of scope because the official renderer depends on CUDA-style `gsplat`.
