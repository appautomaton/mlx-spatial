# Real-Weight Continuation Summary

## Scope

Wire downloaded HY-WorldMirror safetensors into the MLX path and produce first real outputs.

## Result

- Added real safetensors loading for `visual_geometry_transformer.*` plus requested depth/normal/points head tensors.
- Added official DINOv2 patch-embedding execution for native 37x37 checkpoint positional grids.
- Added official VGT block key support for `attn.proj`, `mlp.fc1/fc2`, q/k norms, and layer scale.
- Added official DPT execution for depth, normal, and point heads.
- Preserved exact blocker behavior for unsupported DINO bicubic position interpolation on non-native grids.

## Live Evidence

- `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/trellis2/demo-rgb-background.png --output outputs/hyworld2/real-balanced-depth --heads depth --memory-profile balanced --trace-output outputs/hyworld2/real-balanced-depth/trace.json` -> completed through export.
- `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/trellis2/demo-rgb-background.png --output outputs/hyworld2/real-balanced-dnp --heads depth,normal,points --memory-profile balanced --trace-output outputs/hyworld2/real-balanced-dnp/trace.json` -> completed through export.
- `outputs/hyworld2/real-balanced-dnp/points/points.ply` has 268324 vertices and is non-empty.

## Verification

- `uv run pytest -q tests/test_hyworld2_inference.py tests/test_hyworld2_worldmirror.py tests/test_hyworld2_heads.py tests/test_hyworld2_export.py` -> 51 passed.
- `uv run pytest -q` -> 321 passed, 5 skipped.
- `git diff --check` -> passed.

## Remaining Gaps

- Exact DINO bicubic positional embedding interpolation is still needed for non-518 profiles such as `safe`.
- Official camera head is still blocked in the real-weight path.
- Official Gaussian head/renderer/export remains a follow-up.
