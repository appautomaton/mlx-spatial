---
active_change: hyworld2-worldmirror-mlx-inference
stage: complete
---

# Status

## Current Change

- active change: `hyworld2-worldmirror-mlx-inference`
- current stage: `complete`

## What Is True Now

- The HY-World-2.0 WorldMirror MLX spec, design, and plan are written under `.agent/work/hyworld2-worldmirror-mlx-inference/`.
- Engineering review verdict is `approved_with_risks`.
- Slices 1-9 are complete: CLI/assets, reconstruction blockers/path policy, official-style preprocessing, checkpoint/config routing, MLX `VisualGeometryTransformer` core, MLX camera/DPT dense heads, fixture-backed staged exports, Gaussian attribute staging, and live setup verification.
- Subagent orchestration was used for read-only official/repo discovery and Slice 4/Slice 5/Slice 6/Slice 7 implementation/review.
- Slice 4 quality review found two issues and both were fixed: first-milestone heads no longer require Gaussian groups, and corrupt safetensors now returns a structured checkpoint blocker.
- Slice 5 quality review found two memory/parity issues and both were fixed: q/k RoPE plus official condition/qkv token semantics are covered, and default deterministic fixture tensors are guarded before official-scale allocation.
- Slice 6 quality review found two chunking/runtime issues and both were fixed: dense-head frame chunks now force per-chunk MLX evaluation, and zero-frame inputs return structured blockers.
- Slice 7 quality review found a reused-output state issue and it was fixed: fixture export runs now clear known stale head artifacts before writing new outputs.
- Slice 8 initially exported deterministic Gaussian attributes under `gaussian/attributes.npz`; the follow-on parity slice now writes native 3DGS `gaussians.ply`.
- Slice 9 originally verified that local HY-World weights were missing; the weights have since been downloaded under `weights/hy-world-2/HY-WorldMirror-2.0/`.
- Post-Slice 9 continuation added the real-weight MLX path for DINOv2 patch embedding, official VGT block key aliases, q/k norms, layer scale, real safetensors loading, and real DPT depth/normal/points heads.
- The unrelated TRELLIS SLat full-suite blocker was fixed by restoring the exact chunked self-attention API/guard expected by the existing TRELLIS tests.
- Latest Slice 5 verification passed: `uv run pytest -q tests/test_hyworld2_worldmirror.py tests/test_hyworld2_inference.py` reported 25 passed.
- Latest Slice 6 verification passed: `uv run pytest -q tests/test_hyworld2_heads.py tests/test_hyworld2_worldmirror.py` reported 30 passed.
- Latest Slice 7 verification passed: `uv run pytest -q tests/test_hyworld2_export.py tests/test_hyworld2_inference.py tests/test_hyworld2_tools.py` reported 27 passed.
- Latest Slice 8 verification passed: `uv run pytest -q tests/test_hyworld2_inference.py tests/test_hyworld2_heads.py` reported 29 passed.
- Latest Slice 9 verification passed: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_inference.py tests/test_hyworld2_export.py` reported 27 passed.
- Latest HY-World bundle verification passed: `uv run pytest -q tests/test_hyworld2_tools.py tests/test_hyworld2_preprocess.py tests/test_hyworld2_assets.py tests/test_hyworld2_inference.py tests/test_hyworld2_worldmirror.py tests/test_hyworld2_heads.py tests/test_hyworld2_export.py tests/test_model_assets.py` reported 78 passed.
- Latest TRELLIS SLat verification passed: `uv run pytest -q tests/test_trellis2_slat.py` reported 22 passed.
- Latest real HY-World run passed: `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/trellis2/demo-rgb-background.png --output outputs/hyworld2/real-balanced-dnp --heads depth,normal,points --memory-profile balanced --trace-output outputs/hyworld2/real-balanced-dnp/trace.json` completed through export.
- Latest HY-World focused regression passed: `uv run pytest -q tests/test_hyworld2_*` reported 75 passed.
- Latest official-style Cat_Girl run passed: `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 vendors/HY-World-2.0/examples/worldrecon/stylistic/Cat_Girl --output outputs/hyworld2/cat-girl-518-official --heads camera,depth,normal,points,gs --memory-profile balanced --trace-output outputs/hyworld2/cat-girl-518-official/trace.json` completed with no blocker, `camera_params.json`, `points.ply`, and binary little-endian `gaussians.ply`.
- Latest full regression passed: `uv run pytest -q` reported 323 passed, 5 skipped.
- Final hygiene passed: `git diff --check`.

## Next Step

Clean up commits.

## Open Risks

- Multi-view examples such as `Room_Cat` still need live verification.
- Viewer compatibility may require `.splat` or `.ksplat` conversion depending on the Blender 3DGS add-on.
