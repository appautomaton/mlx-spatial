# Pixal3D MoGe Auto-Camera Spec

## Bounded Goal

Wire Pixal3D auto-camera setup to the existing Torch-free MLX MoGe runtime so generation can derive camera FOV and distance when `--manual-fov` is omitted.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing Torch/CUDA runtime paths.

## Work Scale And Shape

- Scale: small implementation cycle.
- Shape: pipeline integration plus pure camera helper, tests, and docs.
- Selected lenses: engineering, runtime.

## Source Evidence

- Upstream Pixal3D uses MoGe auto-camera in `get_camera_params_wild_moge`, then computes horizontal FOV from normalized intrinsics and Pixal3D distance from the left image edge.
- `mlx-spatial` already has `sam3d_moge.py`, a converted MLX MoGe pointmap/intrinsics runtime with memory profiles and structured blockers.
- The current Pixal3D runtime blocks at `camera-setup` unless `--manual-fov` is passed.

## Constraints And Risks

- Runtime code must stay Torch-free and CUDA-free.
- Reuse the existing MLX MoGe runtime; do not add a Pixal3D-specific MoGe implementation.
- Manual FOV remains the explicit override and must not run MoGe.
- Missing or memory-blocked MoGe assets must return structured Pixal3D blockers with root/profile metadata.
- This cycle must not claim exact upstream MoGe v2 parity; it provides a MoGe-derived MLX auto-camera path using the converted MoGe root available to this package.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXMG-01 | Add a pure helper that converts MoGe normalized intrinsics plus image width into Pixal3D camera params matching the upstream FOV/distance formula. |
| PXMG-02 | Wire `Pixal3DInferencePipeline.generate` to run MLX MoGe auto-camera when `manual_fov` is absent. |
| PXMG-03 | Preserve manual FOV behavior as a no-MoGe override. |
| PXMG-04 | Expose `--moge-root` and `--moge-memory-profile` through the Pixal3D CLI/script. |
| PXMG-05 | Docs and help text must describe auto-camera setup, memory profile, manual override, and remaining asset blockers honestly. |

## Acceptance Criteria

- Missing manual FOV no longer immediately returns the old "auto-camera not implemented" blocker.
- A monkeypatched ready MoGe result lets the pipeline complete `camera-setup` and continue to image conditioning without `manual_fov`.
- Missing MoGe assets return a structured `camera-setup` blocker with MoGe root and memory profile metadata.
- Existing manual-FOV tests still pass and prove MoGe is not invoked when manual FOV is supplied.
- Targeted Pixal3D/SAM3D MoGe tests, full suite, import scan, lock check, diff check, build, artifact checker, and git hygiene pass.

## Anti-Goals

- Do not implement a separate Pixal3D MoGe model.
- Do not add Torch, OpenCV, CUDA, or vendor Pixal3D imports to runtime code.
- Do not download or redistribute MoGe weights.
- Do not change public package versioning during this auto-camera cycle.
