# Pixal3D MLX NAF Bridge Spec

## Bounded Goal

Implement a Torch-free MLX NAF-compatible projection bridge so Pixal3D can produce high-resolution projected DINOv3 features at inference time without caller-supplied NAF feature maps.

## Broader Intent

This is one cycle inside the larger user goal: add first-class MLX support for `TencentARC/Pixal3D` in `mlx-spatial` without breaking existing model families or importing Torch/CUDA runtime paths.

## Work Scale And Shape

- Scale: medium implementation cycle.
- Shape: model-runtime bridge plus conversion utility, pipeline integration, tests, and docs.
- Selected lenses: engineering, runtime.

## Source Evidence

- Upstream Pixal3D calls `torch.hub.load("valeoai/NAF", "naf", pretrained=True)` for the shape and texture projection stages that set `use_naf_upsample=True`.
- Upstream NAF loads a small checkpoint from `valeoai/NAF` and uses an image encoder plus parameter-free neighborhood cross-attention over low-resolution features.
- The current `mlx-spatial` Pixal3D runtime already accepts injected `shape_lr_naf_feature_map`, `shape_hr_naf_feature_map`, and `texture_naf_feature_map`, but normal CLI runs block at `shape-projection-conditioning`.

## Constraints And Risks

- Runtime code must stay Torch-free and CUDA-free.
- NAF checkpoint loading in runtime must use local MLX/safetensors assets, not `torch.hub`.
- Any PyTorch `.pth` to `.safetensors` conversion is a dev/setup utility, not a runtime dependency.
- The implementation must avoid materializing giant `B,C,H,W` NAF output maps when coordinate-sampled projected features are sufficient.
- Apple GPU memory guards must be explicit: coordinate chunk size, NAF target size, and full-map avoidance.
- This cycle must not claim MoGe auto-camera support.

## Required Outcome

| ID | Requirement |
| --- | --- |
| PXNAF-01 | Add a documented local NAF asset layout and validation path for converted NAF safetensors. |
| PXNAF-02 | Add a Torch-free MLX NAF runtime bridge that can load converted NAF image-encoder weights and produce projected HR features for Pixal3D coordinates. |
| PXNAF-03 | Wire the Pixal3D pipeline to use the MLX NAF bridge when explicit NAF feature maps are not supplied. |
| PXNAF-04 | Preserve explicit injected NAF feature maps as a lower-level parity/testing override. |
| PXNAF-05 | Keep shape/HR/texture progression honest: missing NAF assets should produce a structured blocker; successful NAF projection should advance the normal CLI beyond the old shape-projection blocker. |
| PXNAF-06 | Docs and help text must describe NAF assets, conversion, runtime guards, and remaining non-NAF blockers. |

## Acceptance Criteria

- A fake Pixal3D root plus fake converted NAF weights can advance past `shape-projection-conditioning` without passing `shape_lr_naf_feature_map`.
- Existing explicit-NAF tests continue to pass.
- Missing NAF assets return a structured `naf-assets` or projection-conditioning blocker with the expected local path and conversion command.
- The NAF runtime has focused tests for weight validation, image encoder output shape, coordinate-sampled HR features, and no runtime imports of `torch`, `torchvision`, `natten`, or vendor Pixal3D modules.
- Targeted Pixal3D/NAF tests, full suite, import scan, lock check, diff check, build, artifact checker, and git hygiene pass.

## Deferred Scope

- Exact NATTEN parity against the upstream CUDA kernel is not required in this cycle; the MLX implementation should follow the same local-neighborhood attention contract with documented boundary behavior.
- MoGe auto-camera remains separate.
- Publishing or redistributing the upstream NAF checkpoint is out of scope; users convert or download locally into ignored `weights/`.

## Anti-Goals

- Do not add Torch, NATTEN, CUDA, or `torch.hub` to runtime dependencies.
- Do not vendor the NAF repo into the package.
- Do not add Pixal3D-specific fake-zero NAF features just to pass stages.
- Do not change public package versioning during this NAF bridge cycle.
