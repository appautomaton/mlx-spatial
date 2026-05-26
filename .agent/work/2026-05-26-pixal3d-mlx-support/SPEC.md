# Pixal3D MLX Support Spec

## Bounded Goal

Add MLX-native Pixal3D image-to-3D inference support to `mlx-spatial`, using the upstream `TencentARC/Pixal3D` main-branch implementation as the reference while preserving existing package behavior, dependency hygiene, and AppleGPU memory safety.

## Broader Intent

The user wants Pixal3D to become a first-class `mlx-spatial` model family, comparable to the existing TRELLIS.2, MapAnything, SAM3D, HY-WorldMirror, and LiTo surfaces: local assets, explicit validation, runnable inference, documented settings, and no runtime Torch/CUDA dependency.

## Work Scale and Shape

- Scale: capability
- Shape: parity-driven model support and runtime integration

## Selected Lenses

- **product:** Adds a current high-quality 2026 image-to-3D model family to the package.
- **engineering:** Ports Pixal3D inference-time structure without breaking existing model families.
- **runtime:** Keeps AppleGPU execution bounded and avoids CUDA-only assumptions.

## Target User or Stakeholder

Developers running `mlx-spatial` locally on Apple Silicon who want Pixal3D-style single-image GLB generation from local weights.

## Reference Sources

- Upstream repo: `https://github.com/TencentARC/Pixal3D`, observed main branch with 13 commits.
- Local vendor reference: `vendors/Pixal3D` at commit `28efad6`.
- Upstream model card: `https://huggingface.co/TencentARC/Pixal3D`, model revision `0b31f9160aa400719af409098bff7936a932f726`, license metadata `MIT`, `extra_gated_eu_disallowed=true`.
- Existing related MLX code: `src/mlx_spatial/trellis2_*.py`, `src/mlx_spatial/mapanything_*.py`, `src/mlx_spatial/sam3d_moge.py`, `src/mlx_spatial/mlx_memory.py`.

## Required Outcome

`mlx-spatial` gains an MLX Pixal3D runtime path that:

1. Treats Pixal3D as a separate model family with its own CLI, docs, script wrapper, asset validation, and checkpoint inspection.
2. Loads local `TencentARC/Pixal3D` assets directly from safetensors/JSON layout without bundling weights.
3. Reuses and extends existing TRELLIS.2 MLX primitives where compatible instead of copying or importing vendor code.
4. Implements Pixal3D-specific inference features: pixel-aligned projection conditioning, per-stage DINOv3 projection grids, optional NAF boundary handling, MoGe/manual-FOV camera setup, Pixal3D cascade routing, and PBR texture/GLB output handling.
5. Provides guarded dev reference capture against the vendored PyTorch implementation for parity checks.
6. Produces a real generated artifact from at least one Pixal3D sample image: required output is textured GLB when the Mac-native export path supports it; otherwise a concrete runtime blocker must identify the exact missing export boundary while preserving completed MLX predictions.
7. Keeps base runtime dependencies free of Torch, TorchVision, CUDA-only packages, `natten`, `flash_attn`, `o_voxel`, `nvdiffrast`, `cumesh`, and vendor imports.
8. Maintains existing TRELLIS.2, MapAnything, SAM3D, HY-WorldMirror, and LiTo tests and package artifact hygiene.

## Constraints

- Runtime package code must not import Torch, TorchVision, CUDA-only packages, or `vendors/Pixal3D`.
- Upstream Pixal3D training, data toolkit, Gradio app, and CUDA/HF demo wheels are out of runtime scope.
- `weights/`, `vendors/`, `inputs/`, `outputs/`, and `/tmp` captures must not enter built package artifacts.
- HF `TencentARC/Pixal3D` assets are local user-downloaded model assets, not redistributable package data.
- Existing `mlx-spatial` CLIs and scripts must continue to work.
- AppleGPU inference must use staged evaluation, explicit cache clearing where helpful, and token/resolution guards rather than assuming 1536-pipeline execution always fits.

## Risks

- **Model size:** HF Pixal3D includes multiple 5.3 GiB to 5.5 GiB flow checkpoints plus ~0.9 GiB decoders; naive eager loading can exceed practical unified-memory limits.
- **Architecture drift from TRELLIS.2:** Pixal3D main is Trellis.2-based but changes conditioning to projection attention, per-block `proj_linear`, projection grids, NAF-upsampled feature concatenation, and 1024/1536 cascade behavior.
- **CUDA-only postprocess:** Upstream GLB extraction uses `o_voxel` plus CUDA/HF-demo wheels. The MLX implementation may need a Mac-native export compromise or a precise blocker before textured GLB parity.
- **Camera dependency:** Upstream auto-FOV uses MoGe-2. Existing MoGe support can be reused only if the model boundary and required weights are compatible; manual FOV must remain available for smoke and parity.
- **NAF dependency:** Upstream projection conditioning can use NAF upsampling from `torch.hub`. The MLX path must either port/replace this boundary or explicitly stage lower-resolution support until the NAF-equivalent path exists.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| PIXAL3D-01 | Public runtime surface exists | `pyproject.toml` exposes `mlx-spatial-pixal3d`, and `uv run mlx-spatial-pixal3d --help` works. |
| PIXAL3D-02 | Asset tooling exists | CLI can print a HF download command, validate local `weights/pixal3d`, and inspect required Pixal3D checkpoint groups. |
| PIXAL3D-03 | Runtime dependency boundary stays clean | Base dependencies and `src/mlx_spatial` do not import or require Torch, TorchVision, CUDA-only wheels, `natten`, `flash_attn`, `o_voxel`, `nvdiffrast`, `cumesh`, or vendor modules. |
| PIXAL3D-04 | Projection conditioning is implemented | MLX code builds DINOv3 global tokens and projected 3D-grid features for Pixal3D stage configs, with parity or shape checks against vendored reference captures. |
| PIXAL3D-05 | Pixal3D flow/decoder routing is implemented | MLX inference can run Pixal3D sparse-structure, shape cascade, texture cascade, and decoder boundaries using local Pixal3D weights or stops at a concrete unresolved model boundary. |
| PIXAL3D-06 | Camera setup is usable | Runtime supports manual FOV and an auto-camera path or documented MoGe boundary using existing MLX MoGe assets. |
| PIXAL3D-07 | Sample generation path exists | A recommended script runs one `vendors/Pixal3D/assets/images/*` or copied `inputs/pixal3d/*` sample through the MLX pipeline and writes output under `outputs/` or `/tmp`. |
| PIXAL3D-08 | Output contract is explicit | Successful generation writes GLB when supported; otherwise the result includes completed MLX predictions plus a structured blocker for the exact unsupported export step. |
| PIXAL3D-09 | Existing package behavior remains intact | Existing full test suite passes or any failure is proven unrelated and fixed before verification. |
| PIXAL3D-10 | Package hygiene holds | `uv lock --check`, build artifact checks, and git hygiene checks pass without bundling local assets/vendors/generated outputs. |

## Scope Coverage Decisions

- **Included:** inference-time Pixal3D main-branch support, local asset layout, dev-only PyTorch reference capture, projection conditioning, memory-bounded MLX cascade execution, CLI/script/docs/tests, and package hygiene.
- **Deferred:** training, data preparation toolkit, Gradio app, paper-branch Direct3D-S2 implementation, CUDA-only demo wheels, quality benchmarking across datasets, and bundled model redistribution.
- **Anti-goals:** wrapping PyTorch as the runtime, importing `vendors/Pixal3D` from package code, declaring support from asset inspection alone, or breaking existing `mlx-spatial` model families.

## Blocking Questions or Assumptions

No blocking question prevents framing. Planning should assume an incremental execution path: first asset/reference/tooling, then projection conditioning, then cascade model execution, then export. Completion is not proven until a real Pixal3D sample reaches an output artifact or a precise unresolved model/export boundary is demonstrated by current-state evidence.

## Anti-Goals

- Full Pixal3D training or fine-tuning.
- Paper-branch Direct3D-S2 parity.
- Gradio or browser demo reproduction.
- Runtime Torch/CUDA/vendor implementation.
- Weight redistribution in the package.
- Treating a dry-run inspector as final Pixal3D MLX support.
