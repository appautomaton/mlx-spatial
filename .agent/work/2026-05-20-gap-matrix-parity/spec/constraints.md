# Constraints and Risks

## Metal GS Rasterizer

The vendor references use CUDA-only Gaussian splatting (gsplat, diff-gaussian-rasterization) for texture baking and layout optimization. MLX does not have an equivalent. This spec requires a Mac-native GS rasterizer built on Metal compute shaders.

- **Target**: M1 unified memory architecture (minimum). If M1 proves infeasible for performance, minimum target revises to M2 Pro.
- **Approach**: Metal Performance Shaders compute kernel for Gaussian projection and alpha-blended tile rendering. Must produce texture maps matching gsplat reference within a defined pixel tolerance.
- **Risk**: This is a significant standalone project. Unknown whether M1 GPU compute has sufficient bandwidth for real-time GS rendering at reference quality. Mitigation: start with correctness, then optimize.
- **Fallback**: If Metal GS rasterizer cannot match gsplat quality, SAM3D texture baking can use the existing xatlas + trilinear bake path (already ported) as a lower-quality alternative.

## HY-World 2.0 Backbone

The current `hyworld2_worldmirror.py` bundles the model monolithically. The vendor reference has 13+ separate layer modules (attention, block, MLP, patch embed, SwiGLU, RoPE, LayerScale, DropPath, ViT). The spec requires porting each as a separate, testable MLX module.

- **Risk**: The monolithic module may need significant restructuring. The existing tests may not cover individual layer behavior.
- **Mitigation**: Port each layer module independently with its own parity test before integrating.

## Parity Verification

- Parity depends on local PyTorch reference outputs via `tools/hyworld2_dump_torch_reference.py` and `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
- Not CI-gated. Risk: reference outputs may drift across vendor versions.
- The `numeric_parity_verified=false` flag in HY-World 2.0 must remain until all backbone gaps are verified.

## MOT Variant Scope

The SAM 3D MOT (Multi-Object Transformer) variant adds pose-aware inference with quaternion, translation, and scale outputs. This increases SAM3D scope but is explicitly included per user decision.

- **Risk**: MOT may require different checkpoint configurations or conditioning that aren't yet validated.
- **Mitigation**: MOT gaps close after the base SAM3D pipeline is verified; MOT adds conditioning heads on top.

## No Runtime PyTorch/CUDA

All closed gaps must produce matching output via MLX/Metal only. No PyTorch, CUDA, flash-attn, gsplat, or diff-gaussian-rasterization at runtime.

- **Constraint**: Existing tests use `MLX_SPATIAL_RUN_TORCH_PARITY=1` for dev-only reference comparison. Production tests must pass without this flag.