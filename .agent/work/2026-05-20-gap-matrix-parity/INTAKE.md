# Intake

## Work Scale

roadmap

## Work Shape

parity

## Objective Statement

Identify and close all inference gaps in `src/mlx_spatial/` compared to the three vendor reference implementations (`vendors/TRELLIS.2`, `vendors/sam-3d-objects`, `vendors/HY-World-2.0`), maintaining the speed and quality of the original reference implementations. Close gaps in strict priority order across pipeline boundaries using a gap matrix, with Mac-native Metal alternatives for CUDA-blocked features.

## Broader Intent

mlx-spatial aims to be a complete, performance-competitive MLX-native inference library for 3D spatial models on Apple Silicon. This parity audit ensures no inference-relevant functionality is missing and that all ports produce numerically matching output against their PyTorch references.

## Target User or Stakeholder

Developers running TRELLIS.2, SAM 3D Objects, and HY-World 2.0 inference on Apple Silicon workstations who need output fidelity and feature parity with the official implementations.

## Desired Outcome

- Structured gap matrix: pipeline × gap × priority × dependency
- Each closed gap produces numerically matching output against the reference with new parity test coverage
- Metal alternatives for CUDA-blocked features (GS rendering, remeshing)
- All three pipelines reach full inference parity

## Scope Boundary and Anti-Goals

**Included:**
- All inference-relevant gaps across all three pipelines
- Metal-native alternatives for CUDA-blocked features
- Parity tests for each closed gap
- HY-World 2.0 transformer backbone layers (attention, block, MLP, patch embed, SwiGLU, RoPE, LayerScale, DropPath, ViT)
- Camera/rotation/geometry utilities for all pipelines
- GS activation functions and spherical harmonics
- Sparse spatial interpolation
- MOT (Multi-Object Transformer) for SAM 3D pose-aware inference
- Texturing pipeline for TRELLIS.2 (re-texture existing mesh)
- Layout post-optimization for SAM 3D (ICP + render-and-compare)

**Anti-goals:**
- Training infrastructure (trainers, datasets, elastic modules, loss functions)
- Distributed/multi-GPU communication (FSDP, All2All, sequence parallelism)
- Visualization-only features (PBR rendering, voxel rendering, Gradio apps)
- Replacing MLX with another ML framework
- Supporting non-Apple-Silicon hardware

## Rejected Framings

- **Approach A (Pipeline-by-pipeline sequential):** Rejected because shared Metal work would be redesigned per pipeline and no pipeline reaches parity until its turn.
- **Approach B (Shared gaps first):** Rejected because Metal GS rasterizer front-loads risk and blocks all downstream progress.

## Scope Preservation

This spec preserves the full stated intent: close all inference gaps across all three pipelines with Mac-native alternatives. The prioritization by impact (not pipeline boundary) is the accepted reframe from Approach C.

## Scope Coverage

**Included:**
- Gap matrix creation with structured priorities and dependencies
- HY-World 2.0 transformer backbone (critical inference blocker)
- MOT pose-aware inference for SAM 3D (high priority)
- Metal GS rasterizer for texture baking (high priority)
- Camera/rotation/geometry utilities for all pipelines
- GS activation and spherical harmonics
- Sparse spatial interpolation
- TRELLIS.2 texturing pipeline (re-texture existing mesh)
- SAM 3D layout post-optimization
- Shortcut/distillation model for SAM 3D fast inference
- Cross-pipeline shared utilities (mesh postprocessing, render utils)

**Deferred to ROADMAP.md:**
- Performance optimization after parity is verified
- CI and release infrastructure

**Anti-goals:**
- Training, visualization, multi-GPU, Gradio UIs

**Needs decision:**
- Minimum Apple Silicon target for Metal GS rasterizer (M1 baseline vs. M2 Pro)
- Whether the MOT variant is needed now or only when multi-object scenes are requested

## Selected Approach

**Approach C: Gap Matrix With Prioritized Close Order.** Build a structured gap matrix (pipeline × gap × priority × dependency), then close gaps in strict priority order regardless of pipeline boundary. Critical inference blockers close first, then medium-priority gaps, then Metal alternatives alongside their corresponding gaps. This avoids front-loading the GS rasterizer (Approach B's blocker) and avoids pipeline serialization (Approach A's delay).

## Key Assumptions and Risks

- Metal GS rasterizer is feasible on Apple Silicon but is a significant standalone project (risk: may be harder than estimated)
- HY-World 2.0 transformer backbone gaps may require deeper rework than anticipated (risk: scope creep)
- Parity verification depends on having local PyTorch reference outputs (risk: dev-only, not CI-gated)
- Existing parity test patterns (`test_*_parity.py`) are sufficient for gap closure verification
- The `numeric_parity_verified=false` flag in HY-World 2.0 should remain until all gaps are verified

## Deferred Scope

- Performance benchmarks and optimization (separate roadmap phase after parity)
- CI automation and release pipeline
- Training and fine-tuning capabilities