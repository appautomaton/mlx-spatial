# Gap Matrix Parity Spec

## Bounded Goal

Close every inference gap in `src/mlx_spatial/` against the three vendor references (`vendors/TRELLIS.2`, `vendors/sam-3d-objects`, `vendors/HY-World-2.0`) so that all three pipelines produce numerically matching output on Apple Silicon, with Mac-native alternatives for CUDA-blocked features.

## Broader Intent

mlx-spatial aims to be a complete, performance-competitive MLX-native inference library for 3D spatial models on Apple Silicon. This spec closes the inference parity gap so future work can focus on performance optimization rather than missing features.

## Work Scale and Shape

- Scale: roadmap
- Shape: parity

## Selected Lenses

- **product**: Affects what users can produce (missing inference outputs, missing modes)
- **engineering**: Touches architecture, data flow, and MLX/Metal implementation

## Target User or Stakeholder

Developers running TRELLIS.2, SAM 3D Objects, and HY-World 2.0 inference on Apple Silicon workstations who need output fidelity and feature parity with the official implementations.

## Constraints and Risks

See `spec/constraints.md` for full detail. Summary:

- **Metal GS rasterizer**: Must replace CUDA-only gsplat for texture baking. Significant standalone project; feasibility depends on Metal compute shader maturity. Assumed target: M1 baseline (unified memory architecture).
- **Parity verification**: Depends on local PyTorch reference outputs via `test_*_parity.py` files. Not CI-gated. Risk: reference outputs may drift across vendor versions.
- **HY-World 2.0 backbone**: 13+ transformer layer modules need individual MLX ports. Risk: monolithic `hyworld2_worldmirror.py` may need restructuring rather than inline expansion.
- **MOT variant**: Adds pose-aware multi-object inference to SAM 3D. Increases SAM 3D scope but is explicitly included per user decision.
- **No runtime PyTorch/CUDA**: All closed gaps must produce matching output via MLX/Metal only.

## Required Outcome

A structured gap matrix with every inference-relevant gap across all three pipelines, closed in priority order, each producing numerically matching output against the vendor reference with new parity test coverage. The matrix is in `spec/gap-matrix.md`.

The outcome shape is parity conformance: for each gap ID, the MLX implementation must produce output matching the PyTorch reference within a defined tolerance, verified by a dedicated test.

### Gap Priorities (summary)

| Priority | Gaps |
|----------|------|
| **P0 — Critical blockers** | HY-World 2.0 transformer backbone (13+ layer modules), camera/rotation/geometry utilities, GS activation/SH functions |
| **P1 — High** | SAM 3D MOT variant, Metal GS rasterizer, multi-view render utilities, TRELLIS.2 texturing pipeline, FlexiDualGrid encoder |
| **P2 — Medium** | Sparse spatial interpolation, layout post-optimization, shortcut/distillation model, mesh hole-filling, SAM 3D texturing postprocess |
| **P3 — Low** | Renderers (visualization-only), CUDA-specific backends already replaced, training-only modules |

## Acceptance Criteria

Each gap ID in `spec/gap-matrix.md` must pass its verification check. The acceptance criteria are organized by gap, not by pipeline.

| AC ID | Gap ID(s) | Check |
|-------|-----------|-------|
| AC-01 | HW-* (all HY-World backbone gaps) | `uv run pytest tests/test_hyworld2_*.py` passes; `numeric_parity_verified=true` in parity trace metadata for all heads |
| AC-02 | SAM-MOT | `uv run pytest tests/test_sam3d_*.py` passes with MOT pose output matching reference quaternion/translation/scale within tolerance |
| AC-03 | SAM-GS, HW-GS (Metal GS rasterizer) | Metal GS rendering produces texture maps matching gsplat reference within defined pixel tolerance |
| AC-04 | TR-TEX (TRELLIS.2 texturing) | `uv run pytest tests/test_trellis2_*.py` passes with texturing pipeline producing matching textured GLB output |
| AC-05 | All P0/P1 gaps | Numerically matching output against PyTorch reference for standard test inputs, verified by parity test |
| AC-06 | All P2 gaps | Feature-complete implementation with basic correctness tests (parity verification may be deferred to a later phase) |

## Anti-Goals

- Training infrastructure (trainers, datasets, elastic modules, loss functions)
- Multi-GPU/distributed communication (FSDP, All2All, sequence parallelism)
- Visualization-only features (PBR rendering, voxel rendering, Gradio apps)
- Replacing MLX with another ML framework
- Supporting non-Apple-Silicon hardware
- Performance optimization (separate roadmap phase)
- CI and release automation (separate roadmap phase)
- DINOv2 support in TRELLIS.2 (superseded by DINOv3)

## Scope Coverage Decisions

- **Included**: All items from INTAKE scope coverage, plus MOT variant (included per user decision)
- **Deferred to ROADMAP**: Performance optimization, CI/release infrastructure
- **Anti-goals**: Training, visualization, multi-GPU, Gradio UIs, non-MLX backends
- **Resolved decision**: MOT variant is in scope (not deferred until requested)
- **Assumption**: Metal GS rasterizer targets M1 baseline (unified memory). If M1 proves infeasible, minimum target moves to M2 Pro.

## Blocking Questions and Assumptions

- Metal GS rasterizer targets M1 unified memory architecture. If this proves infeasible, the minimum target will be revised to M2 Pro.
- The HY-World 2.0 parity flag (`numeric_parity_verified=false`) remains `false` until all backbone gaps are verified.
- Parity verification is dev-only (local PyTorch checkout) until CI is established in a later phase.