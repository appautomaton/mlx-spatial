# Production Pipeline Parity Spec

## Bounded Goal

Close all remaining inference gaps across SAM 3D, TRELLIS.2, and HY-World 2.0 — completing GS/SH rasterization, the TRELLIS.2 texturing pipeline, sparse interpolation, and cross-pipeline mesh postprocessing with Mac-native alternatives for every CUDA-dependent feature.

## Broader Intent

mlx-spatial aims to be a complete, MLX-native inference library for 3D spatial models on Apple Silicon. This spec closes the final inference parity gap so the library delivers full production output across all three pipelines with no CUDA runtime dependency.

## Work Scale and Shape

- Scale: capability
- Shape: mixed (parity + migration)

## Selected Lenses

- **product**: Affects what users can produce — texturing output mode, rendered GS output, postprocessed mesh quality
- **engineering**: Touches GS rasterizer internals, sparse data structures, mesh algorithms, and Metal compute kernels

## Target User or Stakeholder

Developers running TRELLIS.2, SAM 3D Objects, and HY-World 2.0 inference on Apple Silicon workstations who need output fidelity and feature parity with the official implementations.

## Gap Inventory

10 gaps in 3 groups. Gap IDs from the original matrix in `spec/gap-matrix.md` (P0 closed in Phase 1). Previously-listed SAM-MOT and SAM-SHORTCUT are already ported and removed from scope.

### Group 1: GS/SH Completion

| # | Gap | Source Module | What's Missing |
|---|---|---|---|
| G1.1 | Higher-degree SH in rasterizer | `gs_rasterize.py::_colors_from_sh_features()` | Raises `NotImplementedError` for `sh_degree > 0`. `hyworld2_sh.eval_sh()` exists for degrees 0-3 but is not integrated. |
| G1.2 | SH degree 4 | `hyworld2_sh.py::eval_sh()` | `_C4` constants defined (lines 33-43) but evaluation terms not implemented. Vendor `sh_utils.py` has full degree-4. |
| G1.3 | HY-World GS rasterization | `hyworld2_export.py` / `hyworld2_heads.py` | GS head produces attributes; `rasterize_gaussians()` is never called. Export outputs raw params with metadata note "no CUDA rasterization." |

### Group 2: TRELLIS.2 Texturing Pipeline

| # | Gap | Source Module | What's Missing |
|---|---|---|---|
| G2.1 | `mesh_to_flexible_dual_grid` | New module (replaces `o_voxel.convert`) | Native C++/CUDA function that converts a triangle mesh → FlexiDualGrid (voxel indices + dual vertices + intersected flags). **Critical blocker** for texturing pipeline. Reverse direction (`flexible_dual_grid_to_mesh_np`) already exists in `ovoxel.py`. |
| G2.2 | `Trellis2TexturingPipeline` | New pipeline class | Standalone texturing-only pipeline: preprocess image → preprocess mesh → encode mesh to shape SLat via FlexiDualGrid encoder → sample texture SLat via FlowEuler → decode to PBR voxels → bake texture → export textured GLB. Existing modules (encoder, flow, decoder, export) exist but are not wired into a reusable texturing entry point. |
| G2.3 | Sparse interpolation | New module `spatial_interp.py` | `sparse_nearest_interpolate` and `sparse_trilinear_interpolate` from vendor `spatial2channel.py`. Used by sparse unet Spatial2Channel/Channel2Spatial blocks. |

### Group 3: Mesh Postprocessing

| # | Gap | Source Module | What's Missing |
|---|---|---|---|
| G3.1 | Layout post-optimization | New `sam3d_layout.py` (replaces `layout_post_optimization_utils.py`) | 1295-line vendor file: ICP alignment, render-and-compare, occlusion checks, pose refinement. Depends on PyTorch3D, Open3D, scipy. Needs Mac-native alternatives. Existing `sam3d_render.py` has basic rigid ICP (`optimize_sam3d_layout_alignment`) but not the full occlusion-aware pipeline. |
| G3.2 | Multi-view hole-filling | New `sam3d_postprocess.py` (replaces `_fill_holes` in vendor) | 327-line vendor function: Hammersley sphere sampling → face visibility scoring → igraph mincut on dual graph → cut validation → inner face removal. Depends on igraph, utils3d.torch, pymeshfix, pyvista. Existing `fill_flexible_dual_grid_mesh_holes()` in `ovoxel.py` is a different approach (fan-triangulation, TRELLIS.2-specific). |
| G3.3 | Dual-contouring remeshing | `ovoxel.py` extension (replaces `cumesh.remeshing`) | Vendor uses `cumesh.remeshing.remesh_narrow_band_dc`. MLX uses `fast_simplification` for decimation but has no remeshing. Gap is acknowledged in slice 11 report. |
| G3.4 | Grid utilities | `grid.py` extension | Existing `grid.py` only has `regular_grid()`. Vendor has UV grid creation and sinusoidal Fourier embeddings in `hyworld2/.../grid.py`. |

## Required Outcome

For each gap, the MLX implementation must produce output that either numerically matches the PyTorch vendor reference within defined tolerance (parity target) or produces correct-looking output that matches the vendor's functional behavior when numeric parity is impractical due to replaced native libraries (migration target). Each gap is verified by a dedicated test or set of tests.

The three groups are designed for parallel subagent execution:
- Group 1 touches `gs_rasterize.py`, `hyworld2_sh.py`, `hyworld2_export.py`, and associated tests
- Group 2 touches new modules + `trellis2_inference.py`, `trellis2_decode.py`, `trellis2_slat.py`, `trellis2_export.py`, `ovoxel.py`
- Group 3 touches `sam3d_render.py`, `sam3d_mesh.py`, `ovoxel.py`, `grid.py`, `export_utils.py`

## Constraints

- No runtime PyTorch, CUDA, or vendor-native library dependencies (gsplat, diff-gaussian-rasterization, o_voxel, cumesh, nvdiffrast, flex_gemm, pytorch3d, open3d, igraph)
- Metal compute kernels must target M1 unified memory baseline (no float atomics, threadgroup memory within M1 limits)
- Existing `GaussianSplatRenderer` in `gs_rasterize.py` is the foundation — gaps extend it, not replace it
- Existing test patterns (`test_*_parity.py`, `test_*_*.py`) are sufficient for verification
- Parity verification is dev-only (local PyTorch reference checkout) until CI is established
- G2.1 (mesh_to_flexible_dual_grid) blocks G2.2 (Trellis2TexturingPipeline); no other cross-group dependencies

## Risks

- **G2.1 is the highest-risk gap.** Voxelizing an arbitrary triangle mesh in pure Python/NumPy without `o_voxel` is non-trivial. Mitigation: the reverse direction (`flexible_dual_grid_to_mesh_np`) exists and proves FDG topology is understood. If forward conversion proves infeasible at spec quality, fall back to documenting the limitation and keeping texturing gated on the image-to-3D path (where shape SLat comes from FlowEuler sampling, not mesh encoding).
- **G3.1 is the largest single gap** (1295 vendor lines). Replacing PyTorch3D rasterization, Open3D mesh operations, and scipy image processing with Mac-native alternatives requires significant engineering. Mitigation: `sam3d_render.py` already has basic rigid ICP; extend incrementally.
- **G3.2 depends on igraph** (C library). A pure Python graph mincut or alternative hole-filling strategy may be needed. Mitigation: `ovoxel.py` fan-triangulation approach could be generalized to SAM 3D.
- **G3.3 may not achieve full numeric parity.** `cumesh.remeshing.remesh_narrow_band_dc` is a GPU-accelerated narrow-band dual-contouring algorithm. A Mac-native equivalent may differ in output. Target: functional equivalence (comparable mesh quality), not bitwise match.
- SH degree 4 (G1.2) and HY-World rasterization (G1.3) are low-risk — near-trivial wiring.

## Acceptance Criteria

| AC | Gap(s) | Check |
|---|---|---|
| AC-01 | G1.1 | `sh_degree > 0` no longer raises `NotImplementedError`; rasterizer produces correct output for degrees 1-3 using `hyworld2_sh.eval_sh()`. Parity test: matching RGB output against pre-evaluated reference. |
| AC-02 | G1.2 | `hyworld2_sh.eval_sh(deg=4, ...)` produces correct SH evaluation matching vendor `sh_utils.py` output within float32 tolerance. |
| AC-03 | G1.3 | `hyworld2_export.py` produces rendered GS images via `rasterize_gaussians()` instead of raw params only. Metadata no longer says "no CUDA rasterization." |
| AC-04 | G2.1 | `mesh_to_flexible_dual_grid(vertices, faces, grid_size)` returns voxel indices, dual vertices, and intersected flags that produce a valid mesh when round-tripped through `flexible_dual_grid_to_mesh_np()`. |
| AC-05 | G2.2 | `Trellis2TexturingPipeline.run(image, mesh)` produces a textured GLB file. Test: texture bake output contains base_color/metallic/roughness/alpha channels on a UV-unwrapped mesh. |
| AC-06 | G2.3 | `sparse_nearest_interpolate` and `sparse_trilinear_interpolate` produce output matching vendor reference within tolerance. |
| AC-07 | G3.1 | Layout post-optimization improves scene layout quality (measured by render-and-compare score) compared to unoptimized layout. |
| AC-08 | G3.2 | Multi-view hole-filling removes interior faces from SAM 3D meshes without removing valid exterior faces. Test: hole-filled mesh has fewer interior triangles and comparable exterior surface. |
| AC-09 | G3.3 | Dual-contouring remeshing produces a watertight mesh from FlexiDualGrid fields. Test: output mesh is manifold and comparable in quality to the existing `fast_simplification` path. |
| AC-10 | G3.4 | `grid.py` provides UV grid creation and Fourier embedding utilities. Test: output matches vendor reference within tolerance. |

## Anti-Goals

- Training infrastructure (trainers, datasets, elastic modules, loss functions)
- Distributed/multi-GPU communication (FSDP, All2All, sequence parallelism)
- Visualization-only features (PBR rendering, voxel rendering, Gradio apps)
- Non-Apple-Silicon hardware or non-MLX backends
- DINOv2 support in TRELLIS.2 (superseded by DINOv3)
- Performance optimization and benchmarking (was Phase 5)
- CI, lint, typecheck, and release automation (was Phase 6)
- Tiled/binning GS rasterizer optimization (deferred to future performance phase)

## Scope Coverage Decisions

- **Included**: All 10 gaps across 3 groups as listed in Gap Inventory. Previously-listed SAM-MOT and SAM-SHORTCUT removed after codebase audit confirmed they are already ported.
- **Deferred to ROADMAP.md**: Performance optimization, CI/release infrastructure, tiled Metal GS rendering
- **Anti-goals**: Training, visualization, multi-GPU, Gradio UIs, non-MLX backends, non-Apple-Silicon
- **Already ported (removed from scope)**: SAM-MOT (`sam3d_ss_flow.py`), SAM-SHORTCUT inference (`sam3d_ss_flow.py`), HW-FRUSTUM (training-only)
- **Assumption accepted**: Performance-comparable output (rather than bitwise numeric match) is acceptable for gaps replacing native C++/CUDA libraries (G2.1, G3.1, G3.2, G3.3)

## Multi-Agent Execution Design

The three groups have no cross-group dependencies (only G2.1 → G2.2 is intra-group). This enables parallel subagent execution:

- **Subagent A — GS/SH Completion** (G1.1, G1.2, G1.3): Works in `gs_rasterize.py`, `hyworld2_sh.py`, `hyworld2_export.py`. No external deps.
- **Subagent B — TRELLIS.2 Texturing** (G2.1, G2.2, G2.3): Works in new modules + `trellis2_inference.py`, `ovoxel.py`. Sequential internally (G2.1 before G2.2). G2.3 can run in parallel with G2.1 as a separate sub-task.
- **Subagent C — Mesh Postprocessing** (G3.1, G3.2, G3.3, G3.4): Works in `sam3d_render.py`, `sam3d_mesh.py`, `ovoxel.py`, `grid.py`. All four gaps are independent and can run as parallel sub-tasks.

Each subagent delivers its own test suite. Final integration verifies no regressions across all three pipelines with `uv run pytest`.
