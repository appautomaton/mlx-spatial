# Gap Matrix

Pipeline × gap × priority × dependency. Organized by priority, not by pipeline.

Gap IDs: `TR-*` (TRELLIS.2), `SAM-*` (SAM 3D), `HW-*` (HY-World 2.0), `SHARED-*` (cross-pipeline).

## P0 — Critical Inference Blockers

Gaps without which the pipeline cannot produce correct inference output.

| Gap ID | Pipeline | Module | Vendor Path | Description | Depends On | Verification |
|--------|----------|--------|-------------|-------------|------------|--------------|
| HW-01 | HY-World | VisualGeometryTransformer | `hyworld2/worldrecon/hyworldmirror/models/models/visual_transformer.py` | Alternating-attention transformer backbone. Interleaved frame-attention and global-attention blocks. | HW-02..HW-10 | Parity test: matching token outputs for same input |
| HW-02 | HY-World | Attention layers | `hyworld2/.../layers/attention.py` | Multi-head self-attention with QK-norm, Flash Attention, RoPE | None | Parity test: matching attention output |
| HW-03 | HY-World | Transformer blocks | `hyworld2/.../layers/block.py` | Block, DistBlock, NestedTensorBlock (LayerNorm→Attn→LN→MLP, LayerScale, DropPath) | HW-02 | Parity test: matching block output |
| HW-04 | HY-World | MLP / SwiGLU | `hyworld2/.../layers/mlp.py`, `swiglu_ffn.py` | Standard MLP, SwiGLU, MlpFP32 | None | Parity test: matching MLP output |
| HW-05 | HY-World | Patch embedding | `hyworld2/.../layers/patch_embed.py` | PatchEmbed, PatchEmbed_Mlp, PixelUnshuffle | None | Parity test: matching embedding output |
| HW-06 | HY-World | 2D RoPE | `hyworld2/.../layers/rope.py`, `norm_rope.py` | RotaryPositionEmbedding2D, NormalizedRotaryPositionEmbedding2D | None | Parity test: matching position encoding |
| HW-07 | HY-World | LayerScale | `hyworld2/.../layers/layer_scale.py` | Learnable per-channel scaling | None | Parity test: matching output |
| HW-08 | HY-World | DropPath | `hyworld2/.../layers/drop_path.py` | Stochastic depth (set to 0 at inference) | None | Correctness test: identity at eval |
| HW-09 | HY-World | DinoVisionTransformer (ViT backbone) | `hyworld2/.../layers/vision_transformer.py` | DINOv2-style ViT used as patch embedder | HW-02..HW-08 | Parity test: matching patch tokens |
| HW-10 | HY-World | WorldMirror top-level model | `hyworld2/.../models/models/worldmirror.py` | Assembles VGT + heads; camera parameter conversion | HW-01, HW-11..HW-15 | Parity test: full forward pass matching |
| HW-11 | HY-World | Camera utilities | `hyworld2/.../models/utils/camera_utils.py` | 9-dim vector↔camera matrix conversions | None | Parity test: roundtrip matching |
| HW-12 | HY-World | Rotation utilities | `hyworld2/.../models/utils/rotation.py` | Quaternion↔rotation matrix conversions | None | Parity test: roundtrip matching |
| HW-13 | HY-World | Depth/geometry utilities | `hyworld2/.../models/utils/geometry.py` | Depth-to-camera/world coords, SE(3) inverse | HW-11 | Parity test: matching 3D coordinates |
| HW-14 | HY-World | GS activation functions | `hyworld2/.../models/utils/act_gs.py` | Quaternion normalization, exp scales, sigmoid opacity, SH reshape | None | Parity test: matching parameter outputs |
| HW-15 | HY-World | Spherical harmonics | `hyworld2/.../models/utils/sh_utils.py` | SH evaluation (degrees 0-4), RGB↔SH conversion | None | Parity test: matching SH output |
| HW-16 | HY-World | Prior normalization | `hyworld2/.../models/utils/priors.py` | Percentile-based pose/depth normalization | None | Parity test: matching normalized outputs |
| HW-17 | HY-World | Post-process geometry | `hyworld2/.../utils/geometry.py` | Depth/normal edge detection, points-to-normals, COLMAP↔OpenCV intrinsics | None | Parity test: matching processed outputs |

## P1 — High Priority

Gaps that block important inference modes or output quality.

| Gap ID | Pipeline | Module | Vendor Path | Description | Depends On | Verification |
|--------|----------|--------|-------------|-------------|------------|--------------|
| SAM-MOT | SAM 3D | MOT sparse structure flow | `.../models/mot_sparse_structure_flow.py` | Multi-Object Transformer variant with pose heads (quaternion, translation, scale) | SAM-SS-FLOW (existing) | Parity test: matching pose + shape output |
| SAM-GS | SAM 3D | Gaussian renderer | `.../renderers/gaussian_render.py` | CUDA-only GS rendering for texture baking and layout optimization | HW-14, HW-15 | Metal GS rasterizer produces matching texture maps |
| SAM-RENDER | SAM 3D | Multi-view render utilities | `.../utils/render_utils.py` | Camera setup + multi-view GS rendering | SAM-GS | Correctness test: multi-view renders |
| TR-TEX | TRELLIS.2 | Texturing pipeline | `.../pipelines/trellis2_texturing.py` | Re-texture existing mesh with new image | TR-SLAT (existing), TR-DEC (existing) | Parity test: matching textured GLB output |
| TR-ENC | TRELLIS.2 | FlexiDualGrid encoder | `.../models/sc_vaes/fdg_vae.py` (encoder half) | Encode mesh to SLat (needed by texturing pipeline) | TR-TEX | Parity test: matching SLat encoding |
| TR-INTERP | TRELLIS.2 | Sparse spatial interpolation | `.../modules/sparse/spatial/spatial2channel.py` | `sparse_nearest_interpolate`, `sparse_trilinear_interpolate` | None | Parity test: matching interpolation output |
| HW-GS | HY-World | Gaussian rasterization | `.../models/models/rasterization.py` | GS rendering (render, voxel-merge/prune) for inference output | HW-14, HW-15 | Metal GS produces matching renders |
| SHARED-GS | Shared | Metal GS rasterizer | N/A (new implementation) | Mac-native Gaussian splatting via Metal compute shaders | N/A | Cross-pipeline: matching renders against gsplat reference |

## P2 — Medium Priority

Gaps that improve output quality or enable faster inference but are not blockers for basic output.

| Gap ID | Pipeline | Module | Vendor Path | Description | Depends On | Verification |
|--------|----------|--------|-------------|-------------|------------|--------------|
| SAM-LAYOUT | SAM 3D | Layout post-optimization | `.../pipeline/layout_post_optimization_utils.py` | ICP alignment, render-and-compare, pose refinement | SAM-GS, SAM-RENDER | Correctness test: scene layout improvement |
| SAM-SHORTCUT | SAM 3D | Shortcut/distillation model | `.../generator/shortcut/model.py` | Consistency-distilled model for fewer inference steps | SAM-SS-FLOW (existing), SAM-SLAT-FLOW (existing) | Parity test: matching output with fewer steps |
| SAM-HOLES | SAM 3D | Multi-view hole filling | `.../utils/postprocessing_utils.py::_fill_holes()` | Multi-view rasterization + igraph mincut for interior face removal | SAM-GS | Correctness test: cleaner meshes |
| TR-REGRID | TRELLIS.2 | Dual-contouring remeshing | `o-voxel/o_voxel/postprocess.py` (remesh) | Alternative mesh improvement path (vendor uses cumesh; MLX uses fast-simplification) | None | Quality comparison: not direct parity |
| HW-GRID | HY-World | Positional grid utilities | `.../models/utils/grid.py` | UV grid creation, sinusoidal Fourier embeddings | None | Parity test: matching embeddings |
| HW-FRUSTUM | HY-World | Frustum masking | `.../models/utils/frustum.py` | View frustum intersection for training masks | None | Skip at inference; training-only |

## P3 — Low Priority / Already Covered

Gaps that are training-only, visualization-only, or already covered by an alternative approach.

| Gap ID | Pipeline | Module | Reason |
|--------|----------|--------|--------|
| TR-SS-ENC | TRELLIS.2 | SparseStructureEncoder | Training-only; inference only needs decoder |
| TR-ELASTIC | TRELLIS.2 | ElasticSLatFlowModel | Training-only gradient checkpointing |
| TR-WIN-ATTN | TRELLIS.2 | Windowed sparse attention | Not used by current checkpoints |
| TR-RENDER | TRELLIS.2 | PBR/voxel renderers | Visualization-only |
| SAM-RF | SAM 3D | SLatRadianceFieldDecoder | Commented out in pipeline |
| SAM-OCTREE | SAM 3D | DfsOctree/Strivec | Not used in current pipeline |
| SAM-VIS | SAM 3D | SceneVisualizer/Plotly | Visualization-only |
| HW-COMM | HY-World | All2All/AllGather | Multi-GPU only; not applicable to MLX single-device |
| HW-GRADIO | HY-World | Gradio app | UI, not inference |
| HW-WARN | HY-World | Warning suppression | Python-specific, not needed |
| SHARED-TRAIN | All | Trainers, datasets, data toolkit | Training-only |
| SHARED-CUDA-ALT | All | spconv, torchsparse, flex_gemm backends | Already covered by MLX native `sparse_conv.py` |

## Dependency Graph (P0 → P1)

```
HW-02..HW-08 (layers) → HW-09 (ViT) → HW-01 (VGT) → HW-10 (WorldMirror)
HW-11..HW-13 (cam/rot/geo) → HW-10
HW-14..HW-15 (GS activation/SH) → HW-GS, SAM-GS, SHARED-GS
HW-16..HW-17 (priors/post-geo) → HW-10

SHARED-GS (Metal GS rasterizer) → SAM-GS, HW-GS → SAM-RENDER, SAM-LAYOUT
SAM-SS-FLOW (existing) → SAM-MOT
TR-TEX → TR-ENC
TR-INTERP (standalone)
```