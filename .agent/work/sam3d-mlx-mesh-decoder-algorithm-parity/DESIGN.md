# SAM3D MLX Mesh Decoder Algorithm Design

## Scope

This design covers only the SAM3D Objects mesh path after Stage-2 SLat has already produced sparse coordinates and features. It does not change MoGe, preprocessing, SS, SLat flow, Gaussian PLY export, or introduce PyTorch/CUDA runtime dependencies.

## Runtime Dataflow

The mesh path starts from:

- `slat.coords`: sparse coordinates with shape `[N, 4]` in batch/x/y/z order.
- `slat.feats`: denormalized SLat features with shape `[N, C]`.
- `slat_decoder_mesh` config and safetensors.

The path should run:

1. Shared sparse decoder torso:
   - `input_layer`
   - absolute position embedding
   - decoder transformer blocks
   - final layer norm
2. Mesh-only sparse upsample:
   - `SparseSubdivideBlock3d` at base resolution
   - `SparseSubdivideBlock3d` at doubled resolution
   - final sparse `out_layer`
3. Sparse feature interpretation:
   - `sdf`: 8 corner scalar values per cube
   - `deform`: 8 corner offsets with 3 channels
   - `weights`: 21 FlexiCubes weights
   - optional `color`: 8 corner color/normal attributes when configured
4. SparseFeatures2Mesh-style conversion:
   - add `sdf_bias = -1.0 / extraction_resolution`
   - aggregate sparse cube-corner attrs onto unique vertices by mean
   - assemble dense vertex and cube attributes with memory guards
   - deform dense grid vertices with upstream `tanh(deform)` rule
5. FlexiCubes-style inference:
   - identify surface cubes and surface edges from SDF signs
   - compute case ids and resolve ambiguous cases using official tables
   - compute dual vertices from edge zero crossings and weights
   - triangulate into vertices/faces, carrying optional vertex colors
6. Basic GLB export:
   - write non-empty Blender-readable `.glb` through the existing basic writer
   - record mesh stats and blocker details in trace metadata

## Module Shape

Keep the public CLI unchanged. Implementation can add a focused mesh module, for example `src/mlx_spatial/sam3d_mesh.py`, while reusing:

- `src/mlx_spatial/sam3d_decoder.py` for shared decoder transformer pieces
- `src/mlx_spatial/sam3d_slat.py` sparse tensor, sparse conv, and sparse linear helpers
- `src/mlx_spatial/sam3d_export.py` for the basic GLB writer
- `src/mlx_spatial/sam3d_inference.py` for orchestration and trace wiring

If private helpers need reuse, expose narrow public wrappers rather than duplicating large blocks.

## Memory Guards

Dense extraction can allocate arrays at `resolution * 4`, usually 256 for the active mesh decoder. Guard before materializing:

- dense vertex attrs: `(res + 1)^3 * channels`
- dense cube weights: `res^3 * 21`
- FlexiCubes intermediate surface edge and vertex arrays

Guard failures must return a structured `mesh-decoder` blocker and must not write `.glb`.

## Strictness

The requested GLB must come from `slat_decoder_mesh` and SparseFeatures2Mesh/FlexiCubes-style extraction. Do not fall back to Gaussian splats, point clouds, cube previews, marching-cubes placeholders, or synthetic fixture geometry in the live command.
