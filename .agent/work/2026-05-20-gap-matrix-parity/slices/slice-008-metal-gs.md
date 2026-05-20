# Slice 8: Metal GS Rasterizer Detail

## Objective

Implement a Mac-native Gaussian splatting rasterizer using Metal compute shaders for texture baking and multi-view rendering, replacing the CUDA-only gsplat dependency.

## Architecture

### Python API (`src/mlx_spatial/gs_rasterize.py`)

- `GaussianRasterizeResult`: dataclass with rendered RGBA image, depth buffer, and pixel count
- `rasterize_gaussians(means_3d, quaternions, scales, opacities, sh_features, camera_params, image_size)`: main entry point
- `GaussianSplatRenderer`: class wrapping the Metal pipeline state

### Metal Compute Shader (`src/mlx_spatial/metal/gs_rasterize.metal`)

- Vertex projection kernel: projects 3D Gaussian means to 2D screen coordinates
- Tile-based sorting kernel: assigns Gaussians to tiles, sorts by depth
- Alpha-blended rendering kernel: front-to-back alpha composition per tile

### Integration Points

- SAM 3D: `sam3d_export.py` and `sam3d_inference.py` use `rasterize_gaussians` for texture baking
- HY-World: `hyworld2_export.py` uses `rasterize_gaussians` for multi-view output
- TRELLIS.2: indirect via potential future multi-view checks

## Constraints

- Target: M1 unified memory (Apple Silicon minimum). If M1 performance is insufficient for real-time use, document and revise minimum to M2 Pro.
- Must not import PyTorch, gsplat, or diff-gaussian-rasterization at runtime.
- Must produce output matching gsplat reference within 1% pixel tolerance for standard test images.

## Verification

- `tests/test_gs_rasterize.py`: Unit tests for projection, sorting, and alpha-blended rendering
- Parity test: render same Gaussians with Metal GS and gsplat (via PyTorch dev reference), compare pixel values
- Visual inspection checkpoint: human verifies rendered output looks correct