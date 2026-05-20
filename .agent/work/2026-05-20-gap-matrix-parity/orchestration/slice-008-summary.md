# Slice 8 Coordination Summary

Status: code complete, checkpoint pending

## Scope

Slice 8 implemented a correctness-first Mac-native Gaussian splat rasterizer:

- `src/mlx_spatial/gs_rasterize.py`
- `src/mlx_spatial/metal/gs_rasterize.metal`
- `tests/test_gs_rasterize.py`
- `src/mlx_spatial/__init__.py` exports

## Subagent Coordination

- Discovery confirmed MLX custom Metal kernels are available through `mlx.core.fast.metal_kernel`; float atomics are not viable for the local MLX Metal path.
- Data-contract discovery recommended matrix-first camera inputs, degree-0 SH/direct RGB support, and explicit handling of HY/SAM quaternion conventions.
- Implementer added the rasterizer, Metal kernel, package exports, and tests.
- First spec review requested covariance projection because the initial implementation used isotropic screen-space footprints.
- Follow-up implementation added XYZW quaternion + scale covariance projection, Jacobian projection to screen-space conics, and anisotropic Metal rendering.
- Second spec review approved the bounded implementation as code complete, with human visual/gsplat parity checkpoint still open.
- Quality review requested fixing the public RGBA contract from premultiplied to straight RGB.
- Final quality re-review approved after CPU/Metal unpremultiplication and a straight-RGBA regression test.

## Verification

- `uv run pytest tests/test_gs_rasterize.py -v` - PASS (9 passed)
- Export smoke check for `GaussianCameraParams`, `GaussianRasterizeResult`, `GaussianSplatRenderer`, `rasterize_gaussians`, and `rasterize_gaussians_cpu_reference` - PASS
- Runtime dependency scan over new renderer/test files found no PyTorch, gsplat, diff-gaussian, or CUDA runtime imports.
- `git diff --check` - PASS

## Checkpoint

Keep Slice 8 checkpoint open:

- Human visual inspection is pending.
- Dev-only gsplat/PyTorch reference parity against standard rendered images is pending.
- Current 1% parity evidence is Metal-vs-deterministic CPU reference, including anisotropic quaternion rotation.

## Known Limits

- Correctness-first renderer: Python projects/sorts splats; Metal renders one pixel per thread without float atomics.
- Not a performance-complete tiled/binning renderer.
- Higher-degree SH is not implemented unless callers pre-evaluate RGB. Degree-0 SH and direct RGB are supported.
