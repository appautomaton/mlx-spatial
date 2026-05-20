# Project

## One-Liner

MLX-first 3D/spatial model inference library that ports TRELLIS.2, SAM 3D Objects, and HY-World 2.0 to Apple Silicon, maintaining the speed and quality of the original reference implementations.

## Why This Repo Exists

- Provide native Apple Silicon inference for 3D generation models that originally require PyTorch and CUDA.
- Replace PyTorch/CUDA dependencies with MLX equivalents while preserving numeric accuracy and output quality.
- Offer deterministic, testable spatial primitives that multiple model pipelines can share.
- Enable local-first weight management without automatic network downloads or runtime Hugging Face calls.

Evidence: `README.md:1-4`, `pyproject.toml:6-8`, user's stated goal.

## Current Users or Operators

- Developers running 3D model inference on Apple Silicon workstations.
- CLI users validating assets, converting weights, or generating 3D outputs.
- Parity engineers comparing MLX outputs against PyTorch reference implementations.

## Current System Model

- Request flow: User provides image (+ mask for SAM3D) → pipeline validates assets → staged inference with structured blockers → writes OBJ/GLB/PLY output.
- Primary surfaces: 3 CLI entrypoints + Python library imports.
- Critical dependencies: Apple MLX runtime, local safetensors checkpoints, xatlas for UV unwrapping.

## Major Surfaces

| Surface | Path | Responsibility |
|---------|------|----------------|
| Public API | `src/mlx_spatial/__init__.py` | 257 named exports across all modules |
| TRELLIS.2 pipeline | `src/mlx_spatial/trellis2*.py` (14 modules) | Image→3D: DINOv3 conditioning, RMBG, sparse structure, shape/texture SLat, decoders, FlexiDualGrid mesh, GLB export |
| SAM 3D pipeline | `src/mlx_spatial/sam3d*.py` (14 modules) | Image+mask→Gaussian PLY + textured GLB: MoGe depth, SS/SLat flow, Gaussian decoder, FlexiCubes mesh, xatlas bake |
| HY-World pipeline | `src/mlx_spatial/hyworld2*.py` (7 modules) | Multi-view reconstruction: camera/depth/normal/points/Gaussian heads, WorldMirror parity |
| Shared primitives | `src/mlx_spatial/{ovoxel,topology,sparse_conv,grid,checkpoint,model_assets,export_utils,mlx_memory}.py` | Model-neutral coordinate, topology, sparse conv, mesh, and checkpoint helpers |
| CLI tools | `pyproject.toml:30-32` | `mlx-spatial-trellis2`, `mlx-spatial-sam3d`, `mlx-spatial-hyworld2` |
| Tests | `tests/test_*.py` (~49 files) | Unit, integration, and parity tests |

## Stack Summary

- Python >=3.11, built with hatchling, managed by uv
- MLX (`mlx`) as the sole ML runtime — no PyTorch/CUDA at inference time
- numpy, scipy, pillow, safetensors for data and tensor I/O
- xatlas, fast-simplification for mesh/UV operations
- PyTorch (optional, dev-only) for parity verification

## Commands

- install: `uv sync`
- test: `uv run pytest`
- parity: `MLX_SPATIAL_RUN_TORCH_PARITY=1 uv run pytest -m torch_parity`
- CLI: `uv run mlx-spatial-trellis2`, `uv run mlx-spatial-sam3d`, `uv run mlx-spatial-hyworld2`

## Decision Principles Already Visible In The Repo

- Correctness first: staged pipelines with structured blockers rather than silent fallbacks (`README.md:194-195`).
- No runtime network access: weights managed locally via CLI, not downloaded during import or inference.
- Deterministic primitives: explicit ordering contracts for sparse conv maps, topology offsets, and mesh operations.
- safetensors-only at runtime: no `.pt`/`.pth` loading in production paths (`README.md:116`).
- Parity as dev tooling: opt-in env var `MLX_SPATIAL_RUN_TORCH_PARITY=1`, not a CI gate.
- Vendor code is reference-only: `vendors/` is never imported at runtime.

## Evidence Anchors

- `README.md` — scope, conventions, CLI commands, pipeline documentation
- `pyproject.toml` — dependencies, entry points, test config
- `src/mlx_spatial/__init__.py` — public API surface
- `.gitignore` — excluded paths (weights, vendors, inputs, outputs)
- `tests/` — test inventory