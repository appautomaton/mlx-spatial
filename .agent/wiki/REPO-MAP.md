# Repo Map

## One-Sentence Model

MLX-first 3D/spatial model inference library for Apple Silicon that ports TRELLIS.2, SAM 3D Objects, and HY-World 2.0 to run natively without PyTorch or CUDA.

## What This Repository Owns

- Native MLX inference implementations for three 3D models: TRELLIS.2, SAM 3D Objects, HY-World 2.0
- Shared spatial primitives: O-Voxel coordinates, sparse voxel topology, sparse convolution maps, mesh/export utilities
- CLI tools for asset validation, weight conversion, inference, and parity checking
- Local weight management (no runtime network downloads)

## Runtime Surfaces

| Surface | Path | Role | Entry Points | Notes |
|---------|------|------|--------------|-------|
| Library core | `src/mlx_spatial/` | Python package | `import mlx_spatial` | 47 modules, single package |
| TRELLIS.2 pipeline | `src/mlx_spatial/trellis2*.py` | Image-to-3D generation | CLI: `mlx-spatial-trellis2` | DINOv3 conditioning, RMBG removal, sparse structure flow, shape/texture SLat, decoders, FlexiDualGrid mesh, GLB export |
| SAM 3D Objects pipeline | `src/mlx_spatial/sam3d*.py` | Image+mask→Gaussian PLY/GLB | CLI: `mlx-spatial-sam3d` | MoGe depth, SS/SLat flow, Gaussian decoder, FlexiCubes mesh, xatlas texture bake |
| HY-World 2.0 pipeline | `src/mlx_spatial/hyworld2*.py` | Multi-view reconstruction | CLI: `mlx-spatial-hyworld2` | Camera/depth/normal/points/Gaussian heads, WorldMirror parity |
| Shared primitives | `src/mlx_spatial/{ovoxel,topology,sparse_conv,grid,checkpoint,model_assets,export_utils,mlx_memory}.py` | Model-neutral helpers | `import mlx_spatial` | Coordinate ops, topology, sparse conv, checkpoint load |
| Test suite | `tests/` | Verification | `uv run pytest` | ~49 test files, `torch_parity` marker |
| Dev tools | `tools/` | Reference dumping | `tools/hyworld2_dump_torch_reference.py` | PyTorch oracle for parity |

## Stack and Infrastructure

- Language: Python >=3.11
- ML runtime: Apple MLX (`mlx`)
- Build: hatchling (`pyproject.toml` `[build-system]`)
- Package manager: uv (`uv.lock`, `uv sync`)
- Runtime deps: mlx, numpy, pillow, safetensors, scipy, fast-simplification, xatlas, pyyaml (`pyproject.toml:11-20`)
- Dev deps: pytest>=8, huggingface-hub>=0.36, pt-safe-loader>=0.1.4 (`pyproject.toml:23-27`)
- Checkpoint format: safetensors only (no `.pt`/`.pth` at runtime)

## Commands That Work Today

- install: `uv sync`
- test: `uv run pytest`
- parity test: `MLX_SPATIAL_RUN_TORCH_PARITY=1 uv run pytest -m torch_parity`
- CLI (trellis2): `uv run mlx-spatial-trellis2 {validate,inspect,probe,dinov3-validate,download-command,generate-shape,generate-textured,attempt-forward-trace}`
- CLI (sam3d): `uv run mlx-spatial-sam3d {validate,inspect,download-command,convert,reconstruct}`
- CLI (hyworld2): `uv run mlx-spatial-hyworld2 reconstruct ...`

## Apps, Packages, and Boundaries

- Single package: `src/mlx_spatial/` — no separate apps or monorepo packages
- Vendored references: `vendors/` (gitignored, not imported at runtime, used for parity comparison)
- Local weights: `weights/` (gitignored, managed via CLI tools)
- Local I/O: `inputs/`, `outputs/` (gitignored)

## External Systems and Integrations

- Hugging Face Hub: weight download via `huggingface-cli` (dev-only, not runtime dep)
- Apple MLX framework: hard runtime dependency
- xatlas: Mac-native UV unwrapping (runtime dep)
- fast-simplification: mesh simplification (runtime dep)
- Local PyTorch checkout at `/Users/ac/dev/ai/ai-frameworks/pytorch`: optional parity reference (inferred from README.md:403)
- DINOv3 model weights (`facebook/dinov3-vitl16-pretrain-lvd1689m`): local only, validated via CLI
- RMBG 2.0 model (`briaai/RMBG-2.0`): gated, non-commercial, local only

## Existing Conventions

### Observed

- Exact-mode staged pipelines with structured blockers for unimplemented stages (`README.md:194-195`)
- safetensors-only checkpoint loading at runtime (`README.md:116`)
- Deterministic, reproducible primitives with explicit ordering contracts (sparse conv map rows, topology offsets)
- No automatic network access or model downloads at import/test/runtime
- Vendor code in `vendors/` is reference-only, never imported
- Parity verification is opt-in dev tooling, not a runtime requirement

### Inferred

- Performance optimization is secondary to correctness; the user's stated goal is to maintain speed *and* quality of original implementations
- Module naming convention: `{model}_{concern}.py` (e.g., `trellis2_slat.py`, `sam3d_decoder.py`)

### Needs Confirmation

- Whether there is a CI pipeline or release workflow (none observed in repo)
- Whether `mlx_memory.py` represents a shared memory management strategy across all pipelines

## Verification and Release Surfaces

- Test runner: pytest (`pyproject.toml:38`)
- Custom marker: `torch_parity` for optional PyTorch comparison (`pyproject.toml:40-41`)
- No CI config, no release automation observed (inferred: not yet configured)
- No typecheck or lint commands observed in `pyproject.toml`

## Likely Hotspots for the First Changes

- `src/mlx_spatial/trellis2_forward.py`: TRELLIS.2 staged inference dispatcher
- `src/mlx_spatial/sam3d_inference.py`: SAM 3D pipeline orchestrator
- `src/mlx_spatial/hyworld2_inference.py`: HY-World inference entry
- `src/mlx_spatial/sparse_conv.py`: Shared weighted sparse convolution (performance-critical)
- `src/mlx_spatial/ovoxel.py`: FlexiDualGrid mesh extraction (complex, surface-level)

## Sources Read

- `README.md` — project scope, all pipeline descriptions, CLI commands, conventions
- `pyproject.toml` — package metadata, dependencies, entry points, test config
- `.gitignore` — gitignored paths (weights, inputs, outputs, vendors)
- `src/mlx_spatial/__init__.py` — public API surface (257 exports)
- `src/mlx_spatial/` directory listing — internal module structure
- `tests/` directory listing — test file inventory
- `tools/` directory listing — dev tooling

## Open Questions

- No CI or release pipeline was observed. Is one planned?
- Is `mlx_memory.py` a cross-pipeline memory strategy or specific to one model?
- Are there performance benchmarks or regression tests beyond parity checks?

## Import Verdict

- steering confidence: high
- recommended next skill: `auto-frame`