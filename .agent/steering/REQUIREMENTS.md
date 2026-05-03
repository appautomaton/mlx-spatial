# Requirements

## Observed Constraints

- The root project has no package manifest, source tree, tests, or README observed yet, so implementation planning must begin by defining the library boundary.
- Vendored projects are multiple independent upstream/reference projects and should not be treated as one coherent root application.
- The current reference corpus spans at least four capability areas: image-to-3D generation, object reconstruction, geometric prediction, and Apple Silicon porting patterns.
- `trellis-mac` targets macOS Apple Silicon with Python 3.11+, PyTorch MPS, optional Metal tooling, and documented fallbacks.
- `TRELLIS.2` documents Linux/NVIDIA/CUDA assumptions and should be treated as an upstream CUDA-first reference.
- `sam-3d-objects` is separately packaged as `sam3d_objects` using Hatch/Hatchling and dynamic requirements files.

## Inferred Constraints

- First-party implementation should avoid importing vendor code directly until licensing, API, and porting scope are decided.
- MLX should be the target runtime for new library code, with vendor PyTorch/CUDA/MPS code used as behavioral and architectural reference.
- Early work should prioritize a narrow slice that can be verified locally on Apple Silicon before attempting broad model coverage.

## Unknowns

- Root package name and module layout.
- Initial model target: TRELLIS/O-Voxel, SAM 3D object reconstruction, Hunyuan geometric prediction, or shared spatial primitives.
- Supported hardware floor, memory floor, and acceptable fallbacks.
- Test strategy for numerical parity, shape contracts, and asset output quality.
- License compatibility for adapting code, weights, configs, or assets from vendored projects.
- Whether downloaded checkpoints or generated assets should be managed inside or outside the repository.

## Non-Goals For Now

- Do not claim root install/build/test commands until a root manifest exists.
- Do not collapse vendored projects into a single architecture without selecting a first target.
- Do not port CUDA-only kernels wholesale without first identifying the MLX equivalent or a smaller abstraction boundary.
- Do not treat vendor benchmark numbers as root project performance targets.

## Evidence Anchors

- Root shape: `.`
- Vendor boundary: `vendors/`
- Apple Silicon constraints: `vendors/trellis-mac/README.md`, `vendors/trellis-mac/pyproject.toml`
- CUDA/Linux constraints: `vendors/TRELLIS.2/README.md`, `vendors/HunyuanWorld-Mirror/README.md`
- SAM package constraints: `vendors/sam-3d-objects/pyproject.toml`

List the accepted product and technical constraints.
