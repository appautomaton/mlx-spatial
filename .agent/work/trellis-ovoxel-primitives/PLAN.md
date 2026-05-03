# PLAN: TRELLIS.2 O-Voxel-Inspired MLX Primitives

## Goal

Implement a small `mlx_spatial` O-Voxel/sparse-grid primitive surface with default MLX-only tests and optional local PyTorch parity checks gated behind an explicit marker or environment flag.

## Architecture Approach

Add a focused `mlx_spatial.ovoxel` module that covers coordinate/grid mechanics only, while keeping the public API usable for later SAM3D and Hunyuan-style geometry work:

- Generate dense integer coordinates for an N-D grid.
- Flatten N-D coordinates to row-major linear indices.
- Unflatten row-major linear indices back to N-D coordinates.
- Apply simple coordinate bounds masks.

Usability requirements for this slice:

- Use model-neutral function names and docstrings; TRELLIS.2 motivates the work, but the API should not require TRELLIS.2 concepts to be useful.
- Document row-major coordinate/index ordering in function docstrings and tests so later model integrations do not guess shape conventions.
- Keep return values as plain MLX arrays with predictable shapes and integer dtypes, not custom classes or model-specific containers.
- Avoid names that imply mesh extraction, neural layers, checkpoints, or a complete O-Voxel implementation.

Keep the default runtime dependency list unchanged: `mlx` only. Add pytest marker configuration for optional parity tests, but do not add PyTorch as a dependency. If parity scaffolding is added, it must skip unless explicitly enabled and able to use `/Users/ac/dev/ai/ai-frameworks/pytorch`.

No separate `DESIGN.md` is required; this is a small primitive module plus tests and documentation.

## Ordered Task Sequence

### Slice 1: O-Voxel Primitive Module

**Objective:** Add a small `mlx_spatial.ovoxel` namespace for model-neutral sparse coordinate/grid helpers.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/ovoxel.py`, `src/mlx_spatial/__init__.py`
**Context budget:** ~10% of context window
**Produces:** Exported MLX primitive functions for dense coordinates, coordinate flattening, index unflattening, and bounds masks with model-neutral names and docstrings.
**Acceptance criteria:**
- `mlx_spatial.ovoxel` imports successfully.
- Public helpers use `mlx.core` arrays and do not import vendor code.
- Functions reject invalid shapes or coordinate ranks with clear `ValueError`s.
- Function docstrings state shape conventions, row-major ordering, and return shapes.
- Existing `regular_grid` export remains working.
**Verification:** `uv run python -c "import mlx_spatial; import mlx_spatial.ovoxel; import mlx.core as mx"`
**Auto-continue:** yes

### Slice 2: MLX-Only Primitive Tests

**Objective:** Verify O-Voxel primitive behavior using deterministic MLX-only tests.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `tests/test_ovoxel.py`
**Context budget:** ~10% of context window
**Produces:** Tests for dense coordinate shape/order, flatten/unflatten round trip, bounds masks, invalid inputs, and model-neutral usability.
**Acceptance criteria:**
- Tests assert exact values for small grids.
- Tests verify round-trip behavior between coordinates and linear indices.
- Tests include at least one generic sparse-grid use case that is not named after TRELLIS.2, SAM3D, or Hunyuan.
- Tests do not import Torch, Transformers, Hugging Face, or vendors.
- Full default suite passes.
**Verification:** `uv run pytest`
**Auto-continue:** yes

### Slice 3: Optional Parity Gate Scaffolding

**Objective:** Add a skipped-by-default parity test path that can later compare MLX primitives against local PyTorch.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `pyproject.toml`, `tests/test_ovoxel_parity.py`
**Context budget:** ~10% of context window
**Produces:** A pytest marker and optional parity test scaffolding gated by an environment variable such as `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
**Acceptance criteria:**
- `uv run pytest` skips parity checks by default and still passes without PyTorch.
- Parity scaffolding does not add PyTorch to base dependencies.
- If the environment flag is absent, tests do not require `/Users/ac/dev/ai/ai-frameworks/pytorch`.
- Test code documents that enabled parity must use the local PyTorch checkout path.
**Verification:** `uv run pytest`
**Auto-continue:** yes

### Slice 4: Documentation Update

**Objective:** Document the O-Voxel primitive surface and optional local PyTorch parity workflow.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `README.md`
**Context budget:** ~8% of context window
**Produces:** README notes for sparse-grid/O-Voxel-inspired primitives, default MLX-only tests, optional local PyTorch parity gating, and future model integration boundaries.
**Acceptance criteria:**
- README states O-Voxel-inspired primitives are coordinate/grid helpers, not full TRELLIS.2 inference.
- README states the helpers are intended to stay reusable for future TRELLIS.2, SAM3D, and Hunyuan-family integrations.
- README states default tests remain MLX-only.
- README documents the optional parity flag and local PyTorch checkout path without making it mandatory.
- README keeps Hugging Face/model weights out of this slice.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing And Topology

- Slice 1: direct execution.
- Slice 2: direct execution after Slice 1 verification passes.
- Slice 3: direct execution after Slice 2 verification passes.
- Slice 4: direct execution after Slice 3 verification passes.
- Auto-continue chain: Slice 1 -> Slice 2 -> Slice 3 is safe because each slice has narrow verification and no checkpoint decision.
- Checkpoint boundary: Slice 4 ends the change; stop before adding model inference, Hugging Face download flows, vendor imports, or real Torch dependency installation.
- Parallel-safe groups: none. The files are small but ordered exports/tests/docs reduce rework.
- Subagents: not required. The change is local to one package module, tests, pytest marker config, and README.

## Verification Commands

- Slice 1: `uv run python -c "import mlx_spatial; import mlx_spatial.ovoxel; import mlx.core as mx"`
- Slice 2: `uv run pytest`
- Slice 3: `uv run pytest`
- Slice 4: `uv run pytest`

## Risk Handling

- PyTorch parity is optional and skipped by default, so local PyTorch build issues cannot block default tests.
- The active MLX dependency remains the installable `mlx` package from the current bootstrap unless a later spec changes it.
- Vendor references may inform behavior, but implementation must be first-party and vendor-free at runtime.
- The primitive set deliberately stops before mesh extraction, sparse convolutions, neural layers, checkpoint loading, or TRELLIS.2 inference.
- API usability risk is handled by requiring model-neutral names, explicit coordinate ordering, plain MLX array returns, and documentation that these helpers are shared foundations for future model integrations.

## Context Budget For This Change

Estimated total execution context: ~38% of the context window.

## Recommended Next Skill

`auto-eng-review`

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan keeps the implementation small, MLX-only by default, and explicitly requires model-neutral APIs with documented coordinate/index conventions.
- Concern: The main risk is semantic drift: future model integrations could misuse these helpers if row-major ordering, shape rank, and dtype expectations are not enforced in both docstrings and tests.
- Action: Proceed to `auto-execute` and treat missing ordering/rank/dtype assertions as implementation blockers, not cleanup.
- Verified: canonical plan read, slice dependencies traced, optional PyTorch parity gate checked, API usability constraints reviewed against future TRELLIS.2/SAM3D/Hunyuan reuse.

## Execution Evidence

- Slice 1 complete: `src/mlx_spatial/ovoxel.py` added with model-neutral MLX helpers for dense coordinates, row-major flattening, row-major unflattening, and bounds masks; `src/mlx_spatial/__init__.py` exports the helpers; `uv run python -c "import mlx_spatial; import mlx_spatial.ovoxel; import mlx.core as mx"` passed.
- Slice 2 complete: `tests/test_ovoxel.py` added MLX-only tests for exact coordinate order, int32 dtype, flatten/unflatten round trip, bounds masks, invalid shapes, rank mismatch, and a generic sparse-grid use case; `uv run pytest` passed.
- Slice 3 complete: `pyproject.toml` includes the `torch_parity` marker; `tests/test_ovoxel_parity.py` skips unless `MLX_SPATIAL_RUN_TORCH_PARITY=1` and points to `/Users/ac/dev/ai/ai-frameworks/pytorch`; default `uv run pytest` skipped parity without requiring PyTorch.
- Slice 4 complete: `README.md` documents sparse-grid helpers, model-neutral reuse for TRELLIS.2/SAM3D/Hunyuan-family integrations, MLX-only default tests, optional local PyTorch parity, and no Hugging Face/model-weight requirement; `uv run pytest` passed with 8 passed and 1 skipped.
