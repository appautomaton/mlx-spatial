# PLAN: MLX Spatial Environment Bootstrap

## Goal

Create a root `uv` development environment for `mlx_spatial` that can install, import the first-party package, import MLX, and run initial shape/grid primitive tests while keeping Torch, Transformers, Hugging Face, local framework checkouts, and vendored model setup optional.

## Architecture Approach

Use the smallest root Python package boundary that proves the development loop works:

- `pyproject.toml` defines the root package, `uv` workflow, base MLX dependency, and pytest dev dependency.
- `src/mlx_spatial/` contains the first-party import package.
- `tests/` contains default smoke tests that import `mlx_spatial`, import `mlx.core`, and exercise one minimal shape/grid MLX primitive.
- Optional parity/model-download tooling is documented or isolated for later slices, but not required by default tests.
- `vendors/` remains reference-only and untouched.

The first primitive should be model-neutral shape/grid behavior, not camera, voxel, Gaussian, or model inference code.

## Ordered Task Sequence

### Slice 1: Root Package And uv Metadata

**Objective:** Establish the installable root package boundary for `mlx_spatial`.
**Execution:** direct
**Depends on:** none
**Touches:** `pyproject.toml`, `README.md` if absent or minimal update only
**Context budget:** ~8% of context window
**Produces:** Root package metadata with MLX-first base dependency and pytest dev dependency.
**Acceptance criteria:**
- `pyproject.toml` exists at the repo root.
- The package name/import target is `mlx_spatial`.
- Base dependencies do not require Torch, Transformers, Hugging Face, or local absolute paths.
- Dev/test dependency includes pytest or an equivalent test runner.
**Verification:** `uv sync`
**Auto-continue:** yes

### Slice 2: First-Party Package Skeleton

**Objective:** Add the minimal source package needed for import and a model-neutral shape/grid primitive.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/`
**Context budget:** ~8% of context window
**Produces:** `src/mlx_spatial/__init__.py` and a small primitive module or function using MLX.
**Acceptance criteria:**
- `python -c "import mlx_spatial"` works inside the synced environment.
- The primitive uses `mlx.core` and returns an MLX array or shape value suitable for a smoke test.
- No vendor imports are introduced.
**Verification:** `uv run python -c "import mlx_spatial; import mlx.core as mx"`
**Auto-continue:** yes

### Slice 3: Default Test Loop

**Objective:** Prove the root package and MLX primitive execute under tests.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `tests/`, optional `pyproject.toml` pytest config
**Context budget:** ~8% of context window
**Produces:** Pytest smoke tests for package import, MLX import, and the shape/grid primitive.
**Acceptance criteria:**
- Default tests pass without Torch, Transformers, Hugging Face credentials, local framework checkouts, or vendor setup.
- Tests verify behavior rather than only importing modules.
- Test names make clear that this is bootstrap coverage, not model parity.
**Verification:** `uv run pytest`
**Auto-continue:** yes

### Slice 4: Optional Parity And Download Boundary Documentation

**Objective:** Document how parity/download dependencies are intentionally kept out of the base bootstrap.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `README.md`, optional `pyproject.toml` extras if they remain non-default
**Context budget:** ~8% of context window
**Produces:** Documentation or optional extras explaining Torch/Transformers/Hugging Face/local checkout usage as later parity/model-download resources.
**Acceptance criteria:**
- README or equivalent docs state the base bootstrap commands.
- Docs state that `/Users/ac/dev/ai/ai-frameworks/mlx`, `/Users/ac/dev/ai/ai-frameworks/pytorch`, and `/Users/ac/dev/ai/ai-frameworks/transformers` are local optional resources for this workstation, not required setup.
- Hugging Face tooling is identified as future optional model-download work, not required for base tests.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing And Topology

- Slice 1: direct execution.
- Slice 2: direct execution after Slice 1 verification passes.
- Slice 3: direct execution after Slice 2 verification passes.
- Slice 4: direct execution after Slice 3 verification passes.
- Auto-continue chain: Slice 1 -> Slice 2 -> Slice 3 is safe because each step is small and verified.
- Checkpoint boundary: Slice 4 ends the bootstrap and should be reviewed before adding parity or model-download implementation.
- Parallel-safe groups: none. Slices intentionally build on the package boundary in order.
- Subagents: not needed for the initial implementation; the work touches a small number of root package/test/docs files.

## Verification Commands

- Slice 1: `uv sync`
- Slice 2: `uv run python -c "import mlx_spatial; import mlx.core as mx"`
- Slice 3: `uv run pytest`
- Slice 4: `uv run pytest`

## Product Review Risk Handling

- Hugging Face is not a base dependency in this plan because checkpoint download is outside the import/test bootstrap success signal.
- Torch and Transformers are not base dependencies; they remain optional parity resources for later model slices.
- Local absolute framework paths are documented only as workstation-specific resources and are not required by default setup.
- Vendor projects remain untouched and unimported.

## Context Budget For This Change

Estimated total execution context: ~35% of the context window.

No separate `DESIGN.md` is required; the architecture is a standard Python `src/` package with a small MLX smoke primitive and pytest loop.

## Recommended Next Skill

`auto-eng-review`

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan uses a standard `src/` Python package, direct verification commands, and keeps vendor/local parity dependencies out of the default execution path.
- Concern: `uv sync` may fail if the chosen MLX package/version is unavailable or incompatible with the active Python/macOS environment.
- Action: Proceed to `auto-execute` and resolve any MLX dependency issue by choosing the smallest installable MLX dependency that satisfies the import test without adding Torch, Transformers, Hugging Face, or local absolute paths to the base environment.
- Verified: canonical plan read, slice dependencies traced, product review risks mapped to plan constraints, verification commands checked.

## Execution Evidence

- Slice 1 complete: `pyproject.toml` created for `mlx-spatial` / `mlx_spatial`; `uv sync` passed and installed `mlx==0.31.2`, `mlx-metal==0.31.2`, and `pytest==9.0.3`.
- Slice 2 complete: `src/mlx_spatial/` created with package export and model-neutral `regular_grid` MLX primitive; `uv run python -c "import mlx_spatial; import mlx.core as mx"` passed.
- Slice 3 complete: `tests/test_bootstrap.py` created for package export, MLX array shape/value behavior, and invalid shape rejection; `uv run pytest` passed with 3 tests.
- Slice 4 complete: `README.md` documents base setup, optional local framework resources, future Hugging Face/model-download boundary, and vendor reference boundary; `uv run pytest` passed.
