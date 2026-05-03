# PLAN: TRELLIS.2 Sparse Voxel Topology Helpers

## Goal

Implement `mlx_spatial` sparse voxel topology helpers for deterministic 26-neighbor active adjacency plus grid edge/cell relationships, with MLX-only default tests and optional local PyTorch parity checks.

## Architecture Approach

Add `mlx_spatial.topology` as the public topology module. Keep the API model-neutral and aligned with existing `mlx_spatial.ovoxel` row-major coordinate/index conventions.

The topology contract is:

- Input sparse coordinates: MLX array shaped `(n, 3)` with integer values.
- Input grid shape: positive 3-tuple `(depth, height, width)`.
- Neighbor offsets: deterministic `int32` `(26, 3)` array ordered lexicographically by `(dz, dy, dx)`, excluding `(0, 0, 0)`.
- Adjacency pairs: deterministic `int32` `(m, 2)` array of `(source_index, target_index)` into the input coordinate array, ordered by source input order then neighbor offset order.
- Dense edge/cell helpers: deterministic plain MLX arrays for topology relationships, not mesh extraction.

The implementation may use small Python-side lookup logic for determinism and clarity in this slice, then return MLX arrays. It should not add model weights, vendor imports, PyTorch dependencies, or sparse convolution behavior.

## Ordered Task Sequence

### Slice 1: Topology API And Neighbor Offsets

**Objective:** Add `mlx_spatial.topology` with deterministic 26-neighbor offset generation and validation helpers.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/topology.py`, `src/mlx_spatial/__init__.py`
**Context budget:** ~10% of context window
**Produces:** Importable topology module and exported `neighbor_offsets_26` helper.
**Acceptance criteria:**
- `mlx_spatial.topology` imports successfully.
- `neighbor_offsets_26()` returns shape `(26, 3)` and dtype `int32`.
- Offset ordering is documented as lexicographic `(dz, dy, dx)` and tested later.
- Existing `mlx_spatial.ovoxel` and `regular_grid` exports remain working.
**Verification:** `uv run python -c "import mlx_spatial; import mlx_spatial.topology; import mlx.core as mx"`
**Auto-continue:** yes

### Slice 2: 26-Neighbor Active Adjacency

**Objective:** Implement deterministic 26-neighbor active adjacency for sparse coordinates.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/topology.py`, `tests/test_topology.py`
**Context budget:** ~12% of context window
**Produces:** `adjacency_pairs_26(coordinates, shape)` plus MLX-only tests.
**Acceptance criteria:**
- Adjacency returns `(source_index, target_index)` pairs into input coordinate order.
- Rows are ordered by source input order, then neighbor offset order.
- Function rejects non-`(n, 3)` coordinates, non-positive shapes, out-of-bounds coordinates, and duplicate coordinates with `ValueError`.
- Tests cover face, edge, and corner adjacency in one small sparse coordinate set.
- Tests use only MLX and pytest.
**Verification:** `uv run pytest tests/test_topology.py`
**Auto-continue:** yes

### Slice 3: Dense Grid Edge And Cell Relationships

**Objective:** Add deterministic dense grid edge and cell relationship helpers for topology verification and future geometry work.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `src/mlx_spatial/topology.py`, `tests/test_topology.py`
**Context budget:** ~12% of context window
**Produces:** `grid_edges(shape)` and `grid_cells(shape)` helpers with tests.
**Acceptance criteria:**
- `grid_edges(shape)` returns axis-aligned endpoint index pairs using row-major dense indices.
- `grid_cells(shape)` returns 8-corner cell index relationships using row-major dense indices.
- Helpers reject invalid non-3D or non-positive shapes.
- Tests assert exact values for a small grid.
- Documentation clarifies these are topology relationships, not mesh extraction.
**Verification:** `uv run pytest tests/test_topology.py`
**Auto-continue:** yes

### Slice 4: Optional PyTorch Parity Gate

**Objective:** Add skipped-by-default parity scaffolding for local PyTorch reference checks.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `tests/test_topology_parity.py`, optional `pyproject.toml` marker reuse
**Context budget:** ~8% of context window
**Produces:** Optional parity tests gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
**Acceptance criteria:**
- Default `uv run pytest` skips parity without importing PyTorch.
- Parity scaffolding does not add PyTorch to base dependencies.
- Test code documents that enabled parity must use `/Users/ac/dev/ai/ai-frameworks/pytorch`.
- Parity test compares at least offset ordering or adjacency output to a local PyTorch reference when enabled.
**Verification:** `uv run pytest`
**Auto-continue:** yes

### Slice 5: Documentation Update

**Objective:** Document topology helpers, output contracts, and optional parity behavior.
**Execution:** direct
**Depends on:** Slice 4
**Touches:** `README.md`
**Context budget:** ~8% of context window
**Produces:** README notes for topology helpers and future-model-neutral boundaries.
**Acceptance criteria:**
- README documents 26-neighbor adjacency, offset ordering, adjacency pair ordering, and edge/cell helpers.
- README states helpers are topology primitives, not mesh extraction or TRELLIS.2 inference.
- README states default tests remain MLX-only and optional parity is gated.
- README keeps Hugging Face/model weights out of this slice.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing And Topology

- Slice 1: direct execution.
- Slice 2: direct execution after Slice 1 verification passes.
- Slice 3: direct execution after Slice 2 verification passes.
- Slice 4: direct execution after Slice 3 verification passes.
- Slice 5: direct execution after Slice 4 verification passes.
- Auto-continue chain: Slice 1 -> Slice 2 -> Slice 3 -> Slice 4 is safe because each slice has explicit tests and no external dependency requirement.
- Checkpoint boundary: Slice 5 ends the change; stop before sparse convolution maps, mesh extraction, model loading, or Hugging Face work.
- Parallel-safe groups: none. Slices share the same topology module and should execute serially.
- Subagents: not required, but `auto-eng-review` should evaluate the adjacency/cell contracts before execution.

## Verification Commands

- Slice 1: `uv run python -c "import mlx_spatial; import mlx_spatial.topology; import mlx.core as mx"`
- Slice 2: `uv run pytest tests/test_topology.py`
- Slice 3: `uv run pytest tests/test_topology.py`
- Slice 4: `uv run pytest`
- Slice 5: `uv run pytest`

## Risk Handling

- 26-neighbor adjacency can create ambiguous output ordering; this plan fixes order as source input order then lexicographic neighbor offset order.
- Duplicate coordinates are rejected to avoid nondeterministic source/target lookup behavior.
- Grid edge/cell helpers are dense topology relationships only and must not drift into mesh extraction.
- PyTorch parity remains optional and skipped by default, so local PyTorch issues cannot block the MLX default suite.

## Context Budget For This Change

Estimated total execution context: ~50% of the context window.

## Recommended Next Skill

`auto-eng-review`

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan has clear module boundaries, deterministic 26-neighbor adjacency contracts, optional parity gating, and explicit anti-goals that keep execution weight-free and model-free.
- Concern: Edge and cell helper ordering is less fully specified than adjacency ordering, so implementation must define endpoint/corner order in docstrings and exact-value tests to avoid future integration ambiguity.
- Action: Proceed to `auto-execute` and treat missing edge/cell ordering documentation or exact-value tests as blockers for Slice 3.
- Verified: canonical plan and design read, adjacency data flow traced, duplicate/out-of-bounds edge cases checked, optional PyTorch parity gate reviewed, dense edge/cell contract risk identified.

## Execution Evidence

- Slice 1 complete: `src/mlx_spatial/topology.py` added with `neighbor_offsets_26()` and `src/mlx_spatial/__init__.py` exports; `uv run python -c "import mlx_spatial; import mlx_spatial.topology; import mlx.core as mx"` passed.
- Slice 2 complete: `adjacency_pairs_26(coordinates, shape)` implemented with source-order then offset-order deterministic pairs; `tests/test_topology.py` covers face, edge, and corner adjacency plus invalid coordinate shapes, invalid grid shapes, duplicates, and out-of-bounds coordinates; `uv run pytest tests/test_topology.py` passed.
- Slice 3 complete: `grid_edges(shape)` and `grid_cells(shape)` implemented with documented endpoint/corner ordering and exact-value tests for `(2, 2, 2)`; `uv run pytest tests/test_topology.py` passed with 6 tests.
- Slice 4 complete: `tests/test_topology_parity.py` added optional local PyTorch parity for 26-neighbor offset ordering gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1`; full default suite skipped parity without importing PyTorch.
- Slice 5 complete: `README.md` documents 26-neighbor adjacency, offset ordering, adjacency pair ordering, edge/cell helper ordering, model-neutral boundaries, and optional parity behavior; `uv run pytest` passed with 14 passed and 2 skipped.
