# VERIFY: TRELLIS.2 Sparse Voxel Topology Helpers

## Verification: Sparse Voxel Topology Change

- Criterion: `mlx_spatial.topology` imports successfully.
  - Result: PASS
  - Evidence: `uv run python -c "import mlx_spatial; import mlx_spatial.topology; import mlx.core as mx"` exited successfully.
  - Gap: none

- Criterion: `neighbor_offsets_26()` returns shape `(26, 3)` and dtype `int32`.
  - Result: PASS
  - Evidence: `tests/test_topology.py:11-21` asserts shape and dtype; `uv run pytest tests/test_topology.py` passed.
  - Gap: none

- Criterion: Offset ordering is lexicographic `(dz, dy, dx)`, excluding `[0, 0, 0]`.
  - Result: PASS
  - Evidence: `src/mlx_spatial/topology.py:39-52` defines lexicographic offset generation; `tests/test_topology.py:16-21` asserts exact sentinel offsets and excludes `[0, 0, 0]`.
  - Gap: none

- Criterion: Existing `mlx_spatial.ovoxel` and `regular_grid` exports remain working.
  - Result: PASS
  - Evidence: `src/mlx_spatial/__init__.py:3-17` still exports `regular_grid` and O-Voxel helpers; full `uv run pytest` passed existing bootstrap and O-Voxel tests.
  - Gap: none

- Criterion: `adjacency_pairs_26` returns `(source_index, target_index)` pairs into input coordinate order.
  - Result: PASS
  - Evidence: `src/mlx_spatial/topology.py:55-79` returns pair rows into input coordinate indices; `tests/test_topology.py:24-51` asserts exact pair values.
  - Gap: none

- Criterion: Adjacency rows are ordered by source input order, then neighbor offset order.
  - Result: PASS
  - Evidence: `src/mlx_spatial/topology.py:71-77` loops source coordinates first and offsets second; `tests/test_topology.py:37-51` asserts exact order.
  - Gap: none

- Criterion: Invalid coordinate shapes, invalid shapes, out-of-bounds coordinates, and duplicates raise `ValueError`.
  - Result: PASS
  - Evidence: validation in `src/mlx_spatial/topology.py:16-36`; tests at `tests/test_topology.py:54-76` cover invalid cases.
  - Gap: none

- Criterion: Tests cover face, edge, and corner adjacency in one small sparse coordinate set.
  - Result: PASS
  - Evidence: `tests/test_topology.py:24-51` uses coordinate relationships that include face, edge, and corner neighbors and asserts the exact adjacency output.
  - Gap: none

- Criterion: `grid_edges(shape)` returns axis-aligned endpoint index pairs using row-major dense indices.
  - Result: PASS
  - Evidence: `src/mlx_spatial/topology.py:82-108`; exact `(2, 2, 2)` values asserted at `tests/test_topology.py:79-96`.
  - Gap: none

- Criterion: `grid_cells(shape)` returns 8-corner cell index relationships using row-major dense indices.
  - Result: PASS
  - Evidence: `src/mlx_spatial/topology.py:111-136`; exact `(2, 2, 2)` values asserted at `tests/test_topology.py:99-103`.
  - Gap: none

- Criterion: Edge endpoint ordering and cell corner ordering are documented and exact-value tested.
  - Result: PASS
  - Evidence: `src/mlx_spatial/topology.py:82-90` documents edge ordering; `src/mlx_spatial/topology.py:111-120` documents cell ordering; `tests/test_topology.py:79-103` asserts exact values.
  - Gap: none

- Criterion: Edge/cell helpers reject invalid non-3D or non-positive shapes.
  - Result: PASS
  - Evidence: shared `_shape3` validation at `src/mlx_spatial/topology.py:16-20`; tests at `tests/test_topology.py:106-114` cover invalid shapes.
  - Gap: none

- Criterion: Default tests remain MLX-only and pass without local PyTorch, Transformers, Hugging Face credentials, checkpoints, or vendor setup.
  - Result: PASS
  - Evidence: `uv run pytest` passed with `14 passed, 2 skipped`; `pyproject.toml:11-18` contains only `mlx` and `pytest>=8` dependencies.
  - Gap: none

- Criterion: Optional local PyTorch parity checks are gated and skipped by default.
  - Result: PASS
  - Evidence: `tests/test_topology_parity.py:14-25` skips unless `MLX_SPATIAL_RUN_TORCH_PARITY=1`; full pytest output reports `tests/test_topology_parity.py s`.
  - Gap: none

- Criterion: Parity test compares at least offset ordering to a local PyTorch reference when enabled.
  - Result: PASS
  - Evidence: `tests/test_topology_parity.py:28-42` builds the same offset list with local PyTorch and compares to `neighbor_offsets_26()`.
  - Gap: none

- Criterion: README documents topology helpers, output contracts, optional parity, and model-free boundaries.
  - Result: PASS
  - Evidence: `README.md:32-41` documents topology helpers and contracts; `README.md:55-61` documents optional parity; `README.md:53` keeps Hugging Face/model setup outside the default path.
  - Gap: none

## Commands Run

- `uv run python -c "import mlx_spatial; import mlx_spatial.topology; import mlx.core as mx"`: PASS
- `uv run pytest tests/test_topology.py`: PASS, 6 passed
- `uv run pytest`: PASS, 14 passed and 2 skipped

## Content Checks

- Audience: PASS. README addresses developers using the package with setup, sparse-grid primitives, sparse topology contracts, and optional parity instructions at `README.md:7-61`.
- Thesis: PASS. README maintains the core claim that this is an MLX-first spatial primitive package and positions topology helpers as model-neutral building blocks at `README.md:1-5` and `README.md:32-41`.
- Source policy: PASS. README claims are limited to implemented helpers, local paths established in planning, and dependency boundaries.
- Anti-slop scan: PASS. No promotional claims, significance inflation, vague attribution, or generic conclusion added in the topology section.

## Overall

PASS

## Remaining Gaps

none

## Recommended Next Skill

`auto-frame` for sparse convolution map primitives, mesh-adjacent topology, or the first model-specific TRELLIS.2 parity slice.
