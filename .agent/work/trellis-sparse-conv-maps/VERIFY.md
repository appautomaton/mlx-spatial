# VERIFY: TRELLIS.2 Sparse Convolution Map Primitives

## Verification: Sparse Convolution Map Change

- Criterion: `mlx_spatial.sparse_conv` imports successfully.
  - Result: PASS
  - Evidence: `uv run python -c "import mlx_spatial; import mlx_spatial.sparse_conv; import mlx.core as mx"` exited successfully.
  - Gap: none

- Criterion: `kernel_offsets((3, 3, 3))` returns `(27, 3)` `int32` offsets.
  - Result: PASS
  - Evidence: `tests/test_sparse_conv.py:6-13` asserts shape and dtype; `uv run pytest tests/test_sparse_conv.py` passed.
  - Gap: none

- Criterion: Offset ordering is lexicographic `(dz, dy, dx)` and includes center `[0, 0, 0]`.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:23-42` generates lexicographic offsets; `tests/test_sparse_conv.py:11-13` asserts first, center, and last offsets.
  - Gap: none

- Criterion: Invalid kernel sizes are rejected with `ValueError`.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:16-20`; `tests/test_sparse_conv.py:24-31` covers invalid sizes.
  - Gap: none

- Criterion: `sparse_conv_map` returns deterministic `(target_index, source_index, kernel_index)` rows.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:45-77`; exact row assertions at `tests/test_sparse_conv.py:34-65`.
  - Gap: none

- Criterion: Map rows are ordered by target input order, then kernel offset order.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:69-75` loops targets first and kernel offsets second; exact ordering is asserted at `tests/test_sparse_conv.py:48-65`.
  - Gap: none

- Criterion: Offset sign convention is documented and exact-row tested.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:57-62` documents `source = target + offset`; `README.md:50` documents the same convention; `tests/test_sparse_conv.py:34-65` proves exact rows.
  - Gap: none

- Criterion: Missing neighbors are omitted.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:73-75` appends only present sources; `tests/test_sparse_conv.py:68-78` asserts a sparse two-coordinate map omits missing neighbors.
  - Gap: none

- Criterion: Duplicate coordinates, non-`(n, 3)` coordinates, out-of-bounds coordinates, invalid shapes, and invalid kernel sizes raise `ValueError`.
  - Result: PASS
  - Evidence: sparse map validation delegates to topology validation at `src/mlx_spatial/sparse_conv.py:64-65` and kernel validation at `src/mlx_spatial/sparse_conv.py:67`; invalid cases are tested at `tests/test_sparse_conv.py:81-110`.
  - Gap: none

- Criterion: Tests use only MLX and pytest in the default sparse conv suite.
  - Result: PASS
  - Evidence: `tests/test_sparse_conv.py:1-3` imports only `mlx.core` and `mlx_spatial.sparse_conv`; `uv run pytest tests/test_sparse_conv.py` passed.
  - Gap: none

- Criterion: Optional local PyTorch parity checks are gated and skipped by default.
  - Result: PASS
  - Evidence: `tests/test_sparse_conv_parity.py:14-25` skips unless `MLX_SPATIAL_RUN_TORCH_PARITY=1`; full `uv run pytest` reports `tests/test_sparse_conv_parity.py s`.
  - Gap: none

- Criterion: PyTorch parity scaffolding does not add PyTorch to base dependencies.
  - Result: PASS
  - Evidence: `pyproject.toml:11-18` includes only `mlx` and `pytest>=8` dependencies.
  - Gap: none

- Criterion: README documents map rows, row ordering, kernel index semantics, sign convention, optional parity, and anti-goals.
  - Result: PASS
  - Evidence: `README.md:43-52` documents sparse convolution maps and anti-goals; `README.md:66-72` documents optional parity gating.
  - Gap: none

- Criterion: `uv run pytest` passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, or vendor setup.
  - Result: PASS
  - Evidence: `uv run pytest` passed with `20 passed, 3 skipped`; optional parity tests were skipped by default.
  - Gap: none

## Commands Run

- `uv run python -c "import mlx_spatial; import mlx_spatial.sparse_conv; import mlx.core as mx"`: PASS
- `uv run pytest tests/test_sparse_conv.py`: PASS, 6 passed
- `uv run pytest`: PASS, 20 passed and 3 skipped

## Content Checks

- Audience: PASS. README addresses developers using the package with setup, sparse-grid, topology, sparse-conv map, and optional parity instructions at `README.md:7-72`.
- Thesis: PASS. README maintains the claim that this is an MLX-first spatial primitive package and positions sparse convolution maps as map-construction helpers, not inference or compute, at `README.md:1-5` and `README.md:43-52`.
- Source policy: PASS. README claims are limited to implemented helpers, local paths established in planning, and dependency boundaries.
- Anti-slop scan: PASS. No promotional claims, significance inflation, vague attribution, or generic conclusion added in the sparse conv section.

## Overall

PASS

## Remaining Gaps

none

## Recommended Next Skill

`auto-frame` for feature gather/scatter primitives, strided/dilated map variants, or a first concrete TRELLIS.2 vendor parity slice.
