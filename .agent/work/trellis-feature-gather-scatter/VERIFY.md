# VERIFY: TRELLIS Feature Gather/Scatter Primitives

## Verification: MLX Gather And Scatter Helpers

- Criterion: Helper names are literal and documented in docstrings.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:113-134` defines `gather_sparse_features`; `src/mlx_spatial/sparse_conv.py:137-170` defines `scatter_sparse_features`; both include docstrings with arguments, row contract, and return shapes.
  - Gap: none

- Criterion: Helpers are exposed through the public package surface.
  - Result: PASS
  - Evidence: `src/mlx_spatial/__init__.py:5` imports `gather_sparse_features` and `scatter_sparse_features`; `src/mlx_spatial/__init__.py:14` and `src/mlx_spatial/__init__.py:19` include them in `__all__`.
  - Gap: none

- Criterion: Gather returns source feature rows ordered exactly as map rows appear.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:128-134`; `tests/test_sparse_feature.py:6-28`; fresh command `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py` passed with `12 passed`.
  - Gap: none

- Criterion: Scatter accumulation sums row feature vectors into `target_index` slots.
  - Result: PASS
  - Evidence: `src/mlx_spatial/sparse_conv.py:162-170`; `tests/test_sparse_feature.py:31-60`; fresh command `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py` passed with `12 passed`.
  - Gap: none

- Criterion: Targets with no incoming rows remain zero.
  - Result: PASS
  - Evidence: `tests/test_sparse_feature.py:55-60` asserts untouched target rows remain `[0.0, 0.0]`; `tests/test_sparse_feature.py:73-74` asserts zero-filled scatter output for empty maps.
  - Gap: none

- Criterion: Empty maps produce a correctly shaped empty gather result and zero scatter result.
  - Result: PASS
  - Evidence: `tests/test_sparse_feature.py:63-74`; fresh command `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py` passed with `12 passed`.
  - Gap: none

- Criterion: Duplicate target indices accumulate deterministically.
  - Result: PASS
  - Evidence: `tests/test_sparse_feature.py:31-60` covers repeated target indices `0` and `2`; `src/mlx_spatial/sparse_conv.py:164-168` accumulates in map-row order using a deterministic loop.
  - Gap: none

- Criterion: Invalid map shape, non-integer map rows, invalid feature rank, out-of-bounds source/target indices, invalid target count, and feature-channel/row mismatches raise `ValueError`.
  - Result: PASS
  - Evidence: validation lives at `src/mlx_spatial/sparse_conv.py:35-53`, `src/mlx_spatial/sparse_conv.py:125-134`, and `src/mlx_spatial/sparse_conv.py:156-170`; tests cover invalid maps, ranks, row mismatch, bounds, and negative target count at `tests/test_sparse_feature.py:77-158`.
  - Gap: none

- Criterion: Existing sparse convolution map tests continue to pass.
  - Result: PASS
  - Evidence: fresh command `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py` passed with `12 passed`.
  - Gap: none

## Verification: Documentation And Optional Parity Scaffold

- Criterion: README documents the map row contract, gather ordering, scatter accumulation, empty behavior, and anti-goals.
  - Result: PASS
  - Evidence: `README.md:47-56` documents `sparse_conv_map`, `gather_sparse_features`, `scatter_sparse_features`, row ordering, source convention, empty behavior, and non-goals.
  - Gap: none

- Criterion: Optional PyTorch parity test is gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1` and skipped by default.
  - Result: PASS
  - Evidence: `tests/test_sparse_feature_parity.py:12-18` marks parity and skips unless the environment variable is set; fresh full-suite command reported `tests/test_sparse_feature_parity.py s`.
  - Gap: none

- Criterion: PyTorch remains absent from base dependencies.
  - Result: PASS
  - Evidence: `pyproject.toml:11-18` lists base dependency `mlx` and dev dependency `pytest>=8`; no `torch` dependency is present.
  - Gap: none

- Criterion: Full default test suite passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, vendors, or local absolute paths.
  - Result: PASS
  - Evidence: fresh command `uv run pytest` passed with `26 passed, 4 skipped`; parity tests were skipped by default.
  - Gap: none

## Commands Run

- `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py`: PASS, `12 passed`
- `uv run pytest`: PASS, `26 passed, 4 skipped`

## Content Checks

- Audience: PASS. README speaks to developers consuming `mlx_spatial.sparse_conv` helpers and states setup, semantics, and dependency boundaries at `README.md:43-76`.
- Thesis: PASS. README maintains the package thesis as MLX-first 3D/spatial primitives and positions gather/scatter as feature movement helpers, not model inference or weighted convolution, at `README.md:45-56`.
- Source policy: PASS. Documentation claims are limited to implemented local helpers and dependency boundaries.
- Anti-slop scan: PASS. No promotional claims, significance inflation, vague attribution, generic conclusion, or forced rule-of-three pattern added in the gather/scatter documentation.

## Overall

PASS

## Remaining Gaps

none

## Recommended Next Skill

`auto-office-hours` or `auto-frame` for the next TRELLIS support wedge, likely minimal weighted sparse convolution compute or stride/dilation sparse map variants.
