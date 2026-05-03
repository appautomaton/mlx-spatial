# PLAN: TRELLIS Feature Gather/Scatter Primitives

## Goal

Implement MLX sparse feature gather/scatter primitives that consume `sparse_conv_map` rows and deterministically move or accumulate feature vectors between source and target sparse coordinate slots.

## Architecture Approach

Keep the change in the existing sparse convolution primitive layer. Extend `src/mlx_spatial/sparse_conv.py` with two weight-free helpers:

- a source gather helper that indexes source feature rows using the `source_index` column from `(target_index, source_index, kernel_index)` map rows;
- a target scatter/accumulate helper that sums per-map-row feature vectors into `target_index` slots.

Use correctness-first validation and deterministic accumulation. If MLX lacks a direct stable accumulation primitive for this use case, use a small explicit Python loop over map rows for this primitive slice. Do not add weighted convolution, stride/dilation variants, model layers, vendor imports, or PyTorch runtime dependencies.

No separate `DESIGN.md` is required; the spec and this plan fully define the behavior.

## Ordered Task Sequence

### Slice 1: MLX Gather And Scatter Helpers

**Objective:** Add public MLX helpers for map-row source gather and target accumulation with validation.

**Execution:** direct

**Depends on:** none

**Touches:** `src/mlx_spatial/sparse_conv.py`, `src/mlx_spatial/__init__.py`, `tests/test_sparse_feature.py`

**Context budget:** ~12% of context window

**Produces:** Public gather/scatter helpers and MLX-only unit tests.

**Acceptance criteria:**

- Helper names are literal and documented in docstrings.
- Gather returns source feature rows ordered exactly as map rows appear.
- Scatter accumulation sums row feature vectors into `target_index` slots.
- Targets with no incoming rows remain zero.
- Empty maps produce a correctly shaped empty gather result and zero scatter result.
- Duplicate target indices accumulate deterministically.
- Invalid map shape, non-integer map rows, invalid feature rank, out-of-bounds source/target indices, invalid target count, and feature-channel mismatches raise `ValueError`.
- Existing sparse convolution map tests continue to pass.

**Verification:** `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py`

**Auto-continue:** yes

### Slice 2: Documentation And Optional Parity Scaffold

**Objective:** Document gather/scatter semantics and add an optional local PyTorch parity test skipped by default.

**Execution:** direct

**Depends on:** Slice 1

**Touches:** `README.md`, `tests/test_sparse_feature_parity.py`, `pyproject.toml` if a new pytest marker is required

**Context budget:** ~8% of context window

**Produces:** README coverage and skipped-by-default parity scaffolding.

**Acceptance criteria:**

- README documents the map row contract, gather ordering, scatter accumulation, empty behavior, and anti-goals.
- Optional PyTorch parity test is gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1` and skipped by default.
- PyTorch remains absent from base dependencies.
- Full default test suite passes without local PyTorch, Transformers, Hugging Face credentials, checkpoints, vendors, or local absolute paths.

**Verification:** `uv run pytest`

**Auto-continue:** no

## Execution Routing And Topology

- Slice 1 route: direct.
- Slice 2 route: direct.
- Auto-continue chain: Slice 1 may continue directly into Slice 2 after `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py` passes.
- Checkpoints: stop after Slice 2 for verification and final acceptance.
- Parallel-safe groups: none. Slice 2 depends on helper names and semantics from Slice 1.
- Subagents: none required. The change is small, local, and does not cross risky subsystem boundaries.

## Verification Commands

- Slice 1: `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py`
- Slice 2: `uv run pytest`
- Optional local parity after default verification: `MLX_SPATIAL_RUN_TORCH_PARITY=1 uv run pytest -m torch_parity`

## Execution Evidence

- Slice 1: PASS. `uv run pytest tests/test_sparse_feature.py tests/test_sparse_conv.py` completed with `12 passed`.
- Slice 2: PASS. `uv run pytest` completed with `26 passed, 4 skipped`.
- Optional PyTorch parity remains gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1` and was skipped by default.

## Context Budget For This Change

Estimated total: ~20% of context window.

The implementation should only need the spec, this plan, `src/mlx_spatial/sparse_conv.py`, `src/mlx_spatial/__init__.py`, README sparse sections, and sparse feature tests.

## Recommended Next Skill

`auto-verify`
