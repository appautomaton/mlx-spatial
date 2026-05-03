# PLAN: TRELLIS.2 Sparse Convolution Map Primitives

## Goal

Implement deterministic sparse convolution map primitives that produce kernel offsets and `(target_index, source_index, kernel_index)` map rows from active sparse voxel coordinates, with MLX-only default tests and optional local PyTorch parity checks.

## Architecture Approach

Add `mlx_spatial.sparse_conv` as a small map-construction module. It should use the established coordinate conventions from `mlx_spatial.ovoxel` and validation/ordering style from `mlx_spatial.topology`.

The contract is:

- Coordinates: MLX array shaped `(n, 3)` in `(z, y, x)` order.
- Shape: positive `(depth, height, width)`.
- Kernel size: positive odd `(kz, ky, kx)`.
- Kernel offsets: `int32` array shaped `(kz * ky * kx, 3)`, lexicographic `(dz, dy, dx)`, center included.
- Map rows: `int32` array shaped `(m, 3)` with `(target_index, source_index, kernel_index)`.
- Row order: target input order, then kernel offset order.

This remains a map primitive slice only. No feature gathering, convolution compute, weights, batching, stride, dilation, transposed convolution, checkpoint loading, or TRELLIS.2 inference.

## Ordered Task Sequence

### Slice 1: Kernel Offset Helpers

**Objective:** Add `mlx_spatial.sparse_conv` with deterministic odd-kernel offset generation.
**Execution:** direct
**Depends on:** none
**Touches:** `src/mlx_spatial/sparse_conv.py`, `src/mlx_spatial/__init__.py`
**Context budget:** ~10% of context window
**Produces:** `kernel_offsets(kernel_size)` exported from `mlx_spatial`.
**Acceptance criteria:**
- `mlx_spatial.sparse_conv` imports successfully.
- `kernel_offsets((3, 3, 3))` returns `(27, 3)` `int32` offsets.
- Offset ordering is lexicographic `(dz, dy, dx)` and includes center `[0, 0, 0]`.
- Invalid kernel sizes are rejected with `ValueError`.
**Verification:** `uv run python -c "import mlx_spatial; import mlx_spatial.sparse_conv; import mlx.core as mx"`
**Auto-continue:** yes

### Slice 2: Same-Grid Sparse Conv Map Rows

**Objective:** Implement same-grid stride-1 sparse convolution maps for active sparse coordinates.
**Execution:** direct
**Depends on:** Slice 1
**Touches:** `src/mlx_spatial/sparse_conv.py`, `tests/test_sparse_conv.py`
**Context budget:** ~12% of context window
**Produces:** `sparse_conv_map(coordinates, shape, kernel_size=(3, 3, 3))` returning `(target_index, source_index, kernel_index)` rows.
**Acceptance criteria:**
- Map rows are deterministic and ordered by target input order, then kernel offset order.
- Missing neighbors are omitted.
- Duplicate coordinates, non-`(n, 3)` coordinates, out-of-bounds coordinates, invalid shapes, and invalid kernel sizes raise `ValueError`.
- Tests assert exact rows for a small sparse coordinate set.
- Tests use only MLX and pytest.
**Verification:** `uv run pytest tests/test_sparse_conv.py`
**Auto-continue:** yes

### Slice 3: Optional PyTorch Parity Gate

**Objective:** Add skipped-by-default parity scaffolding for local PyTorch reference checks.
**Execution:** direct
**Depends on:** Slice 2
**Touches:** `tests/test_sparse_conv_parity.py`, optional `pyproject.toml` marker reuse
**Context budget:** ~8% of context window
**Produces:** Optional parity test gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
**Acceptance criteria:**
- Default `uv run pytest` skips parity without importing PyTorch.
- Parity scaffolding does not add PyTorch to base dependencies.
- Test code documents that enabled parity must use `/Users/ac/dev/ai/ai-frameworks/pytorch`.
- Parity compares offset ordering or map rows to a local PyTorch reference when enabled.
**Verification:** `uv run pytest`
**Auto-continue:** yes

### Slice 4: Documentation Update

**Objective:** Document sparse convolution map helpers and boundaries.
**Execution:** direct
**Depends on:** Slice 3
**Touches:** `README.md`
**Context budget:** ~8% of context window
**Produces:** README notes for kernel offsets, map row contract, ordering, optional parity, and anti-goals.
**Acceptance criteria:**
- README documents `(target_index, source_index, kernel_index)` map rows and ordering.
- README states maps are primitives, not convolution compute or TRELLIS.2 inference.
- README states default tests remain MLX-only and optional parity is gated.
- README keeps Hugging Face/model weights out of this slice.
**Verification:** `uv run pytest`
**Auto-continue:** no

## Execution Routing And Topology

- Slice 1: direct execution.
- Slice 2: direct execution after Slice 1 verification passes.
- Slice 3: direct execution after Slice 2 verification passes.
- Slice 4: direct execution after Slice 3 verification passes.
- Auto-continue chain: Slice 1 -> Slice 2 -> Slice 3 is safe because each slice has explicit verification and no required external dependency.
- Checkpoint boundary: Slice 4 ends the change; stop before convolution compute, feature gathering, stride/dilation/transposed maps, model loading, or Hugging Face work.
- Parallel-safe groups: none. The map module and tests should evolve serially.
- Subagents: not required, but `auto-eng-review` should validate the map row contract before execution.

## Verification Commands

- Slice 1: `uv run python -c "import mlx_spatial; import mlx_spatial.sparse_conv; import mlx.core as mx"`
- Slice 2: `uv run pytest tests/test_sparse_conv.py`
- Slice 3: `uv run pytest`
- Slice 4: `uv run pytest`

## Risk Handling

- Map row ordering can be ambiguous; this plan fixes it as target input order, then kernel offset order.
- Kernel index semantics can be ambiguous; this plan defines `kernel_index` as the row index into `kernel_offsets(kernel_size)`.
- Same-grid stride-1 only avoids prematurely designing stride, dilation, transposed, or batched maps.
- PyTorch parity remains optional and skipped by default, so local PyTorch issues cannot block the MLX default suite.

## Context Budget For This Change

Estimated total execution context: ~45% of the context window.

## Recommended Next Skill

`auto-eng-review`

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan cleanly separates sparse map construction from convolution compute and fixes row shapes, row ordering, kernel index semantics, and dependency boundaries.
- Concern: The offset sign convention is not explicit enough, so implementation must document whether each source coordinate is `target + offset` or `target - offset` and assert exact rows in tests.
- Action: Proceed to `auto-execute` and treat missing offset sign documentation or exact map-row tests as blockers for Slice 2.
- Verified: canonical plan and design read, map data flow traced, invalid input coverage checked, optional PyTorch parity gate reviewed, convolution-compute scope boundary confirmed.

## Execution Evidence

- Slice 1 complete: `src/mlx_spatial/sparse_conv.py` added with `kernel_offsets(kernel_size)` and `src/mlx_spatial/__init__.py` exports; `uv run python -c "import mlx_spatial; import mlx_spatial.sparse_conv; import mlx.core as mx"` passed.
- Slice 2 complete: `sparse_conv_map(coordinates, shape, kernel_size=(3, 3, 3))` implemented with `(target_index, source_index, kernel_index)` rows, target-order then kernel-order sorting, and documented `source = target + offset` sign convention; `tests/test_sparse_conv.py` exact row tests passed.
- Slice 3 complete: `tests/test_sparse_conv_parity.py` added optional local PyTorch parity for kernel offset ordering gated by `MLX_SPATIAL_RUN_TORCH_PARITY=1`; full default suite skipped parity without importing PyTorch.
- Slice 4 complete: `README.md` documents sparse convolution maps, row contract, kernel index semantics, `source = target + offset`, optional parity, and anti-goals; `uv run pytest` passed with 20 passed and 3 skipped.
