# Plan: Pixal3D Memory Hygiene

## Goal

Execute [SPEC.md](SPEC.md): reduce avoidable memory lifetime in the Pixal3D
MLX inference path without changing generation behavior.

## Slice 1: Safe Stage Cleanup

**Objective:** Add cleanup boundaries after completed Pixal3D stages.

**Acceptance criteria:**
- Cleanup happens only after downstream tensors/artifacts are retained.
- Trace metadata includes memory snapshots for cleanup boundaries.
- Existing Pixal3D path behavior is unchanged.

**Status:** complete
**Evidence:** changed `src/mlx_spatial/pixal3d_inference.py` to delete completed-stage tensors and call `clear_mlx_cache()` with checkpoint snapshots after retained artifacts/tensors are available; changed `tests/test_pixal3d_pipeline.py` to assert success-path checkpoint labels and snapshot shape.
Verification passed: `uv run pytest tests/test_pixal3d_pipeline.py tests/test_pixal3d_projection.py -q` (33 passed), `uv run pytest tests/test_pixal3d_*.py -q` (79 passed), `git diff --check`, `uv lock --check`, forbidden Torch/CUDA import scan, and `uv run pytest -q` (891 passed, 10 skipped, 27 deselected, 2 warnings).
**Risks / next:** real 8-minute Pixal3D smoke was not rerun during this slice; remeasure peak MLX memory with the existing real-smoke command if we need hard before/after numbers.
