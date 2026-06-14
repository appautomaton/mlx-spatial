# Plan: Pixal3D GPU-Safe Projection

## Goal

Execute [SPEC.md](SPEC.md): remove the unsupported GPU matrix inverse from the
Pixal3D projection path and rerun the real-weight smoke.

## Ordered Slice Sequence

### Slice 1: Rigid Transform Inverse

**Objective:** Replace `mx.linalg.inv` with explicit rigid-transform inverse
math in Pixal3D projection.

**Acceptance criteria:**
- Projection math no longer calls `mx.linalg.inv`.
- Origin/front-view and custom rigid transform projection tests pass.

**Touches:** `src/mlx_spatial/pixal3d_projection.py`, `tests/test_pixal3d_projection.py`

**Verification:** `uv run pytest tests/test_pixal3d_projection.py -q && ! rg -n "mx\\.linalg\\.inv|linalg\\.inv" src/mlx_spatial/pixal3d_projection.py`

**Status:** complete

**Evidence:** replaced the Pixal3D projection `mx.linalg.inv` call with explicit rigid-transform inverse math and added a custom rigid-transform regression. `uv run pytest tests/test_pixal3d_projection.py -q` -> 13 passed; `! rg -n "mx\\.linalg\\.inv|linalg\\.inv" src/mlx_spatial/pixal3d_projection.py` -> passed.

**Risks / next:** real downloaded-weight smoke still needs to prove the previous GPU inverse failure is gone.

### Slice 2: Real-Smoke Verification And Hygiene

**Objective:** Prove the real Pixal3D smoke advances past the prior GPU-inverse
failure and run focused regression checks.

**Acceptance criteria:**
- Real smoke with `--moge-memory-profile balanced` no longer fails at
  `mx.linalg.inv`.
- Focused Pixal3D tests pass.
- Automaton state records the verified outcome.

**Depends on:** Slice 1

**Touches:** `.agent/work/2026-05-26-pixal3d-gpu-safe-projection/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q`

**Status:** complete

**Evidence:** real Pixal3D smoke with downloaded Pixal3D/DINOv3/NAF/MoGe assets and `--moge-memory-profile balanced` no longer fails with the MLX GPU `linalg.inv` error. It reached `shape-slat-cascade` after writing `sparse_projection.npz`, `sparse_structure.npz`, and `shape_slat_lr.npz`, then blocked on the next real limiter: `shape decoder upsample stopped before level 2: token_count=61951 exceeds decoder_token_limit=49152`. `/usr/bin/time -l` reported max RSS `13505675264`; trace metadata recorded MLX peak bytes `63385026372`. `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q` -> 83 passed; `uv run pytest -q` -> 889 passed, 10 skipped, 27 deselected; `uv lock --check`, `git diff --check`, Pixal3D projection inverse scan, git hygiene, build, and artifact checker all passed.

**Risks / next:** real Pixal3D generation is not yet end-to-end GLB for the vendored sample at the documented `max_num_tokens=49152`; the next blocker is HR shape-cascade token-limit handling.
