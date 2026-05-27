# Plan: Pixal3D Shape Upsample Token Guard

## Goal

Execute [SPEC.md](SPEC.md): separate shape-upsample compute guarding from HR
coordinate token selection.

## Ordered Slice Sequence

### Slice 1: Runtime Option

**Objective:** Add a dedicated Pixal3D shape-upsample token limit and pass it to
the shape decoder upsample stage.

**Acceptance criteria:**
- `max_num_tokens` still controls HR coordinate selection.
- `shape_upsample_token_limit` controls only the shape decoder upsample guard.
- Invalid non-positive values return structured input-validation blockers.

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d.py`, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py -q`

**Status:** complete

**Evidence:** added `PIXAL3D_DEFAULT_SHAPE_UPSAMPLE_TOKEN_LIMIT=1000000`, threaded `shape_upsample_token_limit` through the pipeline, CLI, and script, kept `max_num_tokens` as the HR coordinate selection guard, and added validation/metadata tests. `uv run pytest tests/test_pixal3d_pipeline.py -q` -> 19 passed.

**Risks / next:** docs/help and real downloaded-weight smoke still need verification.

### Slice 2: Docs, Real Smoke, Hygiene

**Objective:** Document and verify the new guard with the real Pixal3D assets.

**Acceptance criteria:**
- Docs/help mention `--shape-upsample-token-limit`.
- Real smoke no longer blocks at the old coupled upsample guard.
- Focused and full tests plus release hygiene pass.

**Depends on:** Slice 1

**Touches:** `README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/work/2026-05-26-pixal3d-shape-upsample-token-guard/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q && uv run pytest -q && uv lock --check && git diff --check`

**Status:** complete

**Evidence:** updated README, Pixal3D docs, scripts README, package CLI help, and script help for `--shape-upsample-token-limit`. Real downloaded-weight smoke with the new default `shape_upsample_token_limit=1000000` no longer blocks at the old coupled upsample guard; it reached `shape-slat-cascade:upsample`, wrote `shape_slat_hr_coordinates.npz`, ran HR shape SLat, ran texture SLat, wrote `shape_slat_hr.npz` and `texture_slat.npz`, then blocked at the next limiter: `shape-decoder` stopped before 7-channel output because level 1 had 60822 tokens above `max_num_tokens=49152`. `/usr/bin/time -l` reported max RSS `16126476288`; trace metadata recorded MLX peak bytes `63385026372`. `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q` -> 84 passed; `uv run pytest -q` -> 890 passed, 10 skipped, 27 deselected; `uv lock --check`, `git diff --check`, git hygiene, build, and artifact checker passed.

**Risks / next:** real GLB generation still blocks at the shape-decoder token/reference limit; a separate cycle should decouple or expose shape/texture decoder token limits with the same memory-safety care.
