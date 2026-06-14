# Plan: Pixal3D Decoder Token Guards

## Goal

Execute [SPEC.md](SPEC.md): separate final decoder compute limits from HR token
selection.

## Ordered Slice Sequence

### Slice 1: Runtime And CLI Options

**Objective:** Add dedicated shape and texture decoder token limits and pass
them to the final decoder stages.

**Acceptance criteria:**
- `shape_decoder_token_limit` controls shape field decode.
- `texture_decoder_token_limit` controls texture PBR decode.
- Non-positive limits return structured input-validation blockers.
- CLI/script expose both options.

**Touches:** `src/mlx_spatial/pixal3d_inference.py`, `src/mlx_spatial/pixal3d.py`, `scripts/pixal3d/generate.py`, `tests/test_pixal3d_pipeline.py`

**Verification:** `uv run pytest tests/test_pixal3d_pipeline.py -q`

**Status:** complete

**Evidence:** added dedicated `PIXAL3D_DEFAULT_SHAPE_DECODER_TOKEN_LIMIT` and `PIXAL3D_DEFAULT_TEXTURE_DECODER_TOKEN_LIMIT` defaults, threaded them through the pipeline, package CLI, and script, kept `max_num_tokens` as the HR selection guard, and added validation/metadata tests. `uv run pytest tests/test_pixal3d_pipeline.py -q` -> 20 passed.

**Risks / next:** docs/help and real downloaded-weight smoke still need verification.

### Slice 2: Docs, Real Smoke, Hygiene

**Objective:** Document the decoder guards and verify the next real downloaded
weight boundary.

**Acceptance criteria:**
- README/Pixal3D/script docs describe decoder token guards.
- Real smoke no longer blocks at the old shape decoder `49152` compute limit.
- Focused/full tests and release hygiene pass.

**Depends on:** Slice 1

**Touches:** `README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/work/2026-05-26-pixal3d-decoder-token-guards/PLAN.md`

**Verification:** `uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q && uv run pytest -q && uv lock --check && git diff --check`

**Status:** complete

**Evidence:** documented `--shape-decoder-token-limit` and
`--texture-decoder-token-limit`; bumped the default final decoder guards to
`1100000` after the real 1024 cascade smoke showed `1029574` internal decoder
tokens were required. The rerun completed end to end with downloaded Pixal3D,
DINOv3, MoGe, and NAF assets:
`outputs/pixal3d/real-smoke-moge-balanced-decoders1100k/model.glb` is
`11405740` bytes, `trace.json` has `ready=true`, `blocker=null`, and completed
`artifact:textured_glb`. `/usr/bin/time -l` reported `16093069312` max RSS;
the trace reported MLX `peak_bytes=63385026372`. Verification passed:
`uv run pytest tests/test_pixal3d_*.py tests/test_sam3d_moge.py -q` -> 85
passed; `uv run pytest -q` -> 891 passed, 10 skipped, 27 deselected, 2 known
HyWorld2 warnings; `uv lock --check`, `git diff --check`, package/script help
checks, `python scripts/packaging/check_release_artifacts.py --git-hygiene`,
and `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python
scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz
dist/mlx_spatial-*-py3-none-any.whl` all passed.
