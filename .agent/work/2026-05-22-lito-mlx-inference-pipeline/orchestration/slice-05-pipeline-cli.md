# Slice 5 Pipeline + CLI Orchestration

## Implementer

- Agent: Gibbs (`019e5562-bd98-76b2-825c-fc475f7949ad`)
- Status: completed
- Files changed: `src/mlx_spatial/lito_inference.py`, `src/mlx_spatial/lito.py`, `scripts/lito/generate.py`, `scripts/README.md`, `pyproject.toml`, `uv.lock`, `tests/test_lito_inference.py`, `tests/test_lito_cli.py`, `tests/test_lito_memory_limits.py`

## Coordinator Fixes

- Added dev-only `plyfile>=1.1` and refreshed `uv.lock` so the PLY acceptance check uses the planned validator without adding runtime dependencies.
- Added `--weights-root` as an alias for LiTo `generate` in both package CLI and sample wrapper.
- Moved tokenizer execution into the `tokenize` metrics stage and passed the resulting latent into `dit`.
- Fixed top-level `mlx-spatial-lito --root <path> generate ...` handling by separating `global_root` from subcommand `command_root`.

## Reviews

- Spec review initially requested changes for tokenizer metrics attribution; re-review approved after the stage split.
- Quality review initially requested changes for top-level `--root` shadowing; re-review approved after parser/test fix.

## Verification

- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_lito_inference.py tests/test_lito_cli.py tests/test_lito_memory_limits.py -q` -> `17 passed`
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run mlx-spatial-lito validate weights/lito-mlx` -> `ready=True`, `present=2`, `missing=0`
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output outputs/lito/smoke.ply --memory-profile safe --print-metrics` -> `gaussians=64`, metrics include all seven stages
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run mlx-spatial-lito --root weights/lito-mlx generate inputs/lito/smoke.png --output outputs/lito/smoke-global-root.ply --memory-profile safe --print-metrics` -> `gaussians=64`
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run python scripts/lito/generate.py inputs/lito/smoke.png --weights-root weights/lito-mlx --output outputs/lito/smoke-script.ply --memory-profile safe --print-metrics` -> `gaussians=64`
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run python -c "from plyfile import PlyData; p = PlyData.read('outputs/lito/smoke.ply'); print(p.elements[0].count, 'gaussians')"` -> `64 gaussians`
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> built sdist and wheel
- `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl` -> checked 2 artifacts

## Notes

- Sandboxed MLX test runs can fail during collection with `No Metal device available`; the passing test evidence above was run with escalated execution so MLX can access Metal.
- The implementation remains a source-contract smoke pipeline and does not claim full Apple checkpoint numerical parity.
