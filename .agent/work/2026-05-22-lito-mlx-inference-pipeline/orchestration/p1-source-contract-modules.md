# P1 Source-Contract Modules

Date: 2026-05-23

## Route

Executed in parallel with four worker subagents:

- Slice 1 tokenizer: `src/mlx_spatial/lito_tokenizer.py`, `tests/test_lito_tokenizer.py`
- Slice 2 DiT: `src/mlx_spatial/lito_dit.py`, `tests/test_lito_dit.py`
- Slice 3 conditioner: `src/mlx_spatial/lito_condition.py`, `tests/test_lito_condition.py`
- Slice 4 render: `src/mlx_spatial/lito_render.py`, `tests/test_lito_render.py`

All workers were instructed that CUDA/PyTorch/gsplat upstream code is static source reference only. Runtime modules add no CUDA, Torch, xformers, flash-attention, gsplat, plibs, lito, or vendor imports.

## Review Loop

- Initial spec review requested changes: render needed to exercise the adapter path, P1 probes needed metrics, float32 uses needed inline justification, and HY-World regression evidence was missing.
- Workers fixed those issues; coordinator ran HY-World regression.
- Spec re-review returned `APPROVED`.
- Initial quality review found batch-global reductions in DiT and conditioner; workers fixed per-sample reductions and added batch-isolation tests.
- Quality re-review found batch-global DiT timestep selection; worker fixed per-sample timestep selection and added a regression.
- Final quality re-review found latent/condition batch mismatch could broadcast silently; worker added explicit batch equality validation in forward/sample and regressions.
- Final quality review returned `APPROVED`.

## Verification

- `uv run pytest tests/test_lito_tokenizer.py tests/test_lito_dit.py tests/test_lito_condition.py tests/test_lito_render.py -q` -> `37 passed in 0.27s`
- `uv run python -m compileall -q src/mlx_spatial/lito_tokenizer.py src/mlx_spatial/lito_dit.py src/mlx_spatial/lito_condition.py src/mlx_spatial/lito_render.py tests/test_lito_tokenizer.py tests/test_lito_dit.py tests/test_lito_condition.py tests/test_lito_render.py` -> passed
- `uv run pytest tests/test_gs_rasterize.py tests/test_hyworld2_export.py -q` -> `14 passed in 0.23s`
- `uv run pytest tests/test_hyworld2_*.py -q` -> `155 passed, 2 warnings` from pre-existing `hyworld2_export.py` RuntimeWarnings
- `git diff --quiet -- src/mlx_spatial/gs_rasterize.py && git diff --cached --quiet -- src/mlx_spatial/gs_rasterize.py` -> exit 0

## Result

P1 is complete under the corrected no-CUDA source-contract plan. Residual risk: fixtures are synthetic source-contract fixtures by approved design; they do not prove full vendor numerical parity.
