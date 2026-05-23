# Slice 0B Source-Contract Fixtures

Date: 2026-05-23

## Route

Main agent executed the corrected Slice 0B directly after user clarified that CUDA is not allowed locally or as an acceptance gate. The prior vendor dependency probe remains historical source-analysis evidence only.

## Files Changed

- `scripts/lito/write_contract_fixtures.py`: deterministic local fixture generator using NumPy, PIL, and safetensors only.
- `tests/fixtures/lito/`: generated tokenizer, condition, DiT, and render source-contract fixtures plus `manifest.json`.
- `scripts/lito/validate_fixtures.py`: retained as the vendor-free validator.
- `SPEC.md`, `PLAN.md`, `slices/00b-reference-fixtures.md`, `spec/gap-matrix.md`, `STATUS.md`: updated to remove the CUDA fixture gate and document Torch CPU/MPS as optional only.

## Verification

- `uv run python -m compileall -q scripts/lito/write_contract_fixtures.py scripts/lito/validate_fixtures.py` passed.
- `uv run python scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite` wrote the fixture set.
- `uv run python scripts/lito/validate_fixtures.py tests/fixtures/lito --verbose` passed with 19 files across 4 required groups.
- `grep -E "source-contract|MLX-compatible|no CUDA" .agent/work/2026-05-22-lito-mlx-inference-pipeline/spec/gap-matrix.md` returned the expected no-CUDA/source-contract lines.
- Repository-local scratch scan for `lito-fixtures-*` and `lito-fixture-env-*` returned no leaked scratch directories.

## Result

Slice 0B is complete. P1 can proceed with Slices 1-4 in parallel. Each implementation slice must treat upstream CUDA/PyTorch/gsplat paths as static source reference and port required operations to MLX-compatible code.
