# Slice 0B Dependency Probe

Date: 2026-05-23

## Route

Main agent executed the active Slice 0B probe directly. Subagent `019e54f3-dd62-7e13-9720-91f087bbfe03` ran a parallel read-only dependency/entrypoint review and confirmed the dependency surfaces and lack of upstream avoiders for PyTorch3D/gsplat/xformers in the tokenizer/render paths.

## Files Changed

- `scripts/lito/validate_fixtures.py`: new manifest/fixture validator that does not import MLX or vendor modules.
- `slices/00b-reference-fixtures.md`: recorded isolated env setup, import/install outcomes, tokenizer load attempt, and stop reason.
- `spec/gap-matrix.md`: recorded the narrowed tokenizer fixture capture blocker.
- `PLAN.md`: recorded Slice 0B execution evidence and stop condition.

## Verification And Evidence

- `uv run python -m compileall -q scripts/lito/validate_fixtures.py` passed.
- `uv run python scripts/lito/validate_fixtures.py --help` passed.
- A temporary synthetic manifest/safetensors/image validation probe returned 5 validated files.
- `/tmp/lito-fixture-env` was created with CPython 3.11.15.
- `open3d`, PyTorch3D, `spz`, and `gsplat` installed/imported in the isolated env.
- `lito.trainers.lito_trainer`, `lito.trainers.lito_dit_trainer`, and `plibs.gs_utils` import in the isolated env with expected optional CUDA/TRELLIS warnings.
- Tokenizer checkpoint `weights/lito-raw/lito_new.ckpt` loads on CPU when unused voxel/mesh decoder construction is disabled for tokenizer-boundary probing.

## Stop Reason

Slice 0B reference fixture capture is blocked in this host environment. Running `LightTokenizationTrainer.get_latents(...)` fails because upstream no-CUDA execution forces xformers-backed localized attention. `xformers` fails to build on macOS arm64 here with `clang++: error: unsupported option '-fopenmp'`. This host also has no CUDA, no MPS, and no usable Metal device for upstream MLX paths.

No `tests/fixtures/lito/manifest.json` or reference fixtures were created.

## Addendum — No-CUDA Correction

User clarified after this probe that CUDA is not allowed locally or as an acceptance gate. The probe remains useful as source-analysis evidence: it identifies where upstream execution depends on xformers/flash-attention/CUDA-shaped paths. It does not justify a CUDA fixture gate.

Active Slice 0B now generates deterministic local source-contract fixtures and manifest metadata without vendor runtime imports. Downstream MLX slices must port the needed operations from static source inspection into MLX-compatible implementations. Optional Torch parity may use CPU/MPS only when the path avoids CUDA-only packages.
