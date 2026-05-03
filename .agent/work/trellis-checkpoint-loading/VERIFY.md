# VERIFY: TRELLIS.2 Checkpoint Inspection and MLX Loading

## Verification: Slice 1 Safetensors Fixture and Dependency Boundary

- Criterion: Default dependencies do not include PyTorch, Transformers, or Hugging Face Hub.
  - Result: PASS
  - Evidence: `pyproject.toml:11-15` lists `mlx`, `numpy`, and `safetensors`; `tests/test_checkpoint.py:25-32` asserts `torch`, `transformers`, and `huggingface` are absent; fresh `uv run pytest tests/test_checkpoint.py` passed with `8 passed`.
  - Gap: none

- Criterion: Tests can create a tiny safetensors checkpoint fixture without network access or real weights.
  - Result: PASS
  - Evidence: `tests/test_checkpoint.py:14-22` writes a temp safetensors fixture using MLX arrays; fresh `uv run pytest tests/test_checkpoint.py` passed with `8 passed`.
  - Gap: none

- Criterion: Unsupported dependency assumptions are visible in test setup or package config.
  - Result: PASS
  - Evidence: `pyproject.toml:11-15` declares the lightweight safetensors path dependencies; `.agent/work/trellis-checkpoint-loading/PLAN.md:31-32` records the NumPy correction required by `safetensors.mlx`.
  - Gap: none

## Verification: Slice 2 Checkpoint Inspection API

- Criterion: Inspection returns tensor name, shape, dtype, and source path for fixture tensors.
  - Result: PASS
  - Evidence: `CheckpointTensorInfo` fields are defined at `src/mlx_spatial/checkpoint.py:13-20`; fixture metadata assertion is at `tests/test_checkpoint.py:35-45`; fresh `uv run pytest tests/test_checkpoint.py` passed with `8 passed`.
  - Gap: none

- Criterion: Metadata order is deterministic.
  - Result: PASS
  - Evidence: `inspect_checkpoint` iterates `sorted(tensors.keys())` at `src/mlx_spatial/checkpoint.py:35-36`; deterministic order is asserted at `tests/test_checkpoint.py:41-45`.
  - Gap: none

- Criterion: Exact-name and prefix filters work for inspection.
  - Result: PASS
  - Evidence: filter implementation is at `src/mlx_spatial/checkpoint.py:31-50` and `src/mlx_spatial/checkpoint.py:91-115`; tests are at `tests/test_checkpoint.py:48-56`; fresh checkpoint tests passed.
  - Gap: none

- Criterion: Missing paths, unsupported suffixes, invalid filters, and no-match filters raise clear errors.
  - Result: PASS
  - Evidence: path/format validation is at `src/mlx_spatial/checkpoint.py:82-88`; filter validation is at `src/mlx_spatial/checkpoint.py:100-111`; no-match handling is at `src/mlx_spatial/checkpoint.py:49-50`; tests are at `tests/test_checkpoint.py:59-81`.
  - Gap: none

## Verification: Slice 3 Selected Tensor MLX Loading API

- Criterion: Loader returns a dictionary keyed by tensor name with `mlx.core.array` values.
  - Result: PASS
  - Evidence: loader return contract is at `src/mlx_spatial/checkpoint.py:54-79`; MLX array assertion is at `tests/test_checkpoint.py:84-93`; fresh checkpoint tests passed.
  - Gap: none

- Criterion: Loaded tensor shapes and numeric values match fixture values.
  - Result: PASS
  - Evidence: fixture values are written at `tests/test_checkpoint.py:14-22`; numeric assertions are at `tests/test_checkpoint.py:90-104`.
  - Gap: none

- Criterion: Exact-name and prefix filters work for loading.
  - Result: PASS
  - Evidence: exact-name loading test is at `tests/test_checkpoint.py:84-93`; prefix loading test is at `tests/test_checkpoint.py:96-104`; implementation uses sorted keys and `_matches_filter` at `src/mlx_spatial/checkpoint.py:68-71`.
  - Gap: none

- Criterion: Loading without a selection, missing requested tensors, invalid filters, missing paths, and unsupported suffixes raise clear errors.
  - Result: PASS
  - Evidence: no-selection and missing-tensor handling is at `src/mlx_spatial/checkpoint.py:62-78`; invalid input tests are at `tests/test_checkpoint.py:107-138`.
  - Gap: none

## Verification: Slice 4 TRELLIS.2 Asset and Documentation Workflow

- Criterion: README documents local placement under `weights/trellis2/`.
  - Result: PASS
  - Evidence: `README.md:75` and `README.md:98-102` document `weights/trellis2/`.
  - Gap: none

- Criterion: README documents manual download pattern, inspection API or command snippet, and MLX loading snippet.
  - Result: PASS
  - Evidence: manual HF CLI pattern is at `README.md:83-87`; inspection API and command are at `README.md:91-108`; MLX loading API and command are at `README.md:110-114`.
  - Gap: none

- Criterion: README states unsupported boundaries: no full inference, no block parity, no decoder, no mesh/GLB export, no automatic downloads.
  - Result: PASS
  - Evidence: `README.md:116` names `.safetensors` only and excludes `.pt`/`.pth`, full inference, block parity, decoder execution, mesh extraction, GLB export, and automatic downloads.
  - Gap: none

- Criterion: Asset validation remains deterministic and does not require real files in default tests.
  - Result: PASS
  - Evidence: fresh `uv run pytest tests/test_model_assets.py tests/test_checkpoint.py` passed with `13 passed`; asset tests use `tmp_path` fake files at `tests/test_model_assets.py:19-57`.
  - Gap: none

## Verification: Slice 5 Full Verification

- Criterion: Full test suite passes.
  - Result: PASS
  - Evidence: fresh `uv run pytest` passed with `44 passed, 5 skipped`.
  - Gap: none

- Criterion: No real checkpoint artifacts are present in tracked source paths.
  - Result: PASS
  - Evidence: `Glob weights/**/*` returned no files.
  - Gap: none

- Criterion: Manual real-weight verification steps remain documentation-only.
  - Result: PASS
  - Evidence: README commands at `README.md:83-114` are manual CLI/API examples; default tests passed without real weights, network access, Hugging Face credentials, PyTorch, Transformers, or vendor imports.
  - Gap: none

## Commands Run

- `uv run pytest tests/test_checkpoint.py`: PASS, `8 passed`
- `uv run pytest tests/test_model_assets.py tests/test_checkpoint.py`: PASS, `13 passed`
- `uv run pytest`: PASS, `44 passed, 5 skipped`
- `Glob weights/**/*`: PASS, no files found

## Content Checks

- Audience: PASS. README checkpoint section addresses engineers using local TRELLIS.2 assets with concrete path, inspection, and loading examples at `README.md:91-116`.
- Thesis: PASS. The README claims local safetensors checkpoint inspection and selected MLX loading, and the section supports that claim with API names, command examples, and unsupported boundaries.
- Voice: PASS. The documentation is direct technical reference prose with commands and explicit limits.
- Content anti-goals: PASS. No claim of full TRELLIS.2 inference, block parity, decoder, mesh extraction, GLB export, automatic download, or real-weight availability is made.
- Channel: PASS. The artifact is README reference documentation.
- Source policy: PASS. Claims are limited to implemented APIs, manifest paths, and manual command patterns already in scope.
- Factual risk: PASS. Technical claims are backed by fresh tests and direct code/documentation observation.
- Format: PASS. Markdown sections, bullets, and fenced command examples match the existing README format.
- Anti-slop scan: PASS. No significance inflation, promotional language, vague attribution, generic conclusion, sycophantic framing, or forced rule of three found in the new checkpoint documentation.

## Overall

PASS

## Remaining Gaps

none

## Preserved Risks

- The checkpoint loader supports `.safetensors` only; `.pt`/`.pth` support remains deferred.
- The helper proves metadata and selected tensor loading, not TRELLIS.2 architecture mapping or layer parity.

## Recommended Next Skill

`auto-office-hours` or `auto-frame` for the next TRELLIS slice: exact checkpoint file selection, sparse/transformer block parity, or decoder/mesh path.
