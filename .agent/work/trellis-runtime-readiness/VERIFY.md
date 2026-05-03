# VERIFY: TRELLIS.2 MLX Runtime Readiness

## Verification: Slice 1 Weighted Sparse Convolution Reference

- Criterion: Helper is public and documented with map, feature, weight, and output shape contracts.
  - Result: PASS
  - Evidence: `weighted_sparse_conv` is defined with docstring contract at `src/mlx_spatial/sparse_conv.py:180-199`; public export is covered by full-suite import tests and `src/mlx_spatial/__init__.py` exports.
  - Gap: none

- Criterion: Helper accepts `source_features` `(source_count, in_channels)`, `map_rows` `(m, 3)`, `kernel_weights` `(kernel_count, in_channels, out_channels)`, and `target_count`.
  - Result: PASS
  - Evidence: shape validation and channel checks at `src/mlx_spatial/sparse_conv.py:201-207`; invalid rank/channel tests at `tests/test_sparse_weighted_conv.py:70-89`.
  - Gap: none

- Criterion: Helper computes `source_features[source_index] @ kernel_weights[kernel_index]` per map row and sums into `target_index`.
  - Result: PASS
  - Evidence: implementation at `src/mlx_spatial/sparse_conv.py:212-226`; exact numeric test at `tests/test_sparse_weighted_conv.py:6-39`; fresh command `uv run pytest tests/test_sparse_weighted_conv.py tests/test_sparse_feature.py tests/test_sparse_conv.py` passed with `17 passed`.
  - Gap: none

- Criterion: Duplicate target indices accumulate deterministically.
  - Result: PASS
  - Evidence: deterministic row loop at `src/mlx_spatial/sparse_conv.py:212-226`; duplicate target test at `tests/test_sparse_weighted_conv.py:42-56`.
  - Gap: none

- Criterion: Empty maps return zero output shaped `(target_count, out_channels)`.
  - Result: PASS
  - Evidence: empty map test at `tests/test_sparse_weighted_conv.py:59-67`; output shape derived from `kernel_weights` at `src/mlx_spatial/sparse_conv.py:202-208`.
  - Gap: none

- Criterion: Invalid map shape, non-integer map rows, invalid feature/weight rank, channel mismatch, out-of-bounds source/target/kernel indices, and invalid target count raise `ValueError`.
  - Result: PASS
  - Evidence: validation paths at `src/mlx_spatial/sparse_conv.py:35-53`, `src/mlx_spatial/sparse_conv.py:201-218`; tests at `tests/test_sparse_weighted_conv.py:70-110`; fresh targeted command passed.
  - Gap: none

- Criterion: Existing sparse map and gather/scatter tests continue to pass.
  - Result: PASS
  - Evidence: fresh command `uv run pytest tests/test_sparse_weighted_conv.py tests/test_sparse_feature.py tests/test_sparse_conv.py` passed with `17 passed`.
  - Gap: none

## Verification: Slice 2 Optional Weighted Parity Scaffold

- Criterion: Test is marked `torch_parity` and skips unless `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
  - Result: PASS
  - Evidence: `tests/test_sparse_weighted_conv_parity.py:12-18`; fresh command `uv run pytest tests/test_sparse_weighted_conv_parity.py tests/test_sparse_weighted_conv.py` reported `5 passed, 1 skipped`.
  - Gap: none

- Criterion: Test imports PyTorch only inside the gated loader.
  - Result: PASS
  - Evidence: `tests/test_sparse_weighted_conv_parity.py:15-26` imports `torch` only after the environment gate.
  - Gap: none

- Criterion: Test compares MLX output against an equivalent small PyTorch loop.
  - Result: PASS
  - Evidence: PyTorch reference loop and assertion at `tests/test_sparse_weighted_conv_parity.py:29-53`.
  - Gap: none

- Criterion: PyTorch remains absent from base dependencies.
  - Result: PASS
  - Evidence: `pyproject.toml:11-18` lists only base `mlx` and dev `pytest>=8`.
  - Gap: none

## Verification: Slice 3 TRELLIS.2 Asset Readiness

- Criterion: Manifest/config names TRELLIS.2 and contains expected relative asset paths without storing weights.
  - Result: PASS
  - Evidence: `src/mlx_spatial/model_assets.py:32-49`; relative path tests at `tests/test_model_assets.py:7-17`.
  - Gap: none

- Criterion: Validation helper accepts a local root path and reports deterministic present/missing entries without downloading or importing optional tooling.
  - Result: PASS
  - Evidence: `validate_model_assets` at `src/mlx_spatial/model_assets.py:52-74` uses only filesystem checks; no optional imports exist in `src/mlx_spatial/model_assets.py`; deterministic order is asserted at `tests/test_model_assets.py:19-57`.
  - Gap: none

- Criterion: Missing-file and present-file behavior is covered with temporary fake files.
  - Result: PASS
  - Evidence: tests use `tmp_path` and fake bytes at `tests/test_model_assets.py:19-57`; fresh command `uv run pytest tests/test_model_assets.py` passed with `5 passed`.
  - Gap: none

- Criterion: Local `weights/` artifacts are ignored or docs choose an out-of-repo cache convention.
  - Result: PASS
  - Evidence: `.gitignore:1` contains `weights/`; README documents `weights/trellis2/` and states `weights/` is ignored at `README.md:75`.
  - Gap: none

- Criterion: Default tests do not require network access, Hugging Face credentials, real model weights, vendors, or local absolute paths.
  - Result: PASS
  - Evidence: fresh full-suite command `uv run pytest` passed with `36 passed, 5 skipped`; asset tests use temp files only.
  - Gap: none

## Verification: Slice 4 Runtime Readiness Documentation

- Criterion: README documents weighted sparse convolution shape contracts and deterministic accumulation semantics.
  - Result: PASS
  - Evidence: `README.md:57-64` documents source, map, weight, output shapes and deterministic duplicate target accumulation.
  - Gap: none

- Criterion: README documents local asset convention and validation workflow.
  - Result: PASS
  - Evidence: `README.md:68-81` documents `TRELLIS2_ASSETS`, `validate_model_assets`, `weights/trellis2/`, and validation command.
  - Gap: none

- Criterion: README includes a Hugging Face CLI download command pattern without requiring the CLI or network in default tests.
  - Result: PASS
  - Evidence: `README.md:83-89` includes a placeholder `huggingface-cli download` pattern and states the package does not require CLI, network, login, or model weights for default tests.
  - Gap: none

- Criterion: README clearly states unsupported boundaries: no full TRELLIS.2 inference, no checkpoint loading, no decoder, no mesh/GLB export.
  - Result: PASS
  - Evidence: `README.md:66` states no transformer blocks, checkpoint loading, mesh decode, GLB export, or downloads; `README.md:89` states no CLI, network, login, or model weights required.
  - Gap: none

- Criterion: README names concrete next slices: checkpoint inspection/loading, TRELLIS sparse block parity, decoder/mesh path.
  - Result: PASS
  - Evidence: `README.md:91-95` lists the three next TRELLIS slices.
  - Gap: none

- Criterion: Content follows the specified audience, thesis, voice, source policy, and anti-goals.
  - Result: PASS
  - Evidence: README sections at `README.md:43-95` are technical reference documentation for engineers using implemented APIs; claims are limited to implemented local helpers and documented placeholder CLI pattern; unsupported boundaries are explicit.
  - Gap: none

## Commands Run

- `uv run pytest tests/test_sparse_weighted_conv.py tests/test_sparse_feature.py tests/test_sparse_conv.py`: PASS, `17 passed`
- `uv run pytest tests/test_sparse_weighted_conv_parity.py tests/test_sparse_weighted_conv.py`: PASS, `5 passed, 1 skipped`
- `uv run pytest tests/test_model_assets.py`: PASS, `5 passed`
- `uv run pytest`: PASS, `36 passed, 5 skipped`

## Content Checks

- Audience: PASS. The README addresses engineers using `mlx_spatial` APIs with concrete shape contracts, validation commands, and dependency boundaries at `README.md:43-95`.
- Thesis: PASS. The README supports the claim that the repo is ready for TRELLIS.2 weights through executable sparse compute and asset validation boundaries, not checkpoint download or full inference.
- Voice: PASS. Sentences are direct technical reference statements with commands and API names; no hype framing was added.
- Content anti-goals: PASS. No claims of full TRELLIS.2 inference, no promotional language, no vague next step, and optional dependencies are named explicitly.
- Channel: PASS. The artifact is README reference documentation.
- Source policy: PASS. Claims are limited to existing repository behavior, implemented APIs, and placeholder HF CLI pattern.
- Factual risk: PASS. Technical claims are backed by implemented APIs and fresh tests.
- Format: PASS. Markdown reference sections with commands and unsupported boundaries are present.
- Anti-slop scan: PASS. No significance inflation, promotional language, vague attribution, forced conclusion, or sycophantic framing found in the new runtime readiness documentation.

## Overall

PASS

## Remaining Gaps

none

## Preserved Risks

- TRELLIS.2 asset manifest is provisional until a checkpoint-specific slice selects exact files.
- `weighted_sparse_conv` is a reference layout, not verified TRELLIS model-layer parity.

## Recommended Next Skill

`auto-office-hours` or `auto-frame` for checkpoint inspection/loading or TRELLIS sparse block parity.
