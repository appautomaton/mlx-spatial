# Roadmap

## Phase 1: Gap Matrix and HY-World 2.0 Backbone

- status: done
- change: `2026-05-20-gap-matrix-parity`
- objective: Created the structured gap matrix and closed critical inference blockers for HY-World 2.0 — the transformer backbone layers plus camera/rotation/geometry utilities and GS activation/SH functions.

## Phase 2: Production Pipeline Parity (Final)

- status: done
- change: `2026-05-20-production-pipeline-parity`
- objective: Close all remaining inference gaps across SAM 3D, TRELLIS.2, and HY-World 2.0 — consolidating GS/SH completion, TRELLIS.2 texturing pipeline, sparse interpolation, and cross-pipeline mesh postprocessing into one spec. 10 gaps across three groups.
- evidence: 10 gaps remain after removing already-ported modules (SAM-MOT in `sam3d_ss_flow.py`, SAM-SHORTCUT in `sam3d_flow.py`) and training-only items. See `spec/gap-matrix.md` for original gap IDs.
- exit signal: All remaining gaps produce numerically matching or performance-comparable output against vendor references; all three pipelines reach full inference parity on Apple Silicon.

## Phase 3: LiTo MLX Inference Pipeline

- status: done
- change: `2026-05-22-lito-mlx-inference-pipeline`
- objective: Port Apple LiTo image-to-3DGS inference into `mlx-spatial` as a new pipeline matching the established `<name>_*.py` + `mlx-spatial-<name>` shape, with per-module MLX source-contract probes and no CUDA runtime path.
- evidence: LiTo source/tests/docs landed in `f7d575f`; binary PLY export landed in `14236d5`. Verification passed with `uv run pytest tests/test_lito_*.py -q`, full `uv run pytest -q`, `uv build`, and the no-CUDA runtime guard. Accepted quality evidence uses uncapped checkpoint-backed teacup and beer-mug PLYs viewed in a Gaussian-splat-aware viewer.
- exit signal: `mlx-spatial-lito generate <image>` produces a checkpoint-backed LiTo 3DGS result, not only a source-contract smoke PLY; no existing pipeline regresses. Remaining teacup handle-hole work is a follow-up quality slice, not a Phase 3 blocker.

## Phase 4+ Candidates

- LiTo training and fine-tuning on MLX.
- LiTo weight redistribution via `mlx-community` with a license-compatible model card and no unsupported derivative packaging.
- LiTo mesh extraction from 3DGS+LF outputs through Flexicubes, marching cubes, or an equivalent mesh path.
- LiTo preprocessing/matte-quality pass for handle holes and weak RGBA alpha masks.
- LiTo multi-image and video conditioning.
- M2/M3 LiTo memory profile using tiling or streaming below the 90 GB soft threshold.
- Cross-pipeline ablations comparing LiTo, TRELLIS.2, and Hunyuan3D-2.5.
