# Roadmap

## Phase 1: Gap Matrix and HY-World 2.0 Backbone

- status: done
- change: `2026-05-20-gap-matrix-parity`
- objective: Created the structured gap matrix and closed critical inference blockers for HY-World 2.0 — the transformer backbone layers plus camera/rotation/geometry utilities and GS activation/SH functions.

## Phase 2: Production Pipeline Parity (Final)

- status: pending
- change: `2026-05-20-production-pipeline-parity`
- objective: Close all remaining inference gaps across SAM 3D, TRELLIS.2, and HY-World 2.0 — consolidating GS/SH completion, TRELLIS.2 texturing pipeline, sparse interpolation, and cross-pipeline mesh postprocessing into one spec. 10 gaps across three groups.
- evidence: 10 gaps remain after removing already-ported modules (SAM-MOT in `sam3d_ss_flow.py`, SAM-SHORTCUT in `sam3d_flow.py`) and training-only items. See `spec/gap-matrix.md` for original gap IDs.
- exit signal: All remaining gaps produce numerically matching or performance-comparable output against vendor references; all three pipelines reach full inference parity on Apple Silicon.

## Deferred or Not Now

- Performance optimization (excluded — no user demand for benchmarks at this stage)
- CI and release infrastructure (excluded — no external consumers yet)
- Supporting non-Apple-Silicon hardware or non-MLX backends
- Model fine-tuning or training capabilities
- Cloud deployment or API serving surface
- DINOv2 support in TRELLIS.2 (superseded by DINOv3)
- Visualization-only features (PBR rendering, voxel rendering, Gradio apps)
- Multi-GPU/distributed inference
