---
active_change: 2026-05-20-production-pipeline-parity
stage: plan
---

# Status

## Current Change

- active change: `2026-05-20-production-pipeline-parity`
- current stage: `plan`

## What Is True Now

- PLAN.md written at `.agent/work/2026-05-20-gap-matrix-parity/PLAN.md` with 11 ordered slices.
- Spec and gap matrix are in `.agent/work/2026-05-20-gap-matrix-parity/SPEC.md` and `spec/gap-matrix.md`.
- Slices 1, 2, 3, 4, 5, 6, and 7 are complete; Slice 2 added HY-World transformer block and DINO ViT public facades with extracted-reference parity tests.
- Slice 4 has a local integration pass: WorldMirror now composes the extracted transformer/DINO ViT modules and `uv run pytest tests/test_hyworld2_*.py -v` passes.
- Slice 4 dev-only Torch reference setup is complete (`torch==2.12.0`, `torchvision==0.27.0`, `uv pip check` passes).
- Slice 4 parity compare now passes on Cat_Girl in MLX CPU parity mode: `outputs/hyworld2/cat-girl-518-cpu-parity-report.json` has `passed=true`, 13 tensors checked, and `parity_trace_metadata.numeric_parity_verified=true`.
- Slice 4 correction note: strict Torch CPU numeric parity requires `--mlx-device cpu`; default/GPU MLX still has backend precision drift and is not the Slice 4 evidence path.
- Slice 5 adds MLX sparse nearest/trilinear interpolation helpers and dense-grid reference tests. Plan correction: TRELLIS.2 advertises these names in the sparse registry, but the checked-in vendor `spatial` modules do not define the functions.
- Slice 6 confirms the SAM3D MOT sparse-structure-flow path in `sam3d_ss_flow.py`: merged pose latents, protected shape attention, pose decoding, and reference-value MOT self-attention tests. `uv run pytest tests/test_sam3d_ss_flow.py tests/test_sam3d_ss.py -v` passes with 9 tests.
- Slice 7 adds the named TRELLIS.2 FlexiDualGrid encoder contract and verifies the existing textured GLB path. `uv run pytest tests/test_trellis2_decode.py tests/test_trellis2_inference.py tests/test_trellis2_export.py -v` passes with 82 tests; the extended asset/tool guard set passes with 103 tests.
- Slice 8 Metal GS rasterizer is complete. It adds `src/mlx_spatial/gs_rasterize.py`, `src/mlx_spatial/metal/gs_rasterize.metal`, exports in `src/mlx_spatial/__init__.py`, and `tests/test_gs_rasterize.py`. `uv run pytest tests/test_gs_rasterize.py -v` passes with 9 tests, including anisotropic quaternion projection and Metal-vs-CPU parity within 1%.
- Slice 8 checkpoint was discharged as non-blocking for continuation: `gsplat` is not installed locally, the runtime renderer must stay MLX-only, and available local test/review verification passed. Follow-up validation remains: dev-only gsplat/PyTorch standard-image parity and/or visual inspection.
- Slice 9 is complete. It adds `src/mlx_spatial/sam3d_render.py` for SAM3D orbit cameras, multi-view Gaussian rendering through the Slice 8 rasterizer, SAM3D Gaussian field adaptation, and rigid ICP-style layout post-optimization. `uv run pytest tests/test_sam3d_*.py -k "render or layout or holes" -v` passes with 7 selected tests.
- Slice 10 is complete. It adds SAM3D shortcut parity reporting and HY-World grid utilities. `uv run pytest tests/test_sam3d_flow.py tests/test_hyworld2_*.py -k "shortcut or grid" -v` passes with 7 selected tests.
- Slice 11 is complete. It adds TRELLIS.2 mesh quality metrics and a source-vs-MLX improvement report that records the official `cumesh.remeshing.remesh_narrow_band_dc` remeshing gap. `uv run pytest tests/test_trellis2_export.py -v` passes with 36 tests.
- Final verification passed. Fresh checks: HY-World suite 151 passed, HY-World CPU parity compare 13 tensors passed with `numeric_parity_verified=true`, sparse interpolation suite 13 passed/1 optional local PyTorch checkout parity skipped, SAM3D suite 124 passed, GS rasterizer suite 9 passed, TRELLIS.2 suite 200 passed.

## Next Step

Change is complete. Use `auto-office-hours` to shape the next objective when ready.

## Open Risks

- Slice 8 is correctness-first, not performance-complete: Python projects/sorts splats and the Metal kernel renders one pixel per thread without float atomics. Tiled/binning performance work remains future work.
- Slice 8 higher-degree SH is not implemented unless callers pre-evaluate RGB. Degree-0 SH and direct RGB are supported.
- No CI pipeline exists; quality gate is manual (`uv run pytest`).
- Parity verification depends on local PyTorch reference outputs (dev-only, not CI-gated).
- The optional local PyTorch checkout used by `tests/test_sparse_conv_parity.py` is not importable as installed C extensions, so that single optional parity test still skips even with `MLX_SPATIAL_RUN_TORCH_PARITY=1`.
