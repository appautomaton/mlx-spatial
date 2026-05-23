# quality-audit-decode

## Status

DONE_WITH_CONCERNS

## Confirmed Matches

- Init coordinate threshold/axis/center mostly matches. Upstream `vendors/ml-lito/src/lito/trainers/lito_trainer.py:1134` thresholds occupancy, transposes `(z,y,x)` with `permute(0,3,2,1)`, and centers `(idx + 0.5) * cell_width + min`. Local `src/mlx_spatial/lito_real_backend.py:1120` applies the same transpose and cell-center convention to TRELLIS logits with threshold `0.0`, equivalent to upstream sigmoid threshold `0.5`.
- TRELLIS occupancy logits path is semantically aligned by source. Upstream `vendors/ml-lito/src/lito/integrations/trellis/trellis_sparse_structure.py:123` returns dense logits and `:139` treats `logits > 0` as occupancy. Local `src/mlx_spatial/lito_real_backend.py:1181` consumes sparse-structure logits before `occ_grid_to_lito_init_coord`.
- Gaussian decode equations match. Upstream `vendors/ml-lito/src/lito/models/point_decoder.py:1363` normalizes quaternion, sigmoid-activates scaling, sigmoid-activates opacity with bias/scale, reshapes SH, and applies local xyz offsets. Local `src/mlx_spatial/lito_real_backend.py:786` mirrors those conversions.
- Localized voxel attention grouping and half-cell shift match by static inspection. Upstream precomputes self-attention voxel info with shifts `0` and `0.5 * self_cell_width` in `vendors/ml-lito/src/lito/trainers/lito_trainer.py:1340`; local `src/mlx_spatial/lito_real_backend.py:992` and `:1233` build the same alternating grouped attention metadata.
- LiTo PLY field ordering/conversions match the upstream LiTo export path. Upstream `vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py:445` writes `f_dc`, `f_rest`, inverse-sigmoid opacity, log scale, and raw rotation. Local `src/mlx_spatial/lito_real_backend.py:1330` writes the same schema/order/conversions.

## Confirmed Mismatches

- Local init-coordinate coverage is capped; upstream is not. Upstream `inference_init_coords_for_decoder` keeps all occupied cells after thresholding. Local `src/mlx_spatial/lito_real_backend.py:33` defines `safe=512`, `balanced=2048`, `large=8192`; `:441` passes that cap into init decoding; `:1150` applies per-batch top-k. The baseline command used `--memory-profile safe`, and the resulting `32768` vertices equals `512 * 64`, confirming the cap affected the checkpoint-backed PLY.
- Local quaternion fallback/render convention is inconsistent with upstream rotation code. Upstream `vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py:139` treats quaternion slot `0` as scalar `w`, and TRELLIS initializes `rots_bias[0] = 1` in `vendors/ml-lito/third_party/TRELLIS/trellis/representations/gaussian/gaussian_model.py:66`. Local fallback identity in `src/mlx_spatial/lito_real_backend.py:2633` sets slot `3`, and local rendering expects XYZW scalar-last. This is confirmed for local render/fallback behavior; whether exported checkpoint quaternions need reordering still needs a tensor-level roundtrip check.

## Unknowns

- Whether an uncapped local decode produces a visually coherent PLY. Static source strongly implicates the cap, but runtime comparison was outside this read-only audit.
- Whether checkpoint quaternion tensors are intended WXYZ or XYZW at the PLY boundary. Upstream render/export conventions point to WXYZ, while local render helpers point to XYZW.
- Whether local `probe_sparse_structure_decoder_boundary` is numerically identical to upstream TRELLIS `decode_lowres_latent_to_logits`; only static source parity was audited.

## Fix Target

First fix target: add a quality/no-cap or streaming/chunked Gaussian decode path so Slice 5Q quality export uses all occupied init cells, matching upstream `inference_init_coords_for_decoder`. Re-run the checkpoint-backed PLY before changing Gaussian decode math.

## Verification

- Read-only source inspection with `rg`, `nl`, and `sed` across `vendors/ml-lito`, `vendors/ml-lito/third_party/TRELLIS`, `src/mlx_spatial/lito_real_backend.py`, `src/mlx_spatial/lito_render.py`, `tests/test_lito_real_backend.py`, and `orchestration/quality-baseline.md`.
- No CUDA execution, dependency install, upstream runtime execution, or file edits performed.

## Concerns

- The cap is a strong source-level explanation, but visual causality still needs an uncapped or chunked decode comparison.
- Quaternion convention may mainly affect local preview/rendering; PLY export could still be correct if consumers interpret LiTo `rot_*` as scalar-first.
