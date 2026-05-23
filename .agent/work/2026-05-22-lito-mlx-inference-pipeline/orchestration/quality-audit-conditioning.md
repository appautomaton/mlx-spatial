# quality-audit-conditioning

## Status

DONE_WITH_CONCERNS

## Confirmed Matches

- Upstream checkpoint inference passes RGBA into `LiToDiTTrainer.inference_sample_latent_mlx` and then `get_image_conditioning`: `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py:760`, `:793-808`, `:1085-1123`.
- Local checkpoint-backed path passes RGBA into the real backend, then premultiplies RGB by alpha before DINO/RGBA encoding: `src/mlx_spatial/lito_inference.py:220-239`, `src/mlx_spatial/lito_real_backend.py:612-648`.
- Upstream `dinov2_vitl14_reg_rgba` encoder uses final-layer DINOv2 ViT-L/14 reg tokens, cls/register/patch tokens, concat-token layernorm, RGB+alpha learnable branch, and `learnable_model_first_transforms_rgb=True`: `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py:609-629`.
- Local real backend mirrors those main settings: 518 input, patch 14, 1024 dim, 4 registers, concat-token normalization, cls/register/patch order, learned padding before patch tokens, and final DINO+learnable concat: `src/mlx_spatial/lito_real_backend.py:92-104`, `:2042-2126`.
- Upstream DINO branch normalizes RGB with ImageNet mean/std and selects cls/register/patch tokens from the final block output: `vendors/ml-lito/src/lito/models/dino.py:391-453`, `:773-811`. Local does the same normalization and token order: `src/mlx_spatial/lito_real_backend.py:2049-2091`.
- Upstream learnable branch normalizes premultiplied RGB first when configured, concatenates alpha, applies patch-stride conv, prepends learned paddings, then concatenates with DINO features: `vendors/ml-lito/src/lito/models/dino.py:580-635`, `:838-848`. Local mirrors this in `_run_lito_rgba_learnable_branch`: `src/mlx_spatial/lito_real_backend.py:2094-2126`.

## Confirmed Mismatches

- Preprocessing background removal differs. Upstream demo defaults `remove_bg=true`, reuses transparent RGBA if present, otherwise runs `rembg.remove`, then returns straight RGBA plus premultiplied RGB: `vendors/ml-lito/demos/lito/fastapi_lito_demo.py:141-268`, `:508-533`. Local `_preprocess_image` only converts to RGBA, crops on existing alpha, resizes, and never removes background: `src/mlx_spatial/lito_inference.py:463-472`.
- Crop/framing differs. Upstream default `keep_optical_axis=true` keeps the original image center as the optical axis while sizing crop from the alpha foreground extent: `vendors/ml-lito/src/lito/eval_scripts/st_paper_utils.py:1286-1408`. Local `_crop_and_pad_object` recenters the foreground bounding box on a new square canvas: `src/mlx_spatial/lito_inference.py:475-491`.
- `src/mlx_spatial/lito_condition.py::condition_image` is not upstream conditioning parity: it emits fixture-shaped `(B,17,64)` synthetic tokens from RGB/alpha means, while real LiTo uses DINO/RGBA `(B,1374,2048)` at 518px. This is by design for `source_contract_smoke`, not the checkpoint-backed path: `src/mlx_spatial/lito_condition.py:1-13`, `:67-102`; checkpoint-backed uses `backend.condition_rgba`: `src/mlx_spatial/lito_inference.py:235-239`.

## Unknowns

- Whether the baseline input already had meaningful transparent alpha. If it was opaque RGB/RGBA, the missing background-removal step is a high-probability conditioning cause.
- No local parity fixture compares `DirectMlxLitoBackend.condition_rgba` against upstream `SpatialDinov2` tokens for the same preprocessed RGBA and checkpoint weights.
- DINOv2 positional interpolation internals are not fully proven from vendored Apple source because DINOv2 itself is loaded through `torch.hub`; default 518px likely avoids interpolation, but non-default grids need parity instrumentation.

## Fix Target

Make `src/mlx_spatial/lito_inference.py::_preprocess_image/_crop_and_pad_object` follow upstream demo preprocessing: transparent-alpha detection, background-removal fallback or explicit blocker, `keep_optical_axis=True` crop/pad math, `th_alpha=0.8`, `fill_ratio=0.8`, and 518px RGBA output. DINO/RGBA conditioning math looks close enough from static source to fix preprocessing after the higher-confidence init-coordinate cap is tested.

## Verification

- Read-only inspections only; no CUDA, installs, upstream runtime, or source edits.
- Inspected upstream: `vendors/ml-lito/src/lito/models/dino.py`, `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py`, `vendors/ml-lito/demos/lito/fastapi_lito_demo.py`, `vendors/ml-lito/src/lito/eval_scripts/st_paper_utils.py`, LiTo generator config.
- Inspected local: `src/mlx_spatial/lito_condition.py`, `src/mlx_spatial/lito_real_backend.py`, `src/mlx_spatial/lito_inference.py`, `tests/test_lito_condition.py`, `tests/test_lito_real_backend.py`, `tests/test_lito_inference.py`, and `orchestration/quality-baseline.md`.

## Concerns

- Strongest finding is static-source preprocessing drift, not a token-by-token numerical proof.
- A small upstream-token capture fixture would still help rule out subtle MLX DINO block math differences.
