# LiTo Gap Matrix

Per-module source-contract and parity reference for `LITO-A` through `LITO-F`. Vendor source paths assume `vendors/ml-lito/` after `git clone --depth 1 https://github.com/apple/ml-lito vendors/ml-lito`. CUDA-only upstream paths are static reference only. Slice 0B records deterministic local MLX-compatible fixtures and tolerances so a sub-agent can land a slice cold without CUDA.

## Upstream Reference Layout

| Path | Role |
|---|---|
| `vendors/ml-lito/src/lito/` | PT-Lightning trainers and model definitions; tokenizer + DiT live here |
| `vendors/ml-lito/libraries/plibs/src/plibs/` | 3D utilities including gsplat and nvdiffrast renderers; LF-conditioned GS render lives here |
| `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` | MLX-backend end-to-end inference path — the parity oracle |
| `vendors/ml-lito/LICENSE_MODEL` | Governs local checkpoint conversion to MLX safetensors |
| `vendors/ml-lito/LICENSE_generated_samples` | Governs redistribution of demo sample images |
| `weights/lito/` (local, gitignored) | MLX-ready safetensors when HF hosts them directly |
| `weights/lito-mlx/` (local, gitignored) | Converted-from-PyTorch MLX safetensors when conversion is required |

## HF Search Targets (Slice 0)

The approved plan named `hf search-models`, but the installed `hf` CLI does not expose that command. Slice 0 used the equivalent `huggingface_hub.HfApi.list_models(...)` query:

```
uv run python -c "from huggingface_hub import HfApi; ..."
```

Results: no official or `mlx-community` LiTo safetensors repo was found for `mlx-community lito`, `apple lito`, `apple ml-lito`, `lito_dit_rgba`, or `lito_new`. Active path: Apple CDN `.ckpt` download to `weights/lito-raw/`, local conversion with `pt-safe-loader`, converted safetensors under `weights/lito-mlx/`.

## LITO-A — Vendor + Asset Acquisition + License Review

- **Upstream sources**: `apple/ml-lito` GitHub (shallow clone). `LICENSE_MODEL` and `LICENSE_generated_samples` at repo root. Apple CDN URL set in INTAKE: `https://ml-site.cdn-apple.com/models/lito/` (`lito_new.ckpt` tokenizer, `lito_dit_rgba.ckpt` image-to-3D DiT).
- **HF search**: as above.
- **mlx-spatial reuse**: `hyworld2_assets.py` shape (asset dataclass, `validate`, `inspect`, `download_command`); `sam3d_assets.py` for converter pattern reference; `pt-safe-loader` for safe `.ckpt` deserialization; `safetensors` for write side.
- **New module**: `lito_assets.py`. If conversion is required: a converter callable mirroring `hyworld2_assets` / `sam3d_assets` conventions, output path `weights/lito-mlx/`.
- **Verification**:
  - `lito_assets.validate(root)` returns a frozen dataclass report (mirror `HyWorld2AssetValidation`).
  - `lito_assets.inspect(root)` lists tensor names and shapes via `checkpoint.inspect_checkpoint`.
  - Converter output round-trips through `safetensors.safe_open(...)` with matching keys, shapes, and dtype.

## LITO-B — Image Conditioner Adapter

- **Upstream source**: `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py::load_img_encoder` and `::get_image_conditioning`; implementation uses `vendors/ml-lito/src/lito/models/dino.py::SpatialDinov2`.
- **mlx-spatial reuse**: `trellis2_dinov3_forward.py`, `trellis2_dinov3.py`, `hyworld2_vit.py`. Reuse the encoder weights and forward path if upstream uses a model already in mlx-spatial; otherwise write a thin adapter that loads the upstream encoder via the converter path.
- **New module**: Thin adapter — likely `lito_condition.py` mirroring `sam3d_condition.py`. Final filename TBD by Slice 0.
- **Verification**: Source-contract checks on encoded image feature shape, dtype, token ordering, and normalization vs. `tests/fixtures/lito/cond_*.safetensors`. Initial contract tolerance for local float16 microfixtures: `atol=2e-3, rtol=2e-3` for deterministic preprocessing transforms, `atol=2e-3, rtol=2e-2` only for optional CPU/MPS Torch parity if the upstream path runs without CUDA-only dependencies.

## LITO-C — Tokenizer (point cloud → 8192 × 32 latent)

- **Upstream source**: `vendors/ml-lito/src/lito/trainers/lito_trainer.py` and `vendors/ml-lito/src/lito/models/spoint_encoder.py`. No upstream MLX tokenizer encoder exists; the released MLX path only converts/runs the Gaussian decoder.
- **mlx-spatial reuse**: None at the tokenizer level. Generic ops (attention, MLP, norm) reuse `hyworld2_layers.py` patterns where applicable.
- **New module**: `lito_tokenizer.py`.
- **Verification**: Source-contract probe on tokenized 8192 x 32 latents from >= 3 fixed local inputs. Initial contract tolerance for local float16 microfixtures: `atol=2e-3, rtol=2e-3` for deterministic reductions/projections encoded in the fixture; optional CPU/MPS Torch parity is allowed only if the path avoids CUDA-only xformers/flash-attention requirements.
- **No-CUDA note**: tokenizer fixture execution was probed in `/tmp/lito-fixture-env`; the upstream no-CUDA path fell into xformers-backed localized attention. That implementation is now static source reference only. Slice 1 must port the operation to MLX-compatible attention / neighborhood operations rather than requiring xformers or CUDA.

## LITO-D — Flow-Matching DiT

- **Upstream source**: `vendors/ml-lito/src/lito/mlx/models/dit.py`, `vendors/ml-lito/src/lito/mlx/convert.py`, and `vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py::inference_sample_latent_mlx`. Weights: `lito_dit_rgba.ckpt`.
- **mlx-spatial reuse**: `trellis2_forward.py` and `trellis2_inference.py` flow patterns; `hyworld2_transformer.py` block patterns.
- **New module**: `lito_dit.py`.
- **Verification**: Denoising-trajectory parity at fixed seed and step count against upstream MLX when executable locally; otherwise source-contract microtrajectory fixtures compare intermediate latents at sampled steps (first step, mid-step, final step) plus final latent. Initial drift budget: `atol=5e-2, rtol=5e-2` at intermediate/final latents because errors accumulate through 20 Heun steps; local float16 deterministic microfixtures use `atol=2e-3, rtol=2e-3`.

## LITO-E — LF-Conditioned 3DGS Render

- **Upstream source**: `vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py::render_3dgs_gsplat` for gsplat rendering and `vendors/ml-lito/src/lito/trainers/lito_trainer.py::render_gaussians` for trainer integration.
- **mlx-spatial reuse**: `gs_rasterize.py`, Metal compute kernel under `src/mlx_spatial/metal/`, `hyworld2_sh.py`, `hyworld2_camera.py`.
- **New module**: `lito_render.py` — LF conditioning wrapper around the existing rasterizer. If composition fails, a thin shared base under `gs_rasterize.py` is factored out and both HY-World and LiTo depend on it.
- **Verification**: Source-contract render checks on Gaussian tensor schema, camera conventions, output shape, alpha range, and deterministic local image fixture. Initial local threshold: exact shape match, alpha in `[0, 1]`, mean absolute error <= 1e-5 for local deterministic render fixture. Optional non-CUDA image comparison may use PSNR >= 28 dB or mean absolute error <= 0.02 if a vendor-equivalent CPU/MPS render is available without CUDA-only dependencies.

## LITO-F — Pipeline + CLI + Docs

- **Upstream reference**: `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` for the end-to-end path.
- **mlx-spatial reuse**: `hyworld2.py`, `sam3d.py`, `trellis2.py` CLI patterns; `hyworld2_inference.py` memory-profile system; `HYWORLD2_MEMORY_PROFILES` shape for `LITO_MEMORY_PROFILES`.
- **New modules**: `lito.py` (CLI), `lito_inference.py` (orchestration), `docs/lito.md`.
- **Verification**:
  - `mlx-spatial-lito generate inputs/lito/<sample>.<ext> --output outputs/lito/<sample>.<ext>` produces a 3DGS file.
  - `from mlx_spatial.lito import LitoInferencePipeline` works without `vendors/` present.
  - `mlx-spatial-lito validate weights/lito/` (or `weights/lito-mlx/`) returns OK.

## Verification Commands (scaffold; tightened in Slice 0)

- Per-module parity: `uv run pytest tests/test_lito_<module>.py`
- Optional PyTorch parity (dev-only, CPU/MPS only): `uv run pytest tests/test_lito_<module>.py -m torch_parity`
- End-to-end smoke: `mlx-spatial-lito generate inputs/lito/<sample>.png --output outputs/lito/<sample>.<ext>`
- Asset validation: `mlx-spatial-lito validate weights/lito/` or `mlx-spatial-lito validate weights/lito-mlx/`
- Regression sweep: `uv run pytest` (full suite — all four pipelines)

## Open Items (closed or narrowed at Slice 0)

- Exact vendor file paths are recorded above for tokenizer, DiT, image conditioner, and render.
- Safetensors-direct is unavailable from HF search; active path is `.ckpt`-convert into `weights/lito-mlx/`.
- Vendor demo requests `mlx_compute_dtype="float16"` for both DiT sampling and Gaussian decoding; upstream image conditioning and tokenizer/init-coordinate paths remain PyTorch.
- Upstream MLX coverage is partial: DiT and Gaussian decoder have MLX ports; tokenizer encoder, DINO image conditioner, init-coordinate generation, and render are PyTorch/gsplat.
- Known-good sample inputs are not redistributed from upstream generated samples; generated samples are CC BY-NC-ND and local fixtures remain uncommitted unless their source permits redistribution.
- Slice 0B no longer waits for real CUDA/vendor-reference fixture capture. It generates local source-contract fixtures and manifest metadata with no vendor imports, no MLX requirement, and no CUDA dependency.
- Optional Torch parity is CPU/MPS-only. CUDA, xformers, flash-attention, and CUDA-backed gsplat remain static source references; downstream MLX slices port the needed operations to MLX-compatible code.
