# Slice 0 — Vendor, Assets, Parity Fixtures, Routing Decision (LITO-A)

This slice is the first half of the gating funnel. It records vendor acquisition, licensing, weights, upstream coverage, recommended settings, and routing. Fixture capture moved to [00b-reference-fixtures.md](00b-reference-fixtures.md) after the upstream audit proved LiTo's reference path is hybrid MLX/PyTorch/gsplat, not all vendor-MLX.

## Inputs

- `INTAKE.md` (office-hours context)
- `SPEC.md` (canonical spec)
- `PLAN.md` (this plan)
- `spec/gap-matrix.md` (per-module probe scaffold; this slice closes its Open Items)

## Upstream Files to Read First

After shallow-cloning `vendors/ml-lito/`:

1. `vendors/ml-lito/LICENSE_MODEL`
2. `vendors/ml-lito/LICENSE_generated_samples`
3. `vendors/ml-lito/README.md` (top-level overview)
4. `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` (the MLX-backend end-to-end path — the parity oracle)
5. `vendors/ml-lito/src/lito/` directory listing — identify tokenizer module, DiT module, image-conditioner module
6. `vendors/ml-lito/plibs/` directory listing — identify LF-conditioned render module

## Reference Patterns in mlx-spatial

- `src/mlx_spatial/hyworld2_assets.py` — closest parallel for the asset layer (validation dataclass, `validate`, `inspect`, `download_command`)
- `src/mlx_spatial/sam3d_assets.py` — converter pattern reference
- `src/mlx_spatial/checkpoint.py` — `inspect_checkpoint` and `CheckpointTensorInfo` for tensor introspection
- `src/mlx_spatial/model_assets.py` — generic safetensors loading

## Steps

### 1. Vendor

```bash
git clone --depth 1 --shallow-submodules https://github.com/apple/ml-lito vendors/ml-lito
```

`vendors/` is already gitignored. Confirm `git status` does not list `vendors/ml-lito/` as untracked.

**Scratch policy:** any transient artifact during this slice (debug dumps, intermediate fixture-prep outputs, exploratory probes) uses `/tmp/lito-<stage>-<timestamp>/`. Example helper pattern:

```python
import tempfile, atexit, shutil
scratch = tempfile.mkdtemp(prefix="lito-fixtures-", dir="/tmp")
atexit.register(lambda: shutil.rmtree(scratch, ignore_errors=True))
```

After slice verification, `git status --porcelain` must show no unexpected entries (AC-15).

**Autonomous fallback policy:** transient `hf` CLI flakes, HF rate limits, network errors retry with exponential backoff (e.g., 1s/2s/4s/8s/16s, abort after 5). Lookup-style unknowns (e.g., "what is the current `mlx-community` repo ID for LiTo?", "did Apple update the recommended num_steps?") use `WebFetch` / `WebSearch`; only escalate via the declared decision checkpoints (Risk F, license-blocked, routing re-route). Do **not** stop and ask the user for anything else.

### 2. License Review

Read both license files. Record in this slice file under `## License Decision` (new section):

- May we convert `.ckpt` to MLX `safetensors` locally? (yes/no)
- May we redistribute Apple's demo sample images? (yes/no)
- Any other constraints we must propagate to `docs/lito.md`? (free-form)

If local conversion is barred, the converter path in `lito_assets.py` is omitted; runtime loads `.ckpt` via `pt-safe-loader` and `LITO_DEFAULT_ROOT` points at the raw `.ckpt` layout.

### 3. HF Search

```bash
hf search-models --query "mlx-community lito"
hf search-models --query "apple lito"
hf search-models --query "apple ml-lito"
```

If MLX-ready safetensors are found, record the repo ID and skip the converter path. If only `.ckpt` exists on HF, prefer HF over Apple CDN (better resumability and integrity).

### 4. Download Weights

If safetensors-direct path:

```bash
hf download <repo-id> --local-dir weights/lito
```

If `.ckpt`-convert path:

```bash
hf download <repo-or-cdn> lito_new.ckpt lito_dit_rgba.ckpt --local-dir weights/lito-raw
# then run lito_assets.convert in step 5
```

### 5. Write `src/mlx_spatial/lito_assets.py`

Mirror `hyworld2_assets.py` shape:

- `LITO_REPO_ID: str` — final HF repo ID
- `LITO_DEFAULT_ROOT: str` — `"weights/lito"` (direct) or `"weights/lito-mlx"` (converted)
- `LITO_COMPONENT_GROUPS: tuple[str, ...]` — e.g., `("tokenizer", "dit", "image_conditioner")`
- `@dataclass(frozen=True) class LitoAssetValidation:` — presence report
- `def validate(root: str | Path) -> LitoAssetValidation:`
- `def inspect(root: str | Path, prefixes: Iterable[str] | None = None, limit: int = 20) -> list[CheckpointTensorInfo]:`
- `def download_command(root: str | Path = LITO_DEFAULT_ROOT) -> str:` — prints `hf download …`
- If conversion is permitted: `def convert(src: Path, dst: Path) -> None:` — load `.ckpt` via `pt-safe-loader`, write `safetensors` with matching tensor names. Mirror conversion pattern in `sam3d_assets.py` / `model_assets.py`.

Write `tests/test_lito_assets.py`:

- `test_validate_layout_passes_on_downloaded_weights`
- `test_inspect_lists_expected_tensors`
- `test_download_command_prints_hf_invocation`
- (if convert exists) `test_convert_roundtrip_tensor_names_and_shapes`

### 6. Audit Upstream MLX Backend Coverage

Read `fastapi_lito_demo.py` and trace the call graph through `vendors/ml-lito/src/lito/` and `vendors/ml-lito/plibs/`. For each of {tokenizer, DiT, image-conditioner, LF-render}, record in `## Upstream Coverage Audit` (new section of this file):

- Does an MLX implementation exist in upstream? (yes/no)
- If yes, where? (file:line)
- If no, what does upstream use? (PyTorch + which library)

This is the data-driven check that records which backend is authoritative for each fixture boundary.

### 7. Capture Ground-Truth Fixtures

Moved to Slice 0B. Fixture capture must use the actual upstream reference backend per module:

- tokenizer: PyTorch
- image conditioner: PyTorch `SpatialDinov2`
- DiT: upstream MLX where practical, with PyTorch-produced conditioning; PyTorch sampling is acceptable for stepwise trajectory if MLX exposes only a final latent
- render: PyTorch + gsplat

For each parity boundary, save the resulting tensors to `tests/fixtures/lito/`:

- `tokenizer_input_<n>.safetensors` and `tokenizer_output_<n>.safetensors` — point-cloud → 8192 × 32 latent (≥ 3 fixed inputs)
- `dit_input_<n>.safetensors`, `dit_step_0_<n>.safetensors`, `dit_step_mid_<n>.safetensors`, `dit_step_final_<n>.safetensors` (≥ 1 fixed input × 1 fixed seed)
- `cond_input_<n>.safetensors` and `cond_output_<n>.safetensors` (≥ 3 fixed inputs)
- `render_input_<n>.safetensors` and `render_output_<n>.png` (≥ 1 fixed input)

Whether these fixtures are committed to the repo depends on the license decision in step 2:

- If `LICENSE_generated_samples` permits redistribution → commit fixtures under `tests/fixtures/lito/` (drop from `.gitignore` exclusion)
- If not → keep `tests/fixtures/lito/` gitignored and document fixture generation in `docs/lito.md` so contributors can regenerate

### 8. Tighten Tolerances in `spec/gap-matrix.md`

Moved to Slice 0B. After capturing fixtures, measure vendor's per-module output dtype (float16 vs. bfloat16 vs. float32) and update `spec/gap-matrix.md` "Open Items" → close them with concrete `atol`, `rtol`, or image-similarity thresholds.

Per the lifecycle "append-replace, not stack" rule, edit `spec/gap-matrix.md` in place. Do not duplicate the values into PLAN.md.

### 9. Risk F Audit

Read `vendors/ml-lito/plibs/` LF-render code. Compare to `src/mlx_spatial/gs_rasterize.py`. Identify whether the LF conditioning can be applied as a wrapper around the existing rasterizer or whether `gs_rasterize.py` makes HY-World-specific assumptions that block clean composition. Record in `## Risk F Audit` (new section) as `adapter feasible` or `adapter infeasible — escalate`.

### 10. Capture Upstream Recommended Generation Settings

Read upstream sources in priority order: `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` (live demo defaults) → any `vendors/ml-lito/configs/*.yaml` or `*.toml` → `vendors/ml-lito/README.md` and any model-card or release-notes doc → Apple's published LiTo doc page if any (use `WebFetch` or `WebSearch` autonomously; do **not** stop and ask).

Record in `## Recommended Settings` (new section of this file) — one bullet per setting, with the upstream source named verbatim (file + line range or URL):

- `num_steps` (rectified-flow integration steps for `lito_dit_rgba.ckpt`)
- `seed` policy (fixed, time-based, or upstream's specific recommendation)
- `cfg_scale` or rectified-flow equivalent
- Input image resolution and pre-processing normalization stats (mean, std, color space, alpha handling)
- Sampler-specific knobs (schedule type, step spacing, etc.)
- Any other knob the demo exposes by default

These become `LITO_RECOMMENDED_*` constants in `lito_inference.py` (Slice 5). **Defaults are not invented.** If upstream's recommendations are split across files, record each source and pick the most-recent / most-specific.

### 11. Routing Decision

Synthesize steps 2, 6, 9 into `## Routing Decision` (new section):

- `Approach A continues` if: a viable vendor reference exists for all 4 modules, license permits conversion or fallback works, adapter feasible
- `Approach C re-route: slice X` if: render needs to land first to de-risk the rest
- `Risk F option needed` if: adapter infeasible (auto-execute will surface the A/B decision)
- `Blocked: <reason>` if: licensing or coverage forbids a viable path

## Verification

```bash
test -d vendors/ml-lito
test -f .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
grep -E "^## License Decision" .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
grep -E "^## Upstream Coverage Audit" .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
grep -E "^## Risk F Audit" .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
grep -E "^## Recommended Settings" .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
grep -E "^## Routing Decision" .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
uv run python -c "from mlx_spatial.lito_assets import LITO_REPO_ID, validate; print(LITO_REPO_ID)"
ls tests/fixtures/lito/ | grep -E "tokenizer_|dit_|cond_|render_"
uv run pytest tests/test_lito_assets.py
# AC-15 hygiene: no scratch leaks
git status --porcelain | grep -vE "^\\?\\? \\.agent/work/2026-05-22-lito-mlx-inference-pipeline/" | grep -E "^\\?\\? " && echo "FAIL: scratch leak detected" || echo "OK: no scratch leaks"
```

## Slice-Specific Risks

- **Vendor clone size:** `apple/ml-lito` may include weights or large fixtures. Use `--depth 1 --shallow-submodules`; if size is still excessive, use `--filter=blob:none` and selectively materialize.
- **HF auth scope:** `hf` is logged in (user-confirmed) but the repo may be gated. If access is denied, surface immediately — do not attempt to bypass.
- **Vendor MLX backend dtype mismatch:** if vendor's MLX backend runs at bfloat16 internally and we target float16, fixture tolerance must absorb the dtype-conversion drift. Record this in `spec/gap-matrix.md` step 8.
- **Fixture size:** ground-truth fixtures could be large (8192 × 32 latents are small; render outputs may be 512×512×3 PNGs which are fine; but full DiT trajectories at all steps could bloat). Sample fewer steps if needed.

## Done When

All Slice 0A verification commands pass AND `## Routing Decision` records a specific decision string (not "TBD") AND Slice 0B has enough source/backend detail to capture fixtures.

## License Decision

- Local conversion from `.ckpt` to MLX-readable safetensors: yes for local research/development use. `LICENSE_MODEL` grants use, copy, modify, distribute, and create model derivatives only for Research Purposes, and explicitly excludes commercial exploitation/product use (`LICENSE_MODEL:17-28`). Any redistributed converted checkpoint or model derivative must carry the license and modification disclosure (`LICENSE_MODEL:30-43`).
- Redistribute Apple's generated demo sample images: no for modified/adapted fixtures. `LICENSE_generated_samples` is CC BY-NC-ND 4.0: sharing the licensed material is non-commercial only, and adapted material may be produced/reproduced but not shared (`LICENSE_generated_samples:141-152`). Technical modifications needed to exercise rights are allowed (`LICENSE_generated_samples:162-172`), but generated/modified sample fixtures should stay local/gitignored.
- Docs must propagate: model weights are research-only/non-commercial; generated samples are CC BY-NC-ND; do not use Apple names/logos to endorse derivatives; contributors should download/convert weights locally and regenerate uncommitted fixtures.

## HF Search

- Planned command correction: installed `uv run hf` has no `search-models` command. Equivalent `huggingface_hub.HfApi.list_models(search=..., limit=...)` was used.
- Queries run: `mlx-community lito`, `apple lito`, `apple ml-lito`, `lito_dit_rgba`, `lito_new`.
- Result: no official or `mlx-community` LiTo safetensors repo found. One unrelated `litong2/my-new-shiny-tokenizer` result appeared for `lito_new` and was ignored.
- Active weight path: download `lito_new.ckpt` and `lito_dit_rgba.ckpt` from Apple CDN into `weights/lito-raw/`, then convert locally into `weights/lito-mlx/` with `src/mlx_spatial/lito_assets.py::convert`.
- CDN sizes from HEAD on 2026-05-23: `lito_new.ckpt` 1,158,225,949 bytes; `lito_dit_rgba.ckpt` 7,365,078,343 bytes.

## Weight Conversion

- Raw checkpoint downloads completed into gitignored `weights/lito-raw/`:
  - `lito_new.ckpt` size 1,158,225,949 bytes
  - `lito_dit_rgba.ckpt` size 7,365,078,343 bytes
- Converted output root: gitignored `weights/lito-mlx/`.
  - `tokenizer/lito_new.safetensors` with 1,108 tensors
  - `image_to_3d/lito_dit_rgba.safetensors` with 2,793 tensors
  - conversion metadata under each component's `conversion_metadata/`
- Converter path: `pt-safe-loader` rejected the Apple checkpoint pickle shape with `BUILD with non-empty state is not supported`; `src/mlx_spatial/lito_assets.py::convert` therefore falls back to `torch.load(..., map_location="cpu")`, extracts `state_dict` when present, clones tensors to break shared storage aliases, and writes safetensors with `safetensors.torch.save_file`.
- Verification:
  - `uv run python -m mlx_spatial.lito_assets validate weights/lito-mlx` returns `OK`.
  - `uv run python -m mlx_spatial.lito_assets inspect weights/lito-mlx --limit 3` lists deterministic tensor metadata.
  - `uv run pytest tests/test_lito_assets.py` passes (`4 passed`).

## Upstream Coverage Audit

- Tokenizer: no full upstream MLX tokenizer encoder. Upstream tokenizer trainer and encoder are PyTorch: `LightTokenizationTrainer` imports PyTorch/PyTorch3D/Open3D (`vendors/ml-lito/src/lito/trainers/lito_trainer.py:17-45`), constructs `SPointEncoder` (`lito_trainer.py:219-221`), and `SPointEncoder` is PyTorch/xformers/PyTorch3D based (`vendors/ml-lito/src/lito/models/spoint_encoder.py:10-27`). The released MLX coverage for this area is only the Gaussian decoder conversion/run path (`lito_trainer.py:1256-1526`).
- DiT: yes, partial MLX implementation exists. MLX DiT model is in `vendors/ml-lito/src/lito/mlx/models/dit.py:1-260`; PyTorch-to-MLX conversion is in `vendors/ml-lito/src/lito/mlx/convert.py:28-114`; the demo calls `inference_sample_latent_mlx` on macOS/CPU (`vendors/ml-lito/demos/lito/fastapi_lito_demo.py:335-345`). Caveat: image conditioning tokens are still computed in PyTorch before MLX ODE sampling (`vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py:798-812`).
- Image conditioner: no upstream MLX conditioner. The demo imports and uses PyTorch (`fastapi_lito_demo.py:37-42`); `load_img_encoder` builds `SpatialDinov2` (`lito_dit_trainer.py:563-632`), and `SpatialDinov2` relies on Torch/Torchvision and `torch.hub.load("facebookresearch/dinov2", ...)` (`vendors/ml-lito/src/lito/models/dino.py:12-13`, `dino.py:81`). Slice 3 must port the source contract to MLX; optional Torch parity is CPU/MPS-only if it can run without CUDA-only dependencies.
- LF/render: no upstream MLX render. The demo's primary output path saves a PLY via `plibs.gs_utils.Gaussians.save_ply` (`fastapi_lito_demo.py:391-411`). Training/validation rendering uses `render_gaussians` and `gs_utils.render_3dgs_gsplat` (`lito_trainer.py:1592-1694`, `vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py:867-999`), which import `gsplat` and run through PyTorch tensors.

## Risk F Audit

adapter feasible

Evidence: LiTo render inputs match the existing rasterizer's generic surface: Gaussian centers, XYZW quaternions, scales, opacities, SH features, camera intrinsics, and world-to-camera/camera-to-world parameters. The mlx-spatial renderer accepts those fields directly (`src/mlx_spatial/gs_rasterize.py:216-297`) and evaluates degree > 0 SH features (`src/mlx_spatial/gs_rasterize.py:523-556`). Vendor `render_3dgs_gsplat` consumes the same core fields (`vendors/ml-lito/libraries/plibs/src/plibs/gs_utils.py:867-999`). Known parity risk: vendor uses gsplat antialiasing/`eps2d` and `render_mode` behavior; Slice 4 should wrap `gs_rasterize.py` and compare to fixtures before any shared-base refactor.

## Recommended Settings

- `num_steps`: 20 for the public interactive demo. Source: `/generate-stream` default `sampling_steps: int = Form(20)` (`vendors/ml-lito/demos/lito/fastapi_lito_demo.py:555-560`) and warmup uses `sampling_steps=20` (`fastapi_lito_demo.py:668-670`).
- `seed` policy: no user seed control in the public demo; generation noise is unseeded at runtime. Source: demo exposes no seed form/arg (`fastapi_lito_demo.py:555-560`, `fastapi_lito_demo.py:624-647`) and MLX path uses `mx.random.normal(...)` inside `inference_sample_latent_mlx` (`vendors/ml-lito/src/lito/trainers/lito_dit_trainer.py:836-837`).
- `cfg_scale`: 3.0 for the public demo. Source: `/generate-stream` default `cfg_scale: float = Form(3.0)` (`fastapi_lito_demo.py:555-560`), warmup uses `cfg_scale=3.0` (`fastapi_lito_demo.py:668-670`), and `inference_sample_latent_mlx` defaults to 3.0 (`lito_dit_trainer.py:760-768`).
- Input image resolution: 518. Source: preprocessing default `img_resolution: int = 518` (`fastapi_lito_demo.py:141-148`) and CLI arg `--img_resolution` default 518 (`fastapi_lito_demo.py:641-645`).
- Preprocess: EXIF transpose, optional background removal, object crop/pad with `fill_ratio=0.8`, `keep_optical_axis=True`, alpha threshold 0.8, resize to square 518, convert to `[0, 1]`, and return straight RGBA plus premultiplied RGB. Source: `preprocess_image` (`fastapi_lito_demo.py:141-268`).
- Color normalization for DINO conditioning: Image tensors are normalized with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Source: `vendors/ml-lito/src/lito/models/dino.py:43-55`.
- Sampler: rectified-flow ODE method `heun`; timesteps are `mx.linspace(self.t_eps, 1.0, ode_num_steps)` for `heun`; demo passes `mlx_compute_dtype="float16"`. Sources: demo call (`fastapi_lito_demo.py:338-345`) and `inference_sample_latent_mlx` timestep construction (`lito_dit_trainer.py:880-906`).
- Decode init: demo selects `voxel_decoder` if present, otherwise `sample_xyz`; Gaussian decode uses `steps_for_sample_xyz=50` and `mlx_compute_dtype="float16"`. Source: `fastapi_lito_demo.py:356-379`.
- Output: public demo writes PLY by default; optional SPZ compression is disabled by default. Sources: PLY save path (`fastapi_lito_demo.py:391-411`) and `compress_spz: str = Form("false")` (`fastapi_lito_demo.py:555-560`).

## Source-Contract Fixture Status

- Ground-truth vendor fixtures are not required for Slice 0B because CUDA is not allowed locally or as an acceptance gate.
- Historical dependency probing showed that upstream tokenizer execution falls into xformers/flash-attention-shaped code when CUDA is absent. That implementation is static source reference only.
- Active Slice 0B route: generate deterministic local source-contract fixtures under `tests/fixtures/lito/` with no vendor imports, no MLX imports, and no CUDA dependency. Validate them with `scripts/lito/validate_fixtures.py`.
- Torch with CPU/MPS remains acceptable only for optional `torch_parity` tests when the path runs without CUDA-only packages.
- License handling for fixtures: generated local fixtures must remain synthetic and must not redistribute Apple sample material unless a separate redistributable source image is used.

## Routing Decision

Approach A continues

Reason: license permits local conversion for research use, weight acquisition is public via Apple CDN, DiT and Gaussian decoder have upstream MLX ports, and Risk F is adapter-feasible. The plan must not assume complete upstream MLX coverage: Slices 1 and 3 are source-guided MLX ports, Slice 2 can use upstream MLX source/weights where available, and Slice 4 ports render semantics into the existing MLX/Metal rasterizer. This routing decision is made; Slice 0B now completes by generating local source-contract fixtures instead of waiting for a CUDA vendor run.
