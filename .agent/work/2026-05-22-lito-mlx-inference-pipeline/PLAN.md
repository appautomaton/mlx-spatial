# LiTo MLX Inference Pipeline Plan

## Goal

Per [SPEC.md](SPEC.md) bounded goal — port Apple LiTo inference into `mlx-spatial` as a new pipeline. The vendored LiTo implementation is the source reference; CUDA-only PyTorch/gsplat paths are read-only architecture references, not runnable dependencies or fixture gates.

## Approach

Slice 0 is split into two serial gates. Slice 0A has shallow-cloned the vendor, resolved the license/weight/routing decisions, built the asset layer, and audited upstream coverage. Slice 0B now records source contracts and generates deterministic MLX-compatible contract fixtures at each module boundary without importing vendor runtime code or requiring CUDA. CUDA/PyTorch/gsplat implementations remain source references for the MLX port. Those local fixtures decouple Slices 1–4 from each other and unlock 4-way parallel sub-agent execution. Slice 5 wires the modules into the live pipeline. Slice 6 lands docs and roadmap, parallel-safe with Slice 5.

**Quality refresh (2026-05-23):** Slices 0-6 remain the historical implementation plan, but Slice 5 is re-opened for quality closure. The checkpoint-backed PLY exists, yet user inspection reports broken surfaces and non-credible generation quality. AC-07 is therefore not accepted until Slice 5Q proves the real-object outputs are visually sensible, not just schema-valid. `inputs/lito/smoke.png` remains a color-blob framework probe and cannot be used as qualitative evidence.

```
Slice 0A (assets/routing, complete) -> Slice 0B (source contracts + local fixtures)
   │
   ├── Slice 1: Tokenizer (LITO-C)   ── parallel ──┐
   ├── Slice 2: DiT (LITO-D)         ── parallel ──┤
   ├── Slice 3: Image Conditioner (LITO-B) ── par ─┤
   └── Slice 4: LF Render (LITO-E)   ── parallel ──┤
                                                    ▼
                          Slice 5: Pipeline + CLI + Smoke (LITO-F integration)
                                                    │
                          Slice 6: Docs + ROADMAP (LITO-F docs) ── parallel with Slice 5
                                                    │
                          Slice 5Q: Quality Closure (AC-07 re-opened)
```

## Quality Closure Replan (2026-05-23)

This addendum supersedes any prior "ready for auto-verify" language until Slice 5Q passes. It keeps the scope inside the approved LiTo inference spec: no CUDA execution, no CUDA dependencies, no vendor runtime import, and no training work. Upstream PyTorch/CUDA code is static reference only; optional Torch parity is CPU/MPS-only and must not enter runtime dependencies.

### Slice 5Q-0: Quality Baseline + Inspector

**Objective:** Create a reproducible quality baseline for real-object LiTo outputs and add a small inspector that reports whether a PLY is merely valid or has obvious Gaussian-stat failures.

**Acceptance criteria:**
- Regenerate a baseline from `inputs/trellis2/teacup.png` using checkpoint-backed weights, fixed seed, and safe profile.
- Add `scripts/lito/inspect_quality.py` or equivalent test helper that reads LiTo Gaussian PLY files and reports: vertex count, bbox min/max, opacity distribution, scaling distribution, quaternion norm distribution, finite-value counts, and header comments.
- Record baseline stats under `.agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-baseline.md` with the exact command, output path, and failure classification.
- Do not use `inputs/lito/smoke.png` as quality evidence.

**Verification:**
```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-baseline.ply --memory-profile safe --render-size 12 --num-steps 20 --seed 42 --print-metrics
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-baseline.ply --json /tmp/lito-teacup-quality-baseline.json
uv run pytest tests/test_lito_quality.py -q
```

**Execution:** direct
**Depends on:** Slice 5 and Slice 6 implementation artifacts
**Touches:** `scripts/lito/inspect_quality.py`, `tests/test_lito_quality.py`, `.agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-baseline.md`
**Produces:** baseline PLY, JSON stats, quality-baseline note

**Slice 5Q-0 evidence (2026-05-23):** Added `scripts/lito/inspect_quality.py` and `tests/test_lito_quality.py`; `uv run pytest tests/test_lito_quality.py -q` passed with `3 passed`. Baseline generation command produced `outputs/lito/teacup-quality-baseline.ply` and `.safetensors`; inspector wrote `/tmp/lito-teacup-quality-baseline.json`, reported checkpoint-backed header, `32768` vertices, `62` properties, no inspector flags, and `failure_classification=stats_sane_visual_review_required`. AC-07 remains open because user visual inspection reports broken surfaces despite schema/stat sanity. Detailed evidence lives in `orchestration/quality-baseline.md`.

### Slice 5Q-1: Parallel Source-Contract Audit

**Objective:** Use read-only subagents to locate the highest-probability parity mismatch behind the broken surfaces.

**Acceptance criteria:**
- Three read-only audit notes exist under `.agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/`:
  - `quality-audit-conditioning.md`: upstream image preprocessing, alpha/RGBA handling, DINO token selection, normalization, resize/crop, positional interpolation.
  - `quality-audit-sampler.md`: upstream DiT CFG path, timestep schedule, latent initialization, velocity update, guidance/dropout, latent scaling.
  - `quality-audit-decode.md`: upstream voxel/TRELLIS init-coordinate path, axis/cell-center convention, occupancy threshold/top-k/cap, Gaussian decode field ordering, scaling/opacity/quaternion/SH conventions.
- Each note names exact upstream reference files/functions and exact local files/functions, then lists confirmed matches, confirmed mismatches, and unknowns.
- The coordinator chooses one fix target for Slice 5Q-2 based on evidence, not guesswork.

**Verification:**
```bash
test -f .agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-audit-conditioning.md
test -f .agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-audit-sampler.md
test -f .agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-audit-decode.md
rg -n "Confirmed mismatch|Fix target|No mismatch found" .agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-audit-*.md
```

**Execution:** subagent required
**Depends on:** Slice 5Q-0
**Touches:** `.agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-audit-*.md` only
**Detail:** [slices/05q-quality-closure.md](slices/05q-quality-closure.md)

**Slice 5Q-1 evidence (2026-05-23):** Three read-only subagent audits completed with `DONE_WITH_CONCERNS` and were written to `orchestration/quality-audit-conditioning.md`, `orchestration/quality-audit-sampler.md`, and `orchestration/quality-audit-decode.md`. Conditioning audit found upstream preprocessing/crop drift (no local background removal; local recentering instead of optical-axis crop). Sampler audit found no high-confidence mismatch in the active `heun`/CFG path. Decode audit found the highest-confidence first fix target: safe profile caps init cells at `512`, and the baseline `32768` vertices equals `512 * 64`; upstream keeps all occupied init cells after thresholding. Coordinator selects decode/init-coverage as Slice 5Q-2 pass 1; preprocessing is next if uncapped/chunked decode does not improve quality.

### Slice 5Q-2: Targeted Quality Fix

**Objective:** Patch the single highest-confidence mismatch from Slice 5Q-1, with tests that fail on the previous behavior and pass on the corrected MLX path.

**Acceptance criteria:**
- Exactly one mismatch class is fixed per pass: conditioning, sampler, voxel/init-coordinate, or Gaussian decode/export.
- Runtime code remains MLX-native and no-CUDA. No imports from `torch`, `vendors/ml-lito`, `xformers`, `flash_attn`, `gsplat`, or CUDA packages are added.
- Tests cover the corrected contract using local fixtures, static upstream-derived invariants, or CPU/MPS-only optional parity where available.
- The quality inspector shows a concrete improvement or a narrowed failure mode versus Slice 5Q-0.

**Verification:**
```bash
uv run pytest tests/test_lito_real_backend.py tests/test_lito_inference.py tests/test_lito_quality.py -q
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-fix.ply --memory-profile safe --render-size 12 --num-steps 20 --seed 42 --print-metrics
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-fix.ply --compare /tmp/lito-teacup-quality-baseline.json --json /tmp/lito-teacup-quality-fix.json
bash -lc '! rg -n "import torch|from torch|cuda\\.|xformers|flash_attn|gsplat|vendors/ml-lito|from lito|import lito" src/mlx_spatial/lito_real_backend.py src/mlx_spatial/lito_inference.py'
```

**Execution:** subagent required
**Depends on:** Slice 5Q-1
**Checkpoint after:** none
**Touches:** one bounded code area selected from `src/mlx_spatial/lito_real_backend.py`, `src/mlx_spatial/lito_inference.py`, `src/mlx_spatial/lito_condition.py`, and related `tests/test_lito_*.py`
**Detail:** [slices/05q-quality-closure.md](slices/05q-quality-closure.md)

**Slice 5Q-2 pass 1 evidence (2026-05-23):** Implemented the selected decode/init-coverage fix by adding `--max-init-coords-per-batch {profile|none|N}`. `profile` preserves memory-profile caps, `none` disables top-k capping, and positive integers apply explicit caps. Verification: CLI help shows the flag; `uv run pytest tests/test_lito_real_backend.py tests/test_lito_inference.py tests/test_lito_cli.py tests/test_lito_quality.py -q` passed with `69 passed`; forbidden runtime import scan over `lito_real_backend.py`, `lito_inference.py`, and `lito.py` passed. Quality comparison command with `--max-init-coords-per-batch 1024` produced `outputs/lito/teacup-quality-fix.ply`; inspector reported checkpoint-backed header, `65536` vertices, `62` properties, no flags, and `failure_classification=stats_sane_visual_review_required`. Compared with baseline, vertex count increased by `32768` and bbox span grew by `[0.0265, 0.0349, 0.0146]`. Detailed evidence lives in `orchestration/slice-5q-2-summary.md`.

**Slice 5Q-2 pass 2 evidence (2026-05-23):** Human visual inspection rejected `outputs/lito/teacup-quality-fix.ply`, so the next targeted pass fixed conditioning/preprocessing parity for useful-alpha RGBA inputs. `_crop_and_pad_object` now follows upstream `keep_optical_axis=True` crop/pad math with alpha threshold `0.8`, `fill_ratio=0.8`, original image center, pad ratios `0.5`, and transparent zero padding; no background-removal dependency was added because the teacup/beer-mug inputs already have useful alpha. Verification: `uv run pytest tests/test_lito_inference.py tests/test_lito_quality.py -q` passed with `14 passed`; forbidden import scan for `torch`, CUDA, vendor LiTo, gsplat, and `rembg` passed. Subagent spec review and code-quality review both returned `APPROVED`. New higher-coverage visual candidate is `outputs/lito/teacup-quality-crop-4096.ply`, generated with `--max-init-coords-per-batch 4096`; inspector reported checkpoint-backed header, `262144` vertices, `62` properties, no flags, opacity probability median `0.138184`, scale exp median `0.002623`, quaternion norm median `1.000000`, and `failure_classification=stats_sane_visual_review_required`. Detailed evidence lives in `orchestration/slice-5q-2-pass2-summary.md`.

### Slice 5Q-3: Real-Input Regeneration + Visual Gate

**Objective:** Produce real-object outputs that can be inspected for surface quality before final verification.

**Acceptance criteria:**
- Generate at least two checkpoint-backed real-object outputs: `inputs/trellis2/teacup.png` plus one additional non-blob object image already present in the repo.
- Each output has checkpoint-backed LiTo PLY header comments, finite Gaussian fields, sane inspector stats, and no source-contract smoke header.
- A preview/render artifact is produced if an existing repo tool can render or inspect the PLY without adding CUDA dependencies; otherwise the PLYs and inspector summaries are the handoff artifacts.
- The user visually accepts the output quality before final `auto-verify`.

**Verification:**
```bash
uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-quality-final.ply --memory-profile safe --render-size 12 --num-steps 20 --seed 42 --print-metrics
uv run python scripts/lito/inspect_quality.py outputs/lito/teacup-quality-final.ply --json /tmp/lito-teacup-quality-final.json
uv run pytest tests/test_lito_quality.py -q
```

**Execution:** direct
**Depends on:** Slice 5Q-2
**Checkpoint after:** human-verify
**Checkpoint reason:** The remaining AC-07 question is visual surface credibility; command checks can prove schema, finiteness, and field ranges, but the user must inspect whether the generated object is acceptable.
**Touches:** `outputs/lito/`, `.agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/quality-final.md`

**Slice 5Q-3 evidence (2026-05-23):** Generated two checkpoint-backed real-object outputs without using the color-blob smoke input. Initial teacup candidate `outputs/lito/teacup-quality-fix.ply` was visually rejected by the user, so it is not acceptance evidence. Beer-mug candidate: `outputs/lito/beer-mug-quality-final.ply`, produced from `inputs/trellis2/beer-mug.png`; inspector reported checkpoint-backed header, `65536` vertices, `62` properties, no flags, bbox span `[1.4123, 1.6536, 1.8430]`, opacity probability median `0.005801`, scale exp median `0.001562`, quaternion norm median `1.000000`, and `failure_classification=stats_sane_visual_review_required`. After Slice 5Q-2 pass 2, teacup candidate `outputs/lito/teacup-quality-crop-4096.ply` improved visually but was still missing surface regions. A memory-conscious occupancy diagnostic found `17317` occupied init cells for the teacup latent, so the 4096 candidate covered only about `23.7%` of the occupied structure. Current accepted-looking teacup candidate is `outputs/lito/teacup-quality-crop-uncapped.ply`, generated with `--max-init-coords-per-batch none`; inspector reported checkpoint-backed header, `1108288` vertices, `62` properties, no flags, opacity probability median `0.056885`, scale exp median `0.004650`, quaternion norm median `1.000000`, and `failure_classification=stats_sane_visual_review_required`. Peak active memory remained about `15.28 GB`; file size is about `765 MB` PLY plus `249 MB` safetensors. The second real-object uncapped candidate is `outputs/lito/beer-mug-quality-uncapped.ply`; inspector reported checkpoint-backed header, `925952` vertices, `62` properties, no flags, opacity probability median `0.048492`, scale exp median `0.006928`, quaternion norm median `1.000000`, and `failure_classification=stats_sane_visual_review_required`. Peak active memory again stayed about `15.28 GB`; file size is about `619 MB` PLY plus `208 MB` safetensors. User visual inspection of the uncapped teacup in a proper Gaussian-splat viewer judged it fair-looking, with a known caveat around the teacup handle void/ring. Detailed handoff is in `orchestration/quality-final.md`, `orchestration/slice-5q-2-pass2-summary.md`, and `orchestration/slice-5q-3-uncapped-handoff.md`.

### Slice 5Q-4: Final Verification + Status Closeout

**Objective:** Close the quality replan after the visual gate passes and route the change back to final verification.

**Acceptance criteria:**
- `PLAN.md`, `STATUS.md`, and `docs/lito.md` reflect the real accepted quality evidence and do not claim smoke/blob evidence as qualitative proof.
- Full LiTo tests, full regression, and build pass.
- CUDA/runtime dependency guard remains clean.
- Current automaton stage is ready for `auto-verify`.

**Verification:**
```bash
uv run pytest tests/test_lito_*.py -q
uv run pytest -q
UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build
bash -lc '! rg -n "import torch|from torch|cuda\\.|xformers|flash_attn|gsplat|vendors/ml-lito|from lito|import lito" src/mlx_spatial/lito_real_backend.py src/mlx_spatial/lito_inference.py'
```

**Execution:** direct
**Depends on:** Slice 5Q-3 human verification
**Touches:** `.agent/steering/STATUS.md`, `.agent/work/2026-05-22-lito-mlx-inference-pipeline/PLAN.md`, `docs/lito.md`

**Slice 5Q-4 evidence (2026-05-23):** Final verification passed. `uv run pytest tests/test_lito_*.py -q` passed with `115 passed`; `uv run pytest -q` passed with `762 passed, 5 skipped, 2 warnings` from existing HY-World fixture export tests; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` initially failed inside the network-restricted sandbox while resolving `hatchling`, then passed with approved network access and built `dist/mlx_spatial-0.0.1.tar.gz` plus `dist/mlx_spatial-0.0.1-py3-none-any.whl`; the no-CUDA/runtime import guard over `lito_real_backend.py`, `lito_inference.py`, and `lito.py` passed.

## Plan-Level Defaults

Baked from product-review followups; flip these in plan refresh if needed.

| # | Followup | Default |
|---|---|---|
| 1 | 3DGS output file format | `.ply` (gsplat-standard); `--format {ply,splat,safetensors}` flag on `generate` for the other two. If implementation imports `plyfile` in runtime writer code, add `plyfile` to runtime `dependencies`; if it hand-writes PLY, keep `plyfile` dev-only for validation. |
| 2 | `gs_rasterize.py` blast radius (Risk F) | **Adapter-only**. Slice 4 wraps the rasterizer without modifying it. If Slice 0's render audit reports the adapter pattern infeasible, the plan trips the Risk F decision checkpoint described below. |
| 3 | PyTorch parity source | Optional only. Add `torch` (CPU/MPS-only, no CUDA extras) to `[dependency-groups.dev]` only for tests that can run without CUDA-only packages. Marker stays opt-in via existing `torch_parity` registration. No slice acceptance may require CUDA, xformers, flash-attention, or gsplat CUDA execution. |
| 4 | Sample input license fallback | Document Apple's download URL in `docs/lito.md`; do not redistribute unless `LICENSE_generated_samples` permits. Slice 6 records the license decision. |
| 5 | CLI verb | `generate` (image-to-3D is generative; HY-World's `reconstruct` describes multi-view geometric reconstruction, which is a different operation). |
| 6 | Memory baseline | M4 Max default; document floor and memory profile in `docs/lito.md`. M2/M3 tile/stream profile deferred to Phase 4+ (already in SPEC's deferred scope). |
| 7 | Memory profile naming | `LITO_MEMORY_PROFILES` uses choices `("safe", "balanced", "large")` matching SAM 3D's pattern (see `scripts/sam3d/reconstruct.py`); default `"balanced"`. Cross-profile bound: peak active memory must stay under 90 GB (soft) / 100 GB (hard) on the 128 GB dev system per SPEC's Execution discipline. |
| 8 | Memory monitor implementation | Use `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()` with `mx.eval(...)` / `mx.synchronize()` eval boundaries before measurements. Raise `LitoMemoryLimitExceeded` at 100 GB; log warning at 90 GB. |
| 9 | Scratch dir hygiene | Slices use `tempfile.mkdtemp(prefix="lito-<stage>-", dir="/tmp")` for any transient artifacts; clean up via `atexit` or context manager. `git status --porcelain` after slice verification must show no unexpected files. |
| 10 | Autonomous fallbacks | Slices retry transient failures with exponential backoff; use `WebFetch` / `WebSearch` for lookup-style unknowns; use `playwright-cli` only when browser interaction is required. Only the declared decision checkpoints (Risk F, routing re-route, license-blocked) escalate to the user. |

## Risk F Decision (per Product Review action)

**Default: adapter-only.** Slice 4 wraps `gs_rasterize.py` with LF-conditioning. `gs_rasterize.py` is not modified.

**Trip condition:** Slice 0's render audit (see slice 00 detail) reports that `gs_rasterize.py` makes HY-World-specific assumptions (e.g., fixed SH degree, fixed Gaussian param ordering, hard-coded camera convention) that prevent clean LF-conditioning composition. If tripped:

**Decision checkpoint:** auto-execute pauses. Options:

- **A — Stay adapter-only**: re-route to INTAKE Approach C (render-first risk slice) and rework the LF approach to fit the existing rasterizer, even if suboptimal numerically.
- **B — Allow shared-base refactor**: extract a thin shared base inside `gs_rasterize.py`; HY-World 2.0 regression sweep (`uv run pytest tests/test_hyworld2_*.py`) must remain green after the refactor and before Slice 4 lands.

Default if user does not respond: **A**. `auto-eng-review` reviews this decision before any code in `gs_rasterize.py` changes.

## Execution Discipline

Mirrors SPEC.md `## Constraints → Execution discipline / Performance instrumentation`. Slice-level mapping:

| Slice | Discipline obligation |
|---|---|
| 0A | Use `/tmp/lito-vendor-*/` for any scratch during clone/HF search. Capture upstream's recommended generation settings into `slices/00-vendor-assets-routing.md` § Recommended Settings; later embedded as `LITO_RECOMMENDED_*` constants in Slice 5. |
| 0B | Use `/tmp/lito-fixtures-*/` only for transient local contract-fixture generation. Do not install vendor-only packages into the project `.venv`; do not execute CUDA-only vendor paths. Use autonomous fallbacks (retry, web search, Playwright) for transient or lookup-style failures. |
| 1, 2, 3, 4 | Each parity probe records `(wall_time_s, peak_active_memory_gb)` for its module via `mx.metal.get_peak_memory()` after an `mx.eval(...)` barrier. DiT slice (2) additionally tests that `"balanced"` profile stays under 90 GB on a synthetic-input regression. |
| 5 | `LitoInferencePipeline.generate(...)` returns `LitoGenerationResult.metrics` (per-stage time + peak memory). `LitoMemoryLimitExceeded` raised at 100 GB; warning at 90 GB. `scripts/lito/generate.py` is the standalone sample script. |
| 6 | `docs/lito.md` documents memory profiles, the 90/100 GB thresholds, the `metrics` field, and how to invoke `scripts/lito/generate.py`. |

Sub-agents inherit these obligations from their slice file; the slice files restate the operational specifics so a cold-start agent has them in hand.

## Slice Sequence

Each slice has a detail file under `slices/`. PLAN.md is the index; slice files are the per-slice handoff packets for sub-agent execution (self-contained per the agentic-collab contract).

---

### Slice 0A: Vendor + Assets + Routing (LITO-A)

**Objective:** Shallow-clone vendor, read licenses, decide weight path, write `lito_assets.py`, audit upstream backend coverage, capture recommended settings, and record the routing decision.

**Acceptance criteria:**
- `vendors/ml-lito/` exists from `git clone --depth 1 --shallow-submodules https://github.com/apple/ml-lito vendors/ml-lito`
- License decisions recorded in `slices/00-vendor-assets-routing.md`: (a) local conversion permitted, (b) sample-input redistribution permitted, (c) any other licensed-use constraints
- `hf` CLI searched (`mlx-community/lito*`, `apple/lito*`, `apple/ml-lito`); safetensors-direct vs. `.ckpt`-convert decision recorded
- Weights downloaded into `weights/lito/` (direct) or `weights/lito-mlx/` (converted)
- `src/mlx_spatial/lito_assets.py` written with `LITO_REPO_ID`, `LITO_DEFAULT_ROOT`, `LitoAssetValidation` dataclass, `validate(root)`, `inspect(root)`, `download_command(root)`, optional `convert(src, dst)`
- Vendor MLX backend coverage audit complete; `slices/00-vendor-assets-routing.md` records which of {tokenizer, DiT, image-conditioner, LF-render} have MLX-backend coverage and which require source-guided MLX ports with optional CPU/MPS `torch_parity`
- **Upstream recommended generation settings captured** in a new `## Recommended Settings` section of `slices/00-vendor-assets-routing.md`: `num_steps`, `seed` policy, `cfg_scale` (or rectified-flow equivalent), input image resolution and pre-processing normalization stats, sampler-specific knobs. Each setting names its upstream source (file + line range or doc URL). Slice 5 embeds these as `LITO_RECOMMENDED_*` constants.
- Routing decision recorded: `Approach A continues` | `Approach C re-route: slice X` | `Blocked: <reason>`
- Risk F audit recorded: `adapter feasible` | `adapter infeasible — escalate Risk F decision`
- `git status --porcelain` after Slice 0A shows no unexpected scratch files in the working tree

**Verification:**
```bash
uv run python -c "from mlx_spatial.lito_assets import LITO_REPO_ID, validate; print(LITO_REPO_ID)"
uv run python -m mlx_spatial.lito_assets validate weights/lito 2>/dev/null || uv run python -m mlx_spatial.lito_assets validate weights/lito-mlx
grep -E "^## Routing Decision" .agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00-vendor-assets-routing.md
uv run pytest tests/test_lito_assets.py
```

**Execution:** direct (main agent — Slice 0 is the gating funnel and the routing decision needs main-agent judgment)
**Touches:** `vendors/ml-lito/` (new, gitignored), `weights/lito[-mlx]/` (new, gitignored), `src/mlx_spatial/lito_assets.py` (new), `tests/test_lito_assets.py` (new), `.agent/work/2026-05-22-lito-mlx-inference-pipeline/spec/gap-matrix.md` (update coverage/tolerances)
**Detail:** [slices/00-vendor-assets-routing.md](slices/00-vendor-assets-routing.md)

**Slice 0A evidence recorded during execution:** the installed `hf` CLI does not provide `search-models`; equivalent search was run through `huggingface_hub.HfApi.list_models(...)`. No relevant official or `mlx-community` LiTo checkpoint repo was found for `mlx-community lito`, `apple lito`, `apple ml-lito`, `lito_dit_rgba`, or `lito_new`, so the active weight path is Apple CDN `.ckpt` download into `weights/lito-raw/` followed by local conversion into `weights/lito-mlx/`. `uv run pytest tests/test_lito_assets.py` passes.

**Completion state:** complete except for any re-verification requested by `auto-verify`; source-contract fixture generation moved to Slice 0B.

---

### Slice 0B: Source Contracts + MLX-Compatible Fixtures + Tolerances (LITO-A)

**Objective:** Record module-boundary source contracts from the vendored upstream files and generate deterministic local fixtures that unblock P1 without CUDA or vendor-runtime execution.

**Acceptance criteria:**
- Historical dependency-probe evidence is preserved, but the active contract states: no CUDA, no xformers/flash-attention/gsplat CUDA execution, and no vendor runtime imports for acceptance.
- Fixture source contract is recorded in `slices/00b-reference-fixtures.md`: tokenizer = `LightTokenizationTrainer.get_latents` / `SPointEncoder` source contract, image conditioner = `LiToDiTTrainer.get_image_conditioning` / `SpatialDinov2` source contract, DiT = upstream MLX DiT/ODE source contract, render = `render_gaussians` / `render_3dgs_gsplat` source contract.
- `scripts/lito/write_contract_fixtures.py` generates deterministic local fixtures under `tests/fixtures/lito/` for each boundary: tokenizer input + latent-shape contract output; DiT input + step/final microtrajectory; image-conditioner input + feature-shape contract output; render input + expected image/mask contract output. Fixture file format is `safetensors` with stable tensor names except render output may be PNG.
- `tests/fixtures/lito/manifest.json` records `backend: "source_contract_local"` (or module-specific non-CUDA backend), upstream source file/function, seed, input provenance, dtype, shape, fixture role, and license/distribution status for every fixture.
- Concrete bring-up tolerances are recorded in `spec/gap-matrix.md`, explicitly labeling which checks are source-contract/microfixture checks and which checks are optional non-CUDA numerical parity checks.
- `git status --porcelain` after Slice 0B shows no unexpected scratch files in the working tree.

**Verification:**
```bash
uv run python scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite
test -f tests/fixtures/lito/manifest.json
find tests/fixtures/lito -maxdepth 1 -type f | grep -E "(tokenizer|dit|cond|render)_"
uv run python scripts/lito/validate_fixtures.py tests/fixtures/lito
grep -E "source-contract|MLX-compatible|no CUDA" .agent/work/2026-05-22-lito-mlx-inference-pipeline/spec/gap-matrix.md
find . -maxdepth 3 \( -name 'lito-fixtures-*' -o -name 'lito-fixture-env-*' \) -print -quit | grep . && echo "FAIL: scratch leak detected" || echo "OK: no scratch leaks"
```

**Execution:** direct (main agent — source-contract fixture generation is the serial gate before P1)
**Depends on:** Slice 0A
**Touches:** `tests/fixtures/lito/` (new deterministic local fixtures), `scripts/lito/write_contract_fixtures.py` (new), `scripts/lito/validate_fixtures.py` (existing), `.agent/work/2026-05-22-lito-mlx-inference-pipeline/spec/gap-matrix.md`, `.agent/work/2026-05-22-lito-mlx-inference-pipeline/slices/00b-reference-fixtures.md`
**Detail:** [slices/00b-reference-fixtures.md](slices/00b-reference-fixtures.md)

**Slice 0B historical evidence (2026-05-23):** `scripts/lito/validate_fixtures.py` was added and compiles. A vendor env probe proved that trying to execute upstream PyTorch/gsplat paths locally drifts into xformers/flash-attention/CUDA-shaped dependencies. User clarification after that probe: CUDA is not allowed; those paths are source references only. The active path is therefore local source-contract fixture generation, not remote CUDA fixture capture.

**Correction after user clarification:** do not use the CUDA fixture gate. The downstream slices must port CUDA-only operations to MLX-compatible operations and verify against source-contract fixtures plus optional non-CUDA Torch/MPS or upstream-MLX probes where available.

**Slice 0B completion evidence (2026-05-23):** `scripts/lito/write_contract_fixtures.py` was added and compiled. `uv run python scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite` generated `tests/fixtures/lito/manifest.json` plus 19 local fixture files across tokenizer, condition, DiT, and render groups. `uv run python scripts/lito/validate_fixtures.py tests/fixtures/lito --verbose` passed, and a repository-local scratch scan for `lito-fixtures-*` / `lito-fixture-env-*` returned no leaked scratch directories.

---

### Slice 1: Tokenizer (LITO-C)

**Objective:** Port the LiTo tokenizer (point cloud → 8192 × 32 latent) to MLX float16, using the vendor PyTorch tokenizer/encoder as source reference and verifying against source-contract fixtures plus local MLX probes.

**Acceptance criteria:**
- `src/mlx_spatial/lito_tokenizer.py` implements the tokenizer using MLX `float16` by default
- Source-contract and operation-level checks vs. `tests/fixtures/lito/tokenizer_*.safetensors` within tolerance recorded in `spec/gap-matrix.md`
- ≥ 3 fixed-input fixtures pass

**Verification:**
```bash
uv run pytest tests/test_lito_tokenizer.py -v
```

**Execution:** subagent required (per user directive; this slice is one of the 4-way parallel group)
**Depends on:** Slice 0B
**Touches:** `src/mlx_spatial/lito_tokenizer.py`, `tests/test_lito_tokenizer.py`
**Detail:** [slices/01-tokenizer.md](slices/01-tokenizer.md)

---

### Slice 2: DiT Generator (LITO-D)

**Objective:** Port the LiTo flow-matching DiT (image-conditioned, rectified flow) to MLX float16, matching upstream MLX denoising trajectory at fixed seed and step count. Define `LITO_MEMORY_PROFILES` with 90/100 GB threshold safety.

**Acceptance criteria:**
- `src/mlx_spatial/lito_dit.py` implements the DiT using MLX `float16` by default; `float32` only at numerically-sensitive accumulation points, annotated inline
- Trajectory parity vs. upstream MLX fixtures from `tests/fixtures/lito/dit_*.safetensors`: intermediate latents at sampled steps (recommended: step 0, mid-step, final-step) plus the final latent within tolerance
- `LITO_MEMORY_PROFILES` defined parallel to `HYWORLD2_MEMORY_PROFILES`, with choices `("safe", "balanced", "large")` (default `"balanced"`); each profile's `sample(...)` peak active memory stays under 90 GB on the 128 GB dev system at the upstream-recommended step count. Tested via a synthetic-input memory regression
- Each parity test records `(wall_time_s, peak_active_memory_gb)` for the module via `mx.metal.get_peak_memory()` after `mx.eval(...)` / `mx.synchronize()` barriers

**Verification:**
```bash
uv run pytest tests/test_lito_dit.py -v
```

**Execution:** subagent required
**Depends on:** Slice 0B
**Touches:** `src/mlx_spatial/lito_dit.py`, `tests/test_lito_dit.py`
**Detail:** [slices/02-dit.md](slices/02-dit.md)

---

### Slice 3: Image Conditioner Adapter (LITO-B)

**Objective:** Wire the image conditioner (LiTo `SpatialDinov2` source contract unless later evidence changes it) into the LiTo pipeline. Reuse existing mlx-spatial ViT/DINO code where compatible; otherwise port the source encoder to MLX.

**Acceptance criteria:**
- `src/mlx_spatial/lito_condition.py` (or chosen name) implements the conditioner using MLX `float16`
- Source-contract and operation-level checks vs. `tests/fixtures/lito/cond_*.safetensors` within tolerance
- If reusing an existing encoder, the reuse is explicit (import-and-call, no duplication); if upstream uses a novel encoder, the adapter loads upstream weights via `lito_assets.convert` or direct safetensors load

**Verification:**
```bash
uv run pytest tests/test_lito_condition.py -v
```

**Execution:** subagent required
**Depends on:** Slice 0B
**Touches:** `src/mlx_spatial/lito_condition.py`, `tests/test_lito_condition.py`
**Detail:** [slices/03-image-conditioner.md](slices/03-image-conditioner.md)

---

### Slice 4: LF-Conditioned 3DGS Render (LITO-E)

**Objective:** Port the LF-conditioned 3DGS render to MLX, **adapter-only** wrapping `gs_rasterize.py` (Risk F default). Use vendor gsplat/PyTorch as source reference and verify against local contract fixtures plus non-CUDA render probes.

**Acceptance criteria:**
- `src/mlx_spatial/lito_render.py` wraps the existing `gs_rasterize.py` + Metal kernel + `hyworld2_sh.py` + `hyworld2_camera.py` without modifying them
- LF conditioning is applied as a layer that consumes vendor-reference-equivalent inputs and produces rasterizer-compatible Gaussian parameters
- Per-tensor contract checks and image-similarity probes vs. `tests/fixtures/lito/render_*.safetensors` / `render_*.png` within tolerance
- If Slice 0's render audit tripped the Risk F decision and user picked option B (shared-base refactor), `gs_rasterize.py` changes are guarded by a green HY-World 2.0 regression sweep (`uv run pytest tests/test_hyworld2_*.py`) before and after the refactor

**Verification:**
```bash
uv run pytest tests/test_lito_render.py -v
uv run pytest tests/test_hyworld2_render.py tests/test_hyworld2_export.py  # HY-World regression sweep
```

**Execution:** subagent required
**Depends on:** Slice 0B
**Touches:** `src/mlx_spatial/lito_render.py`, `tests/test_lito_render.py`, **only if Risk F option B**: `src/mlx_spatial/gs_rasterize.py` (extraction of shared base, no behavior change)
**Detail:** [slices/04-render.md](slices/04-render.md)

**P1 completion evidence (2026-05-23):** Slices 1-4 were implemented in parallel via subagents and reviewed in the required order. Final spec review and final quality review returned `APPROVED`. Verification: `uv run pytest tests/test_lito_tokenizer.py tests/test_lito_dit.py tests/test_lito_condition.py tests/test_lito_render.py -q` -> `37 passed`; compileall on the eight P1 files passed; `uv run pytest tests/test_gs_rasterize.py tests/test_hyworld2_export.py -q` -> `14 passed`; `uv run pytest tests/test_hyworld2_*.py -q` -> `155 passed, 2 warnings` from pre-existing HY-World export RuntimeWarnings; `gs_rasterize.py` diff checks exited 0.

**Verification command correction (2026-05-23):** `tests/test_hyworld2_render.py` does not exist in the live repo. Use `uv run pytest tests/test_gs_rasterize.py tests/test_hyworld2_export.py` for the HY-World rasterizer/export regression sweep.

---

### Slice 5: Pipeline + CLI + Sample Script + End-to-End Smoke (LITO-F integration)

**Objective:** Wire Slices 1–4 into a runnable pipeline. Add the `mlx-spatial-lito` CLI entry, the `scripts/lito/generate.py` sample, per-stage performance instrumentation, and the 90/100 GB memory monitor. Produce a 3DGS `.ply` file from a known input image end-to-end at upstream-recommended defaults.

**Acceptance criteria:**
- `src/mlx_spatial/lito_inference.py` orchestrates: image → conditioner → tokenizer → DiT → render → 3DGS export, using the real MLX module outputs (no fixtures). Returns `LitoGenerationResult` dataclass with `metrics: dict[str, dict[str, float]]` capturing per-stage `wall_time_s`, `peak_active_memory_gb`, `peak_cache_memory_gb`. Stages: `preprocess`, `condition`, `tokenize`, `dit`, `decode`, `render`, `export`
- `LITO_RECOMMENDED_NUM_STEPS`, `LITO_RECOMMENDED_SEED_POLICY`, `LITO_RECOMMENDED_RESOLUTION`, and any other settings captured in Slice 0's `## Recommended Settings` section are exposed as module-level constants in `lito_inference.py` and used as CLI / library / `scripts/lito/generate.py` defaults. Each has a one-line comment naming its upstream source
- `LitoMemoryLimitExceeded` exception type defined; orchestration raises it when `mx.metal.get_active_memory()` crosses the 100 GB hard ceiling; logs a warning at the 90 GB soft threshold
- `src/mlx_spatial/lito.py` exposes CLI: `validate`, `inspect`, `download-command`, `generate`. `generate` accepts an image path and writes `.ply` (default) or `--format {ply,splat,safetensors}`; supports `--memory-profile {safe,balanced,large}` (default `balanced`); supports `--num-steps` / `--seed` / etc., all defaulting to the `LITO_RECOMMENDED_*` constants
- `scripts/lito/generate.py` mirrors `scripts/sam3d/reconstruct.py` shape: sys.path bootstrap → argparse with recommended defaults → delegates to `mlx_spatial.lito.main`. Documented in `scripts/README.md`
- `pyproject.toml::[project.scripts]` gains exactly: `mlx-spatial-lito = "mlx_spatial.lito:main"`. No other entry-script changes
- `pyproject.toml::[dependency-groups.dev]` gains `torch` only if optional CPU/MPS `torch_parity` tests need it, and `plyfile` for `.ply` validation. No CUDA extras, xformers, flash-attention, or CUDA-backed gsplat are added. If runtime LiTo code imports `plyfile` for the default writer, `pyproject.toml::dependencies` also gains `plyfile`; otherwise the writer hand-writes PLY and no runtime dependency is added. `uv.lock` updates naturally. `uv build` continues to produce a clean wheel/sdist (verified by running `uv build` after the edit)
- End-to-end smoke: `mlx-spatial-lito generate inputs/lito/<sample>.png --output outputs/lito/<sample>.ply` produces a valid `.ply` file readable by `plyfile`. Smoke is run via `scripts/lito/generate.py` too to validate the wrapper
- `from mlx_spatial.lito import LitoInferencePipeline` works **with `vendors/` absent** (verify by temporarily moving `vendors/ml-lito` and re-running the import)
- `git status --porcelain` after the slice shows only the expected committed paths; no `/tmp/` leaks into the working tree

**Verification:**
```bash
uv run pytest tests/test_lito_inference.py tests/test_lito_cli.py
uv run mlx-spatial-lito validate weights/lito 2>/dev/null || uv run mlx-spatial-lito validate weights/lito-mlx
uv run mlx-spatial-lito generate inputs/lito/<sample>.png --output outputs/lito/<sample>.ply
python -c "import plyfile; p = plyfile.PlyData.read('outputs/lito/<sample>.ply'); print(p.elements[0].count, 'gaussians')"
uv run pytest  # full regression sweep — no other pipeline regresses
```

> **VERIFY-GAP:** Slice 5 smoke output is not an actual checkpoint-backed LiTo result.
> **Evidence:** `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root /tmp/does-not-exist-lito-real-weights --output /tmp/lito-no-weights-proof.ply --memory-profile safe --render-size 12` exits 0 with `gaussians=64`; generated PLY headers include `comment mlx-spatial LiTo source-contract smoke 3DGS export`. Live modules state that tokenizer, conditioner, DiT, and inference are source-contract implementations and do not load converted LiTo tensors.
> **Fix objective:** Implement or explicitly gate a checkpoint-backed LiTo inference path. `generate` must not be counted as AC-07 completion until it fails without required real weights and uses converted checkpoint tensors for conditioner/tokenizer/DiT/decode/render, or the command is renamed/flagged as source-contract smoke and the real-output acceptance remains open.
> **Guard correction evidence (2026-05-23):** `generate` now fails closed by default. Missing weights command `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root /tmp/does-not-exist-lito-real-weights --output /tmp/lito-no-weights-proof-guard-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Present converted weights command `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-unimplemented-proof-guard-20260523.ply --memory-profile safe --render-size 12` validates header sentinels (`image_to_3d/lito_dit_rgba.safetensors=2793`, `tokenizer/lito_new.safetensors=1108`), exits 1 with `LitoRealGenerationNotImplemented`, and leaves no output. Synthetic generation now requires `--source-contract-smoke`; `uv run pytest tests/test_lito_*.py -q` passes (`63 passed`). Real checkpoint-backed AC-07 remains open.
> **Backend-boundary evidence (2026-05-23):** `src/mlx_spatial/lito_real_backend.py` now owns the checkpoint-backed boundary as a direct safetensors-to-MLX path, with no Torch/CUDA/vendor runtime dependency. It records header-only architecture inventory for real converted weights, normalizes LiTo `gs_dict` tensors, and writes a separate checkpoint-backed gsplat-style PLY schema only after a valid real Gaussian dict returns. `lito_inference._generate_checkpoint_backed()` now preprocesses the image, creates that backend, calls `generate_gaussians()`, and exports only after success; backend failure leaves no PLY or sidecar. Verification: `uv run pytest tests/test_lito_inference.py tests/test_lito_cli.py tests/test_lito_real_backend.py -q` -> `26 passed`; `uv run pytest tests/test_lito_*.py -q` -> `70 passed`; `uv run pytest -q` -> `717 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built. `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-not-ready-proof-20260523.ply --memory-profile safe --render-size 12` exits 1 with `LitoBackendUnavailable` after validating real headers and leaves no output. Header inventory: `image_to_3d/lito_dit_rgba.safetensors=2793`, `tokenizer/lito_new.safetensors=1108`, DiT `28 blocks / 8192x32 latent / hidden 1152 / condition dim 2048`, Gaussian decoder `6 blocks / expansion 64 / SH degree 3`, voxel decoder `4 blocks / 16x16x16 init grid`. Attempting to add a Torch probe dependency group was reverted because `uv lock` pulled `cuda-*`, `nvidia-*`, and `triton` packages; dependency parsing shows project deps/dev deps contain no Torch and `uv.lock` has no `torch`, NVIDIA CUDA, or `triton` package names. Real AC-07 remains open.
> **Loader/decode increment evidence (2026-05-23):** `lito_real_backend.py` now selectively loads and remaps real DiT safetensors (`velocity_estimator_ema.module.*` -> local names including `t_proj_linear*`, `t0_proj_linear`, `final_layer.adaLN_linear*`) and real Gaussian decoder safetensors (`gs_decoder.*` -> local names, including sequential MLP remaps and `w12` split into `w1`/`w2`). It also ports the LiTo Gaussian `decode_gs` equations for raw `shape_out`/`color_out` plus caller-supplied `init_coord`. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `13 passed`; `uv run pytest tests/test_lito_*.py -q` -> `76 passed`; `uv run pytest -q` -> `723 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built; real subset load probe prints DiT shapes `(1152, 32)`, `(1152, 64)`, `(2304,)` and Gaussian decoder shapes `(512,195)`, `(512,512)`, `(2048,512)`, `(3136,512)`. `generate` remains fail-closed for full image-to-3D and leaves no output. Real AC-07 remains open until local DINO conditioning, DiT sampling, init-coordinate generation, and Gaussian decoder forward are wired.
> **Gaussian output-head increment evidence (2026-05-23):** `lito_real_backend.py` now runs the real LiTo Gaussian shape/color output MLP heads for caller-supplied decoder query latents using the remapped checkpoint weights, then decodes with explicit init coordinates. This is not the full Gaussian Perceiver/point-query decoder yet; it is the real weighted tail subpath that the Perceiver output will feed. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `15 passed`; `uv run pytest tests/test_lito_*.py -q` -> `78 passed`; `uv run pytest -q` -> `725 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-output-heads-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open until the Gaussian Perceiver/point-query forward, local DINO conditioning, DiT sampling, and init-coordinate generation are wired.
> **Gaussian point-query increment evidence (2026-05-23):** `lito_real_backend.py` now runs the real LiTo coordinate/Fourier point-query stem (`xyz`, `xyz_encoded`, `point_linear`, `point_mlp`) before the output-head/decode tail for caller-supplied init coordinates. This still stops before the Gaussian Perceiver attention over LiTo latents; it is not full checkpoint-backed generation. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `17 passed`; `uv run pytest tests/test_lito_*.py -q` -> `80 passed`; `uv run pytest -q` -> `727 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-query-stem-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open until Gaussian Perceiver attention, local DINO conditioning, DiT sampling, and init-coordinate generation are wired.
> **Gaussian block-0 cross-attention increment evidence (2026-05-23):** `lito_real_backend.py` now runs the real LiTo Gaussian Perceiver block-0 cross-attention plus `ca_mlp` subpath (`kv_linear`, `ca_layer`, residual, `ca_ln`, `ca_mlp`) for caller-supplied query latents and LiTo latent tokens. The helper is explicitly named `cross_only` and raises if asked to include localized-voxel self-attention, so it cannot be counted as a full Gaussian decoder. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `21 passed`; `uv run pytest tests/test_lito_*.py -q` -> `84 passed`; `uv run pytest -q` -> `731 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built after network-approved build-backend fetch; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-cross-only-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open until localized-voxel self-attention/voxel metadata, local DINO conditioning, DiT sampling, and init-coordinate generation are wired.
> **Gaussian block-0 localized-voxel increment evidence (2026-05-23):** `lito_real_backend.py` now builds LiTo localized-voxel self-attention metadata locally from query/init coordinates, matching upstream `PackedPoint.get_bijk_info` integer grouping (`forward_idxs`, `backward_idxs`, `cu_seq_lens`, `max_seq_lens`, `chunk_start_idxs`) with half-cell shift on odd self-attention layers, then runs the real block-0 cross-attention, both localized self-attention layers, both self-MLPs, and the existing output-head/decode tail for caller-supplied init coordinates and latent tokens. This is still a bounded intermediate subpath: the remaining Gaussian Perceiver blocks, local DINO conditioning, DiT sampling, and init-coordinate generation are not wired. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `24 passed`; `uv run pytest tests/test_lito_*.py -q` -> `87 passed`; `uv run pytest -q` -> `734 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built after network-approved build-backend fetch; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-local-voxel-block0-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open.
> **Gaussian all-block localized-voxel increment evidence (2026-05-23):** `lito_real_backend.py` now runs all discovered real Gaussian Perceiver blocks with localized-voxel self-attention for caller-supplied LiTo latent tokens and init coordinates, then feeds the weighted output heads and LiTo `decode_gs` equations. This is the real weighted Gaussian decoder subpath, but still not image-to-3D generation: DINO conditioning, DiT sampling, and voxel/Trellis init-coordinate generation remain unwired. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `26 passed`; `uv run pytest tests/test_lito_*.py -q` -> `89 passed`; `uv run pytest -q` -> `736 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-all-blocks-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open.
> **Voxel/TRELLIS init-coordinate increment evidence (2026-05-23):** `lito_real_backend.py` now loads real `voxel_decoder.*` tensors, runs LiTo's active `SSLatentDecoder -> VectorDecoder` low-res `ss_latent` path for caller-supplied latent tokens, reuses the local no-CUDA TRELLIS sparse-structure decoder to map `ss_latent` to `(B,1,64,64,64)` occupancy logits, and converts occupied cells to packed LiTo Gaussian init coordinates with upstream's axis and cell-center convention. This connects caller-supplied LiTo latents to the real weighted Gaussian decoder subpath, but still not image-to-3D generation: DINO conditioning and DiT sampling remain unwired. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `32 passed`; `uv run pytest tests/test_lito_*.py -q` -> `95 passed`; `uv run pytest -q` -> `742 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-init-coords-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open.
> **DiT caller-supplied-condition increment evidence (2026-05-23):** `lito_real_backend.py` now runs the real LiTo DiT checkpoint path from caller-supplied condition tokens: timestep Fourier embedding, condition token MLP, latent z/pos projection, selected DiT blocks, final AdaLN projection, and `euler`/`heun` ODE sampling. This connects caller-supplied `(B,M,2048)` condition tokens to the real voxel/TRELLIS init-coordinate and Gaussian decoder stages, but still not image-to-3D generation because local DINO/RGBA image conditioning is not wired. Verification: `uv run pytest tests/test_lito_real_backend.py -q` -> `36 passed`; `uv run pytest tests/test_lito_*.py -q` -> `99 passed`; `uv run pytest -q` -> `746 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built; `uv run mlx-spatial-lito generate inputs/lito/smoke.png --weights-root weights/lito-mlx --output /tmp/lito-real-still-failclosed-after-dit-message-20260523.ply --memory-profile safe --render-size 12` exits 1 and leaves no output. Real AC-07 remains open.
> **DINO/RGBA + end-to-end checkpoint evidence (2026-05-23):** `lito_real_backend.py` now loads/remaps `patch_encoder.*` tensors and runs LiTo's real DINOv2 ViT-L/14-reg branch plus the RGBA learnable conv branch in MLX. A full 518px conditioner probe produced finite `(1, 1374, 2048)` condition tokens; a full 8192-token / 28-block DiT velocity probe was finite; a 20-step Heun sampler plus voxel/TRELLIS decode produced occupied cells before safe-profile capping. Default `generate` now wires image -> condition tokens -> DiT sampler -> voxel/TRELLIS init coords -> Gaussian Perceiver/decode -> checkpoint-backed PLY export. Verification on a real object input: `uv run mlx-spatial-lito generate inputs/trellis2/teacup.png --weights-root weights/lito-mlx --output outputs/lito/teacup-real-safe-20260523.ply --memory-profile safe --render-size 12 --num-steps 20 --seed 42 --print-metrics` exits 0, reports `gaussians=512`, and writes a PLY with `comment mlx-spatial LiTo checkpoint-backed 3DGS export`, `element vertex 32768`, 32768 data rows, and 62 columns. `uv run python -c "from plyfile import PlyData; p=PlyData.read('outputs/lito/teacup-real-safe-20260523.ply'); print(p['vertex'].count)"` reports `32768`. `inputs/lito/smoke.png` is only a color-blob framework probe and is not qualitative LiTo evidence. `uv run pytest tests/test_lito_*.py -q` -> `104 passed`; `uv run pytest -q` -> `751 passed, 5 skipped, 2 warnings`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` -> wheel/sdist built. This proves checkpoint-backed execution and export, but user inspection reports broken surfaces, so AC-07 quality remains open until Slice 5Q passes.

**Execution:** subagent recommended (touches > 3 files and crosses subsystem boundaries; main agent may want to validate the integration if Slice 0's routing decision was non-trivial)
**Depends on:** Slices 1, 2, 3, 4
**Touches:** `src/mlx_spatial/lito.py`, `src/mlx_spatial/lito_inference.py` (orchestration + `LITO_RECOMMENDED_*` constants + `LitoMemoryLimitExceeded` + `LitoGenerationResult.metrics`), `scripts/lito/generate.py` (new), `scripts/README.md` (add entry), `pyproject.toml::[project.scripts]` (one line), `pyproject.toml::[dependency-groups.dev]` (`torch`, `plyfile`), `pyproject.toml::dependencies` only if runtime code imports `plyfile`, `tests/test_lito_inference.py`, `tests/test_lito_cli.py`, `tests/test_lito_memory_limits.py` (new — synthetic-input regression for 90/100 GB thresholds), `inputs/lito/` (sample inputs per Slice 0 license decision)
**Detail:** [slices/05-pipeline-cli.md](slices/05-pipeline-cli.md)

---

### Slice 6: Docs + ROADMAP + Architecture Map (LITO-F docs)

**Objective:** Document the pipeline for users and agents; capture the LiTo phase in roadmap; update architecture map.

**Acceptance criteria:**
- `docs/lito.md` exists with: weight acquisition (HF safetensors-direct or `.ckpt`-convert per Slice 0 license decision), sample-input acquisition (download URL per Slice 0 license decision), CLI usage, memory profile + M4 Max requirement, parity probe layout, license summary (what's permitted)
- `docs/architecture.md` updated with LiTo's place in the pipeline shape
- `.agent/steering/ROADMAP.md` Phase 3 entry naming `2026-05-22-lito-mlx-inference-pipeline` with status, objective, exit signal
- Phase 4+ candidates added to ROADMAP: LiTo training, LiTo weight redistribution, LiTo mesh extraction, multi-image / video conditioning

**Verification:**
```bash
test -f docs/lito.md
test -f docs/architecture.md
grep -E "^## Phase 3" .agent/steering/ROADMAP.md
grep -E "lito-mlx-inference-pipeline" .agent/steering/ROADMAP.md
```

**Execution:** subagent recommended (doc-shaped slice; can fan out to a doc-writing sub-agent)
**Depends on:** Slice 0B (for license decision + final tolerances + fixture regeneration docs)
**Touches:** `docs/lito.md` (new), `docs/architecture.md` (update), `.agent/steering/ROADMAP.md` (Phase 3 + Phase 4+)
**Detail:** [slices/06-docs-roadmap.md](slices/06-docs-roadmap.md)

## Execution Routing and Topology

**Default route:** continuation after each slice's verification passes.

**Parallel-safe groups (per user's multi-agent directive):**

- **Group P1** (after Slice 0B): Slices 1, 2, 3, 4 — each consumes source-contract/local fixtures generated in Slice 0B at its input boundary, so write sets are disjoint and there are no live data dependencies between them. Each runs as a separate `subagent required` execution.
- **Group P2** (after Group P1): Slices 5 and 6 — Slice 5 is integration; Slice 6 is documentation. Different file scopes (`src/` vs. `docs/` + `.agent/`).
- **Group P-QUALITY-A** (after Slice 5Q-0): Slice 5Q-1 audit subagents for conditioning, sampler, and decode run in parallel. They are read-only against `src/`, `tests/`, and `vendors/`; each writes only its own `orchestration/quality-audit-*.md`.

**Checkpoints:**

- **After Slice 0A/0B — conditional `decision`:** Only if Slice 0A or 0B reports `Approach C re-route` or `Risk F option B needed` or `Blocked`. A missing CUDA path is not a valid checkpoint. If `Approach A continues` + `adapter feasible` + local contract fixtures validate, **continuation** — auto-execute proceeds directly into Group P1.
- **Checkpoint reason (if tripped):**
  - **Approach C re-route:** Slice 0 found the LF render is the right place to start. Options: (1) keep Slice 0's fixtures and re-order — Slice 4 runs first as a serial slice, then Slices 1/2/3 parallel after; (2) abort and re-frame.
  - **Risk F option B needed:** Slice 0 found `gs_rasterize.py` cannot be wrapped cleanly. Options: (A) stay adapter-only with degraded LF semantics; (B) allow shared-base refactor with HY-World regression sweep gating it.
  - **Blocked (license):** `LICENSE_MODEL` forbids even local conversion. Options: (1) load `.ckpt` at runtime via `pt-safe-loader` only (accept perf hit); (2) abort and surface to user.
- **After Slice 5Q-3 — `human-verify`:** The user inspects the regenerated real-object PLY(s) or preview(s). If accepted, continue to Slice 5Q-4 and then `auto-verify`; if rejected, return to Slice 5Q-2 with the new failure evidence.
- Verification failure in any slice halts execution at that slice; the failure surfaces concrete next actions (per `auto-verify` and slice file).

**Subagent routing:**

- Slices 1, 2, 3, 4: `subagent required` (per user directive). Each slice file is the self-contained handoff packet; sub-agent receives `INTAKE.md` + `SPEC.md` + `PLAN.md` + `slices/<N>-*.md` + the relevant source-contract fixture files under `tests/fixtures/lito/` plus the manifest entries for its module.
- Slice 5: `subagent recommended` (integration touches > 3 files and pyproject; main agent may want to validate).
- Slice 6: `subagent recommended` (doc-shaped, parallel-safe with Slice 5).
- Slice 5Q-1: `subagent required` for the three read-only audit tracks; run them in parallel.
- Slice 5Q-2: `subagent required` for the selected targeted fix, but only one implementer writes `src/mlx_spatial/lito_real_backend.py` at a time.
- Slice 5Q-0, 5Q-3, 5Q-4: `direct` coordinator work because they create shared baselines, own user-facing artifacts, or update state.
- Slice 0A/0B: `direct` (main agent — gating funnel and source-contract fixture gate need orchestrator judgment).

**Subagent model/reasoning guidance:** keep each subagent payload token-efficient: pass only `INTAKE.md`, `SPEC.md`, `PLAN.md`, the target `slices/<N>-*.md`, `spec/gap-matrix.md`, and its fixture paths. Use a high-reasoning coding agent for Slices 2 and 4 (DiT and LF render), medium/high coding agents for Slices 1, 3, and 5, and a lower-cost medium-reasoning agent for Slice 6 docs. If the host cannot spawn subagents, execute the same groups serially without changing write ownership or verification gates.

## Requirement Traceability

| Gap ID (SPEC) | Slice | Acceptance Criteria (SPEC) |
|---|---|---|
| LITO-A | Slice 0A + Slice 0B | AC-01, AC-02, AC-14 (recommended settings capture), source-contract fixture/tolerance finalization for AC-03 through AC-06 |
| LITO-B | Slice 3 | AC-03 |
| LITO-C | Slice 1 | AC-04 |
| LITO-D | Slice 2 | AC-05, AC-12 (memory monitor — DiT is the largest single consumer) |
| LITO-E | Slice 4 | AC-06 |
| LITO-F (integration) | Slice 5 | AC-07, AC-08, AC-09, AC-10, AC-12 (orchestration-level monitor), AC-13 (sample script), AC-14 (constants exposure) |
| LITO-F (quality) | Slice 5Q | AC-05, AC-06, AC-07, AC-08, AC-10, AC-12, AC-15 |
| LITO-F (docs) | Slice 6 | AC-07 (docs portion), AC-11 |
| Cross-cut (regression) | Slice 5Q-4 verification | AC-08 (no pipeline regresses + `uv build` clean after quality closure) |
| Cross-cut (surface) | Slice 5Q verification | AC-09 (pyproject + gitignore), real-output quality evidence for AC-07 |
| Cross-cut (dtype) | Slices 1–5 verification | AC-10 (float16 default + annotated float32) |
| Cross-cut (hygiene) | All slices' verification | AC-15 (`git status --porcelain` shows no scratch leaks) |

## Context Budget for This Change

This change spans multiple sessions and explicitly invokes parallel sub-agent execution. Per-slice budget guidance:

- **Slice 0A**: complete evidence already recorded — vendor clone, license read, HF search, asset code, coverage audit, recommended settings, routing
- **Slice 0B**: ~20% of one session — source-contract audit, local fixture generation, manifest, tolerance finalization
- **Slices 1–4 (each)**: ~15–20% of one sub-agent session — single-module port with source-contract and MLX probes; fixture is the input boundary
- **Slice 5**: ~20% of one session — integration; reuses code from Slices 1–4
- **Slice 6**: ~10% of one session — doc-shaped
- **Slice 5Q**: one quality-debug loop — direct baseline, three parallel read-only audits, then one targeted fix at a time with a human visual gate

Fresh sub-agents for Slices 1–4 start with `INTAKE.md` + `SPEC.md` + `PLAN.md` + their slice file + source-contract fixture file paths + the relevant `tests/fixtures/lito/manifest.json` entry; they do not need to read other slice files.

## Revision Notes — 2026-05-23

**Refresh A (release strategy):** Triggered by `auto-eng-review` verdict `needs_correction` on the prior plan. User clarified: **0.0.1 has not been tagged; LiTo lands in 0.0.2 via normal commits.** SPEC.md's `## Constraints → Cohabitation`, `AC-08`, `AC-09`, and `## Anti-Goals` updated; release-boundary constraint replaced by `uv build` cleanliness gate.

| Engineering concern | Fix |
|---|---|
| Plan modifies `pyproject.toml` dev deps | Boundary relaxed; Slice 5 explicitly touches `[dependency-groups.dev]` for `torch` (CPU/MPS) and `plyfile`. `AC-08` now requires `uv build` clean — the real safety check. |
| Plan modifies `docs/` artifacts | Boundary relaxed; Slice 6 writes `docs/lito.md` + updates `docs/architecture.md` freely. |
| Implicit "separate commit" requirement for script entry | Removed. Normal commits. |

**Refresh B (execution discipline — staff-level spatial-scientist disciplines):** User added autonomous-execution and memory-safety constraints. SPEC.md `## Constraints` gained `Execution discipline` and `Performance instrumentation` sub-sections; new `AC-12` (memory monitor 90/100 GB), `AC-13` (`scripts/lito/generate.py`), `AC-14` (recommended settings as constants), `AC-15` (no scratch in working tree). PLAN.md gained `## Execution Discipline` mapping, Plan-Level Defaults rows 7–10 (memory profile naming, monitor implementation, scratch hygiene, autonomous fallbacks). Slices 0, 2, 4, 5, 6 updated.

| New discipline | Where it lands |
|---|---|
| `/tmp/lito-*` scratch dirs; no working-tree leaks | All slices' verification includes `git status --porcelain` check (AC-15) |
| 128 GB total / 90 GB soft / 100 GB hard memory thresholds | Slice 2 tests `LITO_MEMORY_PROFILES` profiles under 90 GB; Slice 5 wires `LitoMemoryLimitExceeded` |
| Autonomous fallbacks (retry, `WebFetch`, `WebSearch`, `playwright-cli`) | All slices; only declared decision checkpoints escalate |
| Upstream-recommended generation settings preserved | Slice 0 captures into `## Recommended Settings`; Slice 5 embeds as `LITO_RECOMMENDED_*` constants used everywhere |
| Per-stage `(wall_time, peak_memory)` instrumentation | Slices 1–4 record per-module; Slice 5 returns aggregate `LitoGenerationResult.metrics` |
| `scripts/lito/generate.py` standalone sample (mirroring `scripts/sam3d/reconstruct.py`) | Slice 5 |
| MLX-native instrumentation (`mx.metal.get_active_memory`, `mx.metal.get_peak_memory`, `mx.eval` barriers) | Slice 5 orchestration; all slice tests |

**Refresh C (eng-review #2 corrections):** Second `auto-eng-review` pass returned `needs_correction` with two blockers. Both fixed surgically; no scope or topology change.

| Engineering concern | Fix |
|---|---|
| Cross-slice exception hazard: `slices/02-dit.md` referenced `LitoMemoryLimitExceeded`, but that class is defined in `lito_inference.py` (Slice 5). Under the parallel-safe topology, Slice 2 must not depend on Slice 5 artifacts. | `slices/02-dit.md` now uses plain `assert` statements only and explicitly notes ownership: `LitoMemoryLimitExceeded` is a Slice 5 concept. Hard-ceiling raise behavior is tested in Slice 5's `tests/test_lito_memory_limits.py`. SPEC.md `## Multi-Agent Execution Design` now states this ownership rule in the Slice 2 entry. |
| SPEC/PLAN topology drift: SPEC.md `## Multi-Agent Execution Design` still described the original 6-slice sequential map (LITO-E in Slice 3, no LITO-B slice, no separate docs slice) while PLAN.md has the refreshed 7-slice parallel-safe topology. | SPEC.md `## Multi-Agent Execution Design` rewritten to match PLAN.md verbatim — 7 slices, P1 parallel group (1–4), P2 parallel group (5–6), gap → slice mapping aligned (LITO-B in Slice 3, LITO-E in Slice 4, LITO-F split across Slices 5 and 6). |

**Refresh D (runtime dependency and subagent guidance):** Tightened the `.ply` writer dependency rule and added explicit subagent model/reasoning guidance. `plyfile` is runtime only if imported by runtime writer code; otherwise it stays dev-only for validation. Parallel subagents receive minimal artifact payloads with stronger reasoning reserved for DiT/render/integration.

**Refresh E (hybrid reference fixtures):** Triggered by Slice 0A execution evidence and the follow-up engineering review verdict `needs_correction`. Upstream LiTo coverage is hybrid rather than all-MLX: tokenizer and image conditioner are PyTorch, DiT has MLX sampling but PyTorch conditioning, and render is PyTorch + gsplat. SPEC.md, PLAN.md, and slice handoffs now use a vendor-reference fixture contract. Slice 0A is retained as completed asset/routing evidence; new Slice 0B captures fixtures in an isolated Python 3.11/Pixi-style environment before P1 starts.

| Engineering concern | Fix |
|---|---|
| PLAN/SPEC said P1 consumed vendor-MLX fixtures, but Slice 0A proved tokenizer, conditioner, and render references are PyTorch/gsplat. | Replaced the fixture contract with per-module backend provenance and a required `tests/fixtures/lito/manifest.json`. |
| Fixture capture blocker was documented as a note but not executable topology. | Added Slice 0B with explicit environment, fixture, manifest, validation, and tolerance-finalization acceptance criteria. |

No Risk F protocol change. No deferred scope shift. No new gap IDs, requirement IDs, or anti-goals.

The next `auto-eng-review` pass replaces the existing `## Review: Engineering` section per the lifecycle "append-replace, not stack" rule.

**Refresh F (no-CUDA source-reference correction):** User clarified that CUDA is not allowed locally or as an acceptance gate. Torch with MPS/CPU is acceptable for optional parity probes when it runs without CUDA-only packages. CUDA/PyTorch/gsplat implementation is now static source reference for the MLX inference pipeline only.

| Engineering concern | Fix |
|---|---|
| Prior Slice 0B routed around local failure by proposing a CUDA fixture gate. | Removed the gate. Slice 0B now writes deterministic local source-contract fixtures and manifest metadata without vendor runtime imports. |
| P1 wording implied downstream agents could depend on vendor PyTorch/gsplat outputs. | Slices 1-4 now consume local source-contract fixtures and port CUDA-only operations to MLX-compatible implementations. |
| PyTorch dev dependency needed clarification. | Torch remains optional CPU/MPS-only for `torch_parity`; no CUDA extras, xformers, flash-attention, or CUDA-backed gsplat are allowed. |

**Refresh G (quality closure replan):** User inspection rejected the current checkpoint-backed PLY quality. The file is schema-valid, but broken surfaces mean AC-07 is not accepted. PLAN.md now adds Slice 5Q with a baseline inspector, three parallel read-only source-contract audits, one targeted MLX-native fix per pass, real-input regeneration, a human visual gate, and final verification before `auto-verify`.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: Slice 5Q has a concrete no-CUDA quality-debug path: baseline inspector, parallel read-only contract audits, one targeted MLX-native fix per pass, real-input regeneration, and a human visual gate before final verification.
- Concern: Slice 5Q-2 may need more than one mismatch pass because broken surfaces can come from coupled conditioning, sampler, and decode errors, so the plan correctly limits each implementation pass but may loop before AC-07 is accepted.
- Action: Run `auto-execute` starting at Slice 5Q-0, dispatch Slice 5Q-1 as the three read-only audit subagents in parallel, and keep Slice 5Q-2 implementation serial for shared backend files.
- Verified: context diagnostics, STATUS/PLAN stage alignment, no DESIGN artifact required, Slice 5Q acceptance and verification commands, no-CUDA dependency guard, human-verify checkpoint validity, parallel-safe audit write sets, and risk matrix ratings architecture fit 8, data flow clarity 8, edge case coverage 7, test strategy 7, rollback safety 8, dependency risk 8
