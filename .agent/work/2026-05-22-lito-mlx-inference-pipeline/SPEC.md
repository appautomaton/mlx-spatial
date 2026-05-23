# LiTo MLX Inference Pipeline Spec

## Bounded Goal

Port Apple LiTo (Surface Light Field Tokenization, ICLR 2026) image-to-3DGS inference into `mlx-spatial` as a new pipeline matching the established `<name>_*.py` + `mlx-spatial-<name>` shape used by SAM 3D, TRELLIS.2, and HY-World 2.0. The vendored `apple/ml-lito` implementation is the source reference for architecture, tensor contracts, and optional non-CUDA probes; CUDA-only PyTorch/gsplat paths are read-only references and are not execution gates.

## Broader Intent

`mlx-spatial` becomes a complete MLX-native 3D inference library spanning the 2026 SOTA open-weight 3D generative landscape (SAM 3D, TRELLIS.2, HY-World 2.0, LiTo) with consistent pipeline shape, consistent parity discipline, and self-contained per-slice handoffs so any sub-agent (fresh-session Claude, Codex helper, or sub-agent) can land a slice cold from `INTAKE.md` + `SPEC.md` + `PLAN.md` + the slice file.

## Work Scale and Shape

- Scale: capability
- Shape: parity (with a thin slice of new-capability work where upstream's MLX backend does not already cover the path)

## Selected Lenses

- **product**: Adds a new pipeline to the public surface and a new CLI entry (`mlx-spatial-lito`); preserves the "complete MLX-native 3D inference" positioning.
- **engineering**: New modules, asset acquisition path, parity probes, GS-render composition with existing `gs_rasterize.py`, memory-profile system, dtype discipline.

Runtime concerns (Apple Silicon target, M4 Max memory, MLX `float16` default, Metal kernel reuse, no CUDA) are captured in Constraints rather than promoted to a separate lens, matching the Phase 2 precedent in `2026-05-20-production-pipeline-parity/SPEC.md`.

## Target User

Developers running 3D generative inference on Apple Silicon (M-series) workstations who want LiTo image-to-3DGS output without standing up a CUDA box; secondarily, multi-agent contributors landing per-slice parity work in parallel.

## Linked Detail Files

- [spec/gap-matrix.md](spec/gap-matrix.md) — per-module gap inventory: upstream sources, mlx-spatial reuse, parity probes, expected tolerances, slice-0 open items

## Gap Inventory

LiTo decomposes into 6 component groups. Gap IDs `LITO-A` through `LITO-F`. Full per-tensor specifications and tolerances are scaffolded in [spec/gap-matrix.md](spec/gap-matrix.md) and finalized during Slice 0.

| # | Gap | Upstream Reference | mlx-spatial Reuse | New Module(s) | Verification |
|---|---|---|---|---|---|
| LITO-A | Vendor + asset acquisition + license review | `apple/ml-lito` shallow clone; `LICENSE_MODEL`, `LICENSE_generated_samples`; HF search for MLX-ready safetensors | `hyworld2_assets.py` shape; `sam3d_assets.py` converter pattern; `pt-safe-loader` | `lito_assets.py` + optional `.ckpt`→safetensors converter | Asset validation report; shape parity vs. upstream tensors |
| LITO-B | Image conditioner | LiTo image-conditioned DiT input pipeline (exact module from Slice 0 audit) | `trellis2_dinov3*`, `hyworld2_vit.py` | `lito_condition.py` adapter (thin) | Source-contract fixture and MLX tensor-shape/numerics probes; optional non-CUDA PyTorch parity only when available |
| LITO-C | Tokenizer (point cloud → 8192 × 32 latent) | `vendors/ml-lito/src/lito/` tokenizer module | None (architecturally novel) | `lito_tokenizer.py` | Source-contract fixture and local MLX probes on tokenized latents from fixed inputs |
| LITO-D | Flow-matching DiT (image-conditioned, rectified flow) | `vendors/ml-lito/src/lito/` DiT module + `lito_dit_rgba.ckpt` | `trellis2_forward.py`, `trellis2_inference.py` flow patterns; `hyworld2_transformer.py` block patterns | `lito_dit.py` | Denoising-trajectory parity at fixed seed + step count |
| LITO-E | LF-conditioned 3DGS render | `vendors/ml-lito/plibs/` (gsplat + LF) | `gs_rasterize.py`, Metal kernel, `hyworld2_sh.py`, `hyworld2_camera.py` | `lito_render.py` (LF conditioning wrapper) | Source-contract fixture and local MLX render probes; optional vendor image comparison only from non-CUDA references |
| LITO-F | Pipeline + CLI + docs | `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` end-to-end path | `hyworld2.py`, `sam3d.py`, `trellis2.py` CLI patterns; `HYWORLD2_MEMORY_PROFILES` pattern | `lito.py`, `lito_inference.py`, `docs/lito.md` | End-to-end smoke: image → 3DGS file from known input |

## Required Outcome

For each gap, the MLX implementation follows the vendored `apple/ml-lito` source contracts recorded in [spec/gap-matrix.md](spec/gap-matrix.md). Slice 0 records the reference backend per module: DiT and Gaussian decode use upstream MLX source paths where available, while tokenizer, image conditioner, init-coordinate generation, and render use PyTorch/gsplat source as architecture reference only unless a non-CUDA execution path is available. CUDA-only paths must not be executed or required for acceptance. The end-to-end pipeline also produces a sensible 3DGS file from at least one known input image. The pipeline is invocable via `mlx-spatial-lito` and importable as `from mlx_spatial.lito import LitoInferencePipeline`.

Pipeline shape mirrors HY-World 2.0 / SAM 3D / TRELLIS.2 conventions:

- `lito_assets.py` — local weight layout, HF download command, validation, optional `.ckpt`→safetensors conversion
- `lito_<module>.py` — per-module implementations (tokenizer, DiT, render, condition)
- `lito_inference.py` — orchestration with a memory profile system parallel to `HYWORLD2_MEMORY_PROFILES`; choices `("safe", "balanced", "large")` matching the SAM 3D pattern, default `"balanced"`
- `lito.py` — CLI entry exposing `validate`, `inspect`, `download-command`, `generate`
- `scripts/lito/generate.py` — standalone end-to-end sample script mirroring `scripts/sam3d/reconstruct.py` shape, exposing recommended defaults
- `tests/test_lito_*.py` — parity probes per module + end-to-end smoke; each test records `(wall_time, peak_active_memory)` for the operation

## Constraints

**Hardware and runtime**

- Apple Silicon only. Default test target: M4 Max. Baseline target: any M-series with sufficient unified memory for DiT inference.
- Default tensor dtype is MLX `float16`. `float32` only where numerical stability requires it (e.g., RoPE accumulation, softmax denominators) — match the precedence in `hyworld2_*` modules. Each `float32` use is justified inline (one short comment per occurrence).
- No CUDA at runtime, no CUDA in `dependencies` or `dependency-groups.dev`, no CUDA in tests. PyTorch is permitted in the local dev environment as a CPU/MPS parity reference only; PyTorch is never a runtime dependency of `mlx-spatial`.

**Vendor and licensing**

- `apple/ml-lito` is vendored as a **shallow clone** (`git clone --depth 1 …`) under `vendors/ml-lito/`. `vendors/` stays gitignored. Submodules are pulled with `--shallow-submodules` if any.
- `LICENSE_MODEL` and `LICENSE_generated_samples` are read before any local conversion. Local conversion happens only if `LICENSE_MODEL` permits it; otherwise the pipeline loads `.ckpt` at runtime via `pt-safe-loader`. The license decision is recorded in `docs/lito.md`.
- No weight redistribution this change. No HF upload of converted artifacts. `mlx-community/lito-*` publish is deferred to a future change.
- `mlx-spatial` package must function with `vendors/` absent. `vendors/` is a dev-time parity reference and is never imported by anything under `src/mlx_spatial/`.

**Weight acquisition path**

- Use the `hf` CLI (already authenticated) as the primary download channel. Slice 0 searches `mlx-community/lito*`, `apple/lito*`, and `apple/ml-lito` for MLX-ready safetensors. If MLX-ready safetensors are hosted, prefer them and skip conversion.
- If only PyTorch `.ckpt` exists on HF or Apple CDN (`https://ml-site.cdn-apple.com/models/lito/`), download `.ckpt` and convert via `pt-safe-loader` → `safetensors` in `lito_assets.py`, mirroring the conversion pattern in `model_assets.py` / `sam3d_assets.py`.
- Default local layout: `weights/lito/` for direct-from-HF MLX safetensors; `weights/lito-mlx/` for converted artifacts (matching the `*-mlx` suffix convention used by `sam-3d-objects-mlx`, `moge-vitl-mlx`).

**Inputs and outputs**

- Sample input images live under `inputs/lito/` (gitignored, same as `inputs/sam3d/`, `inputs/hyworld2/`, `inputs/trellis2/`). `docs/lito.md` names known-good sample inputs with download instructions; the repo does not redistribute Apple-provided sample images unless `LICENSE_generated_samples` permits it.
- Generated 3DGS outputs go under `outputs/lito/` by default (gitignored).

**Cohabitation**

- **Release strategy (clarified 2026-05-23):** 0.0.1 has not been tagged yet — it is execute-complete locally and awaiting maintainer trigger. LiTo lands in **0.0.2** as part of normal commits. The 0.0.1 release group is **not** frozen for this change: `pyproject.toml` (dev deps + scripts), `docs/`, and `README.md` may be modified as needed for the LiTo pipeline. The only hard constraint is that `uv build` continues to produce a clean wheel and sdist after this change, so the 0.0.1 tag (or 0.0.2 tag) remains producible.
- The remaining prior-release-group files (`.github/workflows/workflow.yaml`, `LICENSE`, `scripts/`, `uv.lock`) are not modified by LiTo unless functionally necessary. `uv.lock` will update naturally when dependencies or dev deps change.
- No existing pipeline (SAM 3D, TRELLIS.2, HY-World 2.0) regresses. `uv run pytest` passes before and after each slice.
- All new source modules follow the `lito_*` prefix. New CLI entry adds exactly one line to `pyproject.toml::[project.scripts]`.

**Agentic-collab**

- Each slice is self-contained. The slice file in `slices/*.md` carries (a) acceptance criteria, (b) verification commands, (c) parity-probe scripts, (d) the upstream files to read first. A fresh-session agent must be able to execute the slice with `INTAKE.md` + `SPEC.md` + `PLAN.md` + that slice file as its only context.

**Parity discipline**

- Per-module verification uses the strongest non-CUDA oracle available: upstream MLX numerical parity where executable locally, local MLX-compatible microfixtures for source-contract coverage where upstream is CUDA-only, and opt-in CPU/MPS PyTorch parity only if the path runs without CUDA-only packages.
- Tolerances are gap-specific and recorded in [spec/gap-matrix.md](spec/gap-matrix.md). DiT uses float16-tight tolerances against upstream MLX outputs when available. Tokenizer, image conditioner, and render use source-contract fixtures plus MLX operation-level numerical probes; any CUDA-only implementation detail is ported to MLX-compatible operations rather than pulled in as a dependency.
- PyTorch parity is opt-in via the existing `torch_parity` pytest marker for broad regression sweeps when it runs on CPU/MPS with no CUDA extras. PyTorch is dev-only and never a runtime dependency of `mlx-spatial`.

**Execution discipline (added 2026-05-23)**

- **Temp-directory hygiene.** Scratch artifacts, debug dumps, and intermediate caches use `/tmp/lito-<stage>-<timestamp>/` (system tmp), never the repo working tree. Persistent fixtures and outputs go to their committed paths (`tests/fixtures/lito/`, `outputs/lito/`, `weights/lito[-mlx]/`). A leaked file under the repo working tree is a slice verification failure.
- **Autonomous fallbacks.** Transient failures (network, HF rate limits, `hf` CLI flakes, retryable I/O) retry with exponential backoff and continue. Lookup-style unknowns (upstream API surface, latest HF repo info, vendor doc updates) use `WebFetch` or `WebSearch`; `playwright-cli` is available when browser interaction is required. Only **genuine decisions** (Risk F mode, license-blocked path, routing re-route) escalate via the decision checkpoints already declared in PLAN.md. No other slice may stop and ask the user.
- **Memory safety.** The dev system has 128 GB unified memory. Soft threshold **90 GB** active; hard ceiling **100 GB** active. `lito_inference.py` instruments every stage via `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()`. Crossing 90 GB logs a warning; crossing 100 GB raises and aborts the generation (slice failure surfaces concrete next action). `LITO_MEMORY_PROFILES` `"safe"` profile stays well under 90 GB on M4 Max; `"balanced"` (default) targets headroom; `"large"` may approach 90 GB but never exceeds it.
- **Recommended upstream settings preserved.** Slice 0 records upstream's recommended generation settings (`num_steps`, `seed` strategy, `cfg_scale` or rectified-flow equivalents, input image resolution, normalization stats, any non-obvious sampler parameters) verbatim. These land as `LITO_RECOMMENDED_*` constants in `lito_inference.py` and become the CLI / library / `scripts/lito/generate.py` defaults. **Defaults are not invented.** If upstream's recommendations are split across files (e.g., README vs. demo vs. config YAML), the slice records each source and picks the most-recent / most-specific. A default that does not trace back to a recorded upstream source is a slice verification failure.

**Performance instrumentation**

- `LitoInferencePipeline.generate(...)` returns a `LitoGenerationResult` dataclass that includes a `metrics: dict[str, dict[str, float]]` field: per-stage `{wall_time_s, peak_active_memory_gb, peak_cache_memory_gb}`. Stages: `preprocess`, `condition`, `tokenize`, `dit`, `decode`, `render`, `export`.
- Each parity probe in `tests/test_lito_*.py` records the same metrics for the module under test and emits them via `caplog` or a fixture. Regression sweeps can compare against a baseline if desired (not required for slice acceptance).
- Staff-level discipline: every stage that allocates large tensors must call `mx.eval(...)` or `mx.synchronize()` before measuring memory; mid-stage measurements without an eval boundary are not load-bearing.

## Risks

- **Risk A — Upstream MLX backend coverage is partial.** INTAKE flags this as load-bearing-but-unverified. **Mitigation:** Slice 0 audits `vendors/ml-lito/demos/lito/fastapi_lito_demo.py` and `vendors/ml-lito/src/lito/` for backend coverage of tokenizer + DiT + render. Components without MLX reference are ported from the source implementation into MLX-compatible operations, with source-contract fixtures and optional non-CUDA parity probes.
- **Risk B — LF-conditioned GS render does not compose with existing `gs_rasterize.py`.** The Metal kernel and `hyworld2_sh.py` were not built for LF conditioning. **Mitigation:** Slice 0 scan; if composition is infeasible, re-route to INTAKE Approach C (render-first risk slice). The re-routing decision is captured in PLAN.md, not SPEC.md.
- **Risk C — `LICENSE_MODEL` forbids local conversion.** **Mitigation:** Slice 0 reads the license before any conversion. Fallback: load `.ckpt` at runtime via `pt-safe-loader` and accept the performance hit; if even runtime load is barred, surface the license decision to the user and pause the change.
- **Risk D — DiT exceeds M4 Max resident memory.** **Mitigation:** Memory profile system mirroring `HYWORLD2_MEMORY_PROFILES`; tile, stream, or quantize layers as needed. `float16` default already halves memory vs. `float32`.
- **Risk E — HF does not host MLX-ready safetensors.** This forces the converter path. **Mitigation:** Already planned via `pt-safe-loader` + `safetensors`. No spec-level impact; it shifts work into LITO-A.
- **Risk F — Hidden coupling between `gs_rasterize.py` and HY-World assumptions.** The rasterizer was developed for HY-World GS output. **Mitigation:** Slice E adds a thin LF-conditioning adapter, not a fork; if coupling is too tight, factor out a thin shared base inside `gs_rasterize.py` and have both pipelines depend on it.

## Acceptance Criteria

| AC | Gap(s) | Check |
|---|---|---|
| AC-01 | LITO-A | `lito_assets.validate(root)` passes against the downloaded weight layout. `lito_assets.inspect(root)` lists expected tensors and shapes. `lito_assets.download_command()` prints a working `hf download …` command. If conversion is needed, `lito_assets.convert(...)` produces MLX safetensors that pass shape and tensor-name parity against the source `.ckpt`. |
| AC-02 | LITO-A (license) | `LICENSE_MODEL` and `LICENSE_generated_samples` are read and `docs/lito.md` records (a) whether local conversion is permitted, (b) whether sample-input redistribution is permitted. The recorded answer drives the code path actually used. |
| AC-03 | LITO-B | Image-conditioner adapter produces `float16` features with shapes, dtype, normalization, and token ordering matching the source contract recorded in `spec/gap-matrix.md`; optional non-CUDA PyTorch parity may tighten tolerance when available. |
| AC-04 | LITO-C | `lito_tokenizer` produces 8192 × 32 latents for ≥ 3 fixed local inputs, with source-derived shape/dtype/range invariants and MLX operation-level probes passing the tolerances recorded in `spec/gap-matrix.md`. |
| AC-05 | LITO-D | `lito_dit` denoising trajectory matches upstream MLX trajectory at identical seed and step count when executable locally; otherwise source-contract trajectory microfixtures compare intermediate latents at sampled steps, not only at the end. |
| AC-06 | LITO-E | `lito_render` produces a rendered GS image that satisfies source-contract image-similarity and shape thresholds for ≥ 1 fixed input. Render reuses `gs_rasterize.py` and Metal kernel; LF conditioning is the only new GS-side surface. |
| AC-07 | LITO-F | `mlx-spatial-lito generate <image>` end-to-end produces a 3DGS file from `inputs/lito/<sample>.png`. `from mlx_spatial.lito import LitoInferencePipeline` works. `docs/lito.md` documents usage, weight acquisition, and the known-good sample input(s). |
| AC-08 | Cross-cut (regression) | `uv run pytest` passes (all existing pipelines + new LiTo tests). `vendors/` remains gitignored. `import mlx_spatial.lito` works with `vendors/` absent. `uv build` still produces a clean wheel and sdist. |
| AC-09 | Cross-cut (surface) | `pyproject.toml::[project.scripts]` gains `mlx-spatial-lito = "mlx_spatial.lito:main"`; no other entry-script changes. Runtime `dependencies` may gain `plyfile` only if the default `.ply` writer imports it at runtime; otherwise the writer hand-writes PLY and `plyfile` stays dev-only for validation. `[dependency-groups.dev]` may gain `torch` (CPU/MPS-only, no CUDA extras) and `plyfile` for opt-in parity and PLY validation. `.gitignore` already covers `vendors/`, `weights/`, `inputs/`, `outputs/` — no `.gitignore` change required (verified). |
| AC-10 | Cross-cut (dtype) | Default tensor dtype throughout LiTo modules is MLX `float16`. `float32` use is justified inline (one short comment per occurrence). A grep over `src/mlx_spatial/lito_*.py` for `float32` returns only annotated lines. |
| AC-11 | ROADMAP | `.agent/steering/ROADMAP.md` gains a Phase 3 entry naming `2026-05-22-lito-mlx-inference-pipeline` with status, objective, and exit signal. Deferred LiTo scope (training, redistribution, mesh extraction, multi-image / video conditioning) is captured as Phase 4+ candidates. |
| AC-12 | Memory safety | `LitoInferencePipeline.generate(...)` instruments per-stage `peak_active_memory_gb` via `mx.metal.get_active_memory()`. A soft warning is emitted at 90 GB; the pipeline raises `LitoMemoryLimitExceeded` and aborts cleanly if peak active memory crosses 100 GB. Test: a synthetic-large-input test triggers the warning path and confirms the abort path is reachable. |
| AC-13 | Sample script | `scripts/lito/generate.py` runs the pipeline end-to-end with upstream's recommended defaults; mirrors `scripts/sam3d/reconstruct.py` shape (sys.path setup → recommended argparse defaults → delegates to `mlx_spatial.lito.main`). Documented in `scripts/README.md`. |
| AC-14 | Recommended settings | Upstream's recommended generation settings (per Slice 0 audit) are preserved as module-level `LITO_RECOMMENDED_*` constants in `lito_inference.py` and used as CLI/library defaults. Each constant has a one-line comment naming the upstream source (file + line range or doc URL). |
| AC-15 | Execution hygiene | No file is created under the repo working tree by any slice's verification, smoke, or parity probe outside committed paths (`src/`, `tests/`, `docs/`, `scripts/`, `.agent/`, `inputs/lito/` if license permits). Scratch goes to `/tmp/`. A `git status --porcelain` after a clean slice verification shows no unexpected files. |

## Anti-Goals

- LiTo training, fine-tuning, or data ingestion
- Weight redistribution (no `mlx-community/lito-*` publish in this change)
- Mesh extraction from 3DGS+LF outputs (Flexicubes, marching cubes) — deferred
- Multi-image and video conditioning — deferred
- CUDA-backed inference paths, even as opt-in
- Performance optimization beyond the `float16` default and the existing Metal GS path
- Breaking the existing `uv build` output for 0.0.1 (the prior release group's runtime artifacts remain producible)
- Permanently vendoring any of `apple/ml-lito` into this repo's tracked tree
- Coupling LiTo release timing to the 0.0.1 PyPI publish
- Visual-parity-only or smoke-only verification (parity probes are non-negotiable)

## Scope Coverage Decisions

- **Included** (all): Vendor shallow-clone under `vendors/ml-lito/`; `hf`-CLI weight acquisition (safetensors-preferred with `.ckpt` fallback); `float16` default; sample inputs under `inputs/lito/`; full LITO-A through LITO-F gaps; per-module non-CUDA probes against the strongest available source-contract or upstream-MLX reference recorded in Slice 0; end-to-end smoke; `docs/lito.md`; ROADMAP Phase 3 entry; new `mlx-spatial-lito` CLI script.
- **Deferred to ROADMAP.md Phase 4+**: Training/fine-tuning on MLX; weight redistribution via `mlx-community`; mesh extraction (Flexicubes / marching cubes) from GS+LF; multi-image and video conditioning; cross-pipeline ablations (LiTo vs. TRELLIS.2 vs. Hunyuan3D-2.5).
- **Anti-goals**: As listed above.
- **Assumption resolved in Slice 0**: Upstream MLX backend is partial. DiT and Gaussian decode have MLX paths; tokenizer encoder, DINO image conditioner, init-coordinate generation, and render use PyTorch/gsplat references. CUDA-only execution is not allowed; affected slices use those source paths as architecture references and consume MLX-compatible contract fixtures generated locally.
- **Assumption accepted (replaces a needs-decision)**: HF discovery happens first; the choice between safetensors-direct and `.ckpt`-convert lives in LITO-A and is resolved at Slice 0, not at framing.

## Multi-Agent Execution Design

Slice ordering follows INTAKE Approach A — tokenizer-first depth — refined into a fixture-decoupled 4-way parallel-safe group (see PLAN.md "Approach"). SPEC fixes the gap → slice mapping and the per-slice exit conditions; PLAN owns per-slice acceptance criteria, verification commands, and topology.

- **Slice 0A — LITO-A assets/routing (funnel, serial, direct execution):** Shallow-clone vendor, read `LICENSE_MODEL` + `LICENSE_generated_samples`, search HF, download weights, decide safetensors-direct vs. convert, write `lito_assets.py` + validators + converter, audit upstream MLX backend coverage, capture upstream's recommended generation settings into `slices/00-vendor-assets-routing.md` § Recommended Settings, and record the routing decision plus Risk F audit.
- **Slice 0B — LITO-A source contracts + MLX-compatible fixtures (serial, direct execution):** Record module-boundary source contracts from the vendored upstream files and generate deterministic local fixtures under `tests/fixtures/lito/` with no CUDA, vendor imports, or MLX requirement. CUDA/PyTorch/gsplat implementations are reference text for the MLX port, not runnable acceptance gates.
- **Slice 1 — LITO-C (parallel group P1):** Tokenizer port + per-tensor parity probes. Tokenizer is the architecturally novel piece.
- **Slice 2 — LITO-D (parallel group P1):** DiT generator port + denoising-trajectory parity. Defines `LITO_MEMORY_PROFILES = ("safe", "balanced", "large")` (default `"balanced"`) and verifies via `assert` that the `"balanced"` profile stays under the 90 GB soft threshold at upstream-recommended `num_steps`. The `LitoMemoryLimitExceeded` exception class is owned by Slice 5 (orchestration layer); Slice 2's tests use plain assertions only.
- **Slice 3 — LITO-B (parallel group P1):** Image-conditioner adapter — reuse `trellis2_dinov3*` / `hyworld2_vit.py` if upstream uses an encoder mlx-spatial already has; otherwise thin port.
- **Slice 4 — LITO-E (parallel group P1):** LF-conditioned GS render. Slice 0's Risk F audit has already confirmed adapter feasibility (or routed to Approach C). Adapter-only by default; shared-base refactor only if Slice 0 reports infeasibility and the user selects Risk F option B.
- **Slice 5 — LITO-F integration (parallel group P2):** `lito_inference.py` orchestration + `LitoMemoryLimitExceeded` definition + `LITO_RECOMMENDED_*` constants + `LitoGenerationResult.metrics` + `lito.py` CLI + `scripts/lito/generate.py` standalone wrapper + `pyproject.toml` script entry, `[dependency-groups.dev]` updates, runtime `plyfile` dependency only if imported by runtime writer code, and end-to-end smoke.
- **Slice 6 — LITO-F docs (parallel group P2):** `docs/lito.md`, architecture map update, ROADMAP Phase 3 entry.

**Topology:** Slice 0 is serial (funnel; main agent, direct execution). Slices 1–4 form parallel-safe group **P1** — each consumes its boundary fixture from Slice 0, write sets are disjoint, no live cross-slice data dependencies. Slices 5 and 6 form parallel-safe group **P2** after P1 completes. Per the user's multi-agent directive, Slices 1–4 are `subagent required`; Slices 5 and 6 are `subagent recommended`.

Each slice's `slices/<n>-<name>.md` file is the self-contained handoff packet for sub-agent execution; a fresh-session agent executes a slice with `INTAKE.md` + `SPEC.md` + `PLAN.md` + that slice file + the relevant fixture file(s) as its only context.

## Blocking Questions or Assumptions

None blocking. Three load-bearing assumptions are explicitly tested in Slice 0:

1. Source-contract fixtures can be generated locally without running CUDA-only PyTorch/gsplat paths.
2. `LICENSE_MODEL` permits local conversion to MLX safetensors (no redistribution required).
3. LF-conditioned GS render composes with `gs_rasterize.py` + Metal kernel + `hyworld2_sh.py`.

If any fails, Slice 0 reports the failure and the change re-frames (not re-scopes) the affected slice via PLAN.md without returning to `auto-frame`.

## Review: Product

- Verdict: approved_with_risks
- Strength: The differentiation is pipeline-shape coherence across SAM3D / TRELLIS.2 / HY-World / LiTo with parity discipline per slice, which is what makes the library defensible and lets multi-agent contributors land work cold — not the LiTo port in isolation, since Apple already ships an MLX backend.
- Concern: Three load-bearing assumptions (upstream MLX backend covers the full pipeline, `LICENSE_MODEL` permits local conversion, LF-conditioned render composes with the existing `gs_rasterize.py` + Metal kernel) are all deferred to Slice 0; any failure forces re-framing of the affected slice rather than the whole change, but Risk F could spill into `gs_rasterize.py` and widen blast radius into HY-World 2.0.
- Action: Run `auto-plan` to expand the 6-slice ordering into per-slice acceptance criteria, verification commands, and parity-probe scripts; capture the Risk F blast-radius decision (adapter-only vs. allowed shared-base refactor) explicitly in PLAN.md so `auto-eng-review` can gate it.
- De-scoped: none new beyond the SPEC's existing Phase 4+ deferrals (training, weight redistribution, mesh extraction, multi-image and video conditioning, cross-pipeline ablations).
