# Slice 6 — Docs + ROADMAP + Architecture Map (LITO-F docs)

Parallel-safe with Slice 5. Depends on Slice 0 (for license decision and final tolerances) and on Slices 1–4 (for parity probe layout in `tests/`).

## Inputs

- `INTAKE.md`, `SPEC.md`, `PLAN.md`, this slice file
- `slices/00-vendor-assets-routing.md` § License Decision and § Routing Decision — drive the docs/lito.md license and weight-acquisition sections
- `spec/gap-matrix.md` — final per-module tolerances for the parity-probe documentation

## Reference Patterns in mlx-spatial

- `docs/sam3d.md` — pipeline-doc shape and tone
- `docs/development.md` — local dev environment expectations
- `docs/architecture.md` — architecture map; this slice adds LiTo to it
- `docs/model-publishing.md` — license/redistribution stance reference
- `.agent/steering/ROADMAP.md` — Phase 1 and Phase 2 entries for shape

## Implementation Outline

### `docs/lito.md` (new)

Sections:

1. **Overview** — what LiTo does, output shape (3DGS), Apple Silicon target
2. **License summary** (drives by Slice 0):
   - Are MLX-converted weights redistributable? (no for this change)
   - Are sample inputs redistributable? (per `LICENSE_generated_samples`)
   - What does the user need to do to obtain weights and inputs?
3. **Weight acquisition**:
   - HF safetensors-direct path (if Slice 0 chose this): `hf download <repo-id> --local-dir weights/lito`
   - Conversion path (if Slice 0 chose this): download `.ckpt` from <source>, run `python -m mlx_spatial.lito_assets convert weights/lito-raw weights/lito-mlx`
4. **Sample input acquisition**:
   - Download URL: `https://ml-site.cdn-apple.com/models/lito/<sample>.png` (or whatever Slice 0 confirmed)
   - Place under `inputs/lito/`
5. **CLI usage**:
   - `mlx-spatial-lito validate <root>`
   - `mlx-spatial-lito inspect <root>`
   - `mlx-spatial-lito download-command`
   - `mlx-spatial-lito generate <image> --output <out>.ply [--format {ply,splat,safetensors}] [--memory-profile {safe,balanced,large}] [--num-steps N] [--cfg-scale F] [--seed N] [--print-metrics]`
6. **Standalone sample script**: `python scripts/lito/generate.py <image> --output <out>.ply` — uses upstream-recommended defaults, mirrors `scripts/sam3d/reconstruct.py` shape
7. **Memory profile and safety thresholds**: M4 Max default. Profiles `("safe", "balanced", "large")` with `balanced` as default. Soft threshold 90 GB, hard ceiling 100 GB on a 128 GB dev system; `LitoMemoryLimitExceeded` raised on ceiling cross. M2/M3 tile/stream profile deferred to Phase 4+.
8. **Performance instrumentation**: `LitoGenerationResult.metrics` returns per-stage `(wall_time_s, peak_active_memory_gb)` for stages `preprocess`, `condition`, `tokenize`, `dit`, `decode`, `render`, `export`. Enable inline logging via `--print-metrics` on the CLI.
9. **Upstream-recommended defaults**: list every `LITO_RECOMMENDED_*` constant exposed in `lito_inference.py` with its value and upstream source (this section is regenerated from the constants' inline comments — keep it in sync if upstream defaults move).
10. **Source-contract probe layout**: where fixtures live (`tests/fixtures/lito/`), how to regenerate (`scripts/lito/write_contract_fixtures.py tests/fixtures/lito --overwrite`), per-module tolerance summary (table mirroring `spec/gap-matrix.md`), and the explicit caveat that these are local source-contract fixtures rather than vendor numerical captures.
11. **Programmatic API**:
    ```python
    from mlx_spatial.lito import LitoInferencePipeline
    from mlx_spatial.lito_inference import LITO_RECOMMENDED_NUM_STEPS

    pipe = LitoInferencePipeline(weights_root="weights/lito", memory_profile="balanced")
    result = pipe.generate("inputs/lito/sample.png", num_steps=LITO_RECOMMENDED_NUM_STEPS, seed=42)
    for stage, m in result.metrics.items():
        print(f"{stage}: {m['wall_time_s']:.2f}s, peak {m['peak_active_memory_gb']:.1f} GB")
    ```
12. **Vendors note**: `vendors/ml-lito/` is a dev-time parity reference and is never required at runtime.
13. **Phase 4+ candidates** (cross-reference to ROADMAP)

### `docs/architecture.md` (update)

Add LiTo to the pipeline matrix. Note that LiTo is image-to-3DGS (no mesh extraction) and reuses `gs_rasterize.py` + `hyworld2_sh.py` (adapter-only by default; if Risk F option B was chosen, document the shared-base extraction here too).

### `.agent/steering/ROADMAP.md` (update)

Append Phase 3 entry:

```markdown
## Phase 3: LiTo MLX Inference Pipeline

- status: done | in-progress | pending  (per current state when this slice runs)
- change: `2026-05-22-lito-mlx-inference-pipeline`
- objective: Port Apple LiTo image-to-3DGS inference into `mlx-spatial` as a new pipeline matching the established `<name>_*.py` + `mlx-spatial-<name>` shape, with per-module numerical parity probes against the vendored LiTo reference implementation.
- exit signal: All 6 slices verified; `mlx-spatial-lito generate <image>` produces a sensible 3DGS file; no existing pipeline regresses.
```

Append Phase 4+ candidates (one bullet each, brief):

- LiTo training / fine-tuning on MLX
- LiTo weight redistribution via `mlx-community` with a model card
- LiTo mesh extraction (Flexicubes / marching cubes) from 3DGS+LF outputs
- LiTo multi-image and video conditioning
- M2/M3 memory profile for LiTo (tile/stream)
- Cross-pipeline ablations: LiTo vs. TRELLIS.2 vs. Hunyuan3D-2.5

## Verification

```bash
test -f docs/lito.md
test -s docs/lito.md  # not empty
grep -E "^## Phase 3" .agent/steering/ROADMAP.md
grep -E "lito-mlx-inference-pipeline" .agent/steering/ROADMAP.md
# Architecture map mentions LiTo
grep -E "LiTo|lito" docs/architecture.md
# License summary in docs
grep -iE "license|redistribut" docs/lito.md
```

## Slice-Specific Risks

- **License-decision drift:** if `docs/lito.md` records a different stance than Slice 0's `License Decision`, agents executing the runtime code path will read one source of truth while users read another. Both must match. This slice reads Slice 0's decision verbatim; do not paraphrase.
- **ROADMAP drift:** if Phase 1 and Phase 2 entries used different shape (e.g., different fields), match those exactly. Do not introduce a new ROADMAP entry shape.

## Done When

All verification commands pass AND `docs/lito.md` is self-sufficient (a fresh user can go from zero to a `.ply` output following only `docs/lito.md`) AND ROADMAP Phase 3 entry mirrors Phase 1/2 entry shape.
