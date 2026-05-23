---
active_change: 2026-05-22-lito-mlx-inference-pipeline
stage: verify
---

# Status

## Current Change

- active change: `2026-05-22-lito-mlx-inference-pipeline`
- current stage: `verify`
- Canonical artifact paths live in `.agent/.automaton/state/current.json` and the artifacts themselves; this file does not duplicate them.

## What Is True Now

- Active change is verification-complete. LiTo source/tests/docs landed in `f7d575f`; the binary PLY storage follow-up landed in `14236d5`. Slice 5Q-4 final verification passed after the uncapped teacup output was judged fair-looking in a proper Gaussian-splat viewer.
- Selected shape: parity/source-contract, scale: capability. Selected lenses: product + engineering. Product review: `approved_with_risks`. Engineering review: `approved_with_risks` after the no-CUDA correction.
- Pipeline shape: LiTo follows the established `<name>_*.py` + `mlx-spatial-<name>` convention used by SAM 3D, TRELLIS.2, and HY-World 2.0.
- Execution topology: Slice 0A is completed vendor/assets/routing evidence. Slice 0B is completed as the serial source-contract fixture gate: `scripts/lito/write_contract_fixtures.py` generated deterministic local `tests/fixtures/lito/` plus `manifest.json`, and `scripts/lito/validate_fixtures.py` passed. Slices 1-6 and 5Q are complete; AC-07 is accepted with the known teacup handle-hole caveat documented as follow-up work.
- Plan-level defaults: `.ply` output (`--format` flag for `splat`/`safetensors`), checkpoint-backed PLY storage defaults to `binary_little_endian` with `--ply-storage ascii` available for debugging, runtime `plyfile` dependency only if runtime writer code imports it, adapter-only as Risk F default, `plyfile` in `[dependency-groups.dev]`, sample-input download URL documented (no redistribution unless license permits), `generate` CLI verb, `LITO_MEMORY_PROFILES = ("safe", "balanced", "large")` with `balanced` default. Torch remains external/transient for CPU/MPS probes; it is not in the project dependency groups because `uv lock` pulled CUDA packages when attempted.
- Execution discipline (Refresh B): `/tmp/lito-*` for scratch (no working-tree leaks per AC-15); 90 GB soft / 100 GB hard memory thresholds enforced via `mx.metal.get_active_memory()` / `mx.metal.get_peak_memory()` with `LitoMemoryLimitExceeded` raised at the hard ceiling; autonomous fallbacks via `WebFetch`/`WebSearch`/`playwright-cli` for transient failures and lookup-style unknowns; only declared decision checkpoints (Risk F, license-blocked, routing re-route) escalate to user.
- Performance instrumentation: `LitoGenerationResult.metrics` returns per-stage `{wall_time_s, peak_active_memory_gb, peak_cache_memory_gb}` for `preprocess`, `condition`, `tokenize`, `dit`, `decode`, `render`, `export`.
- Upstream-recommended generation settings are captured at Slice 0A and embedded as `LITO_RECOMMENDED_*` constants in `lito_inference.py`; CLI / library / `scripts/lito/generate.py` use them as defaults. Defaults are not invented.
- Risk F decision is captured in `PLAN.md` with a conditional checkpoint that trips only if Slice 0 audits the LF-render and reports the adapter pattern infeasible.
- Latest quality-closure evidence: Slice 5Q-2 pass 1 fixed the first high-confidence decode/init-coverage mismatch by adding `--max-init-coords-per-batch {profile|none|N}` while keeping the runtime no-CUDA and MLX-native. Human visual inspection rejected the 1024-cell teacup output, so Slice 5Q-2 pass 2 fixed conditioning/preprocessing parity for RGBA inputs with upstream `keep_optical_axis=True` crop/pad behavior. Human inspection of the 4096-cell crop output showed improvement but missing regions. A memory-conscious occupancy diagnostic found `17317` occupied init cells for the teacup latent, so the 4096 candidate covered only about `23.7%` of occupied structure. The accepted-looking teacup candidate is `outputs/lito/teacup-quality-crop-uncapped.ply`, generated with `--max-init-coords-per-batch none`; it inspects as checkpoint-backed, finite, `1108288` vertices, `62` properties, no structural inspector flags, opacity probability median `0.056885`, scale exp median `0.004650`, quaternion norm median `1.000000`, and `failure_classification=stats_sane_visual_review_required`. Peak active memory stayed about `15.28 GB`; peak cache memory reached about `21.87 GB`. The second real-object uncapped candidate is `outputs/lito/beer-mug-quality-uncapped.ply`; it inspects as checkpoint-backed, finite, `925952` vertices, `62` properties, no structural inspector flags, opacity probability median `0.048492`, scale exp median `0.006928`, quaternion norm median `1.000000`, and `failure_classification=stats_sane_visual_review_required`. The accepted quality note is: output is usable in a Gaussian-splat-aware viewer, but teacup handle-hole topology is not perfect and mesh/GLB extraction is out of scope.
- Handle-hole probe: `.agent/work/2026-05-22-lito-mlx-inference-pipeline/orchestration/handle-hole-probe.md` records the likely next quality target. The teacup PNG's high-alpha component spans nearly the full image and touches edges, so a preprocessing/matte-quality slice is more likely to help the handle void than changing PLY export.

## Release Strategy (clarified 2026-05-23)

The 0.0.1 release-readiness change (`2026-05-22-mlx-spatial-0-0-1-release-readiness`) is execute-complete locally but **has not been tagged**. Maintainer trigger of the PyPI trusted-publishing workflow is pending. LiTo lands in **0.0.2** as part of normal commits.

The 0.0.1 release group is **not frozen** for this change. `pyproject.toml` (scripts + dev deps), `docs/`, `README.md`, `uv.lock` may be modified by LiTo. Files that should not be touched without explicit decision:

- `.github/workflows/workflow.yaml` (the publishing workflow — change requires release ops review)
- `LICENSE` (legal)
- `scripts/` (release scripts — only touch if behavior actually changes)
- `.agent/work/2026-05-22-mlx-spatial-0-0-1-release-readiness/` (immutable historical artifact)

The single hard constraint: `uv build` continues to produce a clean wheel and sdist after LiTo lands, so the 0.0.1 (or 0.0.2) tag remains producible. This is verified in Slice 5.

Known pre-existing dirty worktree state (verified 2026-05-23; carried from before the active change):

- modified automaton infrastructure under `.agent/.automaton/{bin,lib}/`
- modified steering: `.agent/steering/ROADMAP.md` (already), `.agent/steering/STATUS.md` (this file)
- modified older-change artifacts: `.agent/work/2026-05-20-gap-matrix-parity/spec/gap-matrix.md`, `.agent/work/2026-05-20-production-pipeline-parity/PLAN.md`
- new untracked directories: `.agent/work/2026-05-20-production-pipeline-parity/orchestration/`, `.agent/work/2026-05-22-mlx-spatial-0-0-1-release-readiness/`, `.agent/work/2026-05-22-lito-mlx-inference-pipeline/`
- `.agent/.automaton/state/install-manifest.json`

Before the LiTo active change, `src/mlx_spatial/` and `tests/` had no pre-existing runtime/test dirt. LiTo runtime/test/doc edits have been committed; remaining dirty worktree entries outside this closeout are pre-existing or unrelated.

## Next Step

No LiTo implementation blocker remains. Optional next work should start as a new slice: preprocessing/matte cleanup for the teacup handle void, SPZ export, mesh/GLB extraction, or lower-memory profiles.

## Open Risks

- LiTo now has a source-contract smoke pipeline and a checkpoint-backed no-CUDA inference path with accepted quality evidence from uncapped real-object outputs. Upstream CUDA/PyTorch/gsplat paths remain static source references; optional Torch parity is CPU/MPS-only and not a runtime backend. Default `generate` still raises `LitoBackendUnavailable` if converted LiTo/TRELLIS weights are missing or if sampling produces no occupied TRELLIS cells.
- `uv build` passes with `plyfile` dev-only. Runtime `plyfile` was not added because the PLY writer is hand-written; checkpoint-backed PLY now defaults to binary little endian with ASCII retained as an option.
- Memory threshold tests exercise the warning and hard-ceiling abort paths synthetically. The real safe-profile run peaked at about 15.28 GB active MLX memory and caps init cells at 512 before 64x Gaussian expansion; balanced/large profile behavior remains broader performance work.
- Maintainer has not yet triggered the 0.0.1 PyPI publish. LiTo work proceeds in parallel and does not block on that trigger. Eventual tag is **0.0.2**.
- SAM 3D model redistribution remains license/gating-sensitive; LiTo's weight-redistribution stance (deferred to Phase 4+) mirrors the same caution.
