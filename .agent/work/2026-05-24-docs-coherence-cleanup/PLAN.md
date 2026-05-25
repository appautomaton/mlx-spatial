# Docs Coherence Cleanup Plan

## Goal

Execute the bounded docs-only cleanup from `SPEC.md`: make the public docs and script docs tell one consistent, runnable story for `mlx-spatial` model-family usage without changing runtime behavior.

## Ordered Slice Sequence

### Slice 1: Normalize LiTo Defaults And Command Convention

**Objective:** Align LiTo examples and download/validation commands with the actual runtime/script defaults and the repo-checkout command convention.

**Acceptance criteria:**
- LiTo examples in `README.md`, `docs/lito.md`, and `scripts/README.md` consistently present `balanced` as the default runtime path.
- `safe` appears only when explicitly described as smoke, debug, or lower-memory guidance.
- Repo-checkout Hugging Face commands use `uv run hf download`; installed-package command variants are clearly scoped if present.

**Touches:** `README.md`, `docs/lito.md`, `scripts/README.md`.

**Produces:** Updated LiTo setup/run prose and command blocks consistent with `scripts/lito/generate.py` and `src/mlx_spatial/lito_inference.py`.

**Verification:** `rg -n 'LITO_DEFAULT_MEMORY_PROFILE|--memory-profile safe|--memory-profile balanced|(^| )hf download|uv run hf download' README.md docs/lito.md scripts/README.md scripts/lito/generate.py src/mlx_spatial/lito_inference.py`

**Status:** complete
**Evidence:** changed `README.md`, `docs/lito.md`, and `scripts/README.md`; `rg -n 'LITO_DEFAULT_MEMORY_PROFILE|--memory-profile safe|--memory-profile balanced|(^| )hf download|uv run hf download' README.md docs/lito.md scripts/README.md scripts/lito/generate.py src/mlx_spatial/lito_inference.py` passed and shows `balanced` as the default with `safe` only in smoke/debug guidance.
**Risks / next:** none.

### Slice 2: Align Runtime Docs Navigation And Page Shape

**Objective:** Make the docs entry points and runtime pages follow one organized user path across model families.

**Acceptance criteria:**
- `README.md` includes a clear pipeline-choice summary for SAM3D, TRELLIS.2, HY-WorldMirror, and LiTo.
- `docs/README.md` links `scripts/README.md` as a first-class entry point for recommended runnable scripts.
- `docs/trellis2.md` includes assets, validation, inputs, run path, expected outputs, trace or inspection behavior, and export caveats.
- Public runtime docs avoid normal-user guidance that depends on `.agent` or Automaton internals.

**Touches:** `README.md`, `docs/README.md`, `docs/trellis2.md`, `docs/lito.md`; optionally `docs/sam3d.md` or `docs/hyworld2.md` for light section-shape consistency only.

**Produces:** Updated navigation and TRELLIS.2 runtime page with the same reader contract shape used by the other model pages.

**Verification:** `rg -n 'scripts/README.md|SAM 3D Objects|TRELLIS.2|HY-WorldMirror|LiTo|## Assets|## Inputs|## Run|## Outputs|## Trace|## Export Caveat|\\.agent|Automaton' README.md docs/README.md docs/trellis2.md docs/lito.md`

**Status:** complete
**Evidence:** changed `README.md`, `docs/README.md`, `docs/trellis2.md`, `docs/lito.md`, and `docs/sam3d.md`; `rg -n 'scripts/README.md|SAM 3D Objects|TRELLIS.2|HY-WorldMirror|LiTo|## Assets|## Inputs|## Run|## Outputs|## Trace|## Export Caveat|\\.agent|Automaton' README.md docs/README.md docs/trellis2.md docs/lito.md` passed and shows the scripts entry point plus TRELLIS.2 assets, inputs, run, outputs, trace, and export caveat sections without `.agent` or Automaton matches.
**Risks / next:** none.

### Slice 3: Final Docs Consistency Pass

**Objective:** Verify the edited docs satisfy the spec acceptance criteria without adding release, runtime, or stale-version drift.

**Acceptance criteria:**
- No stale `0.0.1` user-facing release drift is introduced in the touched docs.
- Public runtime docs do not mention `.agent/` or Automaton as part of normal user guidance.
- Markdown edits have no trailing whitespace or patch formatting issues.
- If command examples changed materially, CLI/script help still exposes the documented commands or flags without requiring model weights.

**Touches:** touched docs only, unless verification exposes a narrow correction needed to satisfy AC-01 through AC-08.

**Produces:** Final docs diff ready for `auto-verify`.

**Verification:** `git diff --check && ! rg -n '0\\.0\\.1|\\.agent/|Automaton' README.md docs/README.md docs/sam3d.md docs/trellis2.md docs/hyworld2.md docs/lito.md scripts/README.md && rg -n -e '--memory-profile (safe|balanced)' -e '(^| )hf download' -e 'uv run hf download' README.md docs/lito.md scripts/README.md && uv run mlx-spatial-lito generate --help && uv run mlx-spatial-trellis2 generate-textured --help`

**Status:** complete
**Evidence:** changed `README.md`, `docs/README.md`, `docs/sam3d.md`, `docs/trellis2.md`, `docs/lito.md`, and `scripts/README.md`; `git diff --check` passed, `rg -n '0\\.0\\.1|\\.agent/|Automaton' README.md docs/README.md docs/sam3d.md docs/trellis2.md docs/hyworld2.md docs/lito.md scripts/README.md` returned no matches, the command-example `rg` passed, and both `uv run mlx-spatial-lito generate --help` and `uv run mlx-spatial-trellis2 generate-textured --help` passed.
**Risks / next:** none.

## Execution Routing And Topology

Default: direct, serial execution with continuation after each slice verification passes.

Overrides: none.

Parallel-safe groups: none.

Checkpoints: none.

## Aggregate Verification Commands

Run slice-level verification after each slice. At final closeout, run:

```bash
git diff --check
! rg -n '0\.0\.1|\.agent/|Automaton' README.md docs/README.md docs/sam3d.md docs/trellis2.md docs/hyworld2.md docs/lito.md scripts/README.md
rg -n -e '--memory-profile (safe|balanced)' -e '(^| )hf download' -e 'uv run hf download' README.md docs/lito.md scripts/README.md
uv run mlx-spatial-lito generate --help
uv run mlx-spatial-trellis2 generate-textured --help
```

## Review: Engineering

- Verdict: approved
- Strength: The plan is docs-only, serial, and grounded in existing LiTo and TRELLIS.2 CLI defaults with explicit source-and-doc verification commands.
- Concern: The main residual risk is cross-page documentation drift if the final grep and CLI help checks are not rerun after every material command example edit.
- Action: Proceed with `auto-execute` and run each slice verification before advancing to the next slice.
- Verified: Automaton context loaded, PLAN.md reviewed, no DESIGN.md configured, LiTo default traced to `LITO_DEFAULT_MEMORY_PROFILE`, CLI entry points checked in `pyproject.toml`, and LiTo/TRELLIS.2 help commands executed without weights.
