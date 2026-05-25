# mlx-spatial 0.0.1 Release Readiness Plan

## Goal

Execute the release-readiness contract in `SPEC.md`: make `mlx-spatial` credible for a `0.0.1` PyPI release with clean package artifacts, high-signal docs, organized user-facing scripts, and an auditable git/release boundary.

## Requirement Traceability

- AC-01: Slice 1
- AC-02, AC-03, AC-04, AC-05, AC-14: Slice 3
- AC-06, AC-07, AC-08: Slice 2
- AC-09, AC-10, AC-12: Slice 1
- AC-11: Slice 4
- AC-13: Slice 5

## Ordered Slice Sequence

### Slice 1: Package And Release Hygiene

**Objective:** Make the package metadata, build exclusions, license file, and PyPI workflow match the intended `0.0.1` release boundary.

**Acceptance criteria:**
- `pyproject.toml` declares version `0.0.1`, repository metadata, license metadata, and hatch build include/exclude policy.
- Top-level `LICENSE` exists or the selected license decision is explicitly captured in package metadata and docs.
- `.github/workflows/workflow.yaml` or equivalent exists and targets PyPI project `mlx-spatial` with environment `pypi`.
- `uv build` produces an sdist and wheel that exclude `.agent/`, `.codex/`, `.claude/`, `.venv/`, `weights/`, `inputs/`, `outputs/`, `vendors/`, caches, and generated probes.

**Verification:** `uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl`

**Touches:** `pyproject.toml`, `LICENSE`, `.github/workflows/`, `.gitignore`, `scripts/packaging/`

**Produces:** Clean package artifacts and release artifact checker.

**Evidence:** Completed directly. `uv build` produced `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`; `python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl` passed.

### Slice 2: Scripts Surface

**Objective:** Create a small, documented `scripts/` surface for repeatable inference, audit, trace, and packaging tasks without turning one-off probes into public examples.

**Acceptance criteria:**
- `scripts/README.md` explains script categories, support level, input/output conventions, and examples.
- User-facing inference scripts use argparse help, safe defaults under `outputs/`, trace output where supported, and recommended pipeline settings.
- SAM3D inference script defaults to quality gates and primary-mask/sample-friendly behavior; no quality-warning bypass is used by default.
- Reusable support scripts exist for weight audit, trace summary, output quality summary, and package artifact checking, or are explicitly deferred with rationale in `scripts/README.md`.

**Verification:** `python scripts/sam3d/reconstruct.py --help && python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-primary-default/trace.json && python scripts/packaging/check_release_artifacts.py --help`

**Depends on:** Slice 1 for the package checker path.

**Touches:** `scripts/README.md`, `scripts/sam3d/`, `scripts/packaging/`, optional `scripts/audit/`

**Produces:** Public script contract for users and maintainers.

**Evidence:** Completed directly. `python scripts/sam3d/reconstruct.py --help`, `python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-primary-default/trace.json`, and `python scripts/packaging/check_release_artifacts.py --help` passed.

### Slice 3: Documentation Set

**Objective:** Replace bootstrap-era docs with high-signal user and agent references for install, model setup, pipeline usage, development, model publishing, and release process.

**Acceptance criteria:**
- `README.md` is concise and points to durable docs instead of carrying the full implementation history.
- `docs/sam3d.md` documents converted safetensors, MoGe dependency, primary masks, quality gates, PLY expectations, coordinate/viewer caveats, and current recommended commands.
- `docs/architecture.md` maps the major modules and pipeline boundaries for agents.
- `docs/development.md` records test/build commands, local asset conventions, and contribution constraints.
- `docs/model-publishing.md` records AppAutomaton-first naming, model-card content, audit artifacts, license/gating checks, and `mlx-community` deferral.
- `docs/release.md` or `CHANGELOG.md` records the `0.0.1` release checklist.

**Verification:** `python - <<'PY'\nfrom pathlib import Path\nrequired = ['README.md','docs/sam3d.md','docs/architecture.md','docs/development.md','docs/model-publishing.md','docs/release.md']\nmissing = [p for p in required if not Path(p).is_file()]\nassert not missing, missing\nfor p in required:\n    text = Path(p).read_text(encoding='utf-8')\n    assert 'TODO' not in text\nprint('docs ok')\nPY`

**Depends on:** Slice 2, so script docs can point at real paths.

**Touches:** `README.md`, `docs/`, optional `CHANGELOG.md`

**Produces:** User-facing and agent-facing documentation.

**Content constraints:** Channel = repository docs; source policy = current repo state plus upstream links already used for model-license facts; factual risk = medium for technical claims and high for license/gating statements, so license/gating claims need source links.

**Evidence:** Completed directly. Replaced the bootstrap-era `README.md` and added `docs/sam3d.md`, `docs/architecture.md`, `docs/development.md`, `docs/model-publishing.md`, and `docs/release.md`. Verified with the slice documentation check, which passed with `docs ok`.

### Slice 4: Git Boundary And Worktree Organization

**Objective:** Make the release work reviewable by separating release-readiness changes from unrelated implementation dirt and documenting what remains outside scope.

**Acceptance criteria:**
- `git status --short` has no unexplained dirty files relevant to the release work.
- Release-readiness files are commit-ready as a coherent group.
- Unrelated prior implementation changes are either committed separately, left untouched with a written status note, or explicitly excluded from release work.
- `.gitignore` and build exclusions are checked together so generated local files do not become package or commit candidates.

**Verification:** `git status --short && python scripts/packaging/check_release_artifacts.py --git-hygiene`

**Depends on:** Slices 1-3.

**Touches:** `.gitignore`, package checker, `.agent/steering/STATUS.md`, git staging recommendations.

**Checkpoint after:** human-verify

**Checkpoint reason:** Human should confirm the intended commit grouping before any release tag or publish workflow is used.

**Correction:** `check_release_artifacts.py --git-hygiene` now treats `.agent/`, `.codex/`, and `.claude/` as package-artifact blockers but not git-hygiene blockers. This keeps release archives clean while allowing tracked automaton/project state to exist in the worktree. The git-hygiene mode still flags local/generated paths such as `weights/`, `inputs/`, `outputs/`, `vendors/`, `dist/`, caches, virtualenvs, and generated bytecode.

**Evidence:** Completed directly. `git status --short` shows the release-readiness group plus known pre-existing pipeline/runtime dirt; `.agent/steering/STATUS.md` records the intended release commit boundary and out-of-scope dirty groups. `python scripts/packaging/check_release_artifacts.py --git-hygiene` passed. Artifact checking on the existing `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl` also passed after preserving package-artifact blockers.

### Slice 5: Final Release Gate

**Objective:** Run the end-to-end local release gate and record the result without publishing.

**Acceptance criteria:**
- Full tests pass.
- Package build passes.
- Clean artifact checker passes on both sdist and wheel.
- At least one documented lightweight smoke or locally available inference check passes, or gated-weight-dependent checks are listed as manual.
- `docs/release.md` records the verified commands and the remaining manual publish step.

**Verification:** `uv run pytest -q && uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl`

**Depends on:** Slice 4.

**Touches:** `docs/release.md`, release verification notes.

**Checkpoint after:** human-action

**Checkpoint reason:** Publishing to PyPI is intentionally outside this plan; a maintainer must approve and trigger the trusted-publishing workflow.

**Evidence:** Completed directly after the Slice 4 human-verification checkpoint was cleared. `uv run pytest -q` passed with `651 passed, 5 skipped, 2 warnings`; `uv build` rebuilt `dist/mlx_spatial-0.0.1.tar.gz` and `dist/mlx_spatial-0.0.1-py3-none-any.whl`; `python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl` passed; `python scripts/packaging/check_release_artifacts.py --git-hygiene` passed. Lightweight smoke checks passed for `uv run mlx-spatial-sam3d --help`, `uv run mlx-spatial-trellis2 --help`, `uv run mlx-spatial-hyworld2 --help`, and `python scripts/sam3d/reconstruct.py --help`. `docs/release.md` records the verified commands and keeps publishing as a maintainer-triggered trusted-publishing action.

## Execution Routing And Topology

Default execution route is direct. Slices are ordered serially because package/script/docs/git hygiene depend on stable paths and artifact boundaries.

Parallel-safe groups: none.

Checkpoints:
- Slice 4 pauses for human verification of commit grouping.
- Slice 5 pauses for human action before any PyPI publish.

Recommended review: run `auto-eng-review` before execution because the plan changes package release surfaces, public scripts, and workflow configuration.

## Aggregate Verification Commands

```bash
uv run pytest -q
uv build
python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl
python scripts/sam3d/reconstruct.py --help
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-primary-default/trace.json
git status --short
```

## Context Budget For This Change

This plan spans multiple sessions. Load order for execution:

1. `SPEC.md` and this `PLAN.md`
2. `pyproject.toml`, `.gitignore`, current `README.md`
3. Existing CLI modules under `src/mlx_spatial/{sam3d,trellis2,hyworld2}.py`
4. Tests only for the slice being executed
5. Build artifact contents only after Slice 1 changes
