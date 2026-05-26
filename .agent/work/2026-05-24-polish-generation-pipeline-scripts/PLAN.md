# Polish Generation Pipeline Scripts Plan

## Goal

Implement `.agent/work/2026-05-24-polish-generation-pipeline-scripts/SPEC.md`: polish the existing `scripts/` generation pipeline surface so it is self-documenting and consistent without changing runtime behavior or restructuring the tree.

## Execution Routing And Topology

- Default execution: direct.
- Slice order: serial; Slice 2 depends on Slice 1 so docs can match the final script wording and defaults.
- Checkpoints: none.
- Parallel-safe groups: none.

## Ordered Slice Sequence

### Slice 1: Normalize Generation Script Surfaces

**Objective:** Make the user-facing generation scripts self-documenting through names or titles, docstrings, argument help, visible recommended defaults, and consistent effective-setting output where appropriate.

**Acceptance criteria:**
- Each user-facing generation script states the model family, input, output, default model root, and recommended profile/settings in its docstring or help output.
- Help text distinguishes recommended quality settings from smoke/debug overrides where those options exist.
- The implementation stays within the selected Approach A shape; existing wrapper behavior remains acceptable and direct Python pipeline API rewrites are not introduced.
- No model behavior, weight conversion behavior, output format, or default recommended setting changes unless correcting live script/docs drift.

**Verification:**
```bash
uv run ruff check scripts/sam3d/reconstruct.py scripts/trellis2/generate_shape.py scripts/trellis2/generate_textured.py scripts/hyworld2/generate_scene.py scripts/lito/generate.py
uv run python scripts/sam3d/reconstruct.py --help
uv run python scripts/trellis2/generate_shape.py --help
uv run python scripts/trellis2/generate_textured.py --help
uv run python scripts/hyworld2/generate_scene.py --help
uv run python scripts/lito/generate.py --help
```

**Touches:** `scripts/sam3d/reconstruct.py`, `scripts/trellis2/generate_shape.py`, `scripts/trellis2/generate_textured.py`, `scripts/hyworld2/generate_scene.py`, `scripts/lito/generate.py`

**Produces:** A clearer script surface with no heavy inference required.

**Status:** complete
**Evidence:** changed the five generation scripts; `uv run --with ruff ruff check ...` passed; all five planned `uv run python <script> --help` smoke checks passed. LiTo help now stays inspectable in headless Metal-less sessions by importing the runtime only after argparse parses non-help invocations.
**Risks / next:** none; Slice 2 should align docs against the current help output.

### Slice 2: Align Script Documentation And References

**Objective:** Update `scripts/README.md` and adjacent references so the scripts read as the primary repo-local generation examples and maintainer/fixture/packaging tools are clearly separated.

**Acceptance criteria:**
- `scripts/README.md` labels generation scripts separately from fixture, quality-inspection, and packaging scripts.
- `scripts/README.md`, root README references, and model-family docs do not contradict the live script defaults for model roots, memory profiles, output paths, trace behavior, or recommended settings.
- Command examples are copyable from a repo checkout and match the chosen invocation style.
- The docs do not imply a directory reshuffle, a direct API-example rewrite, heavy inference verification, release publishing, or generated asset creation.

**Verification:**
```bash
rg -n "scripts/(sam3d|trellis2|hyworld2|lito)|memory-profile|output-dir|trace|fixture|packaging" scripts/README.md README.md docs
uv run python scripts/sam3d/reconstruct.py --help
uv run python scripts/trellis2/generate_textured.py --help
uv run python scripts/hyworld2/generate_scene.py --help
uv run python scripts/lito/generate.py --help
git diff --check
```

**Depends on:** Slice 1

**Touches:** `scripts/README.md`, `README.md`, relevant `docs/*.md` only where references need alignment.

**Produces:** Aligned user-facing script docs that preserve Approach A scope.

**Status:** complete
**Evidence:** changed `scripts/README.md` and `docs/lito.md`; `rg -n "scripts/(sam3d|trellis2|hyworld2|lito)|memory-profile|output-dir|trace|fixture|packaging" scripts/README.md README.md docs` showed aligned generation, fixture, quality-inspection, and packaging references; all generation script `--help` checks passed; `git diff --check` passed.
**Risks / next:** none.

## Review: Engineering

- Verdict: approved
- Strength: The plan is a right-sized documentation and CLI-surface polish that confines edits to existing scripts and docs without changing model behavior or adding dependencies.
- Concern: The main execution risk is default/help drift across Slice 1 and Slice 2 if documentation is edited without re-reading the generated `--help` output.
- Action: Invoke `auto-execute` and start Slice 1, using the listed help smoke checks before aligning `scripts/README.md` and adjacent docs.
- Verified: canonical plan read, targeted script parser/default surfaces inspected, `scripts/README.md` and docs references searched, and all five planned `--help` smoke checks passed.
