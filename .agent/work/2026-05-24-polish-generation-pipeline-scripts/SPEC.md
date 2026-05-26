# Polish Generation Pipeline Scripts Spec

## Bounded Goal

Make the existing `scripts/` generation pipeline surface self-documenting and consistent as readable `mlx-spatial` repo-local usage examples, without rewriting the scripts into a new API-example architecture.

## Broader Intent

The scripts should make the recommended model-family workflows obvious to humans and coding agents: what each script runs, which MLX model family it demonstrates, what settings are recommended, and where outputs and traces go.

## Work Scale And Shape

- Scale: feature-sized.
- Shape: refactor, limited to script surface clarity and documentation consistency.

## Selected Lenses

- product
- engineering
- content

## Target User Or Stakeholder

Readers are `mlx-spatial` users, maintainers, and coding agents inspecting the repo-local generation scripts to understand how to run SAM3D, TRELLIS.2, HY-WorldMirror, and LiTo with recommended settings.

## Content Target

- Audience: technical users and agents reading scripts as runnable examples.
- Thesis: repo-local generation scripts should be clear enough to understand the intended MLX workflow from the script name, docstring, help output, defaults, and `scripts/README.md`.
- Voice direction: concise engineering-reference prose with exact commands, model roots, input/output expectations, and setting rationale.
- Content anti-goals: no marketing claims, no run-log prose, no inflated quality promises, and no internal Automaton process narration.

## Constraints And Risks

- Keep the existing model-family grouping under `scripts/sam3d/`, `scripts/trellis2/`, `scripts/hyworld2/`, and `scripts/lito/`.
- Keep the existing scripts as the approved Approach A polish pass; do not rewrite them into direct Python pipeline API examples.
- Preserve runtime behavior and recommended settings unless implementation finds script/docs drift against current source constants.
- If a file path rename is considered for clarity, keep it small, update every reference, and avoid churn unless the current name is materially misleading.
- Do not touch unrelated dirty Automaton/skill files already present in the worktree.
- Heavy model inference is out of scope for verification.

## Required Outcome

The script surface should read as a coherent set of generation pipeline examples:

- Generation scripts have clear names or titles, docstrings, examples, argument help, and visible recommended defaults.
- `scripts/README.md` separates user-facing generation scripts from maintainer/fixture/packaging tools and matches live script behavior.
- Recommended settings are discoverable in code and docs for SAM3D, TRELLIS.2, HY-WorldMirror, and LiTo.
- Command examples remain copyable from a repo checkout.
- Existing behavior, model roots, output paths, and trace conventions remain stable unless a documented drift fix is necessary.

## Acceptance Criteria

- AC-01: Each user-facing generation script under `scripts/sam3d/`, `scripts/trellis2/`, `scripts/hyworld2/`, and `scripts/lito/` has a docstring that states the model family, input shape, output artifact, default model root, and recommended profile/settings.
- AC-02: `--help` text for the generation scripts explains non-obvious defaults and distinguishes recommended quality settings from smoke/debug overrides.
- AC-03: Script names, README section labels, and examples make the pipeline role clear; any path rename is justified by a concrete ambiguity and all references are updated.
- AC-04: `scripts/README.md` presents the generation scripts as the primary repo-local pipeline examples and labels maintainer, fixture, quality-inspection, and packaging scripts separately.
- AC-05: `scripts/README.md`, root README references, and model-family docs do not contradict the live script defaults for model roots, memory profiles, output locations, trace behavior, or recommended settings.
- AC-06: The scripts keep the selected Approach A shape: CLI-style wrappers or current implementation style remains acceptable, and direct Python pipeline API rewrites are not required.
- AC-07: The change does not broaden into model-quality tuning, new model outputs, weight conversion behavior, release publishing, or generated assets.
- AC-08: Verification includes `--help` smoke checks for the generation scripts, text consistency checks for changed docs, `ruff` on touched Python scripts, and `git diff --check`.

## Scope Coverage Decisions

Included:

- Existing generation scripts for SAM3D, TRELLIS.2, HY-WorldMirror, and LiTo.
- Script docstrings, help text, section labels, command examples, printed effective settings where already present, and `scripts/README.md` alignment.
- Clarifying recommended settings and the intended repo-local workflow.

Deferred:

- Approach B: rewriting scripts into direct Python pipeline API examples.
- Approach C: restructuring `scripts/` into a formal examples suite plus maintainer-tools split.

Anti-goals:

- No directory reshuffle.
- No model behavior rewrite.
- No heavy inference verification.
- No release/tag/publish work.
- No edits to unrelated dirty files.

## Assumptions

- The existing recommended settings are mostly correct and need clearer presentation rather than recalibration.
- The current `scripts/` layout is good enough for this pass; clarity problems can be solved in place.
