# Docs Coherence Cleanup Spec

## Bounded Goal

Make the public docs and script docs tell one consistent, runnable story for `mlx-spatial` model-family usage without changing runtime behavior.

## Broader Intent

Preserve the release-facing docs quality for the `0.0.2` line by making the documentation easier for humans and coding agents to follow before planning any larger docs rewrite.

## Work Scale And Shape

- Scale: feature-sized.
- Shape: mixed content/refactor, limited to documentation organization and consistency.

## Selected Lenses

- product
- engineering
- content

## Target User Or Stakeholder

Readers are Apple Silicon ML developers, maintainers, and coding agents who know the package is local-runtime focused but need exact commands, asset roots, output expectations, and model-family boundaries before running or modifying it.

## Content Target

- Audience: technical users and agents who need to choose the right model family, set up local weights, and run documented commands from a clean checkout or installed package.
- Thesis: `mlx-spatial` docs should expose one coherent runtime contract: choose the correct pipeline, download or validate the right asset layout, run the supported command, and inspect the expected output without learning internal Automaton process.
- Voice direction: direct engineering reference prose, short sections, exact paths and commands, no promotional framing, no historical transcript style.
- Content anti-goals: no significance inflation, no broad marketing language, no invented quality claims, no internal process narration in public runtime pages.

## Constraints And Risks

- Do not change runtime code, package metadata, release workflow, model cards, generated assets, tests, or validation behavior.
- Keep the cleanup narrow. Primary files are expected to be `README.md`, `docs/README.md`, `docs/trellis2.md`, `docs/lito.md`, and `scripts/README.md`; touch `docs/sam3d.md` or `docs/hyworld2.md` only for light section-shape consistency.
- Match docs to actual CLI and script defaults rather than inventing new defaults. Known example: LiTo default memory profile is `balanced`.
- Keep user-facing commands copyable after `uv sync`; installed-package commands may omit `uv run` only when the text says they are installed-package commands.
- Do not add a new large documentation system or new durable audit artifact unless planning identifies a specific downstream reader.
- Heavy model inference is out of scope for verification.

## Required Outcome

The docs should present one organized user path across model families:

- The root README should help readers choose the right pipeline and then point them to the right detailed page or script docs.
- `docs/README.md` should include all major documentation entry points, including `scripts/README.md`.
- Runtime docs should follow a recognizable pattern: status or purpose, assets, inputs, run command, outputs, options or blockers, and maintainer/dev notes when needed.
- LiTo commands and prose should consistently reflect the runtime default and reserve lower-memory `safe` guidance for explicit smoke/debug cases.
- TRELLIS.2 should meet the same runtime-page contract as the other model-family pages, including validation, expected outputs, trace behavior, and caveats.
- Public runtime pages should not require readers to know Automaton or `.agent` internals.

## Acceptance Criteria

- AC-01: LiTo user-facing examples in the root README, `docs/lito.md`, and `scripts/README.md` consistently document the intended default memory profile, with `safe` used only when explicitly described as smoke/debug/lower-memory guidance.
- AC-02: Download and validation command style is consistent: repo-checkout examples use `uv run` unless a nearby sentence explicitly scopes the command to an installed package.
- AC-03: `docs/trellis2.md` includes the same required runtime-page elements as the other model-family pages: assets, validation, inputs, run path, expected outputs, trace or inspection behavior, and export caveats.
- AC-04: `docs/README.md` exposes `scripts/README.md` as a first-class entry point for recommended runnable scripts.
- AC-05: The root README includes a clear pipeline-choice summary that differentiates object mask reconstruction, object image-to-3D, scene/world reconstruction, and LiTo research 3DGS output.
- AC-06: Public runtime docs do not mention `.agent/` or Automaton as part of normal user guidance.
- AC-07: The cleanup does not introduce new version drift, stale references to `0.0.1`, or release-publication instructions beyond the existing release checklist boundary.
- AC-08: Verification includes text consistency checks and whitespace checks at minimum; CLI help spot checks are included if planning determines command examples changed materially.

## Scope Coverage Decisions

Included:

- Normalize LiTo defaults across root README, LiTo docs, and script docs.
- Standardize command convention for checkout versus installed-package contexts.
- Expand TRELLIS.2 documentation to match the runtime-page contract.
- Update the documentation map so users can find script defaults.
- Remove or relocate public-facing internal process references.
- Add light pipeline-choice clarity in the root README.

Deferred:

- Full docs-site redesign or new navigation system; the current docs are compact and should be corrected in place.
- Model-card rewriting or Hugging Face publishing; those are separate release/model publishing tasks.
- Runtime quality work such as LiTo preprocessing, mesh export, SPZ export, or lower-memory profile changes.

Anti-goals:

- Do not change actual CLI defaults, script behavior, package dependencies, release workflows, or model asset layouts.
- Do not run heavyweight model inference just to verify prose.
- Do not turn stable docs into dated run logs.

## Assumptions

- The existing docs spine is usable; this change fixes coherence drift rather than replacing the documentation set.
- `balanced` remains the LiTo runtime default unless source inspection during planning proves otherwise.
