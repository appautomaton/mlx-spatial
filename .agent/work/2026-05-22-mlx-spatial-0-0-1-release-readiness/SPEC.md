# mlx-spatial 0.0.1 Release Readiness Spec

## Bounded Goal

Prepare `mlx-spatial` for a credible `0.0.1` public release by defining and implementing the docs, user-facing scripts, packaging hygiene, git hygiene, and release workflow needed for PyPI and model-repo users.

## Broader Intent

`mlx-spatial` should become a usable MLX-native 3D inference library whose first public release gives users and coding agents enough reference material to install it, fetch or convert model assets, run documented inference scripts, and continue development without relying on hidden session context.

## Work Scale and Shape

- Scale: capability
- Shape: release-readiness + documentation + tooling organization

## Selected Lenses

- product
- engineering
- content

## Target User or Stakeholder

Apple Silicon developers and coding agents using `mlx-spatial` for SAM3D, TRELLIS.2, and HY-World inference; maintainers preparing AppAutomaton package and model releases.

## Audience

Developers who can run Python/MLX projects but do not know this repo's internal history; the docs and scripts should let them reproduce supported inference paths and understand where model weights live.

## Thesis

The first public release is not ready until the repo has a clean package boundary, high-signal docs, and documented inference scripts that encode recommended settings instead of leaving users to reconstruct them from tests or chat history.

## Voice Direction

Short, direct engineering prose. Prefer commands, paths, and invariants over narrative. Do not use promotional language, broad claims, or filler conclusions.

## Content Anti-Goals

- No marketing-style README sections.
- No vague "getting started" text without exact commands.
- No copied upstream model-card prose beyond attribution, license, and links.
- No hidden assumptions such as local weights existing without explaining how to obtain or convert them.

## Scope Coverage Decisions

- Included: docs, scripts organization, script self-documentation, package metadata, sdist/wheel hygiene, `.gitignore` review, git worktree organization, PyPI trusted-publishing workflow presence, release checklist, and model-card references needed by the package docs.
- Included: user-facing inference scripts under `scripts/` if they are reusable, documented, and encode recommended settings.
- Included: agent-facing docs that preserve development context and module boundaries.
- Deferred: adding new model capabilities, quantized SAM3D runtime support, performance optimization, and publishing to `mlx-community`.
- Anti-goal: shipping generated outputs, large weights, agent state, vendored repos, or local caches inside the Python package.

## Required Outcome

The repository has a release-ready `0.0.1` surface:

- Documentation explains install, model asset setup, conversion, supported inference commands, output expectations, quality gates, model publishing rules, development workflow, and release process.
- `scripts/` exists only for reusable entrypoints and is organized by purpose. Inference scripts are user-facing, self-documented, and use recommended defaults rather than stale experimental flags.
- Package metadata is intentional for `0.0.1`, including license metadata, repository URL, scripts, dependencies, and build configuration.
- Source distribution and wheel contain only intended package/release files.
- `.gitignore` and build exclusions prevent local weights, outputs, vendors, caches, agent state, and generated probes from bleeding into commits or package artifacts.
- Git worktree is organized into reviewable commits or commit-ready change groups with unrelated dirty state identified.
- PyPI trusted publishing workflow is present in the repo and matches the configured PyPI project/environment.

## Constraints and Risks

- Current `pyproject.toml` says `version = "0.1.0"`; `0.0.1` must be an explicit release decision.
- Current local checkout has no `.github/` workflow directory even though PyPI trusted publishing appears configured externally.
- Current `uv build` succeeds, but the sdist includes `.agent/`, `.codex/`, `.claude/`, and other non-package files. This blocks release.
- Current `.gitignore` ignores `weights/`, `inputs/`, `outputs/`, and `vendors/`, but does not by itself control sdist inclusion. Build configuration must also exclude non-release paths.
- `scripts/` currently appears absent. Adding it creates a public support surface; scripts must be stable enough to document.
- SAM3D model redistribution may be gated by upstream Meta terms. Package docs may link to model repos and conversion instructions, but must not imply that large gated weights ship with PyPI.
- Release docs must remain useful to agents, so architecture and development references should live in durable docs, not only `README.md`.
- Existing large local model-card files under ignored `weights/` may be useful for model publishing but are not part of the Python package unless separately managed.

## Acceptance Criteria

| ID | Requirement | Verification |
| --- | --- | --- |
| AC-01 | `pyproject.toml` reflects the intended `0.0.1` package metadata, repository URL, license metadata, dependencies, scripts, and hatch build include/exclude policy. | Inspect `pyproject.toml`; build metadata in wheel matches. |
| AC-02 | Top-level docs are high-signal and sufficient for first use. | `README.md` includes install, supported pipelines, model asset setup, first SAM3D/TRELLIS.2/HY-World commands or explicit support status, and links to detailed docs. |
| AC-03 | Agent/developer docs preserve working context. | `docs/architecture.md`, `docs/development.md`, and pipeline docs exist with module maps, test commands, release commands, and known constraints. |
| AC-04 | SAM3D docs capture the current best-known path. | `docs/sam3d.md` documents converted safetensors, MoGe dependency, primary-mask behavior, quality gates, PLY expectations, and known coordinate/viewer limitations. |
| AC-05 | Model publishing guidance exists. | `docs/model-publishing.md` documents AppAutomaton-first naming, model card requirements, audit artifacts, license/gating checks, and when not to duplicate to `mlx-community`. |
| AC-06 | `scripts/` is organized and documented. | `scripts/README.md` lists script categories, support status, required inputs, and examples. |
| AC-07 | User-facing inference scripts encode recommended settings. | Each inference script has argparse help, examples in docstring or `scripts/README.md`, safe output defaults under `outputs/`, trace output support where applicable, and no experimental flags by default. |
| AC-08 | Reusable audit/package scripts exist where needed. | Scripts for weight audit, trace summary, output quality summary, and package release checks are present or explicitly deferred with rationale. |
| AC-09 | Build artifacts are clean. | `uv build`; inspect sdist and wheel to confirm `.agent/`, `.codex/`, `.claude/`, `.venv/`, `weights/`, `inputs/`, `outputs/`, `vendors/`, and caches are absent. |
| AC-10 | `.gitignore` and build exclusions are aligned. | Review `.gitignore` and hatch config; generated outputs and local assets stay untracked and out of package artifacts. |
| AC-11 | Git state is organized for release review. | `git status --short` has no unexplained dirty files; release work is separated from unrelated prior implementation changes or clearly documented. |
| AC-12 | PyPI trusted publishing workflow is present locally. | `.github/workflows/workflow.yaml` or equivalent exists and targets PyPI project `mlx-spatial` with environment `pypi`. |
| AC-13 | Full test and smoke checks pass. | `uv run pytest -q`; `uv build`; at least one documented inference or lightweight smoke command succeeds or is marked requiring local gated weights. |
| AC-14 | Release checklist exists. | `docs/release.md` or `CHANGELOG.md` records the `0.0.1` release steps, preflight checks, and publish command/trigger. |

## Anti-Goals

- Do not publish to PyPI in this change.
- Do not publish or duplicate model repos in this change.
- Do not add new inference capabilities beyond release-support scripts and docs.
- Do not include model weights, generated outputs, local inputs, vendors, or agent state in the Python package.
- Do not make scripts that silently use non-default, experimental, or quality-warning-bypass settings.
- Do not rewrite core pipeline behavior unless a release script or doc check exposes a blocking defect.
