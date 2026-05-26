# mlx-spatial MapAnything Release Readiness Spec

## Bounded Goal

Prepare the next tagged `mlx-spatial` release so the PyPI package cleanly ships
the MapAnything MLX inference pipeline, with package, docs, scripts, workflow,
and AppleGPU memory evidence aligned.

## Broader Intent

The release should let an Apple Silicon user install the PyPI package, download
the `facebook/map-anything` weights separately, run the documented MapAnything
scene-generation path, and inspect outputs without relying on local chat history,
vendored runtime imports, or hidden development state.

## Work Scale and Shape

- Scale: capability
- Shape: release-readiness, package-surface audit, docs/scripts coherence, and
  release-gate verification

## Selected Lenses

- product
- engineering
- runtime
- content

## Target User or Stakeholder

Apple Silicon developers and maintainers preparing a tagged `mlx-spatial` PyPI
release that includes MapAnything scene inference.

## Required Outcome

The repository is release-ready for the next tag with MapAnything support:

1. Package metadata exposes `mlx-spatial-mapanything` and includes the
   MapAnything runtime modules, docs, script wrapper, and tests in intended
   source distributions without bundling local assets or generated outputs.
2. The release workflow checks artifact names that match the intended version
   instead of stale `0.0.1` filenames.
3. Runtime dependencies remain clean: no base dependency or package runtime
   import on Torch, TorchVision, UniCeption, OpenCV, CUDA-only packages, or
   `vendors/map-anything`.
4. User-facing docs and scripts state the recommended inference path clearly:
   `fixed_mapping`, stride `1`, checkpoint-derived patch size, separate
   `weights/map-anything`, `.npz` scene output, no mesh/3DGS claim.
5. Release verification includes a real local Desk MapAnything run when weights
   are present, records timing and memory evidence, and keeps heavyweight
   Torch-reference parity opt-in.
6. Automaton state points to this release-readiness change, not the previously
   verified scene-generation change.

## Constraints

- Respect `.agent/.automaton/references/FRAMEWORK.md` and
  `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md`.
- Use `sync-status.mjs` for Automaton state mutations; do not edit
  `current.json` by hand.
- Do not publish, push tags, or trigger trusted publishing in this change.
- Do not broaden MapAnything scope to DA3, Apache weights, training, Gradio UI,
  mesh export, or Gaussian splat export.
- Keep local model assets and reference captures ignored and excluded from
  release artifacts.
- Keep default MapAnything script settings memory-conscious; do not expose
  high-resolution knobs in the recommended wrapper.

## Risks

- **Version drift:** `pyproject.toml` and GitHub release workflow artifact names
  can disagree, causing release CI to check nonexistent files.
- **Package omission:** new untracked MapAnything modules/docs/scripts may work
  locally but fail to ship or fail after installation.
- **Dependency creep:** dev-only Torch reference support can accidentally enter
  base dependencies or runtime imports.
- **AppleGPU memory regression:** full DINOv2 giant + info sharing can complete
  in the current Desk run but regress if defaults change or activations are
  retained.
- **Dirty worktree ambiguity:** release work currently coexists with unrelated
  Automaton/scaffold edits and prior model-pipeline changes.

## Acceptance Criteria

| ID | Requirement | Verification |
| --- | --- | --- |
| REL-MA-01 | Automaton state and artifacts identify this release-readiness change. | `node .agent/.automaton/scripts/get-context.mjs` reports this active change with canonical SPEC and PLAN. |
| REL-MA-02 | Package metadata and installed CLIs include MapAnything. | Inspect `pyproject.toml`; `uv run mlx-spatial-mapanything --help` succeeds. |
| REL-MA-03 | Release workflow artifact checks match the current intended version. | Inspect `.github/workflows/workflow.yaml`; no stale hard-coded `0.0.1` artifact paths remain when version is `0.0.2`. |
| REL-MA-04 | Runtime dependency boundary stays clean. | Base dependencies exclude Torch/TorchVision/UniCeption/OpenCV; runtime source scan excludes those imports and vendor paths. |
| REL-MA-05 | MapAnything docs and scripts are high-signal and coherent. | `README.md`, `docs/mapanything.md`, `scripts/README.md`, and `scripts/mapanything/generate_scene.py --help` all state the same defaults and output boundary. |
| REL-MA-06 | Real MapAnything generation path works locally when weights are present. | `uv run python scripts/mapanything/generate_scene.py inputs/map-anything/desk --output-dir /tmp/mapanything-release-smoke` writes `scene.npz` and `trace.json`. |
| REL-MA-07 | AppleGPU memory evidence is captured for release readiness. | `/usr/bin/time -l` or MLX memory counters capture peak memory for the Desk run and the result is recorded in the release plan evidence. |
| REL-MA-08 | Tests and package build pass. | `uv run pytest tests/test_mapanything_*.py -q`; `uv lock --check`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build`. |
| REL-MA-09 | Release artifacts are clean and include intended MapAnything files. | `python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz dist/mlx_spatial-*-py3-none-any.whl`; inspect artifact contents for MapAnything docs/scripts/modules and absence of blocked local paths. |
| REL-MA-10 | Git/review boundary is explicit. | `git status --short` reviewed; release-relevant files are identifiable and unrelated dirty groups are not reverted. |

## Scope Coverage Decisions

Included: MapAnything release surface, package metadata, docs/scripts coherence,
workflow version consistency, tests, build, artifact inspection, memory evidence,
and Automaton state alignment.

Deferred: publishing, tag push, broad benchmark claims, UI/visualizer product
surface, DA3, Apache model variant, training, and non-image input modes.

## Anti-Goals

- Do not claim release readiness from local source-only execution without a
  package artifact check.
- Do not ship local weights, inputs, vendors, outputs, Automaton state, or
  scratch `/tmp` artifacts.
- Do not make Torch reference tooling part of normal runtime or base
  dependencies.
- Do not silently change unrelated Automaton skill edits or pre-existing dirty
  files.
