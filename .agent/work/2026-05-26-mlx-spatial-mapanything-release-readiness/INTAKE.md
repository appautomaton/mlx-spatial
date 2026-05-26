# mlx-spatial MapAnything Release Readiness Intake

## User Objective

Prepare the next tagged `mlx-spatial` release so the PyPI package includes the
MapAnything MLX inference pipeline, while respecting the Automaton lifecycle and
keeping AppleGPU/MLX memory safety visible during release verification.

## Operating Context

- The prior active change, `2026-05-26-mapanything-mlx-scene-generation`, is
  terminal at `stage: verified`.
- This is a new release-readiness change, not more execution inside the verified
  scene-generation change.
- `AUTOMATON.md` is not present in this checkout; the active lifecycle contract
  is `.agent/.automaton/references/FRAMEWORK.md` plus
  `.agent/.automaton/references/ARTIFACT-LIFECYCLE.md`.

## Scope Coverage

Included:

- PyPI package surface for MapAnything runtime code, CLI, script wrapper, docs,
  tests, and artifact hygiene.
- Release workflow consistency with the intended package version.
- Release-gate checks that prove local assets, vendors, inputs, outputs, and
  Automaton state do not ship in wheel/sdist artifacts.
- AppleGPU/MLX memory-safety evidence for the MapAnything Desk scene path.

Deferred:

- Publishing the release or pushing a tag.
- Training, DA3, Apache-weight support, Gradio parity, and mesh/3DGS export for
  MapAnything.
- Broad performance optimization beyond release-blocking memory-safety issues.

Anti-goals:

- Do not bundle `weights/`, `inputs/`, `outputs/`, `vendors/`, Torch reference
  captures, or scratch visualizers.
- Do not add Torch, TorchVision, UniCeption, OpenCV, CUDA packages, or vendored
  Python code to base runtime dependencies.
