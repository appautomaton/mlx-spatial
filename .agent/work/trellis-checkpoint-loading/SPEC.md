# SPEC: TRELLIS.2 Checkpoint Inspection and MLX Loading

## Bounded Goal

Enable local inspection of real TRELLIS.2 checkpoint assets and loading of selected checkpoint tensors into MLX arrays, while keeping default tests independent of network access, real weights, Hugging Face credentials, PyTorch, Transformers, and vendor imports.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Real model weights must remain local-only and untracked; the existing ignored `weights/trellis2/` convention should continue to be used.
- Default tests must use fake/minimal fixtures and must not require network access, Hugging Face login, real TRELLIS.2 weights, PyTorch, Transformers, vendor imports, or local absolute paths.
- Base dependencies should stay minimal; adding a dependency is acceptable only if it is required for checkpoint file parsing and does not pull in PyTorch or Transformers.
- The first loading path is assumed to be safetensors-first; PyTorch `.pt`/`.pth` checkpoint loading is deferred unless inspection proves the selected TRELLIS.2 files cannot be represented that way.
- The implementation should expose model-neutral checkpoint utilities where practical, with TRELLIS.2-specific asset selection kept in manifest or documentation rather than hard-coded throughout the loader.
- Manual download instructions may use Hugging Face CLI patterns, but runtime code and default tests must not invoke downloads.
- The output should support the next slice: selecting tensors for TRELLIS sparse block or transformer-block parity.

## Required Behavior

- Provide a local checkpoint inspection workflow that reports deterministic tensor metadata: tensor name, shape, dtype, and source file.
- Provide a local loading workflow that converts selected checkpoint tensors into `mlx.core.array` values.
- Support filtering by explicit tensor names and/or name prefixes so a user can inspect or load a subset rather than a full checkpoint.
- Preserve deterministic ordering in metadata reports and test assertions.
- Surface clear `ValueError` or `FileNotFoundError` failures for missing files, unsupported formats, missing requested tensors, and invalid filter inputs.
- Update README documentation with the local weights convention, manual download/placement workflow, inspection command or API example, MLX loading example, and unsupported boundaries.
- Keep TRELLIS.2 asset validation compatible with the previous runtime-readiness manifest while allowing this slice to refine exact checkpoint file expectations if needed.

## Acceptance Criteria

- A checkpoint inspection helper exists and returns structured metadata for a fake/minimal checkpoint fixture in deterministic order.
- A checkpoint loading helper exists and returns selected tensors as MLX arrays with preserved shapes and expected numeric values for fake/minimal fixtures.
- Filter behavior is tested for exact names, prefixes, no matches, and invalid filters.
- Failure behavior is tested for missing checkpoint paths and unsupported checkpoint formats.
- TRELLIS.2-facing documentation explains where to place local checkpoint files under `weights/trellis2/` and how to run inspection/loading locally.
- Default `uv run pytest` passes without real weights, network access, Hugging Face credentials, PyTorch, Transformers, vendor imports, or local absolute paths.
- Optional/manual real-weight verification steps are documented separately from default tests.
- No real checkpoint artifacts are committed.

## Blocking Questions Or Assumptions

- Assumption: start with safetensors support because it can load tensors without PyTorch and aligns with minimal dependency constraints.
- Assumption: if TRELLIS.2 requires `.pt`/`.pth` files, that becomes a follow-up decision for an optional loader path rather than part of the default base runtime.
- Assumption: exact Hugging Face repo/file selection may be refined during planning by inspecting vendor docs or known TRELLIS.2 asset layout, but this spec does not require downloading weights during default verification.

## Anti-Goals

- Do not implement full TRELLIS.2 inference.
- Do not execute TRELLIS sparse blocks, transformer blocks, decoders, mesh extraction, or GLB export.
- Do not add PyTorch, Transformers, Hugging Face Hub, or vendor code as base runtime dependencies.
- Do not require real model weights in tests.
- Do not download weights automatically in tests or import-time code.
- Do not commit model weights or generated checkpoint artifacts.
