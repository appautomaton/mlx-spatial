# Slice 7 Implementer: End-To-End Textured Generation Integration

## Scope

Connected the baked texture payload and GLB writer into the `generate-textured` command path.

## Changes

- `generate_textured_glb` now writes the baked texture payload through `write_trellis2_textured_glb`.
- On writer success, the trace appends a `textured_glb` metadata output, marks `mesh-export` complete, returns the artifact, and reports `ready=True`.
- Writer `ValueError` and `OSError` failures now return structured `mesh-export` blockers while preserving bake outputs.
- Added fixture integration tests for:
  - ready/artifact success
  - real non-empty GLB write from fake TRELLIS fixture stages
  - writer ValueError/OSError blockers
  - route metadata still reaching success for `512` and `1536_cascade`
- Fixed `generate-shape` CLI flag forwarding for `--pipeline-type`, `--seed`, and `--max-num-tokens`.

## Verification

- `uv run pytest -q tests/test_trellis2_inference.py tests/test_trellis2_export.py` -> `54 passed`
- `uv run pytest -q tests/test_trellis2_tools.py tests/test_trellis2_inference.py tests/test_trellis2_export.py` -> `68 passed`
- `uv run pytest -q` -> `232 passed, 5 skipped`
- `git diff --check` -> passed

