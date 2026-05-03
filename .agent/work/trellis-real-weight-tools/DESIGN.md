# DESIGN: TRELLIS.2 Real-Weight Tooling

## API Boundary

- Add TRELLIS.2-specific orchestration under a new module such as `mlx_spatial.trellis2`.
- Keep generic checkpoint file parsing in `mlx_spatial.checkpoint` unchanged unless a small reusable summary helper is needed.
- Reuse `TRELLIS2_ASSETS` and `validate_model_assets` for asset layout validation.
- Export public TRELLIS.2 helpers from `mlx_spatial.__init__` only after tests define the contract.

## Tooling Shape

- Provide Python APIs first, with a thin command entrypoint only if it maps directly to those APIs.
- The tool should support three operator actions: validate assets, inspect configured checkpoints, and load-probe named selections.
- Real-weight runs are explicit operator actions; default tests use temporary fake roots and tiny safetensors fixtures.
- Hugging Face CLI is available through dev dependencies for operator workflows, not base runtime imports.

## Probe Groups

- Define named probe groups as data, not hard-coded branch logic.
- Each group should include a name, checkpoint relative path, prefixes and/or exact tensor names, and reference note.
- Use conservative groups informed by `vendors/trellis-mac` and cross-checked conceptually against `vendors/TRELLIS.2`.
- Tests should use fake fixture tensor names that exercise group matching without claiming exact real TRELLIS.2 tensor names unless verified.
- Initial groups: `sparse-structure-flow`, `shape-slat-flow`, `texture-slat-flow`, `shape-decoder`, and `texture-decoder`.
- Reference basis: `vendors/trellis-mac/generate.py` loads `microsoft/TRELLIS.2-4B`; `vendors/TRELLIS.2/trellis2/models/__init__.py` defines paired `.json` and `.safetensors` checkpoint loading; `vendors/TRELLIS.2/trellis2/pipelines/base.py` resolves pipeline model entries through `models.from_pretrained`.

## Output Contracts

- Validation returns deterministic asset readiness details.
- Inspection returns deterministic metadata grouped by checkpoint path or probe group.
- Load-probing returns MLX arrays or compact summaries with name, shape, dtype, and source group in deterministic order.
- CLI output, if added, should be stable plain text or JSON-like data suitable for terminal use.

## Dependency Boundary

- No Hugging Face Hub runtime dependency.
- `huggingface_hub` is allowed in dev dependencies only to provide the `huggingface-cli` command.
- No PyTorch or Transformers dependency.
- No vendor imports.
- No automatic downloads.
- Safetensors remains the only checkpoint format for this slice.
