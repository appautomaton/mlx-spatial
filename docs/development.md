# Development

## Setup

```bash
uv sync
uv run pytest -q
```

Use Python 3.11+ on Apple Silicon with MLX installed through the project dependencies.

## Targeted Checks

SAM3D-focused checks:

```bash
uv run pytest tests/test_sam3d*.py -q
python scripts/sam3d/reconstruct.py --help
python scripts/sam3d/inspect_trace.py --help
```

Package checks:

```bash
uv build
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-0.0.1.tar.gz \
  dist/mlx_spatial-0.0.1-py3-none-any.whl
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

CLI smoke checks:

```bash
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
```

## Local Assets

Keep large and gated assets out of git:

```text
weights/
inputs/
outputs/
vendors/
```

Tests should pass without downloading gated weights unless they are explicitly marked as optional parity or local-inference checks. Runtime commands that need weights should fail with structured blockers instead of fabricating outputs.

## Editing Constraints

- Prefer existing module boundaries over new abstractions.
- Keep model-specific behavior inside the relevant `sam3d_*`, `trellis2_*`, or `hyworld2_*` modules.
- Keep shared primitives model-neutral.
- Do not add generated outputs, converted weights, vendor checkouts, or agent state to package artifacts.
- Use structured parsers for model metadata and safetensors; avoid ad hoc parsing when a local helper exists.
- Keep scripts self-documented with argparse and stable defaults.

## Reference Parity

Reference parity work belongs in targeted tests or dev-only scripts. It should record:

- source checkpoint path
- converted checkpoint path
- tensor count
- missing or extra tensors
- shape mismatches
- max absolute difference

Store heavy audit outputs with ignored model bundles unless the file is small and intentionally part of docs.

## Worktree Hygiene

This repo often has local generated files and experimental model outputs. Before release review:

```bash
git status --short
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

Separate release-readiness changes from unrelated pipeline implementation changes. Do not revert unrelated dirty files unless that is the explicit task.
