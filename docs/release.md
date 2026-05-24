# Release

This is the local checklist for PyPI releases. Publishing itself is a maintainer action.

## Package Metadata

Required state:

- `pyproject.toml` version matches the intended release tag.
- Project name is `mlx-spatial`.
- License metadata points at top-level `LICENSE`.
- Repository URL points at `https://github.com/appautomaton/mlx-spatial`.
- Package CLIs are present for SAM3D, TRELLIS.2, HY-World-2.0, and LiTo.
- Hatch build config excludes local assets, vendors, caches, generated outputs, and agent state.

## Preflight

```bash
uv run pytest -q
uv build
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-*.tar.gz \
  dist/mlx_spatial-*-py3-none-any.whl
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

Smoke checks:

```bash
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
uv run mlx-spatial-lito --help
python scripts/sam3d/reconstruct.py --help
python scripts/lito/generate.py --help
```

When local gated weights are available, run one documented SAM3D reconstruction and inspect the trace:

```bash
python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-release-smoke
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-release-smoke/trace.json
```

If gated weights are not available, record that this smoke was skipped and include the blocker.

## Artifact Boundary

The sdist and wheel must exclude:

```text
.agent/
.claude/
.codex/
.venv/
weights/
inputs/
outputs/
vendors/
dist/
caches
generated probes
```

The artifact checker must pass on both generated files before any publish trigger.

## Trusted Publishing

The workflow lives at:

```text
.github/workflows/workflow.yaml
```

It builds with `uv build` and publishes through `pypa/gh-action-pypi-publish@release/v1` using the `pypi` environment and trusted-publishing OIDC.

## Publish Step

Do not publish from local shell credentials as part of release readiness. A
maintainer should review the worktree and trigger trusted publishing by creating
a GitHub Release for the intended tag or by using the manual workflow dispatch.
Pushing a tag alone does not publish to PyPI.

## Last Local Verification

Last full verification for the initial `0.0.1` release was on 2026-05-22:

```text
uv run pytest -q
647 passed, 5 skipped, 2 warnings

uv build
Successfully built dist/mlx_spatial-0.0.1.tar.gz
Successfully built dist/mlx_spatial-0.0.1-py3-none-any.whl

python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.1.tar.gz dist/mlx_spatial-0.0.1-py3-none-any.whl
checked 2 artifact(s)

python scripts/packaging/check_release_artifacts.py --git-hygiene
git hygiene check passed
```

Lightweight smoke checks passed:

```text
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
python scripts/sam3d/reconstruct.py --help
```

Gated-weight inference smoke completed locally with the bundled SAM3D runtime layout:

```text
python scripts/sam3d/reconstruct.py inputs/sam3d/kidsroom/image.png --mask inputs/sam3d/kidsroom/mask-14.png --output-dir outputs/sam3d/kidsroom-mask14-generated-20260522-160300
completed through ply-export
```

## Post-Publish Check

After publishing:

```bash
python -m pip index versions mlx-spatial
```

Install in a clean environment and run:

```bash
python -c "import mlx_spatial; print(mlx_spatial.__name__)"
mlx-spatial-sam3d --help
```

## 0.0.2 Readiness Notes

The expected delta from `0.0.1` includes the LiTo CLI, checkpoint-backed LiTo image-to-3DGS inference path, binary LiTo PLY export, and the public AppAutomaton LiTo research-weight bundle.

Local `0.0.2` release-candidate verification on 2026-05-24:

```text
uv run pytest -q
763 passed, 5 skipped, 2 warnings

env UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build
Successfully built dist/mlx_spatial-0.0.2.tar.gz
Successfully built dist/mlx_spatial-0.0.2-py3-none-any.whl

python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl
checked 2 artifact(s)

python scripts/packaging/check_release_artifacts.py --git-hygiene
git hygiene check passed
```
