# Release

This is the local checklist for PyPI releases. Publishing itself is a maintainer action through GitHub trusted publishing.

## Package Metadata

Required state:

- `pyproject.toml` version matches the intended release tag.
- Project name is `mlx-spatial`.
- License metadata points at top-level `LICENSE`.
- Repository URL points at `https://github.com/appautomaton/mlx-spatial`.
- Package CLIs are present for SAM3D, TRELLIS.2, HY-World-2.0, LiTo, MapAnything, and Pixal3D.
- Hatch build config excludes local assets, vendors, caches, generated outputs, and agent state.

## Preflight

```bash
uv run pytest -q
rm -rf dist
uv build
python scripts/packaging/check_release_artifacts.py \
  dist/mlx_spatial-*.tar.gz \
  dist/mlx_spatial-*-py3-none-any.whl
python scripts/packaging/check_release_artifacts.py --git-hygiene
```

Use `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache` if the default home cache is not writable in a sandboxed session.

Smoke checks:

```bash
uv run mlx-spatial-sam3d --help
uv run mlx-spatial-trellis2 --help
uv run mlx-spatial-hyworld2 --help
uv run mlx-spatial-lito --help
uv run mlx-spatial-mapanything --help
uv run mlx-spatial-pixal3d --help
python scripts/sam3d/reconstruct.py --help
python scripts/trellis2/generate_textured.py --help
python scripts/hyworld2/generate_scene.py --help
python scripts/lito/generate.py --help
python scripts/mapanything/generate_scene.py --help
python scripts/pixal3d/generate.py --help
```

When local gated weights are available, run one documented SAM3D reconstruction and inspect the trace:

```bash
python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-release-smoke
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-release-smoke/trace.json
```

If gated weights are not available, record that this smoke was skipped and include the structured blocker.

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

## Post-Publish Check

After publishing:

```bash
python -m pip index versions mlx-spatial
```

Install in a clean environment and run:

```bash
python -c "import mlx_spatial; print(mlx_spatial.__name__)"
mlx-spatial-sam3d --help
mlx-spatial-mapanything --help
mlx-spatial-pixal3d --help
```

## Evidence

Do not leave dated release-run transcripts in this stable checklist. Put release-specific evidence in the GitHub Release notes, PR description, or `.agent/work/<change>/` artifact for that release.
