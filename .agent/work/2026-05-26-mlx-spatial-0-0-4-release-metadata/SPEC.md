# mlx-spatial 0.0.4 Release Metadata Spec

## Objective

Make the current Pixal3D end-to-end generation state tag-ready without reusing
an existing release tag or package version.

## Context

- `v0.0.2` exists on the remote.
- A local `v0.0.3` tag already points at an earlier Pixal3D checkpoint.
- The current branch has new commits after that tag, including the completed
  Pixal3D 1024 cascade GLB smoke.

## Acceptance Criteria

- `pyproject.toml` declares `0.0.4`.
- `uv.lock` matches the project version.
- A clean build produces only `mlx_spatial-0.0.4` wheel/sdist artifacts.
- Tests and release artifact hygiene pass after the version change.
- No tag is created in this change; tagging remains a maintainer action.
