# Plan: mlx-spatial 0.0.4 Release Metadata

## Goal

Execute [SPEC.md](SPEC.md): move package metadata to the next valid release
version for the current Pixal3D end-to-end state.

## Slice 1: Version And Release Gates

**Objective:** Bump package metadata to `0.0.4` and verify the tag-ready
artifact boundary.

**Acceptance criteria:**
- `pyproject.toml` and `uv.lock` agree on `0.0.4`.
- Full tests pass after the metadata bump.
- `uv build` creates `dist/mlx_spatial-0.0.4.tar.gz` and
  `dist/mlx_spatial-0.0.4-py3-none-any.whl`.
- Release artifact and git hygiene checks pass.

**Status:** complete

**Evidence:** bumped `pyproject.toml` to `0.0.4` and refreshed `uv.lock`;
`rg -n "name = \"mlx-spatial\"|version = \"0.0.4\"" pyproject.toml uv.lock`
confirmed both files agree. Verification passed: `uv lock --check`,
`git diff --check`, and `uv run pytest -q` -> 891 passed, 10 skipped, 27
deselected, 2 known HyWorld2 warnings. Release build and artifact hygiene
passed: `rm -rf dist && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build &&
python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-*.tar.gz
dist/mlx_spatial-*-py3-none-any.whl && python
scripts/packaging/check_release_artifacts.py --git-hygiene` built and checked
`dist/mlx_spatial-0.0.4.tar.gz` and
`dist/mlx_spatial-0.0.4-py3-none-any.whl`.
