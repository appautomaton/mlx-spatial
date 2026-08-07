#!/usr/bin/env python3
"""Validate mlx-spatial release artifacts and local release hygiene."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


BLOCKED_PATH_PARTS = {
    ".agent",
    ".antigravitycli",
    ".claude",
    ".codex",
    ".opencode",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "inputs",
    "outputs",
    "vendors",
    "weights",
    "__pycache__",
}

BLOCKED_SUFFIXES = {".pyc", ".pyo", ".DS_Store"}
BLOCKED_COMPONENT_SUFFIXES = {".egg-info"}

GIT_HYGIENE_BLOCKED_PATH_PARTS = BLOCKED_PATH_PARTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="*", type=Path, help="sdist .tar.gz or wheel .whl files to inspect")
    parser.add_argument("--git-hygiene", action="store_true", help="check for common generated files in git status")
    args = parser.parse_args(argv)

    errors: list[str] = []
    for artifact in args.artifacts:
        errors.extend(_check_artifact(artifact))
    if args.git_hygiene:
        errors.extend(_check_git_hygiene())

    if errors:
        for error in errors:
            print(f"release-artifact-error: {error}", file=sys.stderr)
        return 1
    if args.artifacts:
        print(f"checked {len(args.artifacts)} artifact(s)")
    if args.git_hygiene:
        print("git hygiene check passed")
    if not args.artifacts and not args.git_hygiene:
        parser.print_help()
    return 0


def _check_artifact(path: Path) -> list[str]:
    if not path.is_file():
        return [f"missing artifact: {path}"]
    if path.suffix == ".whl":
        names = _wheel_names(path)
    elif path.name.endswith(".tar.gz"):
        names = _tar_names(path)
    else:
        return [f"unsupported artifact type: {path}"]

    errors: list[str] = []
    for name in names:
        normalized = name.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        blocked = BLOCKED_PATH_PARTS.intersection(parts)
        if blocked:
            errors.append(f"{path}: blocked path component {sorted(blocked)} in {name}")
        blocked_suffix_parts = [
            part for part in parts if any(part.endswith(suffix) for suffix in BLOCKED_COMPONENT_SUFFIXES)
        ]
        if blocked_suffix_parts:
            errors.append(f"{path}: blocked path component {blocked_suffix_parts} in {name}")
        if any(normalized.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
            errors.append(f"{path}: blocked generated file {name}")
        if normalized.startswith("mlx_spatialkit/") or "/mlx_spatialkit/" in normalized:
            errors.append(f"{path}: obsolete standalone mlx_spatialkit package path {name}")
    if path.suffix == ".whl":
        required_suffixes = (
            "mlx_spatial/spatialkit/__init__.py",
            "mlx_spatial/spatialkit/texture_bake.metal",
        )
        for suffix in required_suffixes:
            if not any(name.replace("\\", "/").endswith(suffix) for name in names):
                errors.append(f"{path}: missing integrated SpatialKit resource {suffix}")
        if not any(
            name.replace("\\", "/").startswith("mlx_spatial/spatialkit/_native")
            and name.endswith(".so")
            for name in names
        ):
            errors.append(f"{path}: missing integrated SpatialKit native extension")
    elif path.name.endswith(".tar.gz"):
        required_suffixes = (
            "/CMakeLists.txt",
            "/native/spatialkit/cpp/bindings.cpp",
            "/native/spatialkit/metal/kernels/texture_bake.metal",
            "/src/mlx_spatial/spatialkit/__init__.py",
        )
        for suffix in required_suffixes:
            if not any(name.replace("\\", "/").endswith(suffix) for name in names):
                errors.append(f"{path}: missing integrated SpatialKit source {suffix}")
    return errors


def _wheel_names(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def _tar_names(path: Path) -> list[str]:
    with tarfile.open(path, mode="r:gz") as archive:
        return archive.getnames()


def _check_git_hygiene() -> list[str]:
    result = subprocess.run(["git", "status", "--short"], check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return [result.stderr.strip() or "git status failed"]

    errors: list[str] = []
    for line in result.stdout.splitlines():
        path = line[3:].strip()
        if not path:
            continue
        normalized = path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        blocked = GIT_HYGIENE_BLOCKED_PATH_PARTS.intersection(parts)
        if normalized == ".agent/work" or normalized.startswith(".agent/work/"):
            blocked.discard(".agent")
        if blocked:
            errors.append(f"git status includes generated/local path {path}")
        if any(part.endswith(suffix) for part in parts for suffix in BLOCKED_COMPONENT_SUFFIXES):
            errors.append(f"git status includes generated/local path {path}")
        if any(normalized.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
            errors.append(f"git status includes generated file {path}")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
