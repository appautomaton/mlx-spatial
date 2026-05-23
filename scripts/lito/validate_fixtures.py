"""Validate LiTo parity fixtures without importing vendor or MLX modules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from safetensors import safe_open


REQUIRED_GROUPS = ("tokenizer", "condition", "dit", "render")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class FileReport:
    path: Path
    kind: str
    count: int
    detail: str


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing manifest: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {manifest_path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    return data


def _group_files(group: str, entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        raise ValueError(f"{group}: manifest entry must be an object")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{group}: files must be a non-empty list")

    paths: list[str] = []
    for index, item in enumerate(files):
        if isinstance(item, str):
            paths.append(item)
            continue
        if isinstance(item, dict):
            path = item.get("path") or item.get("file") or item.get("name")
            if isinstance(path, str):
                paths.append(path)
                continue
        raise ValueError(f"{group}: files[{index}] must be a path string or object with path/file/name")
    return paths


def _resolve_fixture(root: Path, relative_path: str) -> Path:
    fixture_path = Path(relative_path)
    if fixture_path.is_absolute() or ".." in fixture_path.parts:
        raise ValueError(f"fixture path must stay inside {root}: {relative_path}")
    resolved = (root / fixture_path).resolve()
    root_resolved = root.resolve()
    if root_resolved not in (resolved, *resolved.parents):
        raise ValueError(f"fixture path escapes {root}: {relative_path}")
    return resolved


def _validate_safetensors(path: Path) -> FileReport:
    try:
        with safe_open(path, framework="numpy") as tensors:
            keys = list(tensors.keys())
            if not keys:
                raise ValueError("contains no tensors")
            first = tensors.get_tensor(keys[0])
            detail = f"{keys[0]} shape={list(first.shape)} dtype={first.dtype}"
    except Exception as error:
        raise ValueError(f"{path}: invalid safetensors fixture: {error}") from error
    return FileReport(path=path, kind="safetensors", count=len(keys), detail=detail)


def _validate_image(path: Path) -> FileReport:
    try:
        with Image.open(path) as image:
            image.verify()
            detail = f"mode={image.mode} size={image.size[0]}x{image.size[1]}"
    except Exception as error:
        raise ValueError(f"{path}: invalid image fixture: {error}") from error
    return FileReport(path=path, kind="image", count=1, detail=detail)


def validate(root: Path) -> list[FileReport]:
    root = root.expanduser()
    if not root.is_dir():
        raise ValueError(f"fixture root is not a directory: {root}")

    manifest = _load_manifest(root)
    missing_groups = [group for group in REQUIRED_GROUPS if group not in manifest]
    if missing_groups:
        raise ValueError(f"manifest missing required groups: {', '.join(missing_groups)}")

    reports: list[FileReport] = []
    for group in REQUIRED_GROUPS:
        for relative_path in _group_files(group, manifest[group]):
            path = _resolve_fixture(root, relative_path)
            if not path.is_file():
                raise ValueError(f"{group}: missing fixture file: {path}")
            suffix = path.suffix.lower()
            if suffix == ".safetensors":
                reports.append(_validate_safetensors(path))
            elif suffix in IMAGE_SUFFIXES:
                reports.append(_validate_image(path))
            else:
                raise ValueError(f"{group}: unsupported fixture extension for {path}")
    return reports


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate LiTo parity fixtures and manifest.")
    parser.add_argument("root", nargs="?", default="tests/fixtures/lito", help="fixture root containing manifest.json")
    parser.add_argument("--verbose", action="store_true", help="print one line per validated fixture")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        reports = validate(Path(args.root))
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.verbose:
        for report in reports:
            print(f"{report.path.name}: {report.kind} entries={report.count} {report.detail}")
    print(f"OK: validated {len(reports)} LiTo fixture files across {len(REQUIRED_GROUPS)} groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
