"""LiTo asset validation, checkpoint inspection, and local conversion helpers."""

from __future__ import annotations

import argparse
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint
from .safetensors_io import save_safetensors
from .torch_checkpoint import load_torch_zip_state_dict


LITO_REPO_ID = "apple/ml-lito"
LITO_CDN_BASE_URL = "https://ml-site.cdn-apple.com/models/lito"
LITO_RAW_DEFAULT_ROOT = "weights/lito-raw"
LITO_DEFAULT_ROOT = "weights/lito-research-mlx"
LITO_TRELLIS_REPO_ID = "microsoft/TRELLIS-image-large"
LITO_TRELLIS_BUNDLE_PATH = Path("dependencies/trellis")
LITO_COMPONENT_GROUPS = (
    "tokenizer",
    "image_conditioner",
    "dit",
    "gaussian_decoder",
)
LITO_DEFAULT_CHECKPOINTS = (
    ("tokenizer", "lito_new.ckpt", "tokenizer/lito_new.safetensors"),
    ("image_to_3d", "lito_dit_rgba.ckpt", "image_to_3d/lito_dit_rgba.safetensors"),
)
LITO_TRELLIS_REQUIRED_FILES = (
    "ckpts/ss_dec_conv3d_16l8_fp16.json",
    "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
)
LITO_TRELLIS_METADATA_FILES = (
    "LICENSE",
    "SOURCE.json",
)
LITO_MODEL_LICENSE = "Apple Machine Learning Research Model License Agreement"
LITO_SAMPLE_LICENSE = "CC BY-NC-ND 4.0"


@dataclass(frozen=True)
class LitoAssetValidation:
    """Deterministic presence report for a local LiTo safetensors root."""

    root: Path
    checkpoint_paths: tuple[Path, ...]
    runtime_dependency_paths: tuple[Path, ...]
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


def validate(
    root: str | Path = LITO_DEFAULT_ROOT,
    *,
    include_runtime_dependencies: bool = True,
) -> LitoAssetValidation:
    """Validate a LiTo root and, by default, all checkpoint-backed runtime assets."""

    root_path = Path(root)
    checkpoint_paths = tuple(root_path / relative_path for _, _, relative_path in LITO_DEFAULT_CHECKPOINTS)
    runtime_dependency_paths: tuple[Path, ...] = ()
    if include_runtime_dependencies:
        trellis_root = lito_trellis_root(root_path)
        runtime_dependency_paths = tuple(trellis_root / relative_path for relative_path in LITO_TRELLIS_REQUIRED_FILES)
    present: list[str] = []
    missing: list[str] = []
    for path in (*checkpoint_paths, *runtime_dependency_paths):
        relative = _relative_report_path(root_path, path)
        if path.is_file():
            present.append(relative)
        else:
            missing.append(relative)

    return LitoAssetValidation(
        root=root_path,
        checkpoint_paths=checkpoint_paths,
        runtime_dependency_paths=runtime_dependency_paths,
        present=tuple(present),
        missing=tuple(missing),
    )


def inspect(
    root: str | Path = LITO_DEFAULT_ROOT,
    prefixes: Iterable[str] | None = None,
    limit: int = 20,
) -> list[CheckpointTensorInfo]:
    """Inspect tensors across the local LiTo safetensors checkpoints."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    validation = validate(root, include_runtime_dependencies=False)
    if not validation.ready:
        raise FileNotFoundError(f"missing LiTo checkpoint assets: {', '.join(validation.missing)}")

    normalized_prefixes = _normalize_prefixes(prefixes)
    infos: list[CheckpointTensorInfo] = []
    for path in validation.checkpoint_paths:
        for info in inspect_checkpoint(path):
            if normalized_prefixes is not None and not any(info.name.startswith(prefix) for prefix in normalized_prefixes):
                continue
            infos.append(info)
            if len(infos) >= limit:
                break
        if len(infos) >= limit:
            break
    if normalized_prefixes is not None and not infos:
        raise ValueError("checkpoint filters matched no tensors")
    return infos[:limit]


def download_command(root: str | Path = LITO_RAW_DEFAULT_ROOT) -> str:
    """Return shell commands for downloading the official Apple LiTo checkpoints."""

    root_path = Path(root)
    commands = [f"mkdir -p {shlex.quote(str(root_path))}"]
    for _, raw_name, _ in LITO_DEFAULT_CHECKPOINTS:
        output = root_path / raw_name
        url = f"{LITO_CDN_BASE_URL}/{raw_name}"
        commands.append(f"curl -L {shlex.quote(url)} -o {shlex.quote(str(output))}")
    return "\n".join(commands)


def convert(
    src: str | Path,
    dst: str | Path = LITO_DEFAULT_ROOT,
    *,
    overwrite: bool = False,
    max_archive_bytes: int | None = 16 * 1024**3,
    max_tensor_bytes: int | None = 16 * 1024**3,
) -> None:
    """Convert official LiTo `.ckpt` files to a local safetensors mirror."""

    source = Path(src)
    output = Path(dst)
    if source.is_dir():
        for _, raw_name, relative_output in LITO_DEFAULT_CHECKPOINTS:
            _convert_one_checkpoint(
                source / raw_name,
                output / relative_output,
                overwrite=overwrite,
                max_archive_bytes=max_archive_bytes,
                max_tensor_bytes=max_tensor_bytes,
            )
        return

    output_path = output if output.suffix == ".safetensors" else output / source.with_suffix(".safetensors").name
    _convert_one_checkpoint(
        source,
        output_path,
        overwrite=overwrite,
        max_archive_bytes=max_archive_bytes,
        max_tensor_bytes=max_tensor_bytes,
    )


def _convert_one_checkpoint(
    source: Path,
    output: Path,
    *,
    overwrite: bool,
    max_archive_bytes: int | None,
    max_tensor_bytes: int | None,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"LiTo source checkpoint not found: {source}")
    if source.suffix == ".safetensors":
        _copy_file(source, output, overwrite=overwrite)
        return
    if source.suffix not in {".ckpt", ".pt", ".pth", ".bin"}:
        raise ValueError(f"unsupported LiTo checkpoint conversion format: {source.suffix or '<none>'}")
    if output.exists() and not overwrite:
        inspect_checkpoint(output)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = load_torch_zip_state_dict(
        source,
        max_archive_bytes=max_archive_bytes,
        max_tensor_bytes=max_tensor_bytes,
    )
    save_safetensors(output, arrays)
    metadata_output = output.parent / "conversion_metadata" / f"{output.stem}.yaml"
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        f"source: {source.as_posix()}\n"
        "converter: mlx-spatial-restricted-torch-zip\n"
        f"tensor_count: {len(arrays)}\n",
        encoding="utf-8",
    )


def _copy_file(source: Path, output: Path, *, overwrite: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        inspect_checkpoint(output)
        return
    shutil.copy2(source, output)


def _relative_report_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def lito_trellis_root(root: str | Path = LITO_DEFAULT_ROOT) -> Path:
    """Return the bundle-local TRELLIS decoder root for LiTo inference."""

    return Path(root) / LITO_TRELLIS_BUNDLE_PATH


def _normalize_prefixes(prefixes: Iterable[str] | None) -> tuple[str, ...] | None:
    if prefixes is None:
        return None
    if isinstance(prefixes, str):
        raise ValueError("prefixes must be an iterable of non-empty strings")
    normalized = tuple(sorted(set(prefixes)))
    if not normalized or any(not isinstance(prefix, str) or not prefix for prefix in normalized):
        raise ValueError("prefixes must contain only non-empty strings")
    return normalized


def _cmd_validate(args: argparse.Namespace) -> int:
    validation = validate(args.root)
    print(f"root={validation.root}")
    if validation.present:
        print("present=" + ",".join(validation.present))
    if validation.missing:
        print("missing=" + ",".join(validation.missing))
        return 1
    print("OK")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        infos = inspect(args.root, prefixes=args.prefix, limit=args.limit)
    except (FileNotFoundError, ValueError) as error:
        print(f"error={error}")
        return 1
    for info in infos:
        print(f"{info.source}:{info.name} shape={info.shape} dtype={info.dtype}")
    return 0


def _cmd_download_command(args: argparse.Namespace) -> int:
    print(download_command(args.root))
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    convert(args.src, args.dst, overwrite=args.overwrite)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LiTo asset utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate converted LiTo safetensors")
    validate_parser.add_argument("root", nargs="?", default=LITO_DEFAULT_ROOT)
    validate_parser.set_defaults(func=_cmd_validate)

    inspect_parser = subparsers.add_parser("inspect", help="inspect converted LiTo safetensors metadata")
    inspect_parser.add_argument("root", nargs="?", default=LITO_DEFAULT_ROOT)
    inspect_parser.add_argument("--prefix", action="append")
    inspect_parser.add_argument("--limit", type=int, default=20)
    inspect_parser.set_defaults(func=_cmd_inspect)

    download_parser = subparsers.add_parser("download-command", help="print Apple CDN download commands")
    download_parser.add_argument("root", nargs="?", default=LITO_RAW_DEFAULT_ROOT)
    download_parser.set_defaults(func=_cmd_download_command)

    convert_parser = subparsers.add_parser("convert", help="convert official LiTo checkpoints to safetensors")
    convert_parser.add_argument("src")
    convert_parser.add_argument("dst", nargs="?", default=LITO_DEFAULT_ROOT)
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.set_defaults(func=_cmd_convert)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
