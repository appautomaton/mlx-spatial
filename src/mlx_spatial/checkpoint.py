"""Local checkpoint inspection and selected tensor loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


@dataclass(frozen=True)
class CheckpointTensorInfo:
    """Metadata for one tensor in a local checkpoint file."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    source: str


def inspect_checkpoint(
    path: str | Path,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> tuple[CheckpointTensorInfo, ...]:
    """Return deterministic tensor metadata for a local safetensors checkpoint."""

    checkpoint_path = _validate_checkpoint_path(path)
    exact_names, name_prefixes, has_filter = _normalize_filters(names, prefixes)

    infos: list[CheckpointTensorInfo] = []
    with safe_open(checkpoint_path, framework="mlx") as tensors:
        for name in sorted(tensors.keys()):
            if has_filter and not _matches_filter(name, exact_names, name_prefixes):
                continue
            tensor_slice = tensors.get_slice(name)
            infos.append(
                CheckpointTensorInfo(
                    name=name,
                    shape=tuple(tensor_slice.get_shape()),
                    dtype=str(tensor_slice.get_dtype()),
                    source=str(checkpoint_path),
                )
            )

    if has_filter and not infos:
        raise ValueError("checkpoint filters matched no tensors")
    return tuple(infos)


def load_checkpoint_tensors(
    path: str | Path,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> dict[str, mx.array]:
    """Load selected tensors from a local safetensors checkpoint as MLX arrays."""

    import mlx.core as mx

    checkpoint_path = _validate_checkpoint_path(path)
    exact_names, name_prefixes, has_filter = _normalize_filters(names, prefixes)
    if not has_filter:
        raise ValueError("loading checkpoint tensors requires names or prefixes")

    loaded: dict[str, mx.array] = {}
    with safe_open(checkpoint_path, framework="mlx") as tensors:
        for name in sorted(tensors.keys()):
            if _matches_filter(name, exact_names, name_prefixes):
                try:
                    loaded[name] = tensors.get_tensor(name)
                except TypeError as error:
                    if "bfloat16" not in str(error):
                        raise
                    return _load_checkpoint_tensors_with_mlx_load(
                        checkpoint_path,
                        exact_names,
                        name_prefixes,
                    )

    if not loaded:
        raise ValueError("checkpoint filters matched no tensors")
    if exact_names:
        missing = sorted(exact_names.difference(loaded))
        if missing:
            raise ValueError(f"checkpoint is missing requested tensors: {missing}")
    return loaded


def _load_checkpoint_tensors_with_mlx_load(
    checkpoint_path: Path,
    exact_names: set[str],
    prefixes: tuple[str, ...],
) -> dict[str, mx.array]:
    import mlx.core as mx

    tensors = mx.load(str(checkpoint_path))
    loaded = {
        name: tensors[name]
        for name in sorted(tensors)
        if _matches_filter(name, exact_names, prefixes)
    }
    if not loaded:
        raise ValueError("checkpoint filters matched no tensors")
    if exact_names:
        missing = sorted(exact_names.difference(loaded))
        if missing:
            raise ValueError(f"checkpoint is missing requested tensors: {missing}")
    return loaded


def _validate_checkpoint_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint_path}")
    if checkpoint_path.suffix != ".safetensors":
        raise ValueError(f"unsupported checkpoint format: {checkpoint_path.suffix or '<none>'}")
    return checkpoint_path


def _normalize_filters(
    names: Iterable[str] | None,
    prefixes: Iterable[str] | None,
) -> tuple[set[str], tuple[str, ...], bool]:
    exact_names = _normalize_string_filter(names, "names")
    name_prefixes = tuple(sorted(_normalize_string_filter(prefixes, "prefixes")))
    return exact_names, name_prefixes, bool(exact_names or name_prefixes)


def _normalize_string_filter(values: Iterable[str] | None, label: str) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raise ValueError(f"{label} must be an iterable of non-empty strings")

    normalized = set(values)
    if not normalized:
        raise ValueError(f"{label} must not be empty when provided")
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{label} must contain only non-empty strings")
    return normalized


def _matches_filter(name: str, exact_names: set[str], prefixes: tuple[str, ...]) -> bool:
    return name in exact_names or any(name.startswith(prefix) for prefix in prefixes)
