"""MLX-native safetensors I/O and lightweight header inspection."""

from __future__ import annotations

import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import mlx.core as mx
import numpy as np


_HEADER_LENGTH = struct.Struct("<Q")
_MAX_HEADER_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class SafetensorHeader:
    """Metadata for one tensor without materializing its payload."""

    name: str
    shape: tuple[int, ...]
    dtype: str


def inspect_safetensors(path: str | Path) -> tuple[SafetensorHeader, ...]:
    """Read safetensors names, shapes, and dtypes from the JSON header."""

    header, data_size = _read_safetensors_header(path)
    checkpoint = Path(path)

    infos: list[SafetensorHeader] = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(value, dict):
            raise ValueError(f"invalid safetensors tensor entry in {checkpoint}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError(f"invalid safetensors descriptor for {name!r} in {checkpoint}")
        if any(not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ValueError(f"invalid safetensors shape for {name!r} in {checkpoint}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > data_size
        ):
            raise ValueError(f"invalid safetensors data offsets for {name!r} in {checkpoint}")
        infos.append(SafetensorHeader(name=name, shape=tuple(shape), dtype=dtype))
    return tuple(sorted(infos, key=lambda info: info.name))


def read_safetensors_metadata(path: str | Path) -> dict[str, str]:
    """Read string metadata from a safetensors header without loading tensors."""

    header, _ = _read_safetensors_header(path)
    checkpoint = Path(path)
    metadata = header.get("__metadata__", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()
    ):
        raise ValueError(f"invalid safetensors metadata in {checkpoint}")
    return dict(metadata)


def _read_safetensors_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """Return the decoded header and payload size for one safetensors file."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"safetensors file not found: {checkpoint}")

    file_size = checkpoint.stat().st_size
    with checkpoint.open("rb") as handle:
        raw_length = handle.read(_HEADER_LENGTH.size)
        if len(raw_length) != _HEADER_LENGTH.size:
            raise ValueError(f"invalid safetensors header in {checkpoint}")
        (header_size,) = _HEADER_LENGTH.unpack(raw_length)
        if header_size <= 0 or header_size > _MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header size in {checkpoint}: {header_size}")
        data_start = _HEADER_LENGTH.size + header_size
        if data_start > file_size:
            raise ValueError(f"truncated safetensors header in {checkpoint}")
        raw_header = handle.read(header_size)

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid safetensors JSON header in {checkpoint}: {error}") from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header root in {checkpoint}")
    return header, file_size - data_start


def load_mlx_safetensors(path: str | Path) -> dict[str, mx.array]:
    """Load a safetensors file through MLX."""

    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"safetensors file not found: {checkpoint}")
    loaded = mx.load(str(checkpoint))
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a tensor mapping in {checkpoint}")
    return {str(name): tensor for name, tensor in loaded.items()}


def load_numpy_safetensors(
    path: str | Path,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load selected safetensors payloads through MLX and convert them to NumPy."""

    exact_names = set(names or ())
    name_prefixes = tuple(prefixes or ())
    tensors = load_mlx_safetensors(path)
    selected = {
        name: np.asarray(tensor)
        for name, tensor in tensors.items()
        if (
            (not exact_names and not name_prefixes)
            or name in exact_names
            or any(name.startswith(prefix) for prefix in name_prefixes)
        )
    }
    if exact_names:
        missing = sorted(exact_names.difference(selected))
        if missing:
            raise ValueError(f"safetensors file is missing requested tensors: {missing}")
    return selected


def save_safetensors(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Save NumPy or MLX arrays with MLX's native safetensors writer."""

    if not arrays:
        raise ValueError("cannot save an empty safetensors mapping")
    tensors: dict[str, mx.array] = {}
    for name, value in arrays.items():
        tensor_name = str(name)
        if isinstance(value, mx.array):
            tensors[tensor_name] = value
            continue

        source = np.asarray(value)
        tensor = mx.array(source)
        converted_dtype = np.dtype(str(tensor.dtype).rsplit(".", maxsplit=1)[-1])
        if converted_dtype != source.dtype:
            raise ValueError(
                f"MLX cannot preserve dtype {source.dtype} for tensor {tensor_name!r}; "
                f"it would be stored as {converted_dtype}"
            )
        tensors[tensor_name] = tensor
    destination = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".safetensors",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        mx.save_safetensors(
            str(temporary),
            tensors,
            metadata=dict(metadata) if metadata is not None else None,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
