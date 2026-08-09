"""Restricted reader for tensor-only PyTorch ZIP checkpoints."""

from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class _TorchStorageType:
    def __init__(self, name: str):
        self.name = name


@dataclass(frozen=True)
class _TorchStorageRef:
    storage_type: _TorchStorageType
    key: str
    location: str
    size: int


@dataclass(frozen=True)
class _TorchTensorRef:
    storage: _TorchStorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor_v2(
    storage: _TorchStorageRef,
    storage_offset: int,
    size: Sequence[int],
    stride: Sequence[int],
    requires_grad: bool,
    backward_hooks: object,
) -> _TorchTensorRef:
    del requires_grad, backward_hooks
    return _TorchTensorRef(
        storage=storage,
        storage_offset=int(storage_offset),
        size=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
    )


class _RestrictedTorchZipUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _TorchStorageType(name)
        raise pickle.UnpicklingError(f"unsupported PyTorch checkpoint global: {module}.{name}")

    def persistent_load(self, persistent_id: object) -> _TorchStorageRef:
        if not isinstance(persistent_id, tuple) or len(persistent_id) < 5:
            raise pickle.UnpicklingError(f"unsupported persistent id: {persistent_id!r}")
        tag, storage_type, key, location, size = persistent_id[:5]
        if tag != "storage" or not isinstance(storage_type, _TorchStorageType):
            raise pickle.UnpicklingError(f"unsupported storage persistent id: {persistent_id!r}")
        return _TorchStorageRef(
            storage_type=storage_type,
            key=str(key),
            location=str(location),
            size=int(size),
        )


_TORCH_STORAGE_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("<i1"),
    "ByteStorage": np.dtype("<u1"),
    "BoolStorage": np.dtype("?"),
}


@dataclass(frozen=True)
class TorchCheckpointTensorInfo:
    """Tensor metadata available from a PyTorch ZIP without loading storage."""

    name: str
    shape: tuple[int, ...]
    dtype: str


def inspect_torch_zip_checkpoint(
    source: str | Path,
    *,
    max_archive_bytes: int | None = 16 * 1024**3,
    max_tensor_bytes: int | None = 16 * 1024**3,
) -> tuple[TorchCheckpointTensorInfo, ...]:
    """Inspect tensor names, shapes, and dtypes without reading tensor payloads."""

    checkpoint = _validate_checkpoint(source, max_archive_bytes=max_archive_bytes)
    with zipfile.ZipFile(checkpoint) as archive:
        state_dict = _load_pickled_state_dict(archive)
        infos: list[TorchCheckpointTensorInfo] = []
        for name, tensor in state_dict.items():
            if not isinstance(tensor, _TorchTensorRef):
                continue
            dtype = _TORCH_STORAGE_DTYPES.get(tensor.storage.storage_type.name)
            if dtype is None:
                raise ValueError(f"unsupported PyTorch storage type: {tensor.storage.storage_type.name}")
            _validate_tensor_layout(tensor)
            storage_bytes = tensor.storage.size * dtype.itemsize
            if max_tensor_bytes is not None and storage_bytes > max_tensor_bytes:
                raise ValueError(
                    f"tensor storage {tensor.storage.key!r} is {storage_bytes} bytes, "
                    f"limit is {max_tensor_bytes}"
                )
            infos.append(
                TorchCheckpointTensorInfo(
                    name=str(name),
                    shape=tensor.size,
                    dtype=dtype.name,
                )
            )
        if not infos:
            raise ValueError("restricted PyTorch checkpoint reader found no tensor state_dict entries")
        return tuple(sorted(infos, key=lambda info: info.name))


def load_torch_zip_state_dict(
    source: str | Path,
    *,
    max_archive_bytes: int | None = 16 * 1024**3,
    max_tensor_bytes: int | None = 16 * 1024**3,
) -> dict[str, np.ndarray]:
    """Load tensor values without importing Torch or executing checkpoint code."""

    checkpoint = _validate_checkpoint(source, max_archive_bytes=max_archive_bytes)

    with zipfile.ZipFile(checkpoint) as archive:
        state_dict = _load_pickled_state_dict(archive)
        data_pickle = _torch_zip_data_pickle_name(archive)
        prefix = data_pickle[: -len("/data.pkl")] if data_pickle.endswith("/data.pkl") else ""
        arrays: dict[str, np.ndarray] = {}
        storage_cache: dict[tuple[str, str], np.ndarray] = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, _TorchTensorRef):
                arrays[str(key)] = _tensor_ref_to_numpy(
                    archive,
                    prefix,
                    tensor,
                    storage_cache,
                    max_tensor_bytes=max_tensor_bytes,
                )
        if not arrays:
            raise ValueError("restricted PyTorch checkpoint reader found no tensor state_dict entries")
        return arrays


def _validate_checkpoint(source: str | Path, *, max_archive_bytes: int | None) -> Path:
    checkpoint = Path(source)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint}")
    if max_archive_bytes is not None and checkpoint.stat().st_size > max_archive_bytes:
        raise ValueError(f"archive is {checkpoint.stat().st_size} bytes, limit is {max_archive_bytes}")
    return checkpoint


def _load_pickled_state_dict(archive: zipfile.ZipFile) -> OrderedDict[str, object] | dict[str, object]:
    data_pickle = _torch_zip_data_pickle_name(archive)
    root = _RestrictedTorchZipUnpickler(archive.open(data_pickle)).load()
    return _extract_state_dict(root)


def _torch_zip_data_pickle_name(archive: zipfile.ZipFile) -> str:
    candidates = sorted(name for name in archive.namelist() if name.endswith("data.pkl"))
    if not candidates:
        raise ValueError("PyTorch zip checkpoint does not contain data.pkl")
    return candidates[0]


def _extract_state_dict(root: object) -> OrderedDict[str, object] | dict[str, object]:
    if isinstance(root, OrderedDict):
        return root
    if isinstance(root, dict):
        for key in ("state_dict", "model"):
            value = root.get(key)
            if isinstance(value, OrderedDict | dict):
                return value
        if all(isinstance(value, _TorchTensorRef) for value in root.values()):
            return root
    raise ValueError(f"unsupported PyTorch checkpoint root object: {type(root).__name__}")


def _tensor_ref_to_numpy(
    archive: zipfile.ZipFile,
    prefix: str,
    tensor: _TorchTensorRef,
    storage_cache: dict[tuple[str, str], np.ndarray],
    *,
    max_tensor_bytes: int | None,
) -> np.ndarray:
    dtype = _TORCH_STORAGE_DTYPES.get(tensor.storage.storage_type.name)
    if dtype is None:
        raise ValueError(f"unsupported PyTorch storage type: {tensor.storage.storage_type.name}")
    _validate_tensor_layout(tensor)
    cache_key = (tensor.storage.storage_type.name, tensor.storage.key)
    if cache_key not in storage_cache:
        member = f"{prefix}/data/{tensor.storage.key}" if prefix else f"data/{tensor.storage.key}"
        info = archive.getinfo(member)
        expected_bytes = tensor.storage.size * dtype.itemsize
        if info.file_size != expected_bytes:
            raise ValueError(
                f"tensor storage {tensor.storage.key!r} contains {info.file_size} bytes, "
                f"expected {expected_bytes}"
            )
        if max_tensor_bytes is not None and info.file_size > max_tensor_bytes:
            raise ValueError(f"tensor storage {tensor.storage.key!r} is {info.file_size} bytes, limit is {max_tensor_bytes}")
        storage_cache[cache_key] = np.frombuffer(archive.read(member), dtype=dtype)
    storage = storage_cache[cache_key]
    offset = tensor.storage_offset
    if any(dimension == 0 for dimension in tensor.size):
        return np.empty(tensor.size, dtype=dtype)
    if not tensor.size:
        return np.array(storage[offset], dtype=dtype)
    byte_strides = tuple(stride * dtype.itemsize for stride in tensor.stride)
    view = np.lib.stride_tricks.as_strided(
        storage[offset:],
        shape=tensor.size,
        strides=byte_strides,
        writeable=False,
    )
    return np.array(view, copy=True)


def _validate_tensor_layout(tensor: _TorchTensorRef) -> None:
    """Reject malformed views before NumPy is allowed to construct them."""

    if tensor.storage.size < 0:
        raise ValueError(f"invalid storage size for tensor: {tensor.storage.size}")
    if len(tensor.size) != len(tensor.stride):
        raise ValueError("tensor shape and stride ranks do not match")
    if any(dimension < 0 for dimension in tensor.size):
        raise ValueError(f"invalid tensor shape: {tensor.size}")
    if any(stride < 0 for stride in tensor.stride):
        raise ValueError(f"negative tensor strides are unsupported: {tensor.stride}")

    offset = tensor.storage_offset
    if any(dimension == 0 for dimension in tensor.size):
        if offset < 0 or offset > tensor.storage.size:
            raise ValueError(f"invalid storage offset for empty tensor: {offset}")
        return
    if offset < 0 or offset >= tensor.storage.size:
        raise ValueError(f"invalid storage offset for tensor: {offset}")
    final_index = offset + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(tensor.size, tensor.stride, strict=True)
    )
    if final_index >= tensor.storage.size:
        raise ValueError(
            f"tensor view exceeds storage: final index {final_index}, "
            f"storage size {tensor.storage.size}"
        )
