"""Small test helpers backed by the project's MLX-native safetensors I/O."""

from __future__ import annotations

import pickle
import sys
import types
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mlx_spatial.safetensors_io import load_mlx_safetensors, save_safetensors


def save_file(arrays: Mapping[str, Any], path: str | Path) -> None:
    save_safetensors(path, arrays)


def load_file(path: str | Path) -> dict[str, np.ndarray]:
    return {name: np.asarray(tensor) for name, tensor in load_mlx_safetensors(path).items()}


def load_mlx_file(path: str | Path):
    return load_mlx_safetensors(path)


class _StorageProxy:
    def __init__(self, storage_type: type, key: str, size: int):
        self.storage_type = storage_type
        self.key = key
        self.size = size


class _TensorProxy:
    def __init__(self, rebuild, storage: _StorageProxy, shape: tuple[int, ...]):
        self.rebuild = rebuild
        self.storage = storage
        self.shape = shape

    def __reduce__(self):
        stride = _contiguous_stride(self.shape)
        return self.rebuild, (self.storage, 0, self.shape, stride, False, OrderedDict())


class _TorchPickler(pickle.Pickler):
    def persistent_id(self, value: object):
        if isinstance(value, _StorageProxy):
            return ("storage", value.storage_type, value.key, "cpu", value.size)
        return None


def write_torch_zip_checkpoint(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a minimal tensor-only PyTorch ZIP fixture without importing Torch."""

    storage_names = {
        np.dtype("float32"): "FloatStorage",
        np.dtype("float16"): "HalfStorage",
        np.dtype("int64"): "LongStorage",
        np.dtype("int32"): "IntStorage",
        np.dtype("uint8"): "ByteStorage",
        np.dtype("bool"): "BoolStorage",
    }
    torch_module = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch._utils")

    def rebuild_tensor_v2(*_):
        raise AssertionError("fixture rebuild function must not execute while writing")

    rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
    rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    rebuild_tensor_v2.__module__ = "torch._utils"
    torch_utils._rebuild_tensor_v2 = rebuild_tensor_v2

    state = OrderedDict()
    payloads: dict[str, bytes] = {}
    for index, (name, value) in enumerate(arrays.items()):
        array = np.ascontiguousarray(value)
        storage_name = storage_names.get(array.dtype)
        if storage_name is None:
            raise ValueError(f"unsupported test checkpoint dtype: {array.dtype}")
        storage_type = getattr(torch_module, storage_name, None)
        if storage_type is None:
            storage_type = type(storage_name, (), {})
            storage_type.__module__ = "torch"
            setattr(torch_module, storage_name, storage_type)
        key = str(index)
        state[str(name)] = _TensorProxy(
            rebuild_tensor_v2,
            _StorageProxy(storage_type, key, int(array.size)),
            tuple(int(dim) for dim in array.shape),
        )
        payloads[key] = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")

    previous_torch = sys.modules.get("torch")
    previous_utils = sys.modules.get("torch._utils")
    sys.modules["torch"] = torch_module
    sys.modules["torch._utils"] = torch_utils
    try:
        from io import BytesIO

        buffer = BytesIO()
        _TorchPickler(buffer, protocol=2).dump({"state_dict": state})
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
        if previous_utils is None:
            sys.modules.pop("torch._utils", None)
        else:
            sys.modules["torch._utils"] = previous_utils

    checkpoint = Path(path)
    with zipfile.ZipFile(checkpoint, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        for key, payload in payloads.items():
            archive.writestr(f"archive/data/{key}", payload)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))
