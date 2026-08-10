import numpy as np
import pytest

from mlx_spatial.safetensors_io import (
    inspect_safetensors,
    load_numpy_safetensors,
    read_safetensors_metadata,
    save_safetensors,
)


def test_mlx_safetensors_round_trip_preserves_supported_dtypes(tmp_path):
    path = tmp_path / "weights.safetensors"
    expected = {
        "half.weight": np.array([[1.0, 2.0]], dtype=np.float16),
        "float.bias": np.array([3.0], dtype=np.float32),
    }

    save_safetensors(path, expected, metadata={"source": "test"})

    infos = inspect_safetensors(path)
    loaded = load_numpy_safetensors(path)
    assert read_safetensors_metadata(path) == {"source": "test"}
    assert [(info.name, info.shape, info.dtype) for info in infos] == [
        ("float.bias", (1,), "F32"),
        ("half.weight", (1, 2), "F16"),
    ]
    for name, value in expected.items():
        assert loaded[name].dtype == value.dtype
        np.testing.assert_array_equal(loaded[name], value)


def test_mlx_safetensors_writer_rejects_silent_dtype_conversion(tmp_path):
    path = tmp_path / "weights.safetensors"
    save_safetensors(path, {"weight": np.array([1.0], dtype=np.float32)})
    original = path.read_bytes()

    with pytest.raises(ValueError, match="cannot preserve dtype float64"):
        save_safetensors(
            path,
            {"weight": np.array([1.0], dtype=np.float64)},
        )
    assert path.read_bytes() == original
