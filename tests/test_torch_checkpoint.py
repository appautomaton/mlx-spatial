import numpy as np
import pytest

from mlx_spatial.torch_checkpoint import inspect_torch_zip_checkpoint, load_torch_zip_state_dict
from tests.safetensors_test_utils import write_torch_zip_checkpoint


def test_restricted_torch_zip_reader_preserves_names_shapes_dtypes_and_values(tmp_path):
    path = tmp_path / "model.ckpt"
    expected = {
        "half.weight": np.array([[1.0, 2.0]], dtype=np.float16),
        "float.bias": np.array([3.0], dtype=np.float32),
        "empty.buffer": np.empty((0, 2), dtype=np.float32),
    }
    write_torch_zip_checkpoint(path, expected)

    infos = inspect_torch_zip_checkpoint(path)
    arrays = load_torch_zip_state_dict(path)

    assert [(info.name, info.shape, info.dtype) for info in infos] == [
        ("empty.buffer", (0, 2), "float32"),
        ("float.bias", (1,), "float32"),
        ("half.weight", (1, 2), "float16"),
    ]
    for name, value in expected.items():
        np.testing.assert_array_equal(arrays[name], value)


def test_restricted_torch_zip_reader_enforces_archive_and_tensor_limits(tmp_path):
    path = tmp_path / "model.ckpt"
    write_torch_zip_checkpoint(path, {"weight": np.ones((8,), dtype=np.float32)})

    with pytest.raises(ValueError, match="archive is"):
        inspect_torch_zip_checkpoint(path, max_archive_bytes=1)
    with pytest.raises(ValueError, match="tensor storage"):
        inspect_torch_zip_checkpoint(path, max_tensor_bytes=1)
