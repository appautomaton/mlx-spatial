import tomllib

import mlx.core as mx
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.checkpoint import (
    CheckpointTensorInfo,
    inspect_checkpoint,
    load_checkpoint_tensors,
)


def _write_checkpoint(path):
    save_file(
        {
            "decoder.bias": mx.array([7.0, 8.0], dtype=mx.float32),
            "encoder.block.weight": mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32),
            "encoder.norm.weight": mx.array([5.0, 6.0], dtype=mx.float32),
        },
        path,
    )


def test_base_dependencies_exclude_heavy_model_frameworks():
    config = tomllib.loads(open("pyproject.toml", "rb").read().decode())
    dependencies = "\n".join(config["project"]["dependencies"]).lower()

    assert "safetensors" in dependencies
    assert "torch" not in dependencies
    assert "transformers" not in dependencies
    assert "huggingface" not in dependencies


def test_inspect_checkpoint_returns_deterministic_metadata(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    _write_checkpoint(checkpoint)

    infos = inspect_checkpoint(checkpoint)

    assert infos == (
        CheckpointTensorInfo("decoder.bias", (2,), "F32", str(checkpoint)),
        CheckpointTensorInfo("encoder.block.weight", (2, 2), "F32", str(checkpoint)),
        CheckpointTensorInfo("encoder.norm.weight", (2,), "F32", str(checkpoint)),
    )


def test_inspect_checkpoint_filters_by_exact_names_and_prefixes(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    _write_checkpoint(checkpoint)

    by_name = inspect_checkpoint(checkpoint, names=["decoder.bias"])
    by_prefix = inspect_checkpoint(checkpoint, prefixes=["encoder."])

    assert [info.name for info in by_name] == ["decoder.bias"]
    assert [info.name for info in by_prefix] == ["encoder.block.weight", "encoder.norm.weight"]


def test_inspect_checkpoint_rejects_invalid_inputs(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    unsupported = tmp_path / "tiny.pt"
    _write_checkpoint(checkpoint)
    unsupported.write_bytes(b"not a safetensors file")

    invalid_cases = (
        lambda: inspect_checkpoint(tmp_path / "missing.safetensors"),
        lambda: inspect_checkpoint(unsupported),
        lambda: inspect_checkpoint(checkpoint, names="decoder.bias"),
        lambda: inspect_checkpoint(checkpoint, names=[]),
        lambda: inspect_checkpoint(checkpoint, prefixes=[""]),
        lambda: inspect_checkpoint(checkpoint, prefixes=["missing."]),
    )
    expected_errors = (FileNotFoundError, ValueError, ValueError, ValueError, ValueError, ValueError)

    for call, expected_error in zip(invalid_cases, expected_errors, strict=True):
        try:
            call()
        except expected_error:
            pass
        else:
            raise AssertionError("invalid checkpoint inspection input should be rejected")


def test_load_checkpoint_tensors_loads_selected_mlx_arrays(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    _write_checkpoint(checkpoint)

    tensors = load_checkpoint_tensors(checkpoint, names=["decoder.bias"])

    assert list(tensors) == ["decoder.bias"]
    assert isinstance(tensors["decoder.bias"], mx.array)
    assert tensors["decoder.bias"].shape == (2,)
    assert tensors["decoder.bias"].tolist() == [7.0, 8.0]


def test_load_checkpoint_tensors_filters_by_prefix(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    _write_checkpoint(checkpoint)

    tensors = load_checkpoint_tensors(checkpoint, prefixes=["encoder."])

    assert list(tensors) == ["encoder.block.weight", "encoder.norm.weight"]
    assert tensors["encoder.block.weight"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert tensors["encoder.norm.weight"].tolist() == [5.0, 6.0]


def test_load_checkpoint_tensors_rejects_invalid_inputs(tmp_path):
    checkpoint = tmp_path / "tiny.safetensors"
    unsupported = tmp_path / "tiny.pth"
    _write_checkpoint(checkpoint)
    unsupported.write_bytes(b"not a safetensors file")

    invalid_cases = (
        lambda: load_checkpoint_tensors(checkpoint),
        lambda: load_checkpoint_tensors(tmp_path / "missing.safetensors", names=["decoder.bias"]),
        lambda: load_checkpoint_tensors(unsupported, names=["decoder.bias"]),
        lambda: load_checkpoint_tensors(checkpoint, names="decoder.bias"),
        lambda: load_checkpoint_tensors(checkpoint, names=[]),
        lambda: load_checkpoint_tensors(checkpoint, names=["missing.weight"]),
        lambda: load_checkpoint_tensors(checkpoint, prefixes=["missing."]),
    )
    expected_errors = (
        ValueError,
        FileNotFoundError,
        ValueError,
        ValueError,
        ValueError,
        ValueError,
        ValueError,
    )

    for call, expected_error in zip(invalid_cases, expected_errors, strict=True):
        try:
            call()
        except expected_error:
            pass
        else:
            raise AssertionError("invalid checkpoint loading input should be rejected")


def test_checkpoint_helpers_are_public_exports():
    assert mlx_spatial.CheckpointTensorInfo is CheckpointTensorInfo
    assert mlx_spatial.inspect_checkpoint is inspect_checkpoint
    assert mlx_spatial.load_checkpoint_tensors is load_checkpoint_tensors
