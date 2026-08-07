import sys
import types

import numpy as np
from safetensors.numpy import save_file

from mlx_spatial.checkpoint import inspect_checkpoint
from mlx_spatial.lito_assets import (
    LITO_DEFAULT_CHECKPOINTS,
    LITO_REPO_ID,
    LITO_TRELLIS_REQUIRED_FILES,
    convert,
    download_command,
    inspect,
    validate,
)


def _write_lito_fixture(root):
    for _, _, relative_path in LITO_DEFAULT_CHECKPOINTS:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if "lito_new" in path.name:
            tensors = {
                "fpoint_encoder.weight": np.array([1.0], dtype=np.float32),
                "gs_decoder.weight": np.array([2.0], dtype=np.float32),
            }
        else:
            tensors = {
                "patch_encoder.weight": np.array([3.0], dtype=np.float32),
                "velocity_estimator.blocks.0.weight": np.array([4.0], dtype=np.float32),
            }
        save_file(tensors, path)


def test_validate_layout_passes_on_downloaded_weights(tmp_path):
    root = tmp_path / "lito-research-mlx"
    _write_lito_fixture(root)
    _write_trellis_fixture(tmp_path / "trellis2/microsoft/TRELLIS-image-large")

    validation = validate(root)

    assert validation.ready
    assert validation.missing == ()
    assert validation.present == (
        "tokenizer/lito_new.safetensors",
        "image_to_3d/lito_dit_rgba.safetensors",
        "trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json",
        "trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    )


def test_validate_reports_missing_trellis_runtime_dependency(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "lito-research-mlx"
    _write_lito_fixture(root)

    validation = validate(root)

    assert not validation.ready
    assert validation.missing == (
        "trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json",
        "trellis2/microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    )


def test_inspect_lists_expected_tensors(tmp_path):
    _write_lito_fixture(tmp_path)

    infos = inspect(tmp_path, prefixes=("velocity_estimator",), limit=5)

    assert len(infos) == 1
    assert infos[0].name == "velocity_estimator.blocks.0.weight"
    assert infos[0].shape == (1,)
    assert infos[0].source.endswith("image_to_3d/lito_dit_rgba.safetensors")


def _write_trellis_fixture(root):
    for relative_path in LITO_TRELLIS_REQUIRED_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")


def test_download_command_prints_cdn_invocation_when_hf_has_no_repo():
    command = download_command("weights/lito-raw")

    assert LITO_REPO_ID == "apple/ml-lito"
    assert "curl -L" in command
    assert "lito_new.ckpt" in command
    assert "lito_dit_rgba.ckpt" in command
    assert "weights/lito-raw" in command

def test_convert_roundtrip_tensor_names_and_shapes(tmp_path, monkeypatch):
    source = tmp_path / "lito_new.ckpt"
    source.write_bytes(b"fake checkpoint")
    output_root = tmp_path / "converted"
    output = output_root / "lito_new.safetensors"

    class FakePtCheckpoint:
        @classmethod
        def load(cls, path, **kwargs):
            assert path.endswith("lito_new.ckpt")
            assert kwargs["max_archive_bytes"] == 16 * 1024**3
            assert kwargs["max_tensor_bytes"] == 16 * 1024**3
            return cls()

        def export(self, *, format, dir):
            assert format == "safetensors"
            weights = tmp_path / "exported.safetensors"
            metadata = tmp_path / "exported.yaml"
            save_file({"model.weight": np.array([[1.0, 2.0]], dtype=np.float32)}, weights)
            metadata.write_text("source_sha256: fixture\n", encoding="utf-8")
            return {"weights_path": str(weights), "metadata_path": str(metadata)}

    monkeypatch.setitem(sys.modules, "pt_loader", types.SimpleNamespace(PtCheckpoint=FakePtCheckpoint))

    convert(source, output_root)

    infos = inspect_checkpoint(output)
    assert [info.name for info in infos] == ["model.weight"]
    assert infos[0].shape == (1, 2)
    assert (output.parent / "conversion_metadata" / "lito_new.yaml").is_file()
