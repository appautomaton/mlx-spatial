import mlx.core as mx
from safetensors.mlx import save_file

import mlx_spatial
from mlx_spatial.trellis2_rmbg import (
    Rmbg2KeyInventory,
    Rmbg2PortAssessment,
    Rmbg2PortBlocker,
    Rmbg2TensorProbe,
    assess_rmbg2_mlx_port,
    inspect_rmbg2_checkpoint,
    inspect_rmbg2_key_inventory,
    load_rmbg2_tensors,
)


def _write_rmbg_root(root):
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "backbone.stem.weight": mx.array([[1.0, 2.0]], dtype=mx.float32),
            "backbone.block.0.bias": mx.array([3.0], dtype=mx.float32),
            "decoder.head.weight": mx.array([4.0, 5.0], dtype=mx.float32),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text("{}")
    (root / "BiRefNet_config.py").write_text("config = {}\n")
    (root / "birefnet.py").write_text("class BiRefNet: pass\n")


def _write_rmbg_root_with_deform_conv(root):
    _write_rmbg_root(root)
    (root / "birefnet.py").write_text("from torchvision.ops import deform_conv2d\n")


def test_inspect_rmbg2_checkpoint_returns_deterministic_metadata(tmp_path):
    _write_rmbg_root(tmp_path)

    infos = inspect_rmbg2_checkpoint(tmp_path)

    assert [info.name for info in infos] == [
        "backbone.block.0.bias",
        "backbone.stem.weight",
        "decoder.head.weight",
    ]
    assert infos[0].shape == (1,)
    assert infos[0].dtype == "F32"


def test_load_rmbg2_tensors_loads_selected_mlx_arrays(tmp_path):
    _write_rmbg_root(tmp_path)

    tensors = load_rmbg2_tensors(tmp_path, prefixes=["decoder."])

    assert [tensor.name for tensor in tensors] == ["decoder.head.weight"]
    assert tensors[0].checkpoint_path == "model.safetensors"
    assert tensors[0].shape == (2,)
    assert tensors[0].dtype == "float32"
    assert tensors[0].array.tolist() == [4.0, 5.0]


def test_inspect_rmbg2_key_inventory_summarizes_prefixes_and_samples(tmp_path):
    _write_rmbg_root(tmp_path)

    inventory = inspect_rmbg2_key_inventory(tmp_path)

    assert inventory.tensor_count == 3
    assert inventory.top_level_prefixes == ("backbone", "decoder")
    assert inventory.sample_keys == (
        "backbone.block.0.bias",
        "backbone.stem.weight",
        "decoder.head.weight",
    )


def test_rmbg2_helpers_reject_invalid_inputs(tmp_path):
    invalid_cases = (
        lambda: inspect_rmbg2_checkpoint(tmp_path / "missing"),
        lambda: load_rmbg2_tensors(tmp_path / "missing", names=["x"]),
    )

    for call in invalid_cases:
        try:
            call()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing RMBG checkpoint should be rejected")


def test_assess_rmbg2_mlx_port_reports_deform_conv_blocker(tmp_path):
    _write_rmbg_root_with_deform_conv(tmp_path)

    assessment = assess_rmbg2_mlx_port(tmp_path)

    assert not assessment.ready
    assert assessment.tensor_count == 3
    assert assessment.top_level_prefixes == ("backbone", "decoder")
    assert assessment.blocker is not None
    assert assessment.blocker.stage == "image-preprocessing-background"
    assert assessment.blocker.operation == "MLX BiRefNet deformable convolution"
    assert "DeformConv2d" in assessment.blocker.reason


def test_assess_rmbg2_mlx_port_reports_missing_key_prefixes(tmp_path):
    _write_rmbg_root(tmp_path)

    assessment = assess_rmbg2_mlx_port(tmp_path)

    assert not assessment.ready
    assert assessment.blocker is not None
    assert assessment.blocker.operation == "MLX BiRefNet checkpoint key mapping"
    assert "squeeze_module" in assessment.blocker.reason


def test_rmbg2_helpers_are_public_exports():
    assert mlx_spatial.Rmbg2KeyInventory is Rmbg2KeyInventory
    assert mlx_spatial.Rmbg2PortAssessment is Rmbg2PortAssessment
    assert mlx_spatial.Rmbg2PortBlocker is Rmbg2PortBlocker
    assert mlx_spatial.Rmbg2TensorProbe is Rmbg2TensorProbe
    assert mlx_spatial.assess_rmbg2_mlx_port is assess_rmbg2_mlx_port
    assert mlx_spatial.inspect_rmbg2_checkpoint is inspect_rmbg2_checkpoint
    assert mlx_spatial.inspect_rmbg2_key_inventory is inspect_rmbg2_key_inventory
    assert mlx_spatial.load_rmbg2_tensors is load_rmbg2_tensors
