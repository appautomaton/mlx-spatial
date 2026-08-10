from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.safetensors_io import inspect_safetensors, read_safetensors_metadata, save_safetensors
from mlx_spatial.sam3d_quantization import (
    SAM3D_CHECKPOINT_PATHS,
    SAM3D_QUANTIZATION_METADATA_KEY,
    Sam3dQuantizedMatrix,
    inspect_logical_sam3d_safetensors,
    load_sam3d_checkpoint_tensors,
    quantize_sam3d_checkpoint,
    quantize_sam3d_weights,
    read_sam3d_quantization_spec,
    sam3d_linear,
    should_quantize_sam3d_tensor,
)


@pytest.mark.parametrize(
    ("checkpoint", "name"),
    (
        (
            "checkpoints/ss_generator.safetensors",
            "_base_models.condition_embedder.module_list.0.backbone.blocks.0.attn.qkv.weight",
        ),
        (
            "checkpoints/ss_generator.safetensors",
            "_base_models.condition_embedder.module_list.2.blocks.0.attn.qkv.weight",
        ),
        (
            "checkpoints/slat_generator.safetensors",
            "_base_models.generator.reverse_fn.backbone.blocks.0.self_attn.to_qkv.weight",
        ),
        ("checkpoints/slat_decoder_gs.safetensors", "blocks.0.attn.to_qkv.weight"),
        ("checkpoints/slat_decoder_gs_4.safetensors", "blocks.0.mlp.mlp.0.weight"),
        ("checkpoints/slat_decoder_mesh.safetensors", "blocks.0.attn.to_out.weight"),
        ("moge/model.safetensors", "backbone.blocks.0.mlp.fc1.weight"),
    ),
)
def test_sam3d_quantization_policy_selects_internal_block_matrices(checkpoint, name):
    assert should_quantize_sam3d_tensor(checkpoint, name, (128, 64), "F32")


@pytest.mark.parametrize(
    ("checkpoint", "name", "shape", "dtype"),
    (
        ("checkpoints/ss_decoder.safetensors", "blocks.0.conv.weight", (128, 64), "F32"),
        (
            "checkpoints/ss_generator.safetensors",
            "_base_models.generator.reverse_fn.backbone.input_layer.weight",
            (128, 64),
            "F32",
        ),
        (
            "checkpoints/ss_generator.safetensors",
            "_base_models.condition_embedder.module_list.0.backbone.patch_embed.proj.weight",
            (128, 64),
            "F32",
        ),
        ("checkpoints/slat_decoder_mesh.safetensors", "out_layer.weight", (101, 64), "F32"),
        ("moge/model.safetensors", "head.projects.0.weight", (128, 64), "F32"),
        ("moge/model.safetensors", "backbone.blocks.0.attn.qkv.bias", (128,), "F32"),
        ("moge/model.safetensors", "backbone.blocks.0.attn.qkv.weight", (128, 63), "F32"),
        ("moge/model.safetensors", "backbone.blocks.0.attn.qkv.weight", (128, 64), "I32"),
    ),
)
def test_sam3d_quantization_policy_preserves_boundaries_and_ineligible_tensors(
    checkpoint,
    name,
    shape,
    dtype,
):
    assert not should_quantize_sam3d_tensor(checkpoint, name, shape, dtype)


def test_sam3d_quantized_checkpoint_preserves_logical_names_and_runs_linear(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    name = "backbone.blocks.0.attn.qkv.weight"
    values = np.linspace(-1.0, 1.0, 128 * 64, dtype=np.float32).reshape(128, 64)
    bias = np.linspace(-0.1, 0.1, 128, dtype=np.float32)
    boundary = np.ones((64, 64), dtype=np.float32)
    save_safetensors(
        source,
        {
            name: values,
            "backbone.blocks.0.attn.qkv.bias": bias,
            "head.projects.0.weight": boundary,
        },
        metadata={"source": "test"},
    )

    result = quantize_sam3d_checkpoint(
        source,
        output,
        checkpoint="moge/model.safetensors",
    )

    assert result.quantized_tensor_count == 1
    physical_names = {header.name for header in inspect_safetensors(output)}
    assert name not in physical_names
    assert name + ".__mlx_qweight__" in physical_names
    assert {header.name for header in inspect_logical_sam3d_safetensors(output)} == {
        name,
        "backbone.blocks.0.attn.qkv.bias",
        "head.projects.0.weight",
    }
    assert read_safetensors_metadata(output)["source"] == "test"
    spec = read_sam3d_quantization_spec(output)
    assert spec is not None
    assert spec.tensors[name].shape == values.shape

    loaded = load_sam3d_checkpoint_tensors(output, prefixes=("backbone.blocks.0.",))
    packed = loaded[name]
    assert isinstance(packed, Sam3dQuantizedMatrix)
    inputs = mx.array(np.linspace(-0.5, 0.5, 2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64))
    actual = sam3d_linear(inputs, packed, loaded["backbone.blocks.0.attn.qkv.bias"])
    expected = inputs @ mx.transpose(mx.array(values)) + mx.array(bias)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-2, atol=2e-2)


def test_quantize_sam3d_weights_preserves_root_layout_and_checkpoint_names(tmp_path):
    source = tmp_path / "sam3d"
    output = tmp_path / "sam3d-int8"
    source.mkdir()
    (source / "checkpoints").mkdir()
    (source / "moge").mkdir()
    (source / "checkpoints/pipeline.yaml").write_text("target: test\n", encoding="utf-8")
    (source / "README.md").write_text("test\n", encoding="utf-8")

    for relative_path in SAM3D_CHECKPOINT_PATHS:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.name == "ss_decoder.safetensors":
            tensors = {"blocks.0.conv.weight": np.ones((2, 2, 1, 1, 1), dtype=np.float32)}
        elif relative_path.as_posix().endswith("moge/model.safetensors"):
            tensors = {"backbone.blocks.0.attn.qkv.weight": np.ones((64, 64), dtype=np.float32)}
        elif relative_path.name.startswith("slat_decoder_"):
            tensors = {"blocks.0.attn.to_qkv.weight": np.ones((64, 64), dtype=np.float32)}
        else:
            tensors = {
                "_base_models.generator.reverse_fn.backbone.blocks.0.self_attn.to_qkv.weight": np.ones(
                    (64, 64), dtype=np.float32
                )
            }
        save_safetensors(path, tensors)

    results = quantize_sam3d_weights(source, output)

    assert tuple(result.checkpoint for result in results) == SAM3D_CHECKPOINT_PATHS
    assert (output / "checkpoints/pipeline.yaml").read_text(encoding="utf-8") == "target: test\n"
    assert (output / "README.md").read_text(encoding="utf-8") == "test\n"
    for relative_path in SAM3D_CHECKPOINT_PATHS:
        assert (output / relative_path).is_file()
    assert read_sam3d_quantization_spec(output / "checkpoints/ss_decoder.safetensors") is None
    metadata = json.loads(
        read_safetensors_metadata(output / "checkpoints/ss_generator.safetensors")[
            SAM3D_QUANTIZATION_METADATA_KEY
        ]
    )
    assert metadata["checkpoint"] == "checkpoints/ss_generator.safetensors"
