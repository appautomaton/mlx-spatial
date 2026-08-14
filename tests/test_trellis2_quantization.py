import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.checkpoint import inspect_checkpoint, load_checkpoint_tensors
from mlx_spatial.safetensors_io import inspect_safetensors, read_safetensors_metadata, save_safetensors
from mlx_spatial.trellis2_quantization import (
    TRELLIS2_DINOV3_CHECKPOINT_PATH,
    TRELLIS2_MAIN_CHECKPOINT_PATHS,
    TRELLIS2_QUANTIZATION_METADATA_KEY,
    TRELLIS2_RMBG_CHECKPOINT_PATH,
    Trellis2QuantizedMatrix,
    inspect_logical_trellis2_safetensors,
    quantize_trellis2_checkpoint,
    quantize_trellis2_weights,
    read_trellis2_quantization_spec,
    should_quantize_trellis2_tensor,
    trellis2_linear,
)


@pytest.mark.parametrize(
    ("checkpoint", "name"),
    (
        (
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "blocks.0.self_attn.to_qkv.weight",
        ),
        (
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            "blocks.29.cross_attn.to_out.weight",
        ),
        (
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
            "blocks.12.mlp.mlp.2.weight",
        ),
        (
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "blocks.1.4.mlp.0.weight",
        ),
        ("dinov3/model.safetensors", "layer.23.attention.q_proj.weight"),
        ("rmbg/model.safetensors", "bb.layers.2.blocks.4.mlp.fc2.weight"),
    ),
)
def test_trellis2_quantization_policy_selects_internal_matrices(checkpoint, name):
    assert should_quantize_trellis2_tensor(checkpoint, name, (128, 64), "F32")


@pytest.mark.parametrize(
    ("checkpoint", "name", "shape", "dtype"),
    (
        ("ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", "input_layer.weight", (1536, 8), "BF16"),
        (
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "blocks.0.norm2.weight",
            (1536,),
            "BF16",
        ),
        (
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "blocks.0.0.conv.weight",
            (1024, 3, 3, 3, 1024),
            "F16",
        ),
        ("dinov3/model.safetensors", "embeddings.patch_embeddings.weight", (1024, 3, 16, 16), "F32"),
        ("rmbg/model.safetensors", "bb.layers.0.downsample.reduction.weight", (256, 512), "F32"),
        (
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "blocks.0.self_attn.to_qkv.weight",
            (128, 63),
            "F32",
        ),
    ),
)
def test_trellis2_quantization_policy_preserves_boundaries(checkpoint, name, shape, dtype):
    assert not should_quantize_trellis2_tensor(checkpoint, name, shape, dtype)


def test_trellis2_quantized_checkpoint_preserves_logical_names_and_runs_linear(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    name = "blocks.0.self_attn.to_qkv.weight"
    values = np.linspace(-1.0, 1.0, 128 * 64, dtype=np.float32).reshape(128, 64)
    bias = np.linspace(-0.1, 0.1, 128, dtype=np.float32)
    boundary = np.ones((64, 64), dtype=np.float32)
    save_safetensors(
        source,
        {
            name: values,
            "blocks.0.self_attn.to_qkv.bias": bias,
            "input_layer.weight": boundary,
        },
        metadata={"source": "test"},
    )

    result = quantize_trellis2_checkpoint(
        source,
        output,
        checkpoint="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    )

    assert result.quantized_tensor_count == 1
    physical_names = {header.name for header in inspect_safetensors(output)}
    assert name not in physical_names
    assert name + ".__mlx_qweight__" in physical_names
    assert {header.name for header in inspect_logical_trellis2_safetensors(output)} == {
        name,
        "blocks.0.self_attn.to_qkv.bias",
        "input_layer.weight",
    }
    assert read_safetensors_metadata(output)["source"] == "test"

    spec = read_trellis2_quantization_spec(output)
    assert spec is not None
    assert spec.bits == 8
    assert spec.group_size == 64
    assert spec.tensors[name].shape == values.shape

    assert {info.name for info in inspect_checkpoint(output)} == {
        name,
        "blocks.0.self_attn.to_qkv.bias",
        "input_layer.weight",
    }
    loaded = load_checkpoint_tensors(output, prefixes=("blocks.0.",))
    packed = loaded[name]
    assert isinstance(packed, Trellis2QuantizedMatrix)
    inputs = mx.array(np.linspace(-0.5, 0.5, 2 * 3 * 64, dtype=np.float32).reshape(2, 3, 64))
    actual = trellis2_linear(inputs, packed, loaded["blocks.0.self_attn.to_qkv.bias"])
    expected = inputs @ mx.transpose(mx.array(values)) + mx.array(bias)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-2, atol=2e-2)


def test_quantize_trellis2_weights_preserves_layout_and_bundles_auxiliary_models(tmp_path):
    source = tmp_path / "trellis2"
    output = tmp_path / "trellis2-mlx-8bit"
    dino = tmp_path / "dinov3"
    rmbg = tmp_path / "rmbg"
    source.mkdir()
    dino.mkdir()
    rmbg.mkdir()
    (source / "pipeline.json").write_text('{"name":"test"}\n', encoding="utf-8")
    (dino / "config.json").write_text("{}\n", encoding="utf-8")
    (rmbg / "config.json").write_text("{}\n", encoding="utf-8")

    for relative_path in TRELLIS2_MAIN_CHECKPOINT_PATHS:
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.name in {
            "shape_dec_next_dc_f16c32_fp16.safetensors",
            "shape_enc_next_dc_f16c32_fp16.safetensors",
            "tex_dec_next_dc_f16c32_fp16.safetensors",
            "tex_enc_next_dc_f16c32_fp16.safetensors",
        }:
            tensors = {"blocks.0.0.mlp.0.weight": np.ones((64, 64), dtype=np.float32)}
        elif relative_path.name == "ss_dec_conv3d_16l8_fp16.safetensors":
            tensors = {"blocks.0.conv.weight": np.ones((2, 2, 1, 1, 1), dtype=np.float32)}
        else:
            tensors = {"blocks.0.self_attn.to_qkv.weight": np.ones((64, 64), dtype=np.float32)}
        save_safetensors(path, tensors)

    save_safetensors(
        dino / "model.safetensors",
        {"layer.0.attention.q_proj.weight": np.ones((64, 64), dtype=np.float32)},
    )
    save_safetensors(
        rmbg / "model.safetensors",
        {"bb.layers.0.blocks.0.attn.qkv.weight": np.ones((64, 64), dtype=np.float32)},
    )

    results = quantize_trellis2_weights(source, output, dinov3_root=dino, rmbg_root=rmbg)

    assert tuple(result.checkpoint for result in results) == TRELLIS2_MAIN_CHECKPOINT_PATHS + (
        TRELLIS2_DINOV3_CHECKPOINT_PATH,
        TRELLIS2_RMBG_CHECKPOINT_PATH,
    )
    assert json.loads((output / "pipeline.json").read_text(encoding="utf-8")) == {"name": "test"}
    assert (output / "dinov3/config.json").is_file()
    assert (output / "rmbg/config.json").is_file()
    for result in results:
        assert result.output.is_file()
    sparse_decoder = output / "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"
    assert read_trellis2_quantization_spec(sparse_decoder) is None
    dino_metadata = json.loads(
        read_safetensors_metadata(output / TRELLIS2_DINOV3_CHECKPOINT_PATH)[
            TRELLIS2_QUANTIZATION_METADATA_KEY
        ]
    )
    assert dino_metadata["checkpoint"] == "dinov3/model.safetensors"


def test_quantized_linear_rejects_input_channel_mismatch(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    save_safetensors(
        source,
        {"blocks.0.mlp.mlp.0.weight": np.ones((64, 64), dtype=np.float32)},
    )
    quantize_trellis2_checkpoint(
        source,
        output,
        checkpoint="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
    )
    packed = load_checkpoint_tensors(output, names=("blocks.0.mlp.mlp.0.weight",))[
        "blocks.0.mlp.mlp.0.weight"
    ]
    with pytest.raises(ValueError, match="expected 64"):
        trellis2_linear(mx.ones((1, 63)), packed, None)
