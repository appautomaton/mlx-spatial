from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.lito_quantization import (
    LITO_QUANTIZATION_METADATA_KEY,
    LITO_RUNTIME_METADATA_KEY,
    LitoQuantizedMatrix,
    inspect_logical_lito_safetensors,
    is_lito_runtime_tensor,
    load_logical_lito_safetensors,
    prune_lito_checkpoint,
    quantize_lito_checkpoint,
    read_lito_quantization_spec,
    should_quantize_lito_tensor,
)
from mlx_spatial.safetensors_io import inspect_safetensors, read_safetensors_metadata, save_safetensors


@pytest.mark.parametrize(
    "name",
    (
        "velocity_estimator_ema.module.blocks.0.attn.linear_qkv.weight",
        "gs_decoder.perceiver.blocks.0.ca_layer.linear_q.weight",
        "gs_decoder.perceiver.blocks.5.sa_layers.1.linear_out.weight",
        "gs_decoder.perceiver.blocks.2.ca_mlp.w12.weight",
        "voxel_decoder.net.encoder.blocks.0.ca_layer.linear_kv.weight",
        "voxel_decoder.net.encoder.blocks.1.mlp_layers.0.fc2.weight",
    ),
)
def test_lito_quantization_policy_selects_internal_linear_weights(name):
    assert should_quantize_lito_tensor(name, (128, 64), "F32")


@pytest.mark.parametrize(
    ("name", "shape", "dtype"),
    (
        ("patch_encoder.dinov2_model.model.blocks.0.attn.qkv.weight", (192, 64), "F32"),
        ("velocity_estimator.blocks.27.mlp.w2.weight", (128, 64), "F32"),
        ("velocity_estimator_ema.module.z_proj.weight", (64, 64), "F32"),
        ("velocity_estimator_ema.module.final_layer.linear.weight", (32, 64), "F32"),
        ("gs_decoder.point_linear.weight", (512, 195), "F32"),
        ("gs_decoder.gs_output_shape_mlp.2.linear.weight", (640, 512), "F32"),
        ("voxel_decoder.final_layer.linear.weight", (8, 512), "F32"),
        ("velocity_estimator_ema.module.blocks.0.attn.linear_qkv.bias", (192,), "F32"),
        ("velocity_estimator_ema.module.blocks.0.attn.linear_qkv.weight", (192, 63), "F32"),
        ("velocity_estimator_ema.module.blocks.0.attn.linear_qkv.weight", (192, 64), "I32"),
    ),
)
def test_lito_quantization_policy_preserves_sensitive_or_ineligible_tensors(name, shape, dtype):
    assert not should_quantize_lito_tensor(name, shape, dtype)


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("velocity_estimator_ema.module.blocks.0.attn.linear_qkv.weight", True),
        ("patch_encoder.dinov2_model.model.cls_token", True),
        ("gs_decoder.point_linear.weight", True),
        ("voxel_decoder.input_linear.weight", True),
        ("velocity_estimator.blocks.0.attn.linear_qkv.weight", False),
        ("pretrained_tokenizer.gs_decoder.point_linear.weight", False),
        ("mesh_decoder.weight", False),
        ("fpoint_encoder.weight", False),
        ("lpips_model.weight", False),
        ("velocity_decoder.weight", False),
        ("flow_t_encoder.weight", False),
    ),
)
def test_lito_runtime_tensor_inventory(name, expected):
    assert is_lito_runtime_tensor(name) is expected


def test_prune_lito_checkpoint_atomically_replaces_source_with_runtime_tensors(tmp_path):
    path = tmp_path / "lito.safetensors"
    runtime = {
        "velocity_estimator_ema.module.z_proj.weight": np.ones((4, 4), dtype=np.float32),
        "patch_encoder.dinov2_model.model.cls_token": np.ones((1, 1, 4), dtype=np.float32),
    }
    stale = {
        "velocity_estimator.z_proj.weight": np.ones((4, 4), dtype=np.float32),
        "pretrained_tokenizer.gs_decoder.weight": np.ones((4, 4), dtype=np.float32),
    }
    save_safetensors(path, runtime | stale, metadata={"source": "test"})

    result = prune_lito_checkpoint(path, path, overwrite=True)

    assert result.source_tensor_count == 4
    assert result.runtime_tensor_count == 2
    assert result.removed_tensor_count == 2
    assert result.removed_payload_bytes == sum(value.nbytes for value in stale.values())
    assert {header.name for header in inspect_safetensors(path)} == set(runtime)
    metadata = read_safetensors_metadata(path)
    assert metadata["source"] == "test"
    runtime_metadata = json.loads(metadata[LITO_RUNTIME_METADATA_KEY])
    assert runtime_metadata["policy"] == "main-inference-prefixes-v1"


def test_quantized_checkpoint_round_trip_executes_packed_affine_matmul(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "output.safetensors"
    name = "velocity_estimator_ema.module.blocks.0.attn.linear_qkv.weight"
    values = np.linspace(-1.0, 1.0, 128 * 64, dtype=np.float32).reshape(128, 64)
    untouched = np.arange(64, dtype=np.float32)
    stale = np.ones((128, 64), dtype=np.float32)
    save_safetensors(
        source,
        {
            name: values,
            "velocity_estimator_ema.module.final_layer.linear.bias": untouched,
            "velocity_estimator.blocks.0.attn.linear_qkv.weight": stale,
            "pretrained_tokenizer.gs_decoder.point_linear.weight": stale,
        },
    )

    result = quantize_lito_checkpoint(source, output, bits=8, group_size=64)

    assert result.quantized_tensor_count == 1
    assert result.source_tensor_count == 4
    assert result.runtime_tensor_count == 2
    assert result.removed_tensor_count == 2
    assert result.removed_source_bytes == stale.nbytes * 2
    physical_names = {header.name for header in inspect_safetensors(output)}
    assert name not in physical_names
    assert name + ".__mlx_qweight__" in physical_names
    assert name + ".__mlx_scales__" in physical_names
    assert name + ".__mlx_biases__" in physical_names
    assert {header.name for header in inspect_logical_lito_safetensors(output)} == {
        name,
        "velocity_estimator_ema.module.final_layer.linear.bias",
    }

    spec = read_lito_quantization_spec(output)
    assert spec is not None
    assert spec.bits == 8
    assert spec.group_size == 64
    assert spec.tensors[name].shape == values.shape
    metadata = json.loads(read_safetensors_metadata(output)[LITO_QUANTIZATION_METADATA_KEY])
    assert metadata["policy"] == "runtime-only-selective-internal-linear-v1"

    loaded = load_logical_lito_safetensors(output)
    quantized = loaded[name]
    assert isinstance(quantized, LitoQuantizedMatrix)
    inputs = mx.array(np.linspace(-0.5, 0.5, 3 * 64, dtype=np.float32).reshape(3, 64))
    actual = mx.quantized_matmul(
        inputs,
        quantized.qweight,
        quantized.scales,
        quantized.biases,
        transpose=True,
        group_size=quantized.group_size,
        bits=quantized.bits,
        mode=quantized.mode,
    )
    expected = inputs @ mx.transpose(mx.array(values))
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-2, atol=2e-2)


def test_gaussian_loader_splits_packed_fused_swiglu_rows(tmp_path):
    from mlx_spatial.lito_real_backend import load_lito_gaussian_decoder_weight_arrays

    root = tmp_path / "source"
    output_root = tmp_path / "quantized"
    source = root / "tokenizer/lito_new.safetensors"
    output = output_root / "tokenizer/lito_new.safetensors"
    source.parent.mkdir(parents=True)
    name = "gs_decoder.perceiver.blocks.0.ca_mlp.w12.weight"
    values = np.linspace(-1.0, 1.0, 128 * 64, dtype=np.float32).reshape(128, 64)
    save_safetensors(source, {name: values})
    quantize_lito_checkpoint(source, output)

    loaded = load_lito_gaussian_decoder_weight_arrays(output_root)

    first = loaded["perceiver.blocks.0.ca_mlp.w1.weight"]
    second = loaded["perceiver.blocks.0.ca_mlp.w2.weight"]
    assert isinstance(first, LitoQuantizedMatrix)
    assert isinstance(second, LitoQuantizedMatrix)
    assert first.shape == (64, 64)
    assert second.shape == (64, 64)
    assert tuple(first.qweight.shape) == (64, 16)
    assert tuple(second.qweight.shape) == (64, 16)
