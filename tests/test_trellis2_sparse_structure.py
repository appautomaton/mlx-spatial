import json

import mlx.core as mx
import numpy as np
import pytest
from safetensors.mlx import save_file

from mlx_spatial.trellis2_sparse_structure import (
    SPARSE_STRUCTURE_BLOCK0_INSPECTION_NAMES,
    SPARSE_STRUCTURE_DECODER_TENSOR_NAMES,
    SPARSE_STRUCTURE_INPUT_TENSOR_NAMES,
    SparseStructureDecoderConfig,
    SparseStructureFlowConfig,
    expected_sparse_noise_shape,
    extract_sparse_structure_coordinates,
    fake_sparse_structure_sampling_metadata,
    flow_euler_schedule,
    probe_sparse_structure_decoder_boundary,
    probe_sparse_structure_forward_boundary,
    read_sparse_structure_decoder_config,
    read_sparse_structure_flow_config,
    validate_sparse_noise_shape,
)


def _write_sparse_flow_config(root):
    path = root / "ckpts/ss_flow_img_dit_1_3B_64_bf16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "SparseStructureFlowModel",
                "args": {
                    "resolution": 16,
                    "in_channels": 8,
                    "out_channels": 8,
                    "model_channels": 1536,
                    "cond_channels": 1024,
                    "num_blocks": 30,
                    "num_heads": 12,
                    "mlp_ratio": 5.3334,
                    "pe_mode": "rope",
                    "share_mod": True,
                    "initialization": "scaled",
                    "qk_rms_norm": True,
                    "qk_rms_norm_cross": True,
                    "dtype": "bfloat16",
                },
            }
        )
    )
    return "ckpts/ss_flow_img_dit_1_3B_64_bf16.json"


def _small_sparse_flow_config():
    return SparseStructureFlowConfig(
        name="SparseStructureFlowModel",
        resolution=2,
        in_channels=2,
        out_channels=2,
        model_channels=6,
        cond_channels=6,
        num_blocks=1,
        num_heads=1,
        mlp_ratio=2.0,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="float32",
    )


def _write_sparse_flow_checkpoint(path, config=None, *, omit=(), bad_input_weight_shape=False):
    config = config or _small_sparse_flow_config()
    tensors = {
        "input_layer.weight": mx.ones(
            (config.model_channels, config.in_channels)
            if not bad_input_weight_shape
            else (config.model_channels + 1, config.in_channels),
            dtype=mx.float32,
        ),
        "input_layer.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "out_layer.weight": mx.zeros((config.out_channels, config.model_channels), dtype=mx.float32),
        "out_layer.bias": mx.zeros((config.out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": mx.zeros((config.model_channels, 256), dtype=mx.float32),
        "t_embedder.mlp.0.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "t_embedder.mlp.2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": mx.zeros((config.model_channels * 6, config.model_channels), dtype=mx.float32),
        "adaLN_modulation.1.bias": mx.zeros((config.model_channels * 6,), dtype=mx.float32),
        "blocks.0.modulation": mx.zeros((config.model_channels * 6,), dtype=mx.float32),
        "blocks.0.norm2.weight": mx.ones((config.model_channels,), dtype=mx.float32),
        "blocks.0.norm2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.weight": mx.zeros((config.model_channels * 3, config.model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_qkv.bias": mx.zeros((config.model_channels * 3,), dtype=mx.float32),
        "blocks.0.self_attn.q_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.self_attn.k_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.self_attn.to_out.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "blocks.0.self_attn.to_out.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_q.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.weight": mx.zeros((config.model_channels * 2, config.cond_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_kv.bias": mx.zeros((config.model_channels * 2,), dtype=mx.float32),
        "blocks.0.cross_attn.q_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.cross_attn.k_rms_norm.gamma": mx.ones(
            (config.num_heads, config.model_channels // config.num_heads), dtype=mx.float32
        ),
        "blocks.0.cross_attn.to_out.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "blocks.0.cross_attn.to_out.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.weight": mx.zeros((int(config.model_channels * config.mlp_ratio), config.model_channels), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.bias": mx.zeros((int(config.model_channels * config.mlp_ratio),), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.weight": mx.zeros((config.model_channels, int(config.model_channels * config.mlp_ratio)), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
    }
    for name in omit:
        tensors.pop(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _write_sparse_decoder_config(root):
    path = root / "ckpts/ss_dec_conv3d_16l8_fp16.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "SparseStructureDecoder",
                "args": {
                    "out_channels": 1,
                    "latent_channels": 2,
                    "num_res_blocks": 1,
                    "channels": [4, 8],
                    "num_res_blocks_middle": 2,
                    "norm_type": "layer",
                    "use_fp16": True,
                },
            }
        )
    )
    return "ckpts/ss_dec_conv3d_16l8_fp16.json"


def _small_sparse_decoder_config():
    return SparseStructureDecoderConfig(
        name="SparseStructureDecoder",
        out_channels=1,
        latent_channels=2,
        num_res_blocks=1,
        channels=(4, 8),
        num_res_blocks_middle=2,
        norm_type="layer",
        use_fp16=False,
    )


def _write_sparse_decoder_checkpoint(path, config=None, *, omit=()):
    config = config or _small_sparse_decoder_config()
    tensors = {
        "input_layer.weight": mx.ones((config.channels[0], config.latent_channels, 3, 3, 3), dtype=mx.float32),
        "input_layer.bias": mx.zeros((config.channels[0],), dtype=mx.float32),
        "out_layer.0.weight": mx.ones((config.channels[-1],), dtype=mx.float32),
        "out_layer.0.bias": mx.zeros((config.channels[-1],), dtype=mx.float32),
        "out_layer.2.weight": mx.zeros((config.out_channels, config.channels[-1], 3, 3, 3), dtype=mx.float32),
        "out_layer.2.bias": mx.zeros((config.out_channels,), dtype=mx.float32),
    }
    for index in range(config.num_res_blocks_middle):
        tensors.update(_decoder_resblock_tensors(f"middle_block.{index}", config.channels[0]))
    block_index = 0
    for level, channels in enumerate(config.channels):
        for _ in range(config.num_res_blocks):
            tensors.update(_decoder_resblock_tensors(f"blocks.{block_index}", channels))
            block_index += 1
        if level < len(config.channels) - 1:
            next_channels = config.channels[level + 1]
            tensors[f"blocks.{block_index}.conv.weight"] = mx.zeros((next_channels * 8, channels, 3, 3, 3), dtype=mx.float32)
            tensors[f"blocks.{block_index}.conv.bias"] = mx.zeros((next_channels * 8,), dtype=mx.float32)
            block_index += 1
    for name in omit:
        tensors.pop(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def _decoder_resblock_tensors(prefix, channels):
    return {
        f"{prefix}.norm1.weight": mx.ones((channels,), dtype=mx.float32),
        f"{prefix}.norm1.bias": mx.zeros((channels,), dtype=mx.float32),
        f"{prefix}.conv1.weight": mx.zeros((channels, channels, 3, 3, 3), dtype=mx.float32),
        f"{prefix}.conv1.bias": mx.zeros((channels,), dtype=mx.float32),
        f"{prefix}.norm2.weight": mx.ones((channels,), dtype=mx.float32),
        f"{prefix}.norm2.bias": mx.zeros((channels,), dtype=mx.float32),
        f"{prefix}.conv2.weight": mx.zeros((channels, channels, 3, 3, 3), dtype=mx.float32),
        f"{prefix}.conv2.bias": mx.zeros((channels,), dtype=mx.float32),
    }


def test_read_sparse_structure_flow_config_maps_local_fields(tmp_path):
    config_path = _write_sparse_flow_config(tmp_path)

    config = read_sparse_structure_flow_config(tmp_path, config_path)

    assert config.name == "SparseStructureFlowModel"
    assert config.resolution == 16
    assert config.in_channels == 8
    assert config.out_channels == 8
    assert config.model_channels == 1536
    assert config.cond_channels == 1024
    assert config.num_blocks == 30
    assert config.num_heads == 12
    assert config.pe_mode == "rope"
    assert config.share_mod is True
    assert config.dtype == "bfloat16"


def test_flow_euler_schedule_matches_reference_rescale_shape():
    schedule = flow_euler_schedule(steps=4, rescale_t=5.0, guidance_interval=(0.6, 1.0))

    assert schedule.steps == 4
    assert len(schedule.pairs) == 4
    assert schedule.pairs[0] == (1.0, pytest.approx(0.9375))
    assert schedule.pairs[-1] == (pytest.approx(0.625), 0.0)
    assert schedule.guidance_active == (True, True, True, True)


def test_sparse_noise_shape_validation_uses_flow_config(tmp_path):
    config = read_sparse_structure_flow_config(tmp_path, _write_sparse_flow_config(tmp_path))

    shape = expected_sparse_noise_shape(config)

    assert shape == (1, 8, 16, 16, 16)
    validate_sparse_noise_shape(shape, config)
    with pytest.raises(ValueError, match="sparse noise shape mismatch"):
        validate_sparse_noise_shape((1, 4, 16, 16, 16), config)


def test_fake_sparse_sampling_metadata_reports_schedule_and_shapes(tmp_path):
    config = read_sparse_structure_flow_config(tmp_path, _write_sparse_flow_config(tmp_path))

    metadata = fake_sparse_structure_sampling_metadata(
        config,
        steps=12,
        rescale_t=5.0,
        guidance_interval=(0.6, 1.0),
    )

    assert metadata.noise_shape == (1, 8, 16, 16, 16)
    assert metadata.sample_shape == (1, 8, 16, 16, 16)
    assert metadata.dtype == "bfloat16"
    assert metadata.steps == 12
    assert metadata.guidance_active_steps == 10


def test_sparse_flow_config_reports_invalid_required_fields(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"name": "SparseStructureFlowModel", "args": {"resolution": 16}}))

    with pytest.raises(ValueError, match="sparse structure flow config is invalid"):
        read_sparse_structure_flow_config(tmp_path, "bad.json")


def test_read_sparse_structure_decoder_config_maps_local_fields(tmp_path):
    config_path = _write_sparse_decoder_config(tmp_path)

    config = read_sparse_structure_decoder_config(tmp_path, config_path)

    assert config.name == "SparseStructureDecoder"
    assert config.out_channels == 1
    assert config.latent_channels == 2
    assert config.num_res_blocks == 1
    assert config.channels == (4, 8)
    assert config.num_res_blocks_middle == 2
    assert config.norm_type == "layer"
    assert config.use_fp16 is True


def test_sparse_forward_probe_runs_input_projection_and_block0_forward(tmp_path):
    config = _small_sparse_flow_config()
    checkpoint = tmp_path / "flow.safetensors"
    _write_sparse_flow_checkpoint(checkpoint, config)

    conditioning = mx.zeros((1, 3, config.cond_channels), dtype=mx.float32)
    probe = probe_sparse_structure_forward_boundary(checkpoint, config, conditioning=conditioning)

    assert probe.token_shape == (1, 8, 2)
    assert probe.input_projection_shape == (1, 8, 6)
    assert probe.block0_output_shape == (1, 8, 6)
    assert probe.completed_blocks == 1
    assert probe.stack_output_shape == (1, 8, 6)
    assert probe.output_projection_shape == (1, 8, 2)
    assert set(SPARSE_STRUCTURE_INPUT_TENSOR_NAMES).issubset(probe.loaded_tensor_names)
    assert set(probe.inspected_tensor_names) == set(SPARSE_STRUCTURE_BLOCK0_INSPECTION_NAMES)
    assert probe.sampled_latent_shape == (1, 2, 2, 2, 2)
    assert probe.blocker_operation == "sparse structure decoder handoff"
    assert "block-0 ModulatedTransformerCrossBlock executed" in probe.blocker_detail
    assert "all 1 sparse flow transformer blocks executed" in probe.blocker_detail
    assert "FlowEuler sampler executed 1 steps" in probe.blocker_detail


def test_sparse_forward_probe_reports_missing_requested_tensor(tmp_path):
    config = _small_sparse_flow_config()
    checkpoint = tmp_path / "flow.safetensors"
    _write_sparse_flow_checkpoint(checkpoint, config, omit=("blocks.0.cross_attn.to_kv.weight",))

    with pytest.raises(ValueError, match="blocks.0.cross_attn.to_kv.weight"):
        probe_sparse_structure_forward_boundary(checkpoint, config)


def test_sparse_forward_probe_reports_input_projection_shape_mismatch(tmp_path):
    config = _small_sparse_flow_config()
    checkpoint = tmp_path / "flow.safetensors"
    _write_sparse_flow_checkpoint(checkpoint, config, bad_input_weight_shape=True)

    with pytest.raises(ValueError, match="input_layer.weight shape mismatch"):
        probe_sparse_structure_forward_boundary(checkpoint, config)


def test_sparse_decoder_probe_reports_upstream_latent_boundary(tmp_path):
    config = _small_sparse_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_sparse_decoder_checkpoint(checkpoint, config)

    probe = probe_sparse_structure_decoder_boundary(checkpoint, config)

    assert set(SPARSE_STRUCTURE_DECODER_TENSOR_NAMES).issubset(probe.loaded_tensor_names)
    assert set(SPARSE_STRUCTURE_DECODER_TENSOR_NAMES).issubset(probe.inspected_tensor_names)
    assert probe.latent_shape is None
    assert probe.decoded_shape is None
    assert probe.coordinates_shape is None
    assert probe.blocker_operation == "sparse structure decoder upstream latent availability"


def test_sparse_decoder_probe_reports_conv_stack_boundary_with_latent(tmp_path):
    config = _small_sparse_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_sparse_decoder_checkpoint(checkpoint, config)
    latent = mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)

    probe = probe_sparse_structure_decoder_boundary(checkpoint, config, sparse_latent=latent)

    assert probe.latent_shape == (1, 2, 2, 2, 2)
    assert probe.decoded_shape == (1, 1, 4, 4, 4)
    assert probe.coordinates_shape == (0, 4)
    assert probe.target_resolution == 4
    assert probe.blocker_operation == "sparse structure decoder coordinate extraction"
    assert "thresholding produced sparse coordinate shape" in probe.blocker_detail


def test_sparse_decoder_probe_applies_target_resolution(tmp_path):
    config = _small_sparse_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_sparse_decoder_checkpoint(checkpoint, config)
    latent = mx.zeros((1, 2, 2, 2, 2), dtype=mx.float32)

    probe = probe_sparse_structure_decoder_boundary(checkpoint, config, sparse_latent=latent, target_resolution=2)

    assert probe.decoded_shape == (1, 1, 4, 4, 4)
    assert probe.target_resolution == 2
    assert "target_resolution=2" in probe.blocker_detail


def test_sparse_decoder_probe_reports_missing_checkpoint_tensor(tmp_path):
    config = _small_sparse_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_sparse_decoder_checkpoint(checkpoint, config, omit=("out_layer.2.weight",))

    with pytest.raises(ValueError, match="out_layer.2.weight"):
        probe_sparse_structure_decoder_boundary(checkpoint, config)


def test_extract_sparse_structure_coordinates_uses_batch_zyx_order():
    array = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    array[0, 0, 0, 1, 1] = 0.25
    array[0, 0, 1, 0, 1] = 0.5
    logits = mx.array(array)

    result = extract_sparse_structure_coordinates(logits)

    assert result.decoded_shape == (1, 1, 2, 2, 2)
    assert result.target_resolution == 2
    assert result.coordinate_shape == (2, 4)
    assert result.coordinates.tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]


def test_extract_sparse_structure_coordinates_pools_to_target_resolution():
    array = np.zeros((1, 1, 4, 4, 4), dtype=np.float32)
    array[0, 0, 2, 3, 1] = 1.0
    logits = mx.array(array)

    result = extract_sparse_structure_coordinates(logits, target_resolution=2)

    assert result.target_resolution == 2
    assert result.coordinate_shape == (1, 4)
    assert result.coordinates.tolist() == [[0, 1, 1, 0]]
