import json

import mlx.core as mx
import numpy as np
from safetensors.mlx import save_file

from mlx_spatial.trellis2_slat import (
    SLatFlowConfig,
    _slat_cross_attention,
    probe_shape_slat_forward_boundary,
    read_slat_flow_config,
)
from mlx_spatial.trellis2_sparse_structure import (
    SparseStructureFlowConfig,
    _sparse_structure_cross_attention,
    probe_sparse_structure_forward_boundary,
    read_sparse_structure_flow_config,
)


def test_pixal3d_sparse_flow_config_parses_projection_attention(tmp_path):
    path = tmp_path / "ss.json"
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
                    "image_attn_mode": "proj",
                    "dtype": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )

    config = read_sparse_structure_flow_config(tmp_path, "ss.json")

    assert config.image_attn_mode == "proj"
    assert config.proj_in_channels is None


def test_pixal3d_slat_flow_config_parses_projection_channels(tmp_path):
    path = tmp_path / "slat.json"
    path.write_text(
        json.dumps(
            {
                "name": "ElasticSLatFlowModel",
                "args": {
                    "resolution": 32,
                    "in_channels": 32,
                    "out_channels": 32,
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
                    "image_attn_mode": "proj",
                    "proj_in_channels": 2048,
                    "dtype": "bfloat16",
                },
            }
        ),
        encoding="utf-8",
    )

    config = read_slat_flow_config(tmp_path, "slat.json")

    assert config.name == "ElasticSLatFlowModel"
    assert config.image_attn_mode == "proj"
    assert config.proj_in_channels == 2048


def test_pixal3d_sparse_project_attention_adds_projected_context():
    config = SparseStructureFlowConfig(
        name="SparseStructureFlowModel",
        resolution=2,
        in_channels=2,
        out_channels=2,
        model_channels=2,
        cond_channels=4,
        num_blocks=1,
        num_heads=1,
        mlp_ratio=2.0,
        pe_mode="ape",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=False,
        qk_rms_norm_cross=False,
        dtype="bfloat16",
        image_attn_mode="proj",
        proj_in_channels=3,
    )
    hidden = mx.zeros((1, 3, 2), dtype=mx.float32)
    conditioning = {
        "global": mx.zeros((1, 1, 4), dtype=mx.float32),
        "proj": mx.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], dtype=mx.float32),
    }
    tensors = _project_attention_tensors(model_channels=2, cond_channels=4, proj_in_channels=3)

    output = _sparse_structure_cross_attention(hidden, conditioning, config, tensors, block_index=0)

    np.testing.assert_allclose(
        np.array(output[0]),
        [[1.5, 1.5], [4.5, 4.5], [7.5, 7.5]],
        atol=1e-6,
    )


def test_pixal3d_slat_project_attention_adds_sparse_projected_context():
    config = SLatFlowConfig(
        name="ElasticSLatFlowModel",
        resolution=32,
        in_channels=32,
        out_channels=32,
        model_channels=2,
        cond_channels=4,
        num_blocks=1,
        num_heads=1,
        mlp_ratio=2.0,
        pe_mode="ape",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=False,
        qk_rms_norm_cross=False,
        dtype="bfloat16",
        image_attn_mode="proj",
        proj_in_channels=3,
    )
    hidden = mx.zeros((3, 2), dtype=mx.float32)
    conditioning = {
        "global": mx.zeros((1, 1, 4), dtype=mx.float32),
        "proj": mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=mx.float32),
    }
    tensors = _project_attention_tensors(model_channels=2, cond_channels=4, proj_in_channels=3)

    output = _slat_cross_attention(hidden, conditioning, config, tensors, block_index=0)

    np.testing.assert_allclose(
        np.array(output),
        [[1.5, 1.5], [4.5, 4.5], [7.5, 7.5]],
        atol=1e-6,
    )


def test_pixal3d_sparse_flow_probe_loads_project_attention_checkpoint(tmp_path):
    config = _tiny_sparse_proj_config()
    checkpoint = tmp_path / "ss.safetensors"
    save_file(_tiny_flow_checkpoint(config), checkpoint)
    conditioning = {
        "global": mx.zeros((1, 1, config.cond_channels), dtype=mx.float32),
        "proj": mx.zeros((1, config.resolution**3, config.proj_in_channels), dtype=mx.float32),
    }

    probe = probe_sparse_structure_forward_boundary(checkpoint, config, conditioning=conditioning)

    assert probe.block0_output_shape == (1, config.resolution**3, config.model_channels)
    assert f"blocks.0.cross_attn.cross_attn_block.to_q.weight" in probe.loaded_tensor_names
    assert f"blocks.0.cross_attn.proj_linear.weight" in probe.loaded_tensor_names


def test_pixal3d_slat_flow_probe_loads_project_attention_checkpoint(tmp_path):
    config = _tiny_slat_proj_config()
    checkpoint = tmp_path / "slat.safetensors"
    save_file(_tiny_flow_checkpoint(config), checkpoint)
    coordinates = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    conditioning = {
        "global": mx.zeros((1, 1, config.cond_channels), dtype=mx.float32),
        "proj": mx.zeros((2, config.proj_in_channels), dtype=mx.float32),
    }

    probe = probe_shape_slat_forward_boundary(checkpoint, config, coordinates, conditioning=conditioning)

    assert probe.block0_output_shape == (2, config.model_channels)
    assert "blocks.0.cross_attn.cross_attn_block.to_q.weight" in probe.loaded_tensor_names
    assert "blocks.0.cross_attn.proj_linear.weight" in probe.loaded_tensor_names


def _project_attention_tensors(*, model_channels: int, cond_channels: int, proj_in_channels: int) -> dict[str, mx.array]:
    prefix = "blocks.0.cross_attn"
    return {
        f"{prefix}.cross_attn_block.to_q.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.cross_attn_block.to_q.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn_block.to_kv.weight": mx.zeros((model_channels * 2, cond_channels), dtype=mx.float32),
        f"{prefix}.cross_attn_block.to_kv.bias": mx.zeros((model_channels * 2,), dtype=mx.float32),
        f"{prefix}.cross_attn_block.to_out.weight": mx.zeros((model_channels, model_channels), dtype=mx.float32),
        f"{prefix}.cross_attn_block.to_out.bias": mx.zeros((model_channels,), dtype=mx.float32),
        f"{prefix}.proj_linear.weight": mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=mx.float32),
        f"{prefix}.proj_linear.bias": mx.array([0.5, -0.5], dtype=mx.float32),
    }


def _tiny_sparse_proj_config() -> SparseStructureFlowConfig:
    return SparseStructureFlowConfig(
        name="SparseStructureFlowModel",
        resolution=2,
        in_channels=2,
        out_channels=2,
        model_channels=6,
        cond_channels=4,
        num_blocks=1,
        num_heads=1,
        mlp_ratio=2.0,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="bfloat16",
        image_attn_mode="proj",
        proj_in_channels=3,
    )


def _tiny_slat_proj_config() -> SLatFlowConfig:
    return SLatFlowConfig(
        name="ElasticSLatFlowModel",
        resolution=32,
        in_channels=32,
        out_channels=32,
        model_channels=6,
        cond_channels=4,
        num_blocks=1,
        num_heads=1,
        mlp_ratio=2.0,
        pe_mode="rope",
        share_mod=True,
        initialization="scaled",
        qk_rms_norm=True,
        qk_rms_norm_cross=True,
        dtype="bfloat16",
        image_attn_mode="proj",
        proj_in_channels=3,
    )


def _tiny_flow_checkpoint(config: SparseStructureFlowConfig | SLatFlowConfig) -> dict[str, mx.array]:
    prefix = "blocks.0"
    head_dim = config.model_channels // config.num_heads
    return {
        "input_layer.weight": mx.ones((config.model_channels, config.in_channels), dtype=mx.float32),
        "input_layer.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "out_layer.weight": mx.zeros((config.out_channels, config.model_channels), dtype=mx.float32),
        "out_layer.bias": mx.zeros((config.out_channels,), dtype=mx.float32),
        "t_embedder.mlp.0.weight": mx.zeros((config.model_channels, 256), dtype=mx.float32),
        "t_embedder.mlp.0.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "t_embedder.mlp.2.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        "t_embedder.mlp.2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        "adaLN_modulation.1.weight": mx.zeros((config.model_channels * 6, config.model_channels), dtype=mx.float32),
        "adaLN_modulation.1.bias": mx.zeros((config.model_channels * 6,), dtype=mx.float32),
        f"{prefix}.modulation": mx.zeros((config.model_channels * 6,), dtype=mx.float32),
        f"{prefix}.norm2.weight": mx.ones((config.model_channels,), dtype=mx.float32),
        f"{prefix}.norm2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        f"{prefix}.self_attn.to_qkv.weight": mx.zeros((config.model_channels * 3, config.model_channels), dtype=mx.float32),
        f"{prefix}.self_attn.to_qkv.bias": mx.zeros((config.model_channels * 3,), dtype=mx.float32),
        f"{prefix}.self_attn.q_rms_norm.gamma": mx.ones((config.num_heads, head_dim), dtype=mx.float32),
        f"{prefix}.self_attn.k_rms_norm.gamma": mx.ones((config.num_heads, head_dim), dtype=mx.float32),
        f"{prefix}.self_attn.to_out.weight": mx.zeros((config.model_channels, config.model_channels), dtype=mx.float32),
        f"{prefix}.self_attn.to_out.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_q.weight": mx.zeros(
            (config.model_channels, config.model_channels), dtype=mx.float32
        ),
        f"{prefix}.cross_attn.cross_attn_block.to_q.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_kv.weight": mx.zeros(
            (config.model_channels * 2, config.cond_channels), dtype=mx.float32
        ),
        f"{prefix}.cross_attn.cross_attn_block.to_kv.bias": mx.zeros((config.model_channels * 2,), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.q_rms_norm.gamma": mx.ones((config.num_heads, head_dim), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.k_rms_norm.gamma": mx.ones((config.num_heads, head_dim), dtype=mx.float32),
        f"{prefix}.cross_attn.cross_attn_block.to_out.weight": mx.zeros(
            (config.model_channels, config.model_channels), dtype=mx.float32
        ),
        f"{prefix}.cross_attn.cross_attn_block.to_out.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        f"{prefix}.cross_attn.proj_linear.weight": mx.zeros(
            (config.model_channels, config.proj_in_channels or config.cond_channels), dtype=mx.float32
        ),
        f"{prefix}.cross_attn.proj_linear.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
        f"{prefix}.mlp.mlp.0.weight": mx.zeros((int(config.model_channels * config.mlp_ratio), config.model_channels), dtype=mx.float32),
        f"{prefix}.mlp.mlp.0.bias": mx.zeros((int(config.model_channels * config.mlp_ratio),), dtype=mx.float32),
        f"{prefix}.mlp.mlp.2.weight": mx.zeros((config.model_channels, int(config.model_channels * config.mlp_ratio)), dtype=mx.float32),
        f"{prefix}.mlp.mlp.2.bias": mx.zeros((config.model_channels,), dtype=mx.float32),
    }
