import numpy as np
import mlx.core as mx
from safetensors.mlx import save_file

from mlx_spatial.sam3d_decoder import (
    Sam3dMeshDecoderConfig,
    Sam3dSLatDecoderConfig,
    load_sam3d_mesh_decoder_tensors,
    read_sam3d_mesh_decoder_config,
    _window_partition,
    run_sam3d_slat_decoder_network,
    run_sam3d_slat_decoder_torso,
)
from mlx_spatial.sam3d_mesh import run_sam3d_mesh_decoder_features


def test_sam3d_decoder_window_partition_groups_shifted_windows():
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 4, 0, 0],
        ],
        dtype=np.int32,
    )

    _fwd, _bwd, seq_lens = _window_partition(coords, window_size=4, shift_window=0)

    assert sorted(seq_lens) == [1, 2]


def test_sam3d_slat_decoder_network_runs_tiny_windowed_fixture():
    config = Sam3dSLatDecoderConfig(
        resolution=4,
        model_channels=2,
        latent_channels=2,
        num_blocks=1,
        num_heads=1,
        window_size=4,
    )
    tensors = {
        "input_layer.weight": mx.eye(2, dtype=mx.float32),
        "input_layer.bias": mx.zeros((2,), dtype=mx.float32),
        "blocks.0.attn.to_qkv.weight": mx.array(np.concatenate([np.eye(2, dtype=np.float32)] * 3, axis=0)),
        "blocks.0.attn.to_qkv.bias": mx.zeros((6,), dtype=mx.float32),
        "blocks.0.attn.to_out.weight": mx.eye(2, dtype=mx.float32),
        "blocks.0.attn.to_out.bias": mx.zeros((2,), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.weight": mx.zeros((4, 2), dtype=mx.float32),
        "blocks.0.mlp.mlp.0.bias": mx.zeros((4,), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.weight": mx.zeros((2, 4), dtype=mx.float32),
        "blocks.0.mlp.mlp.2.bias": mx.zeros((2,), dtype=mx.float32),
        "out_layer.weight": mx.eye(2, dtype=mx.float32),
        "out_layer.bias": mx.zeros((2,), dtype=mx.float32),
    }
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    feats = mx.ones((2, 2), dtype=mx.float32)

    out = run_sam3d_slat_decoder_network(coords, feats, tensors, config)

    assert tuple(out.shape) == (2, 2)


def test_sam3d_mesh_decoder_config_reads_active_target_and_representation(tmp_path):
    config_path = tmp_path / "slat_decoder_mesh.yaml"
    config_path.write_text(
        """
_target_: sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh.SLatMeshDecoderTdfyWrapper
resolution: 64
model_channels: 768
latent_channels: 8
num_blocks: 12
num_heads: 12
window_size: 8
representation_config:
  use_color: true
""".strip(),
        encoding="utf-8",
    )

    config = read_sam3d_mesh_decoder_config(config_path)

    assert config.resolution == 64
    assert config.model_channels == 768
    assert config.latent_channels == 8
    assert config.num_blocks == 12
    assert config.num_heads == 12
    assert config.window_size == 8
    assert config.use_color is True
    assert config.representation_config == {"use_color": True}


def test_sam3d_mesh_decoder_tensor_loader_requires_mesh_prefixes(tmp_path):
    path = tmp_path / "slat_decoder_mesh.safetensors"
    save_file(
        {
            "input_layer.weight": mx.ones((1,), dtype=mx.float32),
            "blocks.0.weight": mx.ones((1,), dtype=mx.float32),
            "upsample.0.weight": mx.ones((1,), dtype=mx.float32),
            "out_layer.weight": mx.ones((1,), dtype=mx.float32),
            "offset_perturbation": mx.ones((1,), dtype=mx.float32),
        },
        path,
    )

    tensors = load_sam3d_mesh_decoder_tensors(path)

    assert set(tensors) == {
        "input_layer.weight",
        "blocks.0.weight",
        "upsample.0.weight",
        "out_layer.weight",
    }


def test_sam3d_slat_decoder_torso_can_feed_distinct_output_heads():
    config = Sam3dSLatDecoderConfig(
        resolution=4,
        model_channels=2,
        latent_channels=2,
        num_blocks=0,
        num_heads=1,
        window_size=4,
    )
    tensors = {
        "input_layer.weight": mx.eye(2, dtype=mx.float32),
        "input_layer.bias": mx.zeros((2,), dtype=mx.float32),
    }
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    feats = mx.ones((2, 2), dtype=mx.float32)

    torso = run_sam3d_slat_decoder_torso(coords, feats, tensors, config)
    gaussian_weight = mx.ones((3, 2), dtype=mx.float32)
    mesh_weight = mx.ones((5, 2), dtype=mx.float32)
    gaussian = torso @ mx.transpose(gaussian_weight)
    mesh = torso @ mx.transpose(mesh_weight)

    assert tuple(torso.shape) == (2, 2)
    assert tuple(gaussian.shape) == (2, 3)
    assert tuple(mesh.shape) == (2, 5)


def test_sam3d_mesh_decoder_features_trace_records_subdivision_counts():
    config = Sam3dMeshDecoderConfig(
        resolution=4,
        model_channels=2,
        latent_channels=2,
        num_blocks=0,
        num_heads=1,
        window_size=4,
    )
    tensors = {
        "input_layer.weight": mx.eye(2, dtype=mx.float32),
        "input_layer.bias": mx.zeros((2,), dtype=mx.float32),
        **_mesh_upsample_tensors("upsample.0.", 2, 2, skip_projection=False),
        **_mesh_upsample_tensors("upsample.1.", 2, 1, skip_projection=True),
        "out_layer.weight": mx.ones((5, 1), dtype=mx.float32),
        "out_layer.bias": mx.zeros((5,), dtype=mx.float32),
    }
    coords = np.array([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    feats = mx.array([[1.0, 3.0], [2.0, 4.0]], dtype=mx.float32)

    out = run_sam3d_mesh_decoder_features(coords, feats, tensors, config)

    assert out.coords.shape == (128, 4)
    assert tuple(out.feats.shape) == (128, 5)
    assert out.metadata["subdivisions"] == [
        {"block": 0, "input_tokens": 2, "output_tokens": 16},
        {"block": 1, "input_tokens": 16, "output_tokens": 128},
    ]


def _mesh_upsample_tensors(prefix: str, in_channels: int, out_channels: int, *, skip_projection: bool) -> dict[str, mx.array]:
    tensors = {
        f"{prefix}act_layers.0.weight": mx.ones((in_channels,), dtype=mx.float32),
        f"{prefix}act_layers.0.bias": mx.zeros((in_channels,), dtype=mx.float32),
        f"{prefix}out_layers.0.conv.weight": mx.zeros((out_channels, 3, 3, 3, in_channels), dtype=mx.float32),
        f"{prefix}out_layers.0.conv.bias": mx.zeros((out_channels,), dtype=mx.float32),
        f"{prefix}out_layers.1.weight": mx.ones((out_channels,), dtype=mx.float32),
        f"{prefix}out_layers.1.bias": mx.zeros((out_channels,), dtype=mx.float32),
        f"{prefix}out_layers.3.conv.weight": mx.zeros((out_channels, 3, 3, 3, out_channels), dtype=mx.float32),
        f"{prefix}out_layers.3.conv.bias": mx.zeros((out_channels,), dtype=mx.float32),
    }
    if skip_projection:
        skip = np.zeros((out_channels, 1, 1, 1, in_channels), dtype=np.float32)
        skip[0, 0, 0, 0, 0] = 1.0
        tensors[f"{prefix}skip_connection.conv.weight"] = mx.array(skip)
        tensors[f"{prefix}skip_connection.conv.bias"] = mx.zeros((out_channels,), dtype=mx.float32)
    return tensors
