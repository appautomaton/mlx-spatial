import json

import mlx.core as mx
import pytest
from safetensors.mlx import save_file

from mlx_spatial.trellis2_decode import (
    STRUCTURED_LATENT_DECODER_TENSOR_NAMES,
    StructuredLatentDecoderConfig,
    probe_decode_latents_boundary,
    probe_structured_latent_decoder_boundary,
    read_structured_latent_decoder_config,
    run_shape_decoder_to_fields,
    run_shape_decoder_upsample_coordinates,
)


def _small_decoder_config(*, name="SparseUnetVaeDecoder", out_channels=6, pred_subdiv=False):
    return StructuredLatentDecoderConfig(
        name=name,
        latent_channels=32,
        model_channels=(16, 8),
        num_blocks=(1, 0),
        block_type=("SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"),
        up_block_type=("SparseResBlockC2S3d",),
        use_fp16=False,
        out_channels=out_channels,
        resolution=256 if name == "FlexiDualGridVaeDecoder" else None,
        pred_subdiv=pred_subdiv,
    )


def _write_decoder_config(path, *, name="SparseUnetVaeDecoder", out_channels=6, pred_subdiv=False):
    args = {
        "model_channels": [16, 8],
        "latent_channels": 32,
        "num_blocks": [1, 0],
        "block_type": ["SparseConvNeXtBlock3d", "SparseConvNeXtBlock3d"],
        "up_block_type": ["SparseResBlockC2S3d"],
        "block_args": [{}, {}],
        "use_fp16": False,
    }
    if name == "FlexiDualGridVaeDecoder":
        args["resolution"] = 256
    else:
        args["out_channels"] = out_channels
        args["pred_subdiv"] = pred_subdiv
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "args": args}))


def _write_decoder_checkpoint(path, config=None, *, omit=(), bad_from_latent_shape=False):
    config = config or _small_decoder_config()
    tensors = {
        "from_latent.weight": mx.ones(
            (config.model_channels[0], config.latent_channels)
            if not bad_from_latent_shape
            else (config.model_channels[0] + 1, config.latent_channels),
            dtype=mx.float32,
        ),
        "from_latent.bias": mx.zeros((config.model_channels[0],), dtype=mx.float32),
        "output_layer.weight": mx.zeros((config.out_channels, config.model_channels[-1]), dtype=mx.float32),
        "output_layer.bias": mx.zeros((config.out_channels,), dtype=mx.float32),
        "blocks.0.0.conv.weight": mx.zeros((config.model_channels[0], 3, 3, 3, config.model_channels[0]), dtype=mx.float32),
        "blocks.0.0.conv.bias": mx.zeros((config.model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.weight": mx.ones((config.model_channels[0],), dtype=mx.float32),
        "blocks.0.0.norm.bias": mx.zeros((config.model_channels[0],), dtype=mx.float32),
        "blocks.0.0.mlp.0.weight": mx.zeros((config.model_channels[0] * 4, config.model_channels[0]), dtype=mx.float32),
        "blocks.0.0.mlp.0.bias": mx.zeros((config.model_channels[0] * 4,), dtype=mx.float32),
        "blocks.0.0.mlp.2.weight": mx.zeros((config.model_channels[0], config.model_channels[0] * 4), dtype=mx.float32),
        "blocks.0.0.mlp.2.bias": mx.zeros((config.model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.weight": mx.ones((config.model_channels[0],), dtype=mx.float32),
        "blocks.0.1.norm1.bias": mx.zeros((config.model_channels[0],), dtype=mx.float32),
        "blocks.0.1.conv1.weight": mx.zeros(
            (config.model_channels[1] * 8, 3, 3, 3, config.model_channels[0]),
            dtype=mx.float32,
        ),
        "blocks.0.1.conv1.bias": mx.zeros((config.model_channels[1] * 8,), dtype=mx.float32),
        "blocks.0.1.conv2.weight": mx.zeros(
            (config.model_channels[1], 3, 3, 3, config.model_channels[1]),
            dtype=mx.float32,
        ),
        "blocks.0.1.conv2.bias": mx.zeros((config.model_channels[1],), dtype=mx.float32),
    }
    if config.pred_subdiv:
        tensors["blocks.0.1.to_subdiv.weight"] = mx.zeros((8, config.model_channels[0]), dtype=mx.float32)
        tensors["blocks.0.1.to_subdiv.bias"] = mx.ones((8,), dtype=mx.float32)
    for name in omit:
        tensors.pop(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)


def test_read_structured_latent_decoder_config_maps_shape_decoder_fields(tmp_path):
    path = tmp_path / "shape.json"
    _write_decoder_config(path, name="FlexiDualGridVaeDecoder")

    config = read_structured_latent_decoder_config(tmp_path, "shape.json")

    assert config.name == "FlexiDualGridVaeDecoder"
    assert config.resolution == 256
    assert config.out_channels == 7
    assert config.latent_channels == 32
    assert config.model_channels == (16, 8)
    assert config.up_block_type == ("SparseResBlockC2S3d",)
    assert config.pred_subdiv is True


def test_read_structured_latent_decoder_config_maps_texture_decoder_fields(tmp_path):
    path = tmp_path / "tex.json"
    _write_decoder_config(path, name="SparseUnetVaeDecoder", out_channels=6, pred_subdiv=False)

    config = read_structured_latent_decoder_config(tmp_path, "tex.json")

    assert config.name == "SparseUnetVaeDecoder"
    assert config.out_channels == 6
    assert config.pred_subdiv is False
    assert config.resolution is None


def test_structured_latent_decoder_probe_runs_from_latent_projection(tmp_path):
    config = _small_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_decoder_checkpoint(checkpoint, config)
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    feats = mx.zeros((2, 32), dtype=mx.float32)

    probe = probe_structured_latent_decoder_boundary(checkpoint, config, coords, feats, latent_name="shape_slat")

    assert probe.coordinate_shape == (2, 4)
    assert probe.feature_shape == (2, 32)
    assert probe.input_projection_shape == (2, 16)
    assert probe.convnext0_output_shape == (2, 16)
    assert probe.level0_completed_blocks == 1
    assert probe.level0_output_shape == (2, 16)
    assert probe.first_upblock_output_shape is None
    assert set(STRUCTURED_LATENT_DECODER_TENSOR_NAMES).issubset(set(probe.loaded_tensor_names))
    assert set(STRUCTURED_LATENT_DECODER_TENSOR_NAMES).issubset(set(probe.inspected_tensor_names))


def test_structured_latent_decoder_probe_runs_first_shape_c2s_upblock(tmp_path):
    config = _small_decoder_config(name="FlexiDualGridVaeDecoder", out_channels=7, pred_subdiv=True)
    checkpoint = tmp_path / "decoder.safetensors"
    _write_decoder_checkpoint(checkpoint, config)
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    feats = mx.zeros((2, 32), dtype=mx.float32)

    probe = probe_structured_latent_decoder_boundary(checkpoint, config, coords, feats, latent_name="shape_slat")

    assert probe.input_projection_shape == (2, 16)
    assert probe.level0_output_shape == (2, 16)
    assert probe.first_upblock_coordinate_shape == (16, 4)
    assert probe.first_upblock_output_shape == (16, 8)
    assert probe.first_upblock_subdivision_shape == (2, 8)
    assert probe.completed_levels == 2
    assert probe.subdivision_shapes == ((2, 8),)
    assert probe.decoder_output_coordinate_shape == (16, 4)
    assert probe.decoder_output_shape == (16, 7)
    assert probe.reference_stop is None
    assert "blocks.0.1.to_subdiv.weight" in probe.loaded_tensor_names


def test_run_shape_decoder_to_fields_returns_full_7_channel_output(tmp_path):
    config = _small_decoder_config(name="FlexiDualGridVaeDecoder", out_channels=7, pred_subdiv=True)
    checkpoint = tmp_path / "decoder.safetensors"
    _write_decoder_checkpoint(checkpoint, config)
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    feats = mx.zeros((2, 32), dtype=mx.float32)

    result = run_shape_decoder_to_fields(checkpoint, config, coords, feats)

    assert result.coordinates.shape == (16, 4)
    assert result.fields.shape == (16, 7)
    assert result.probe.completed_levels == 2
    assert result.probe.subdivision_shapes == ((2, 8),)


def test_run_shape_decoder_to_fields_allows_final_zero_block_above_conv_limit(tmp_path):
    config = StructuredLatentDecoderConfig(
        name="FlexiDualGridVaeDecoder",
        latent_channels=32,
        model_channels=(8,),
        num_blocks=(0,),
        block_type=("SparseConvNeXtBlock3d",),
        up_block_type=(),
        use_fp16=False,
        out_channels=7,
        resolution=256,
        pred_subdiv=True,
    )
    checkpoint = tmp_path / "decoder.safetensors"
    save_file(
        {
            "from_latent.weight": mx.ones((8, 32), dtype=mx.float32),
            "from_latent.bias": mx.zeros((8,), dtype=mx.float32),
            "output_layer.weight": mx.zeros((7, 8), dtype=mx.float32),
            "output_layer.bias": mx.zeros((7,), dtype=mx.float32),
        },
        checkpoint,
    )
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 4]], dtype=mx.int32)
    feats = mx.zeros((3, 32), dtype=mx.float32)

    result = run_shape_decoder_to_fields(checkpoint, config, coords, feats, decoder_token_limit=2)

    assert result.coordinates.shape == (3, 4)
    assert result.fields.shape == (3, 7)
    assert result.probe.completed_levels == 1


def test_run_shape_decoder_upsample_coordinates_returns_cascade_coords(tmp_path):
    config = _small_decoder_config(name="FlexiDualGridVaeDecoder", out_channels=7, pred_subdiv=True)
    checkpoint = tmp_path / "decoder.safetensors"
    _write_decoder_checkpoint(checkpoint, config)
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    feats = mx.zeros((2, 32), dtype=mx.float32)

    result = run_shape_decoder_upsample_coordinates(checkpoint, config, coords, feats, upsample_times=1, decoder_token_limit=2)

    assert result.input_coordinate_shape == (2, 4)
    assert result.output_coordinate_shape == (16, 4)
    assert result.completed_upsamples == 1
    assert result.subdivisions[0].shape == (2, 8)


def test_structured_latent_decoder_probe_reports_bad_latent_layout(tmp_path):
    config = _small_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_decoder_checkpoint(checkpoint, config)

    with pytest.raises(ValueError, match="feature width mismatch"):
        probe_structured_latent_decoder_boundary(
            checkpoint,
            config,
            mx.array([[0, 0, 0, 0]], dtype=mx.int32),
            mx.zeros((1, 16), dtype=mx.float32),
            latent_name="shape_slat",
        )


def test_structured_latent_decoder_probe_reports_missing_tensor(tmp_path):
    config = _small_decoder_config()
    checkpoint = tmp_path / "decoder.safetensors"
    _write_decoder_checkpoint(checkpoint, config, omit=("output_layer.weight",))
    coords = mx.array([[0, 0, 0, 0]], dtype=mx.int32)
    feats = mx.zeros((1, 32), dtype=mx.float32)

    with pytest.raises(ValueError, match="output_layer.weight"):
        probe_structured_latent_decoder_boundary(checkpoint, config, coords, feats, latent_name="shape_slat")


def test_decode_latents_probe_validates_both_decoders_and_blocks_at_shape_stack(tmp_path):
    shape_config = _small_decoder_config(name="FlexiDualGridVaeDecoder", out_channels=7, pred_subdiv=True)
    texture_config = _small_decoder_config(name="SparseUnetVaeDecoder", out_channels=6, pred_subdiv=False)
    shape_checkpoint = tmp_path / "shape.safetensors"
    texture_checkpoint = tmp_path / "tex.safetensors"
    _write_decoder_checkpoint(shape_checkpoint, shape_config)
    _write_decoder_checkpoint(texture_checkpoint, texture_config)
    coords = mx.array([[0, 0, 0, 0], [0, 1, 2, 3]], dtype=mx.int32)
    feats = mx.zeros((2, 32), dtype=mx.float32)

    probe = probe_decode_latents_boundary(
        shape_checkpoint,
        shape_config,
        texture_checkpoint,
        texture_config,
        shape_slat_coordinates=coords,
        shape_slat_features=feats,
        texture_slat_coordinates=coords,
        texture_slat_features=feats,
        resolution=1024,
    )

    assert probe.resolution == 1024
    assert probe.shape_probe.input_projection_shape == (2, 16)
    assert probe.shape_probe.convnext0_output_shape == (2, 16)
    assert probe.shape_probe.first_upblock_output_shape == (16, 8)
    assert probe.texture_probe.input_projection_shape == (2, 16)
    assert probe.texture_probe.convnext0_output_shape == (2, 16)
    assert probe.texture_probe.first_upblock_output_shape == (16, 8)
    assert probe.texture_probe.subdivision_shapes == ()
    assert probe.texture_probe.decoder_output_shape == (16, 6)
    assert probe.blocker_operation == "MLX shape latent decoder SparseConvNeXt/FlexiDualGrid forward"
    assert "first C2S up-block produced" in probe.blocker_detail
    assert "large-token sparse ConvNeXt/up-block decoder execution" in probe.blocker_detail
