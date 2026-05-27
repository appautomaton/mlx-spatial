from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.mlx import save_file

from mlx_spatial.naf import (
    NAF_WEIGHTS_FILENAME,
    NafRuntimeConfig,
    load_naf_tensors,
    naf_conversion_command,
    naf_required_tensor_shapes,
    project_naf_features_at_points,
    run_naf_image_encoder,
    validate_naf_assets,
    validate_naf_tensors,
)


def test_naf_required_shapes_match_release_checkpoint_contract():
    shapes = naf_required_tensor_shapes()

    assert shapes["image_encoder.encoder.0.weight"] == (128, 3, 1, 1)
    assert shapes["image_encoder.sem_encoder.0.weight"] == (128, 3, 3, 3)
    assert shapes["image_encoder.encoder.2.conv2.weight"] == (128, 128, 1, 1)
    assert shapes["image_encoder.sem_encoder.2.conv2.weight"] == (128, 128, 3, 3)
    assert shapes["image_encoder.rope.periods"] == (16,)


def test_naf_asset_validation_and_load_roundtrip(tmp_path):
    config = _tiny_config()
    root = tmp_path / "naf"
    root.mkdir()
    save_file(_tiny_naf_tensors(config), root / NAF_WEIGHTS_FILENAME)

    validation = validate_naf_assets(root)
    tensors = load_naf_tensors(root, config=config)

    assert validation.ready
    assert sorted(tensors) == sorted(naf_required_tensor_shapes(config))
    assert "scripts/pixal3d/convert_naf.py" in " ".join(naf_conversion_command(root))


def test_validate_naf_tensors_rejects_missing_required_key():
    config = _tiny_config()
    tensors = _tiny_naf_tensors(config)
    tensors.pop("image_encoder.rope.periods")

    try:
        validate_naf_tensors(tensors, config=config)
    except ValueError as error:
        assert "missing required tensor" in str(error)
    else:
        raise AssertionError("expected missing NAF tensor validation failure")


def test_run_naf_image_encoder_returns_rope_encoded_target_shape():
    config = _tiny_config()
    image = mx.ones((1, 3, 4, 4), dtype=mx.float32)

    encoded = run_naf_image_encoder(image, _tiny_naf_tensors(config), output_size=(4, 4), config=config)

    assert encoded.shape == (1, 8, 4, 4)
    np.testing.assert_allclose(np.array(encoded), np.zeros((1, 8, 4, 4), dtype=np.float32), atol=1e-6)


def test_project_naf_features_at_points_uses_coordinate_sampled_attention():
    config = _tiny_config()
    image = mx.zeros((1, 3, 4, 4), dtype=mx.float32)
    lr_features = mx.array(
        np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2),
    )
    points = mx.array([[[0.0, 0.0], [3.0, 3.0]]], dtype=mx.float32)

    projected = project_naf_features_at_points(
        image,
        lr_features,
        points,
        image_resolution=4,
        output_size=(4, 4),
        tensors=_tiny_naf_tensors(config),
        chunk_size=1,
        config=config,
    )

    assert projected.features.shape == (1, 2, 4)
    assert projected.target_size == (4, 4)
    assert projected.point_count == 2
    lr_np = np.array(lr_features)
    expected = np.stack((lr_np[0, :, 0, 0], lr_np[0, :, 1, 1]), axis=0)
    np.testing.assert_allclose(np.array(projected.features[0]), expected, atol=1e-6)


def _tiny_config() -> NafRuntimeConfig:
    return NafRuntimeConfig(dim=8, heads_attn=2, heads_rope=2, kernel_size=1, img_layers=0, group_count=2)


def _tiny_naf_tensors(config: NafRuntimeConfig) -> dict[str, mx.array]:
    tensors = {}
    for name, shape in naf_required_tensor_shapes(config).items():
        if name.endswith(".periods"):
            tensors[name] = mx.ones(shape, dtype=mx.float32)
        else:
            tensors[name] = mx.zeros(shape, dtype=mx.float32)
    return tensors
