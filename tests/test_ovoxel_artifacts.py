import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.ovoxel_artifacts import (
    write_decoded_ovoxel_shape_npz,
    write_decoded_ovoxel_texture_npz,
)


def test_write_decoded_ovoxel_shape_npz_records_contract_and_metadata(tmp_path):
    artifact = write_decoded_ovoxel_shape_npz(
        tmp_path / "shape_decoder_fields.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        mx.ones((2, 7), dtype=mx.float32),
        subdivisions=(mx.zeros((2, 8), dtype=mx.float32),),
        metadata={"model_family": "pixal3d"},
    )

    assert artifact.coordinates_shape == (2, 4)
    assert artifact.fields_shape == (2, 7)
    assert artifact.subdivision_shapes == ((2, 8),)
    with np.load(artifact.path) as payload:
        assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
        assert payload["fields"].shape == (2, 7)
        assert payload["subdivision_0"].shape == (2, 8)
        metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_decoder_fields"
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["model_family"] == "pixal3d"


def test_write_decoded_ovoxel_shape_npz_requires_flexidualgrid_width(tmp_path):
    with pytest.raises(ValueError, match=r"shape \(n, 7\)"):
        write_decoded_ovoxel_shape_npz(
            tmp_path / "shape_decoder_fields.npz",
            mx.zeros((2, 4), dtype=mx.int32),
            mx.zeros((2, 6), dtype=mx.float32),
        )


def test_write_decoded_ovoxel_texture_npz_records_pbr_contract(tmp_path):
    artifact = write_decoded_ovoxel_texture_npz(
        tmp_path / "texture_decoder_pbr.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        mx.ones((2, 6), dtype=mx.float32),
        spatial_shape=(2, 2, 2),
        batch_size=1,
        decode_resolution=1024,
        voxel_size=1.0 / 1024.0,
        metadata={"model_family": "trellis2"},
    )

    assert artifact.coordinates_shape == (2, 4)
    assert artifact.attributes_shape == (2, 6)
    assert artifact.spatial_shape == (2, 2, 2)
    assert artifact.batch_size == 1
    assert artifact.decode_resolution == 1024
    with np.load(artifact.path) as payload:
        assert payload["attributes"].shape == (2, 6)
        assert payload["spatial_shape"].tolist() == [2, 2, 2]
        metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "texture_decoder_pbr"
    assert metadata["attribute_channels"] == [
        "base_color_r",
        "base_color_g",
        "base_color_b",
        "metallic",
        "roughness",
        "alpha",
    ]
    assert metadata["model_family"] == "trellis2"
