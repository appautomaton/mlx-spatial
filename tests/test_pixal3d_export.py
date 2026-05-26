import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.pixal3d_export import (
    write_pixal3d_projection_npz,
    write_pixal3d_shape_decoder_npz,
    write_pixal3d_shape_hr_coordinates_npz,
    write_pixal3d_shape_slat_npz,
    write_pixal3d_sparse_structure_npz,
    write_pixal3d_texture_decoder_npz,
    write_pixal3d_texture_slat_npz,
)
from mlx_spatial.pixal3d_projection import Pixal3DProjectionConditioning, pixal3d_projection_stage_config


def test_write_pixal3d_projection_npz_records_features_and_metadata(tmp_path):
    conditioning = Pixal3DProjectionConditioning(
        stage=pixal3d_projection_stage_config("ss"),
        global_tokens=mx.ones((1, 5, 3), dtype=mx.float32),
        projected_features=mx.zeros((1, 8, 3), dtype=mx.float32),
    )

    artifact = write_pixal3d_projection_npz(
        tmp_path / "sparse_projection.npz",
        conditioning,
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.global_shape == (1, 5, 3)
    assert artifact.projected_shape == (1, 8, 3)
    payload = np.load(artifact.path)
    assert payload["global_tokens"].shape == (1, 5, 3)
    assert payload["projected_features"].shape == (1, 8, 3)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "ss"
    assert metadata["pipeline_type"] == "1024_cascade"


def test_write_pixal3d_projection_npz_requires_completed_projection(tmp_path):
    conditioning = Pixal3DProjectionConditioning(
        stage=pixal3d_projection_stage_config("shape_512"),
        global_tokens=mx.ones((1, 5, 3), dtype=mx.float32),
        projected_features=None,
    )

    with pytest.raises(ValueError, match="global and projected features"):
        write_pixal3d_projection_npz(tmp_path / "missing.npz", conditioning)


def test_write_pixal3d_sparse_structure_npz_records_coordinates_and_metadata(tmp_path):
    artifact = write_pixal3d_sparse_structure_npz(
        tmp_path / "sparse_structure.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        decoded_shape=(1, 1, 2, 2, 2),
        target_resolution=2,
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.coordinates_shape == (2, 4)
    assert artifact.decoded_shape == (1, 1, 2, 2, 2)
    assert artifact.target_resolution == 2
    payload = np.load(artifact.path)
    assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert payload["decoded_shape"].tolist() == [1, 1, 2, 2, 2]
    assert int(payload["target_resolution"]) == 2
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "sparse_structure"
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["pipeline_type"] == "1024_cascade"


def test_write_pixal3d_sparse_structure_npz_requires_coordinate_shape(tmp_path):
    with pytest.raises(ValueError, match=r"shape \(n, 4\)"):
        write_pixal3d_sparse_structure_npz(
            tmp_path / "sparse_structure.npz",
            mx.zeros((2, 3), dtype=mx.int32),
            decoded_shape=(1, 1, 2, 2, 2),
            target_resolution=2,
        )


def test_write_pixal3d_shape_slat_npz_records_coordinates_features_and_metadata(tmp_path):
    artifact = write_pixal3d_shape_slat_npz(
        tmp_path / "shape_slat_lr.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        mx.ones((2, 32), dtype=mx.float32),
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.coordinates_shape == (2, 4)
    assert artifact.features_shape == (2, 32)
    payload = np.load(artifact.path)
    assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert payload["features"].shape == (2, 32)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_slat_lr"
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["pipeline_type"] == "1024_cascade"


def test_write_pixal3d_shape_slat_npz_requires_matching_token_count(tmp_path):
    with pytest.raises(ValueError, match="token mismatch"):
        write_pixal3d_shape_slat_npz(
            tmp_path / "shape_slat_lr.npz",
            mx.zeros((2, 4), dtype=mx.int32),
            mx.zeros((1, 32), dtype=mx.float32),
        )


def test_write_pixal3d_shape_hr_coordinates_npz_records_resolution_guard(tmp_path):
    artifact = write_pixal3d_shape_hr_coordinates_npz(
        tmp_path / "shape_slat_hr_coordinates.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        requested_hr_resolution=1536,
        actual_hr_resolution=1024,
        actual_hr_grid_resolution=64,
        max_num_tokens=49152,
        raw_upsampled_shape=(4, 4),
        metadata={"pipeline_type": "1536_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.coordinates_shape == (2, 4)
    assert artifact.requested_hr_resolution == 1536
    assert artifact.actual_hr_resolution == 1024
    assert artifact.actual_hr_grid_resolution == 64
    assert artifact.token_count == 2
    payload = np.load(artifact.path)
    assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert int(payload["actual_hr_resolution"]) == 1024
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_slat_hr_coordinates"
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["raw_upsampled_shape"] == [4, 4]
    assert metadata["token_count"] == 2


def test_write_pixal3d_texture_slat_npz_records_coordinates_features_and_metadata(tmp_path):
    artifact = write_pixal3d_texture_slat_npz(
        tmp_path / "texture_slat.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        mx.ones((2, 32), dtype=mx.float32),
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.coordinates_shape == (2, 4)
    assert artifact.features_shape == (2, 32)
    payload = np.load(artifact.path)
    assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert payload["features"].shape == (2, 32)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "texture_slat"
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["pipeline_type"] == "1024_cascade"


def test_write_pixal3d_shape_decoder_npz_records_fields_subdivisions_and_metadata(tmp_path):
    artifact = write_pixal3d_shape_decoder_npz(
        tmp_path / "shape_decoder_fields.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        mx.ones((2, 7), dtype=mx.float32),
        subdivisions=(mx.zeros((2, 8), dtype=mx.float32),),
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.coordinates_shape == (2, 4)
    assert artifact.fields_shape == (2, 7)
    assert artifact.subdivision_shapes == ((2, 8),)
    payload = np.load(artifact.path)
    assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert payload["fields"].shape == (2, 7)
    assert payload["subdivision_0"].shape == (2, 8)
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "shape_decoder_fields"
    assert metadata["coordinate_order"] == "batch,z,y,x"
    assert metadata["pipeline_type"] == "1024_cascade"


def test_write_pixal3d_shape_decoder_npz_requires_flexidualgrid_width(tmp_path):
    with pytest.raises(ValueError, match=r"shape \(n, 7\)"):
        write_pixal3d_shape_decoder_npz(
            tmp_path / "shape_decoder_fields.npz",
            mx.zeros((2, 4), dtype=mx.int32),
            mx.zeros((2, 6), dtype=mx.float32),
        )


def test_write_pixal3d_texture_decoder_npz_records_pbr_voxels_and_metadata(tmp_path):
    artifact = write_pixal3d_texture_decoder_npz(
        tmp_path / "texture_decoder_pbr.npz",
        mx.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=mx.int32),
        mx.ones((2, 6), dtype=mx.float32),
        spatial_shape=(2, 2, 2),
        batch_size=1,
        decode_resolution=1024,
        voxel_size=1.0 / 1024.0,
        metadata={"pipeline_type": "1024_cascade"},
    )

    assert artifact.path.is_file()
    assert artifact.coordinates_shape == (2, 4)
    assert artifact.attributes_shape == (2, 6)
    assert artifact.spatial_shape == (2, 2, 2)
    assert artifact.batch_size == 1
    assert artifact.decode_resolution == 1024
    payload = np.load(artifact.path)
    assert payload["coordinates"].tolist() == [[0, 0, 1, 1], [0, 1, 0, 1]]
    assert payload["attributes"].shape == (2, 6)
    assert payload["spatial_shape"].tolist() == [2, 2, 2]
    assert int(payload["decode_resolution"]) == 1024
    metadata = json.loads(payload["metadata_json"].item())
    assert metadata["stage"] == "texture_decoder_pbr"
    assert metadata["attribute_channels"] == ["base_color_r", "base_color_g", "base_color_b", "metallic", "roughness", "alpha"]
    assert metadata["pipeline_type"] == "1024_cascade"
