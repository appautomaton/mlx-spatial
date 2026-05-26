import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.pixal3d_export import write_pixal3d_projection_npz, write_pixal3d_sparse_structure_npz
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
