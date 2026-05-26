import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_spatial.pixal3d_export import write_pixal3d_projection_npz
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
