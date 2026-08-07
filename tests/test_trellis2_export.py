from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_spatial
from mlx_spatial.ovoxel import FlexibleDualGridMesh
from mlx_spatial.trellis2 import TRELLIS2_GLB_DEFAULT_FACE_TARGET
from mlx_spatial.trellis2_export import (
    SUPPORTED_TRELLIS2_EXPORT_SUFFIXES,
    Trellis2ExportArtifact,
    sparse_coordinates_to_obj_payload,
    validate_trellis2_export_path,
    write_flexible_dual_grid_obj,
    write_sparse_coordinate_preview_obj,
)


def _fixture_mesh() -> FlexibleDualGridMesh:
    return FlexibleDualGridMesh(
        vertices=np.array(
            [
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
                [-0.25, 0.25, 0.0],
                [0.25, 0.25, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64),
    )


def test_validate_export_path_accepts_arbitrary_destination_without_root(tmp_path: Path) -> None:
    output = tmp_path / "artifacts" / "model.obj"

    assert validate_trellis2_export_path(output) == output.resolve()


def test_validate_export_path_can_enforce_explicit_root(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    inside = outputs_root / "model.obj"

    assert validate_trellis2_export_path(inside, outputs_root=outputs_root) == inside.resolve()
    with pytest.raises(ValueError, match="must stay under"):
        validate_trellis2_export_path(tmp_path / "outside" / "model.obj", outputs_root=outputs_root)


def test_validate_export_path_rejects_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported TRELLIS.2 export format"):
        validate_trellis2_export_path(tmp_path / "model.ply")


def test_sparse_coordinate_preview_writes_coarse_obj(tmp_path: Path) -> None:
    coordinates = mx.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=mx.int32)
    payload = sparse_coordinates_to_obj_payload(coordinates, grid_size=2)

    text = payload.decode("utf-8")
    assert "coarse voxel OBJ" in text
    assert text.count("\nv ") == 40
    assert text.count("\nf ") == 10

    output = tmp_path / "preview.obj"
    artifact = write_sparse_coordinate_preview_obj(coordinates, output, grid_size=2)
    assert artifact.path == output.resolve()
    assert artifact.bytes_written == len(payload)


def test_sparse_coordinate_preview_validates_contract() -> None:
    with pytest.raises(ValueError, match="shape"):
        sparse_coordinates_to_obj_payload(mx.zeros((2, 3), dtype=mx.int32))
    with pytest.raises(ValueError, match="at least one token"):
        sparse_coordinates_to_obj_payload(mx.zeros((0, 4), dtype=mx.int32))
    with pytest.raises(ValueError, match="batch index 0"):
        sparse_coordinates_to_obj_payload(mx.array([[1, 0, 0, 0]], dtype=mx.int32))


def test_write_flexible_dual_grid_obj_supports_temp_paths(tmp_path: Path) -> None:
    output = tmp_path / "shape.obj"

    artifact = write_flexible_dual_grid_obj(_fixture_mesh(), output)

    text = output.read_text(encoding="utf-8")
    assert artifact.path == output.resolve()
    assert artifact.format == "obj"
    assert text.count("\nv ") == 4
    assert text.count("\nf ") == 2


def test_shape_export_helpers_are_public() -> None:
    assert mlx_spatial.SUPPORTED_TRELLIS2_EXPORT_SUFFIXES == SUPPORTED_TRELLIS2_EXPORT_SUFFIXES
    assert mlx_spatial.TRELLIS2_GLB_DEFAULT_FACE_TARGET == TRELLIS2_GLB_DEFAULT_FACE_TARGET
    assert mlx_spatial.Trellis2ExportArtifact is Trellis2ExportArtifact
    assert mlx_spatial.sparse_coordinates_to_obj_payload is sparse_coordinates_to_obj_payload
    assert mlx_spatial.validate_trellis2_export_path is validate_trellis2_export_path
    assert mlx_spatial.write_flexible_dual_grid_obj is write_flexible_dual_grid_obj
    assert mlx_spatial.write_sparse_coordinate_preview_obj is write_sparse_coordinate_preview_obj
