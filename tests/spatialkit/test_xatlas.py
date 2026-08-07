from __future__ import annotations

import numpy as np
import pytest

from mlx_spatial.spatialkit import (
    make_xatlas_uvs,
    resolve_xatlas_parallel_chunks,
    unwrap_xatlas_spatial,
)


def _quad_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    return vertices, faces


def test_resolve_xatlas_parallel_chunks_contract() -> None:
    assert resolve_xatlas_parallel_chunks(100_000, 3) == 3
    assert resolve_xatlas_parallel_chunks(2, 8) == 2
    assert resolve_xatlas_parallel_chunks(50_000, 0) == 1
    assert resolve_xatlas_parallel_chunks(
        120_000,
        0,
        face_target=50_000,
        max_auto_chunks=8,
    ) == 3
    with pytest.raises(ValueError, match="requested_chunks"):
        resolve_xatlas_parallel_chunks(10, -1)


def test_spatial_xatlas_preserves_face_geometry_and_reports_partition_cuts() -> None:
    vertices, faces = _quad_mesh()

    result = unwrap_xatlas_spatial(vertices, faces, chunks=2)

    assert result.stats["backend"] == "xatlas-parallel-spatial"
    assert result.stats["chunks"] == 2
    assert result.stats["chunk_faces"] == (1, 1)
    assert result.stats["spatial_partition_shared_edge_count"] == 1
    assert result.stats["spatial_partition_cut_edge_count"] == 1
    assert result.stats["spatial_partition_cut_edge_ratio"] == 1.0
    np.testing.assert_allclose(
        result.vertices[result.faces],
        vertices[faces[result.source_face_ids]],
    )
    np.testing.assert_array_equal(np.sort(result.source_face_ids), np.arange(faces.shape[0]))
    assert np.all(result.chart_ids >= 0)
    assert np.all(result.uvs >= 0.0)
    assert np.all(result.uvs <= 1.0)


def test_make_xatlas_uvs_measures_spatial_output_in_its_face_order() -> None:
    vertices, faces = _quad_mesh()

    mesh = make_xatlas_uvs(vertices, faces, parallel_chunks=2)

    assert mesh.stats["backend"] == "xatlas-parallel-spatial"
    assert mesh.stats["unassigned_surface_area_ratio"] == 0.0
    assert mesh.stats["uv_degenerate_surface_area_ratio"] == 0.0
    assert mesh.stats["spatial_partition_cut_edge_count"] == 1
    assert mesh.faces.shape == faces.shape


def test_make_xatlas_uvs_rejects_conflicting_partition_modes() -> None:
    vertices, faces = _quad_mesh()

    with pytest.raises(ValueError, match="mutually exclusive"):
        make_xatlas_uvs(vertices, faces, clustered=True, parallel_chunks=2)
