from __future__ import annotations

import numpy as np
import pytest

from mlx_spatialkit import clean_mesh, mesh_metrics, simplify_mesh


def _messy_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [5.0, 5.0, 0.0],
            [6.0, 5.0, 0.0],
            [5.0, 6.0, 0.0],
            [9.0, 9.0, 9.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 1, 0],
            [0, 0, 1],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def test_mesh_metrics_reports_actionable_topology_diagnostics() -> None:
    vertices, faces = _messy_mesh()

    metrics = mesh_metrics(vertices, faces)

    assert metrics["vertex_count"] == 8
    assert metrics["face_count"] == 5
    assert metrics["degenerate_faces"] == 1
    assert metrics["duplicate_faces"] == 1
    assert metrics["connected_components"] == 2
    assert metrics["boundary_edges"] > 0
    assert metrics["nonmanifold_edges"] > 0
    assert "degenerate_faces_present" in metrics["export_blocking_reasons"]
    assert "duplicate_faces_present" in metrics["export_blocking_reasons"]
    assert "nonmanifold_edges_present" in metrics["export_blocking_reasons"]


def test_mesh_metrics_separates_boundary_edges_from_nonmanifold_edges() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

    metrics = mesh_metrics(vertices, faces)

    assert metrics["boundary_edges"] == 4
    assert metrics["nonmanifold_edges"] == 0
    assert metrics["export_blocking_reasons"] == []


def test_mesh_metrics_reports_true_nonmanifold_edges_as_export_blockers() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)

    metrics = mesh_metrics(vertices, faces)

    assert metrics["nonmanifold_edges"] == 1
    assert "nonmanifold_edges_present" in metrics["export_blocking_reasons"]


def test_clean_mesh_removes_degenerates_duplicates_unreferenced_vertices_and_small_components() -> None:
    vertices, faces = _messy_mesh()

    mesh, stats = clean_mesh(vertices, faces, min_component_faces=2)

    assert stats["degenerate_faces_removed"] == 1
    assert stats["duplicate_faces_removed"] == 1
    assert stats["components_removed"] == 1
    assert stats["component_faces_removed"] == 1
    assert stats["unreferenced_vertices_removed"] == 4
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)
    np.testing.assert_array_equal(mesh.faces, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))


def test_mesh_processing_handles_empty_face_results() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 0, 1]], dtype=np.int64)

    mesh, stats = clean_mesh(vertices, faces, min_component_faces=1)
    metrics = mesh_metrics(np.zeros((1, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))

    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)
    assert stats["final_faces"] == 0
    assert metrics["export_blocking_reasons"] == ["no_faces"]


def test_simplify_mesh_reduces_faces_with_native_owned_interface() -> None:
    vertices = np.array(
        [[float(index), float(index % 2), 0.0] for index in range(8)],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
        ],
        dtype=np.int64,
    )

    mesh, stats = simplify_mesh(vertices, faces, target_faces=3, min_component_faces=1)

    assert stats["simplified"] == 1
    assert stats["source_faces"] == 6
    assert stats["target_faces"] == 3
    assert mesh.faces.shape[0] == 3
    assert mesh.vertices.shape[0] <= vertices.shape[0]


def test_mesh_processing_rejects_invalid_face_indices() -> None:
    vertices = np.zeros((2, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="outside the vertex array"):
        mesh_metrics(vertices, faces)


def test_simplify_mesh_rejects_invalid_target() -> None:
    vertices, faces = _messy_mesh()

    with pytest.raises(ValueError, match="target_faces must be positive"):
        simplify_mesh(vertices, faces, target_faces=0)
