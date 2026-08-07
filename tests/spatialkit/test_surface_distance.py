from __future__ import annotations

import numpy as np
import pytest

from mlx_spatial.spatialkit.mesh import (
    bidirectional_surface_distance_metrics,
    point_to_mesh_distances,
    sampled_surface_to_mesh_distance_metrics,
)


def test_point_to_mesh_distances_uses_exact_triangle_surface() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    queries = np.array(
        [[0.25, 0.25, 0.0], [0.25, 0.25, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    distances, stats = point_to_mesh_distances(queries, vertices, faces)

    np.testing.assert_allclose(distances, [0.0, 1.0, np.sqrt(0.5)], atol=1.0e-6)
    assert stats["backend"] == "native-cpu-triangle-bvh"
    assert stats["distance_kind"] == "exact-point-to-triangle-unsigned"
    assert stats["query_count"] == 3
    assert stats["workers"] >= 1

    detailed_distances, closest_points, closest_faces, detailed_stats = point_to_mesh_distances(
        queries,
        vertices,
        faces,
        return_closest=True,
    )
    np.testing.assert_allclose(detailed_distances, distances)
    np.testing.assert_allclose(closest_points[1], [0.25, 0.25, 0.0], atol=1.0e-6)
    np.testing.assert_array_equal(closest_faces, [0, 0, 0])
    assert detailed_stats == stats


def test_bidirectional_surface_distance_identical_mesh_is_zero() -> None:
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [[0, 2, 3], [2, 1, 3], [1, 0, 3], [0, 1, 2]],
        dtype=np.int64,
    )

    metrics = bidirectional_surface_distance_metrics(
        vertices,
        faces,
        vertices,
        faces,
        max_samples_per_mesh=8,
        voxel_size=0.25,
    )

    assert metrics["source_samples"]["total"] == 8
    assert metrics["candidate_samples"]["total"] == 8
    assert metrics["symmetric"]["sampled_chamfer_l1"] == pytest.approx(0.0, abs=1.0e-7)
    assert metrics["symmetric"]["sampled_hausdorff_voxels"] == pytest.approx(0.0, abs=1.0e-7)


def test_bidirectional_surface_distance_reports_offset_in_voxels() -> None:
    source_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    candidate_vertices = source_vertices + np.array([0.0, 0.0, 0.25], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    metrics = bidirectional_surface_distance_metrics(
        source_vertices,
        faces,
        candidate_vertices,
        faces,
        max_samples_per_mesh=4,
        voxel_size=0.125,
    )

    assert metrics["candidate_to_source"]["p95"] == pytest.approx(0.25, abs=1.0e-6)
    assert metrics["source_to_candidate"]["p95"] == pytest.approx(0.25, abs=1.0e-6)
    assert metrics["symmetric"]["sampled_p95_max_voxels"] == pytest.approx(2.0, abs=1.0e-6)
    assert metrics["source_to_candidate"]["max_displacement"][2] == pytest.approx(-0.25, abs=1.0e-6)

    one_way = sampled_surface_to_mesh_distance_metrics(
        source_vertices,
        faces,
        candidate_vertices,
        faces,
        max_samples=4,
        normalization_vertices=source_vertices,
        voxel_size=0.125,
    )
    assert one_way["distance_kind"] == "unsigned-one-way-sampled-surface-distance"
    assert one_way["p95_voxels"] == pytest.approx(2.0, abs=1.0e-6)


def test_surface_distance_rejects_invalid_sampling_configuration() -> None:
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="max_samples_per_mesh"):
        bidirectional_surface_distance_metrics(
            vertices,
            faces,
            vertices,
            faces,
            max_samples_per_mesh=0,
        )
