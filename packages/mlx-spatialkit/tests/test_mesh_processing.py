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
    assert metrics["boundary_vertices"] == 4
    assert metrics["boundary_loop_count"] == 1
    assert metrics["boundary_open_chain_count"] == 0
    assert metrics["boundary_small_loop_count"] == 1
    assert metrics["boundary_small_loop_edge_count"] == 4
    assert metrics["boundary_small_loop_threshold_edges"] == 32
    assert metrics["boundary_open_chain_edge_count"] == 0
    assert metrics["boundary_small_open_chain_count"] == 0
    assert metrics["boundary_small_open_chain_edge_count"] == 0
    assert metrics["boundary_simple_open_chain_count"] == 0
    assert metrics["boundary_branched_open_chain_count"] == 0
    assert metrics["boundary_open_chain_endpoint_count"] == 0
    assert metrics["boundary_open_chain_branch_vertex_count"] == 0
    assert metrics["boundary_max_loop_edges"] == 4
    assert metrics["boundary_max_open_chain_edges"] == 0
    assert metrics["boundary_max_component_edges"] == 4
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
    assert metrics["boundary_edges"] == 6
    assert metrics["boundary_loop_count"] == 0
    assert metrics["boundary_open_chain_count"] == 1
    assert metrics["boundary_open_chain_edge_count"] == 6
    assert metrics["boundary_small_open_chain_count"] == 1
    assert metrics["boundary_small_open_chain_edge_count"] == 6
    assert metrics["boundary_simple_open_chain_count"] == 0
    assert metrics["boundary_branched_open_chain_count"] == 1
    assert metrics["boundary_open_chain_endpoint_count"] == 0
    assert metrics["boundary_open_chain_branch_vertex_count"] == 2
    assert metrics["boundary_max_open_chain_edges"] == 6
    assert metrics["boundary_max_component_edges"] == 6
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
    vertices, faces = _grid_mesh(10)

    mesh, stats = simplify_mesh(vertices, faces, target_faces=40, min_component_faces=1)
    metrics = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["simplified"] is True
    assert stats["backend"] == "spatial-cluster"
    assert stats["algorithm"] == "native_spatial_vertex_clustering"
    assert stats["quality_tier"] == "geometry_aware_preview"
    assert stats["production_ready"] is False
    assert stats["requested_backend"] == "spatial-cluster"
    assert stats["backend_selection_status"] == "selected"
    assert stats["source_faces"] == faces.shape[0]
    assert stats["source_vertices"] == vertices.shape[0]
    assert stats["target_faces"] == 40
    assert stats["final_faces"] == mesh.faces.shape[0]
    assert stats["final_vertices"] == mesh.vertices.shape[0]
    assert stats["cluster_count"] > 0
    assert stats["grid_resolution"] >= 2
    assert stats["degenerate_faces_removed"] > 0
    assert stats["duplicate_faces_removed"] >= 0
    assert stats["nonmanifold_faces_removed"] >= 0
    assert stats["target_reached"] is True
    assert 0 < mesh.faces.shape[0] <= 40
    assert mesh.vertices.shape[0] <= vertices.shape[0]
    assert metrics["degenerate_faces"] == 0
    assert metrics["duplicate_faces"] == 0
    assert metrics["nonmanifold_edges"] == 0
    assert "degenerate_faces_present" not in metrics["export_blocking_reasons"]


def test_simplify_mesh_uses_distinct_topology_aware_backend() -> None:
    vertices, faces = _warped_grid_mesh(10)

    preview_mesh, preview_stats = simplify_mesh(vertices, faces, target_faces=80, min_component_faces=1)
    mesh, stats = simplify_mesh(vertices, faces, target_faces=80, min_component_faces=1, backend="topology-aware")
    metrics = mesh_metrics(mesh.vertices, mesh.faces)

    assert preview_stats["backend"] == "spatial-cluster"
    assert stats["requested_backend"] == "topology-aware"
    assert stats["backend"] == "topology-aware"
    assert stats["algorithm"] == "native_topology_aware_representative_clustering"
    assert stats["algorithm"] != preview_stats["algorithm"]
    assert stats["quality_tier"] == "production"
    assert stats["production_ready"] is True
    assert stats["production_blockers"] == []
    assert stats["backend_selection_status"] == "selected"
    assert stats["backend_selection_reason"] == "topology_aware_backend_requested"
    assert stats["candidate_faces_considered"] == faces.shape[0]
    assert stats["accepted_faces"] == mesh.faces.shape[0]
    assert stats["representative_vertices_selected"] > 0
    assert stats["target_reached"] is True
    assert metrics["export_blocking_reasons"] == []
    if preview_mesh.vertices.shape == mesh.vertices.shape:
        assert not np.allclose(preview_mesh.vertices, mesh.vertices)


def test_simplify_mesh_topology_aware_fills_triangular_boundary_loop() -> None:
    vertices, faces = _triangular_hole_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 2,
        min_component_faces=1,
        backend="topology-aware",
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["boundary_edges"] == 9
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_algorithm"] == "projected-ear-clipping"
    assert stats["small_boundary_loop_fill_fallback_algorithm"] == "centroid-fan"
    assert stats["small_boundary_loop_fill_fallback_enabled"] is True
    assert stats["small_boundary_loop_fill_fallback_max_edges"] == 8
    assert stats["small_boundary_loop_fill_fallback_policy_max_edges"] == 8
    assert stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 8
    assert stats["small_boundary_loop_repair_max_passes"] == 2
    assert stats["small_boundary_loop_repair_pass_count"] == 2
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["small_boundary_loop_fill_face_budget"] == 2
    assert stats["small_boundary_loops_considered"] == 3
    assert stats["small_boundary_loops_filled"] == 1
    assert stats["small_boundary_loops_filled_by_ear_clipping"] == 1
    assert stats["small_boundary_loops_centroid_fan_attempted"] == 0
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 0
    assert stats["small_boundary_loops_rejected"] == 0
    assert stats["small_boundary_loops_rejected_ordering"] == 0
    assert stats["small_boundary_loops_rejected_triangulation"] == 0
    assert stats["small_boundary_loops_rejected_fallback_cap"] == 0
    assert stats["small_boundary_loops_rejected_degenerate"] == 0
    assert stats["small_boundary_loops_rejected_duplicate"] == 0
    assert stats["small_boundary_loops_rejected_nonmanifold"] == 0
    assert stats["small_boundary_loops_budget_limited"] == 2
    assert stats["small_boundary_loop_faces_added"] == 1
    assert stats["final_faces"] == faces.shape[0] + 1
    assert stats["target_reached"] is True
    assert after["boundary_loop_count"] == 1
    assert after["boundary_edges"] == 6
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_can_disable_small_boundary_loop_fill() -> None:
    vertices, faces = _triangular_hole_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 2,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=0,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["small_boundary_loop_fill_enabled"] is False
    assert stats["small_boundary_loop_fill_fallback_algorithm"] == "centroid-fan"
    assert stats["small_boundary_loop_fill_fallback_enabled"] is False
    assert stats["small_boundary_loop_fill_fallback_max_edges"] == 8
    assert stats["small_boundary_loop_fill_fallback_policy_max_edges"] == 8
    assert stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 0
    assert stats["small_boundary_loop_repair_max_passes"] == 2
    assert stats["small_boundary_loop_repair_pass_count"] == 0
    assert stats["small_boundary_loop_fill_max_edges"] == 0
    assert stats["small_boundary_loops_considered"] == 0
    assert stats["small_boundary_loops_filled"] == 0
    assert stats["small_boundary_loops_centroid_fan_attempted"] == 0
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 0
    assert stats["small_boundary_loop_faces_added"] == 0
    assert after["boundary_loop_count"] == before["boundary_loop_count"]
    assert after["boundary_edges"] == before["boundary_edges"]
    assert after["nonmanifold_edges"] == 0


def test_simplify_mesh_topology_aware_fills_four_edge_boundary_loop_by_default() -> None:
    vertices, faces = _grid_mesh_with_missing_cell(6, missing_x=3, missing_y=3)
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 4,
        min_component_faces=1,
        backend="topology-aware",
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["boundary_edges"] == 28
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_algorithm"] == "projected-ear-clipping"
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["small_boundary_loops_considered"] == 1
    assert stats["small_boundary_loops_filled"] == 1
    assert stats["small_boundary_loop_faces_added"] == 2
    assert stats["final_faces"] == faces.shape[0] + 2
    assert after["boundary_loop_count"] == 1
    assert after["boundary_edges"] == 24
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_tries_alternative_triangulation_for_topology_blocked_loop() -> None:
    vertices, faces = _quad_hole_with_blocked_primary_diagonal_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 4,
        min_component_faces=1,
        backend="topology-aware",
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["nonmanifold_edges"] == 0
    assert stats["small_boundary_loops_alternative_triangulation_attempted"] == 1
    assert stats["small_boundary_loops_filled_by_alternative_triangulation"] == 1
    assert stats["small_boundary_loops_filled_by_ear_clipping"] == 1
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 0
    assert stats["small_boundary_loop_faces_added"] == 2
    assert after["boundary_loop_count"] == 1
    assert after["boundary_edges"] == before["boundary_edges"] - 4
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_topology_aware_fills_eight_edge_concave_boundary_loop() -> None:
    vertices, faces = _grid_mesh_with_missing_cells(6, {(2, 2), (3, 2), (2, 3)})
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 8,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=8,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["boundary_edges"] == 32
    assert before["boundary_small_loop_count"] == 2
    assert before["boundary_small_loop_edge_count"] == 32
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_algorithm"] == "projected-ear-clipping"
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["small_boundary_loops_considered"] == 1
    assert stats["small_boundary_loops_filled"] == 1
    assert stats["small_boundary_loop_faces_added"] == 6
    assert stats["final_faces"] == faces.shape[0] + 6
    assert after["boundary_loop_count"] == 1
    assert after["boundary_edges"] == 24
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_topology_aware_fills_branched_small_cycles() -> None:
    vertices, faces = _pinched_triangular_hole_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 2,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=8,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_simple_open_chain_count"] == 0
    assert before["boundary_branched_open_chain_count"] == 1
    assert before["boundary_open_chain_branch_vertex_count"] == 1
    assert stats["small_boundary_branched_cycle_fill_enabled"] is True
    assert stats["small_boundary_branched_cycle_fill_max_edges"] == 6
    assert stats["small_boundary_branched_cycle_fill_policy_max_edges"] == 6
    assert stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 6
    assert stats["small_boundary_loop_repair_max_passes"] == 2
    assert stats["small_boundary_loop_repair_pass_count"] == 1
    assert stats["small_boundary_branched_cycle_candidates"] >= 2
    assert stats["small_boundary_branched_cycles_filled"] == 2
    assert stats["small_boundary_loop_faces_added"] == 2
    assert stats["final_faces"] == faces.shape[0] + 2
    assert after["boundary_branched_open_chain_count"] == 0
    assert after["boundary_open_chain_count"] == 0
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_branched_cycles_respect_public_loop_cap() -> None:
    vertices, faces = _pinched_quad_hole_mesh()
    before = mesh_metrics(vertices, faces)

    capped_mesh, capped_stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 20,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=3,
    )
    capped_after = mesh_metrics(capped_mesh.vertices, capped_mesh.faces)

    repaired_mesh, repaired_stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 20,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=4,
    )
    repaired_after = mesh_metrics(repaired_mesh.vertices, repaired_mesh.faces)

    assert before["boundary_branched_open_chain_count"] == 1
    assert before["boundary_open_chain_branch_vertex_count"] == 1
    assert capped_stats["small_boundary_branched_cycle_fill_max_edges"] == 6
    assert capped_stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 3
    assert capped_stats["small_boundary_branched_cycle_candidates"] == 0
    assert capped_stats["small_boundary_branched_cycles_filled"] == 0
    assert capped_after["boundary_branched_open_chain_count"] == before["boundary_branched_open_chain_count"]
    assert capped_after["boundary_open_chain_edge_count"] == before["boundary_open_chain_edge_count"]
    assert capped_after["nonmanifold_edges"] == 0

    assert repaired_stats["small_boundary_branched_cycle_fill_max_edges"] == 6
    assert repaired_stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 4
    assert repaired_stats["small_boundary_branched_cycle_candidates"] >= 2
    assert repaired_stats["small_boundary_branched_cycles_filled"] == 2
    assert repaired_stats["small_boundary_loop_faces_added"] == 4
    assert repaired_after["boundary_branched_open_chain_count"] == 0
    assert repaired_after["boundary_open_chain_edge_count"] == 0
    assert repaired_after["nonmanifold_edges"] == 0


def test_simplify_mesh_centroid_fallback_uses_eight_edge_policy_cap() -> None:
    vertices, faces = _self_intersecting_octagon_hole_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 20,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=8,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["boundary_max_loop_edges"] == 8
    assert stats["small_boundary_loop_fill_fallback_max_edges"] == 8
    assert stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 8
    assert stats["small_boundary_loops_centroid_fan_attempted"] == 2
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 2
    assert stats["small_boundary_loops_rejected_fallback_cap"] == 0
    assert stats["small_boundary_loop_faces_added"] == 16
    assert after["boundary_loop_count"] == 0
    assert after["nonmanifold_edges"] == 0


def test_simplify_mesh_topology_aware_respects_triangle_only_boundary_loop_cap() -> None:
    vertices, faces = _grid_mesh_with_missing_cell(6, missing_x=3, missing_y=3)
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 4,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=3,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["boundary_edges"] == 28
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_max_edges"] == 3
    assert stats["small_boundary_loops_considered"] == 0
    assert stats["small_boundary_loops_filled"] == 0
    assert stats["small_boundary_loop_faces_added"] == 0
    assert after["boundary_loop_count"] == before["boundary_loop_count"]
    assert after["boundary_edges"] == before["boundary_edges"]
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def _grid_mesh(quads_per_axis: int) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    for y in range(quads_per_axis + 1):
        for x in range(quads_per_axis + 1):
            vertices.append([float(x), float(y), 0.0])

    def vid(x: int, y: int) -> int:
        return y * (quads_per_axis + 1) + x

    faces: list[list[int]] = []
    for y in range(quads_per_axis):
        for x in range(quads_per_axis):
            faces.append([vid(x, y), vid(x + 1, y), vid(x, y + 1)])
            faces.append([vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int64)


def _triangular_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 1.0, 0.0],
            [-0.866, -0.5, 0.0],
            [0.866, -0.5, 0.0],
            [0.0, 2.0, 0.0],
            [-1.4, 1.0, 0.0],
            [-1.8, -1.0, 0.0],
            [0.0, -2.0, 0.0],
            [1.8, -1.0, 0.0],
            [1.4, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 3, 4],
            [0, 4, 1],
            [1, 4, 5],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 8],
            [2, 8, 0],
            [0, 8, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _pinched_triangular_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    first_vertices, first_faces = _triangular_hole_mesh()
    second_vertices, second_faces = _triangular_hole_mesh()
    second_vertices = second_vertices.copy()
    second_vertices[:, 0] += 5.0
    second_vertices[0] = first_vertices[0]

    vertices = np.concatenate([first_vertices, second_vertices[1:]], axis=0)
    remap = {0: 0}
    for old_index in range(1, second_vertices.shape[0]):
        remap[old_index] = first_vertices.shape[0] + old_index - 1
    second_remapped = np.array([[remap[int(value)] for value in face] for face in second_faces], dtype=np.int64)
    faces = np.concatenate([first_faces, second_remapped], axis=0)
    return vertices.astype(np.float32), faces.astype(np.int64)


def _pinched_quad_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    first_vertices, first_faces = _grid_mesh_with_missing_cell(3, missing_x=1, missing_y=1)
    second_vertices, second_faces = _grid_mesh_with_missing_cell(3, missing_x=1, missing_y=1)
    second_vertices = second_vertices.copy()
    second_vertices[:, 0] += 5.0

    shared_vertex = 5
    second_vertices[shared_vertex] = first_vertices[shared_vertex]
    vertices = list(first_vertices)
    remap = {shared_vertex: shared_vertex}
    for old_index in range(second_vertices.shape[0]):
        if old_index == shared_vertex:
            continue
        remap[old_index] = len(vertices)
        vertices.append(second_vertices[old_index])

    second_remapped = np.array([[remap[int(value)] for value in face] for face in second_faces], dtype=np.int64)
    faces = np.concatenate([first_faces, second_remapped], axis=0)
    return np.array(vertices, dtype=np.float32), faces.astype(np.int64)


def _quad_hole_with_blocked_primary_diagonal_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _grid_mesh_with_missing_cell(3, missing_x=1, missing_y=1)
    vertex_list = vertices.tolist()
    face_list = faces.tolist()

    # The native ear pass first triangulates the center quad through diagonal
    # 6-9. This closed sidecar makes that diagonal unavailable while preserving
    # manifold edge counts, so the alternate diagonal is the valid repair.
    vertex_list.extend([[1.5, 1.5, 1.0], [1.5, 1.5, -1.0]])
    top_vertex = len(vertex_list) - 2
    bottom_vertex = len(vertex_list) - 1
    face_list.extend(
        [
            [6, 9, top_vertex],
            [9, 6, bottom_vertex],
            [6, top_vertex, bottom_vertex],
            [9, bottom_vertex, top_vertex],
        ]
    )
    return np.array(vertex_list, dtype=np.float32), np.array(face_list, dtype=np.int64)


def _self_intersecting_octagon_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    angles = np.deg2rad(np.array([0, 135, 270, 45, 180, 315, 90, 225], dtype=np.float32))
    inner = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
    outer = inner * 2.5
    vertices = np.concatenate([inner, outer], axis=0).astype(np.float32)

    faces: list[list[int]] = []
    loop_size = int(inner.shape[0])
    for index in range(loop_size):
        next_index = (index + 1) % loop_size
        faces.append([index, loop_size + index, loop_size + next_index])
        faces.append([index, loop_size + next_index, next_index])
    return vertices, np.array(faces, dtype=np.int64)


def _grid_mesh_with_missing_cell(
    quads_per_axis: int,
    *,
    missing_x: int,
    missing_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    return _grid_mesh_with_missing_cells(quads_per_axis, {(missing_x, missing_y)})


def _grid_mesh_with_missing_cells(
    quads_per_axis: int,
    missing_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    for y in range(quads_per_axis + 1):
        for x in range(quads_per_axis + 1):
            vertices.append([float(x), float(y), 0.0])

    def vid(x: int, y: int) -> int:
        return y * (quads_per_axis + 1) + x

    faces: list[list[int]] = []
    for y in range(quads_per_axis):
        for x in range(quads_per_axis):
            if (x, y) in missing_cells:
                continue
            faces.append([vid(x, y), vid(x + 1, y), vid(x, y + 1)])
            faces.append([vid(x + 1, y), vid(x + 1, y + 1), vid(x, y + 1)])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int64)


def _warped_grid_mesh(quads_per_axis: int) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _grid_mesh(quads_per_axis)
    vertices[:, 2] = 0.05 * vertices[:, 0] * vertices[:, 0] + 0.02 * vertices[:, 1]
    return vertices, faces


def test_mesh_processing_rejects_invalid_face_indices() -> None:
    vertices = np.zeros((2, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    with pytest.raises(ValueError, match="outside the vertex array"):
        mesh_metrics(vertices, faces)


def test_simplify_mesh_rejects_invalid_target() -> None:
    vertices, faces = _messy_mesh()

    with pytest.raises(ValueError, match="target_faces must be positive"):
        simplify_mesh(vertices, faces, target_faces=0)


def test_simplify_mesh_rejects_invalid_backend() -> None:
    vertices, faces = _messy_mesh()

    with pytest.raises(ValueError, match="simplifier backend"):
        simplify_mesh(vertices, faces, target_faces=4, backend="bad-backend")


def test_simplify_mesh_rejects_negative_small_boundary_loop_fill_cap() -> None:
    vertices, faces = _messy_mesh()

    with pytest.raises(ValueError, match="small_boundary_loop_fill_max_edges"):
        simplify_mesh(vertices, faces, target_faces=4, small_boundary_loop_fill_max_edges=-1)
