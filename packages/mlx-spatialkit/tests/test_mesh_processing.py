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


def test_mesh_metrics_reports_simple_open_chain_boundary_components() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int64,
    )

    metrics = mesh_metrics(vertices, faces)

    assert metrics["nonmanifold_edges"] == 1
    assert metrics["boundary_open_chain_count"] == 1
    assert metrics["boundary_open_chain_edge_count"] == 2
    assert metrics["boundary_simple_open_chain_count"] == 1
    assert metrics["boundary_branched_open_chain_count"] == 0
    assert metrics["boundary_open_chain_endpoint_count"] == 2
    assert metrics["boundary_open_chain_branch_vertex_count"] == 0
    assert metrics["boundary_max_open_chain_edges"] == 2
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
    assert stats["algorithm"] == "native_topology_aware_quadric_representative_clustering"
    assert stats["algorithm"] != preview_stats["algorithm"]
    assert stats["quality_tier"] == "production_candidate_blocked"
    assert stats["production_ready"] is False
    assert stats["production_blockers"] == [
        "missing_qem_edge_collapse_simplification",
        "missing_narrow_band_dc_remesh",
    ]
    assert stats["remesh_backend"] == "not_implemented"
    assert stats["remesh_equivalence_status"] == "blocked_missing_narrow_band_dc"
    assert stats["qem_simplification_backend"] == "not_implemented"
    assert stats["qem_equivalence_status"] == "qem_scored_not_edge_collapse"
    assert stats["reference_geometry_backend_status"] == "blocked_missing_reference_geometry"
    assert stats["reference_geometry_blockers"] == stats["production_blockers"]
    assert stats["backend_selection_status"] == "selected"
    assert stats["backend_selection_reason"] == "topology_aware_backend_requested"
    assert stats["candidate_faces_considered"] == faces.shape[0]
    assert stats["accepted_faces"] == mesh.faces.shape[0]
    assert stats["representative_vertices_selected"] > 0
    assert stats["representative_selection_strategy"] == "cluster_quadric_error_minimizer"
    assert stats["quadric_representative_candidates_evaluated"] == vertices.shape[0]
    assert stats["quadric_representative_nonfinite_candidates"] == 0
    assert stats["quadric_representative_error_sum"] >= 0.0
    assert stats["quadric_representative_error_max"] >= 0.0
    assert stats["target_reached"] is True
    assert metrics["export_blocking_reasons"] == []
    if preview_mesh.vertices.shape == mesh.vertices.shape:
        assert not np.allclose(preview_mesh.vertices, mesh.vertices)


def test_simplify_mesh_topology_aware_fills_triangular_boundary_loop() -> None:
    vertices, faces = _small_triangular_hole_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 3,
        min_component_faces=1,
        backend="topology-aware",
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert before["boundary_loop_count"] == 2
    assert before["boundary_edges"] == 9
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_algorithm"] == "cumesh-perimeter-centroid-fan"
    assert stats["small_boundary_loop_fill_max_perimeter"] == pytest.approx(0.03)
    assert stats["small_boundary_loop_fill_fallback_algorithm"] == "disabled"
    assert stats["small_boundary_loop_fill_fallback_enabled"] is False
    assert stats["small_boundary_loop_fill_fallback_max_edges"] == 0
    assert stats["small_boundary_loop_fill_fallback_policy_max_edges"] == 0
    assert stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 0
    assert stats["small_boundary_loop_repair_max_passes"] == 3
    assert 1 <= stats["small_boundary_loop_repair_pass_count"] <= 3
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["small_boundary_loop_fill_face_budget"] == 3
    assert stats["small_boundary_loops_considered"] == 2
    assert stats["small_boundary_loops_filled"] == 1
    assert stats["small_boundary_loops_filled_by_ear_clipping"] == 0
    assert stats["small_boundary_loops_alternative_triangulation_attempted"] == 0
    assert stats["small_boundary_loops_filled_by_alternative_triangulation"] == 0
    assert stats["small_boundary_loops_centroid_fan_attempted"] == 1
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 1
    assert stats["small_boundary_loops_rejected"] == 1
    assert stats["small_boundary_loops_rejected_ordering"] == 0
    assert stats["small_boundary_loops_rejected_triangulation"] == 0
    assert stats["small_boundary_loops_rejected_perimeter"] == 1
    assert stats["small_boundary_loops_rejected_edge_cap"] == 0
    assert stats["small_boundary_loops_rejected_fallback_cap"] == 0
    assert stats["small_boundary_loops_rejected_degenerate"] == 0
    assert stats["small_boundary_loops_rejected_duplicate"] == 0
    assert stats["small_boundary_loops_rejected_nonmanifold"] == 0
    assert stats["small_boundary_loops_budget_limited"] == 0
    assert stats["small_boundary_loop_faces_added"] == 3
    assert stats["final_faces"] == faces.shape[0] + 3
    assert mesh.vertices.shape[0] == vertices.shape[0] + 1
    assert stats["target_reached"] is True
    assert after["boundary_loop_count"] == 1
    assert after["boundary_edges"] == 6
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_can_disable_small_boundary_loop_fill() -> None:
    vertices, faces = _small_triangular_hole_mesh()
    before = mesh_metrics(vertices, faces)

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=faces.shape[0] + 3,
        min_component_faces=1,
        backend="topology-aware",
        small_boundary_loop_fill_max_edges=0,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["small_boundary_loop_fill_enabled"] is False
    assert stats["small_boundary_loop_fill_algorithm"] == "cumesh-perimeter-centroid-fan"
    assert stats["small_boundary_loop_fill_max_perimeter"] == pytest.approx(0.03)
    assert stats["small_boundary_loop_fill_fallback_algorithm"] == "disabled"
    assert stats["small_boundary_loop_fill_fallback_enabled"] is False
    assert stats["small_boundary_loop_fill_fallback_max_edges"] == 0
    assert stats["small_boundary_loop_fill_fallback_policy_max_edges"] == 0
    assert stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 0
    assert stats["small_boundary_loop_repair_max_passes"] == 3
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
    vertices, faces = _small_quad_hole_mesh()
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
    assert before["boundary_edges"] == 8
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_algorithm"] == "cumesh-perimeter-centroid-fan"
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["small_boundary_loop_fill_max_perimeter"] == pytest.approx(0.03)
    assert stats["small_boundary_loops_considered"] == 2
    assert stats["small_boundary_loops_filled"] == 1
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 1
    assert stats["small_boundary_loops_filled_by_ear_clipping"] == 0
    assert stats["small_boundary_loops_rejected_perimeter"] == 1
    assert stats["small_boundary_loop_faces_added"] == 4
    assert stats["final_faces"] == faces.shape[0] + 4
    assert after["boundary_loop_count"] == 1
    assert after["boundary_edges"] == 4
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_topology_aware_prefills_reference_sized_loop_before_clustering() -> None:
    vertices, faces = _small_dodecagon_hole_mesh()

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=18,
        min_component_faces=1,
        backend="topology-aware",
    )

    assert stats["source_faces"] == faces.shape[0]
    assert stats["pre_simplify_hole_fill_enabled"] is True
    assert stats["pre_simplify_hole_fill_algorithm"] == "reference-clean-boundary-centroid-fan"
    assert stats["pre_simplify_hole_fill_max_edges"] == 64
    assert stats["pre_simplify_hole_fill_max_perimeter"] == pytest.approx(0.03)
    assert stats["pre_simplify_hole_fill_boundary_edges_before"] == 24
    assert stats["pre_simplify_hole_fill_clean_boundary_loops"] == 2
    assert stats["pre_simplify_hole_fill_filled_loops"] == 1
    assert stats["pre_simplify_hole_fill_skipped_large_loops"] == 1
    assert stats["pre_simplify_hole_fill_skipped_complex_components"] == 0
    assert stats["pre_simplify_hole_fill_vertices_added"] == 1
    assert stats["pre_simplify_hole_fill_faces_added"] == 12
    assert stats["pre_simplify_faces"] == faces.shape[0] + 12
    assert stats["candidate_faces_considered"] == faces.shape[0] + 12
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["final_faces"] == mesh.faces.shape[0]


def test_simplify_mesh_disables_alternative_triangulation_for_reference_hole_fill() -> None:
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
    assert stats["small_boundary_loop_fill_algorithm"] == "cumesh-perimeter-centroid-fan"
    assert stats["small_boundary_loops_alternative_triangulation_attempted"] == 0
    assert stats["small_boundary_loops_filled_by_alternative_triangulation"] == 0
    assert stats["small_boundary_loops_filled_by_ear_clipping"] == 0
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 0
    assert stats["small_boundary_loops_rejected_perimeter"] > 0
    assert stats["small_boundary_loop_faces_added"] == 0
    assert after["boundary_loop_count"] == before["boundary_loop_count"]
    assert after["boundary_edges"] == before["boundary_edges"]
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_topology_aware_rejects_large_concave_boundary_loop_by_perimeter() -> None:
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
    assert stats["small_boundary_loop_fill_algorithm"] == "cumesh-perimeter-centroid-fan"
    assert stats["small_boundary_loop_fill_max_edges"] == 8
    assert stats["small_boundary_loop_fill_max_perimeter"] == pytest.approx(0.03)
    assert stats["small_boundary_loops_considered"] == 2
    assert stats["small_boundary_loops_filled"] == 0
    assert stats["small_boundary_loops_rejected_perimeter"] == 2
    assert stats["small_boundary_loop_faces_added"] == 0
    assert stats["final_faces"] == faces.shape[0]
    assert after["boundary_loop_count"] == before["boundary_loop_count"]
    assert after["boundary_edges"] == before["boundary_edges"]
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_topology_aware_fills_branched_small_cycles() -> None:
    vertices, faces = _pinched_triangular_hole_mesh()
    vertices = vertices * 0.001
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

    assert before["boundary_simple_open_chain_count"] == 0
    assert before["boundary_branched_open_chain_count"] == 1
    assert before["boundary_open_chain_branch_vertex_count"] == 1
    assert stats["small_boundary_branched_cycle_fill_enabled"] is True
    assert stats["small_boundary_branched_cycle_fill_max_edges"] == 8
    assert stats["small_boundary_branched_cycle_fill_policy_max_edges"] == 8
    assert stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 8
    assert stats["small_boundary_loop_repair_max_passes"] == 3
    assert 1 <= stats["small_boundary_loop_repair_pass_count"] <= 3
    assert stats["small_boundary_branched_cycle_candidates"] == 2
    assert stats["small_boundary_branched_cycles_filled"] == 2
    assert stats["small_boundary_loop_faces_added"] > 0
    assert stats["final_faces"] > faces.shape[0]
    assert after["boundary_branched_open_chain_count"] == 0
    assert after["boundary_open_chain_count"] == 0
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []


def test_simplify_mesh_branched_cycles_respect_public_loop_cap() -> None:
    vertices, faces = _pinched_quad_hole_mesh()
    vertices = vertices * 0.001
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
    assert capped_stats["small_boundary_branched_cycle_fill_enabled"] is True
    assert capped_stats["small_boundary_branched_cycle_fill_max_edges"] == 8
    assert capped_stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 3
    assert capped_stats["small_boundary_branched_cycle_candidates"] == 0
    assert capped_stats["small_boundary_branched_cycles_filled"] == 0
    assert capped_after["boundary_branched_open_chain_count"] == before["boundary_branched_open_chain_count"]
    assert capped_after["boundary_open_chain_edge_count"] == before["boundary_open_chain_edge_count"]
    assert capped_after["nonmanifold_edges"] == 0

    assert repaired_stats["small_boundary_branched_cycle_fill_enabled"] is True
    assert repaired_stats["small_boundary_branched_cycle_fill_max_edges"] == 8
    assert repaired_stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 4
    assert repaired_stats["small_boundary_branched_cycle_candidates"] == 2
    assert repaired_stats["small_boundary_branched_cycles_filled"] == 2
    assert repaired_stats["small_boundary_loop_faces_added"] > 0
    assert repaired_after["boundary_branched_open_chain_count"] == 0
    assert repaired_after["boundary_open_chain_count"] == 0
    assert repaired_after["nonmanifold_edges"] == 0


def test_simplify_mesh_centroid_fan_is_primary_and_fallback_is_disabled() -> None:
    vertices, faces = _small_quad_hole_mesh()
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
    assert before["boundary_max_loop_edges"] == 4
    assert stats["small_boundary_loop_fill_fallback_enabled"] is False
    assert stats["small_boundary_loop_fill_fallback_max_edges"] == 0
    assert stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 0
    assert stats["small_boundary_loops_centroid_fan_attempted"] == 1
    assert stats["small_boundary_loops_filled_by_centroid_fan"] == 1
    assert stats["small_boundary_loops_rejected_fallback_cap"] == 0
    assert stats["small_boundary_loop_faces_added"] == 4
    assert after["boundary_loop_count"] == 1
    assert after["nonmanifold_edges"] == 0


def test_simplify_mesh_topology_aware_respects_public_edge_cap_after_perimeter_gate() -> None:
    vertices, faces = _small_quad_hole_mesh()
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
    assert before["boundary_edges"] == 8
    assert stats["small_boundary_loop_fill_enabled"] is True
    assert stats["small_boundary_loop_fill_max_edges"] == 3
    assert stats["small_boundary_loop_fill_max_perimeter"] == pytest.approx(0.03)
    assert stats["small_boundary_loops_considered"] == 2
    assert stats["small_boundary_loops_filled"] == 0
    assert stats["small_boundary_loops_rejected_perimeter"] == 1
    assert stats["small_boundary_loops_rejected_edge_cap"] == 1
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


def _small_triangular_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = _triangular_hole_mesh()
    vertices = vertices.copy()
    vertices[:3] *= 0.002
    return vertices, faces


def _small_quad_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    inner_radius = 0.003
    inner = np.array(
        [
            [-inner_radius, -inner_radius, 0.0],
            [inner_radius, -inner_radius, 0.0],
            [inner_radius, inner_radius, 0.0],
            [-inner_radius, inner_radius, 0.0],
        ],
        dtype=np.float32,
    )
    outer = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    vertices = np.concatenate([inner, outer], axis=0)

    faces: list[list[int]] = []
    for index in range(4):
        next_index = (index + 1) % 4
        outer_index = 4 + index
        outer_next = 4 + next_index
        faces.append([index, outer_index, outer_next])
        faces.append([index, outer_next, next_index])
    return vertices, np.array(faces, dtype=np.int64)


def _small_dodecagon_hole_mesh() -> tuple[np.ndarray, np.ndarray]:
    loop_size = 12
    angles = np.linspace(0.0, 2.0 * np.pi, loop_size, endpoint=False, dtype=np.float32)
    inner_radius = 0.003
    inner = np.stack(
        [
            np.cos(angles) * inner_radius,
            np.sin(angles) * inner_radius,
            np.zeros_like(angles),
        ],
        axis=1,
    )
    outer = np.stack(
        [
            np.cos(angles),
            np.sin(angles),
            np.zeros_like(angles),
        ],
        axis=1,
    )
    vertices = np.concatenate([inner, outer], axis=0).astype(np.float32)

    faces: list[list[int]] = []
    for index in range(loop_size):
        next_index = (index + 1) % loop_size
        outer_index = loop_size + index
        outer_next = loop_size + next_index
        faces.append([index, outer_index, outer_next])
        faces.append([index, outer_next, next_index])
    return vertices, np.array(faces, dtype=np.int64)


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


def test_mesh_metrics_nonmanifold_vertices_detects_pinch_invisible_to_edge_metric() -> None:
    # Two tetrahedra sharing exactly one apex vertex (vertex 0).
    # Tetra A: vertices {0, 1, 2, 3}.  Tetra B: vertices {0, 4, 5, 6}.
    # The shared vertex 0 is a pinch: its incident triangles form two
    # disjoint fans (one per tetrahedron), so nonmanifold_vertices >= 1.
    # Crucially, no edge is shared between the two tetrahedra, so
    # nonmanifold_edges == 0.  Both tetrahedra are individually closed, so
    # boundary_loop_count == 0.
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],   # 0 — shared apex
            [1.0, 0.0, 0.0],   # 1  \ tetra A
            [0.0, 1.0, 0.0],   # 2  |
            [0.0, 0.0, 1.0],   # 3  /
            [-1.0, 0.0, 0.0],  # 4  \ tetra B
            [0.0, -1.0, 0.0],  # 5  |
            [0.0, 0.0, -1.0],  # 6  /
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            # Tetra A (outward-consistent winding for a closed shell)
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
            # Tetra B
            [0, 4, 5],
            [0, 5, 6],
            [0, 6, 4],
            [4, 6, 5],
        ],
        dtype=np.int64,
    )

    metrics = mesh_metrics(vertices, faces)

    assert metrics["nonmanifold_vertices"] >= 1, (
        "Pinched vertex shared by two separate tetra fans must be detected"
    )
    assert metrics["nonmanifold_edges"] == 0, (
        "No edge is shared between the two tetrahedra, so edge metric must be 0"
    )
    assert metrics["boundary_loop_count"] == 0, (
        "Both tetrahedra are individually closed; no boundary loops expected"
    )


def test_mesh_metrics_nonmanifold_vertices_is_zero_on_clean_closed_manifold() -> None:
    # Regular octahedron: 6 vertices, 8 triangular faces, fully closed 2-manifold.
    # Every vertex has exactly one fan of incident triangles.
    s = 1.0
    vertices = np.array(
        [
            [s, 0.0, 0.0],
            [-s, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, -s, 0.0],
            [0.0, 0.0, s],
            [0.0, 0.0, -s],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [0, 5, 2],
            [2, 5, 1],
            [1, 5, 3],
            [3, 5, 0],
        ],
        dtype=np.int64,
    )

    metrics = mesh_metrics(vertices, faces)

    assert metrics["nonmanifold_vertices"] == 0
    assert metrics["nonmanifold_edges"] == 0
    assert metrics["boundary_loop_count"] == 0


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


def test_simplify_mesh_rejects_invalid_small_boundary_loop_fill_perimeter() -> None:
    vertices, faces = _messy_mesh()

    with pytest.raises(ValueError, match="small_boundary_loop_fill_max_perimeter"):
        simplify_mesh(vertices, faces, target_faces=4, small_boundary_loop_fill_max_perimeter=0.0)


# ---------------------------------------------------------------------------
# Native QEM edge-collapse simplifier (slice 2).
#
# Each adversarial input below is a CLOSED manifold (boundary_loop_count == 0,
# boundary_open_chain_count == 0, nonmanifold_edges == 0, nonmanifold_vertices
# == 0). The QEM backend must reduce face count toward the target while keeping
# all four topology metrics at zero (no tearing, no pinch). The
# nonmanifold_vertices oracle comes from slice 1's mesh_metrics.
# ---------------------------------------------------------------------------


def _assert_closed_manifold(metrics: dict[str, object]) -> None:
    assert metrics["boundary_loop_count"] == 0
    assert metrics["boundary_open_chain_count"] == 0
    assert metrics["nonmanifold_edges"] == 0
    assert metrics["nonmanifold_vertices"] == 0


def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed manifold sphere built by subdividing an icosahedron."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    for _ in range(subdivisions):
        mid: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in mid:
                return mid[key]
            pa, pb = verts[a], verts[b]
            m = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2]
            norm = (m[0] ** 2 + m[1] ** 2 + m[2] ** 2) ** 0.5 or 1.0
            m = [m[0] / norm, m[1] / norm, m[2] / norm]
            idx = len(verts)
            verts.append(m)
            mid[key] = idx
            return idx

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _thin_double_sided_sheet(grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Two near-coincident triangle layers sealed by a rim — a closed, very thin
    box that provokes fold/pinch failures under naive collapse."""
    eps = 1.0e-3
    top = []
    bottom = []
    top_idx = {}
    bottom_idx = {}
    verts = []
    for j in range(grid + 1):
        for i in range(grid + 1):
            x = i / grid
            y = j / grid
            top_idx[(i, j)] = len(verts)
            verts.append([x, y, eps])
            top.append((i, j))
    for j in range(grid + 1):
        for i in range(grid + 1):
            x = i / grid
            y = j / grid
            bottom_idx[(i, j)] = len(verts)
            verts.append([x, y, -eps])
            bottom.append((i, j))

    faces = []
    # Top layer (CCW from above).
    for j in range(grid):
        for i in range(grid):
            a = top_idx[(i, j)]
            b = top_idx[(i + 1, j)]
            c = top_idx[(i + 1, j + 1)]
            d = top_idx[(i, j + 1)]
            faces += [[a, b, c], [a, c, d]]
    # Bottom layer (reversed winding so the closed solid is outward-oriented).
    for j in range(grid):
        for i in range(grid):
            a = bottom_idx[(i, j)]
            b = bottom_idx[(i + 1, j)]
            c = bottom_idx[(i + 1, j + 1)]
            d = bottom_idx[(i, j + 1)]
            faces += [[a, c, b], [a, d, c]]
    # Rim: seal the four borders connecting top and bottom into a closed solid.
    border = []
    for i in range(grid):
        border.append(((i, 0), (i + 1, 0)))
    for j in range(grid):
        border.append(((grid, j), (grid, j + 1)))
    for i in range(grid, 0, -1):
        border.append(((i, grid), (i - 1, grid)))
    for j in range(grid, 0, -1):
        border.append(((0, j), (0, j - 1)))
    for (i0, j0), (i1, j1) in border:
        ta = top_idx[(i0, j0)]
        tb = top_idx[(i1, j1)]
        ba = bottom_idx[(i0, j0)]
        bb = bottom_idx[(i1, j1)]
        faces += [[ta, ba, bb], [ta, bb, tb]]
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _high_valence_hub(spokes: int) -> tuple[np.ndarray, np.ndarray]:
    """A sealed bipyramid: a ring of `spokes` vertices fanned to two apex hubs,
    giving each apex a high degree (the hub) on a closed manifold."""
    verts = []
    ring = []
    for k in range(spokes):
        angle = 2.0 * np.pi * k / spokes
        ring.append(len(verts))
        verts.append([float(np.cos(angle)), float(np.sin(angle)), 0.0])
    top = len(verts)
    verts.append([0.0, 0.0, 1.0])
    bottom = len(verts)
    verts.append([0.0, 0.0, -1.0])
    faces = []
    for k in range(spokes):
        a = ring[k]
        b = ring[(k + 1) % spokes]
        faces.append([a, b, top])
        faces.append([b, a, bottom])
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _subdivided_tetrahedron(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A tetrahedron (valence-3 corners) optionally subdivided — closed manifold
    that stresses the valence-3 / fold guard."""
    verts = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
    faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
    for _ in range(subdivisions):
        mid: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in mid:
                return mid[key]
            pa, pb = verts[a], verts[b]
            m = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2]
            idx = len(verts)
            verts.append(m)
            mid[key] = idx
            return idx

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def test_qem_input_fixtures_are_closed_manifolds() -> None:
    # Sanity: every adversarial QEM input starts as a clean closed manifold so a
    # post-output regression is attributable to the collapser, not the fixture.
    for vertices, faces in (
        _icosphere(3),
        _thin_double_sided_sheet(6),
        _high_valence_hub(24),
        _subdivided_tetrahedron(3),
    ):
        _assert_closed_manifold(mesh_metrics(vertices, faces))


def test_qem_simplifies_icosphere_preserving_closed_manifold() -> None:
    vertices, faces = _icosphere(3)
    target = 200

    mesh, stats = simplify_mesh(vertices, faces, target_faces=target, backend="qem")
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["backend"] == "qem"
    assert stats["requested_backend"] == "qem"
    assert stats["algorithm"] == "native_qem_edge_collapse"
    assert stats["qem_simplification_backend"] == "native-qem-edge-collapse"
    assert stats["qem_equivalence_status"] == "edge-collapse"
    assert stats["source_faces"] == faces.shape[0]
    assert stats["qem_input_faces"] == faces.shape[0]
    assert stats["target_faces"] == target
    assert stats["final_faces"] == mesh.faces.shape[0]
    assert stats["qem_collapses_applied"] > 0
    assert mesh.faces.shape[0] < faces.shape[0]
    assert mesh.faces.shape[0] <= target
    assert stats["target_reached"] is True
    _assert_closed_manifold(after)


def test_qem_simplifies_thin_double_sided_sheet_without_pinch() -> None:
    vertices, faces = _thin_double_sided_sheet(6)

    mesh, stats = simplify_mesh(vertices, faces, target_faces=80, backend="qem")
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["backend"] == "qem"
    assert mesh.faces.shape[0] < faces.shape[0]
    _assert_closed_manifold(after)


def test_qem_simplifies_high_valence_hub_without_tearing() -> None:
    vertices, faces = _high_valence_hub(24)

    mesh, stats = simplify_mesh(vertices, faces, target_faces=16, backend="qem")
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["backend"] == "qem"
    assert mesh.faces.shape[0] <= faces.shape[0]
    _assert_closed_manifold(after)


def test_qem_simplifies_subdivided_tetrahedron_without_fold() -> None:
    vertices, faces = _subdivided_tetrahedron(3)

    mesh, stats = simplify_mesh(vertices, faces, target_faces=32, backend="qem")
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["backend"] == "qem"
    assert mesh.faces.shape[0] < faces.shape[0]
    _assert_closed_manifold(after)


def test_qem_is_deterministic_across_repeated_calls() -> None:
    vertices, faces = _icosphere(3)

    mesh_a, _ = simplify_mesh(vertices, faces, target_faces=150, backend="qem")
    mesh_b, _ = simplify_mesh(vertices, faces, target_faces=150, backend="qem")

    np.testing.assert_array_equal(mesh_a.vertices, mesh_b.vertices)
    np.testing.assert_array_equal(mesh_a.faces, mesh_b.faces)


def test_qem_input_already_under_target_returns_cleanly_via_early_return() -> None:
    vertices, faces = _subdivided_tetrahedron(1)
    target = faces.shape[0] + 100  # input already <= target

    mesh, stats = simplify_mesh(vertices, faces, target_faces=target, backend="qem")
    after = mesh_metrics(mesh.vertices, mesh.faces)

    assert stats["backend"] == "qem"
    assert stats["qem_simplification_backend"] == "native-qem-edge-collapse"
    assert stats["qem_equivalence_status"] == "edge-collapse"
    assert stats["target_reached"] is True
    assert stats["simplified"] is False
    assert stats["qem_collapses_applied"] == 0
    assert mesh.faces.shape[0] == faces.shape[0]
    _assert_closed_manifold(after)


def test_qem_stats_keyset_matches_clustering_plus_qem_fields() -> None:
    # M4: the forked qem stat dict's keyset must be a superset of EVERY clustering
    # backend's keyset (no clustering field silently dropped), and the extras must
    # be exactly the expected qem_* diagnostics — confirmed for both clustering
    # backends (spatial-cluster and topology-aware).
    vertices, faces = _icosphere(2)

    _, sc_stats = simplify_mesh(
        vertices, faces, target_faces=120, min_component_faces=1, backend="spatial-cluster"
    )
    _, ta_stats = simplify_mesh(
        vertices, faces, target_faces=120, min_component_faces=1, backend="topology-aware"
    )
    _, qem_stats = simplify_mesh(
        vertices, faces, target_faces=120, min_component_faces=1, backend="qem"
    )

    sc_keys = set(sc_stats.keys())
    ta_keys = set(ta_stats.keys())
    qem_keys = set(qem_stats.keys())
    expected_qem_only = {
        "qem_collapses_applied",
        "qem_collapses_rejected_by_guard",
        "qem_geometric_error_mean",
        "qem_geometric_error_max",
        "qem_input_faces",
        "qem_pre_fill_residual_boundary_loops",
    }

    # Every clustering key must be present in qem — no KeyError downstream.
    assert sc_keys - qem_keys == set(), (
        f"qem missing spatial-cluster keys: {sc_keys - qem_keys}"
    )
    assert ta_keys - qem_keys == set(), (
        f"qem missing topology-aware keys: {ta_keys - qem_keys}"
    )

    # qem-only extras must be exactly the documented qem_* diagnostics (vs topology-aware,
    # which has the same shared keyset as spatial-cluster).
    assert qem_keys - ta_keys == expected_qem_only, (
        f"unexpected qem-only extras vs topology-aware: {qem_keys - ta_keys - expected_qem_only}"
    )
    # spatial-cluster has the same shared keyset — confirm symmetry.
    assert qem_keys - sc_keys == expected_qem_only, (
        f"unexpected qem-only extras vs spatial-cluster: {qem_keys - sc_keys - expected_qem_only}"
    )


def test_qem_production_blocker_contract_main_path() -> None:
    # QEM-04 main decimation path: missing_qem_edge_collapse_simplification must be
    # ABSENT; missing_narrow_band_dc_remesh must be PRESENT (remesh is the remaining gap).
    vertices, faces = _icosphere(3)
    target = 200

    _, stats = simplify_mesh(vertices, faces, target_faces=target, backend="qem")

    assert stats["qem_simplification_backend"] == "native-qem-edge-collapse"
    assert stats["qem_equivalence_status"] == "edge-collapse"
    production_blockers = list(stats["production_blockers"])
    assert "missing_qem_edge_collapse_simplification" not in production_blockers, (
        "qem backend must not self-report missing_qem_edge_collapse_simplification"
    )
    assert "missing_narrow_band_dc_remesh" in production_blockers, (
        "qem backend must still report missing_narrow_band_dc_remesh until remesh runs"
    )


def test_qem_production_blocker_contract_early_return_path() -> None:
    # QEM-04 NR7: input already <= target triggers the early-return code path.
    # The blocker contract must hold identically on that path.
    vertices, faces = _subdivided_tetrahedron(1)
    target = faces.shape[0] + 100  # input already satisfies the target

    _, stats = simplify_mesh(vertices, faces, target_faces=target, backend="qem")

    assert stats["simplified"] is False
    assert stats["qem_collapses_applied"] == 0
    assert stats["qem_simplification_backend"] == "native-qem-edge-collapse"
    assert stats["qem_equivalence_status"] == "edge-collapse"
    production_blockers = list(stats["production_blockers"])
    assert "missing_qem_edge_collapse_simplification" not in production_blockers, (
        "early-return qem must not self-report missing_qem_edge_collapse_simplification"
    )
    assert "missing_narrow_band_dc_remesh" in production_blockers, (
        "early-return qem must still report missing_narrow_band_dc_remesh"
    )


def test_topology_aware_still_emits_not_implemented_and_blockers() -> None:
    # QEM-04 isolation: clustering backends must be undisturbed — topology-aware
    # must still emit qem_simplification_backend=="not_implemented" and both
    # production blockers.
    vertices, faces = _warped_grid_mesh(10)

    _, stats = simplify_mesh(vertices, faces, target_faces=80, min_component_faces=1, backend="topology-aware")

    assert stats["qem_simplification_backend"] == "not_implemented"
    assert stats["qem_equivalence_status"] == "qem_scored_not_edge_collapse"
    assert "missing_qem_edge_collapse_simplification" in list(stats["production_blockers"])
    assert "missing_narrow_band_dc_remesh" in list(stats["production_blockers"])


# ---------------------------------------------------------------------------
# S4 QEM input-prep: small boundary loop fill before edge collapse
# ---------------------------------------------------------------------------

def _icosphere_with_small_hole(subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """An icosphere (scaled small) with one face removed to create a small 3-edge boundary loop.

    Vertices are scaled to 0.01 so the triangular hole perimeter is ~0.019, which is
    below kPreSimplifyCleanBoundaryLoopFillMaxPerimeter (0.03). This models the
    situation after remesh(repair_nonmanifold=True), which opens small boundary loops.
    """
    vertices, faces = _icosphere(subdivisions)
    # Scale to make hole perimeter small enough for the pre-QEM reference fill
    # (perimeter ~1.89 * 0.01 = 0.019, under the 0.03 max).
    vertices = vertices * 0.01
    # Remove face 0 — this creates a triangular boundary loop.
    faces_with_hole = np.delete(faces, 0, axis=0)
    return vertices, faces_with_hole


def test_qem_input_prep_fills_small_boundary_loop_before_collapse() -> None:
    # S4: a mesh that has a small boundary loop (like one opened by
    # remesh(repair_nonmanifold=True)) must end watertight after
    # simplify_mesh(..., backend="qem") because the pre-QEM fill closes it.
    vertices, faces = _icosphere_with_small_hole(3)
    before = mesh_metrics(vertices, faces)
    target = 200

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=target,
        min_component_faces=1,
        backend="qem",
        small_boundary_loop_fill_max_edges=8,
        small_boundary_loop_fill_max_perimeter=0.03,
    )
    after = mesh_metrics(mesh.vertices, mesh.faces)

    # Input had a hole.
    assert before["boundary_loop_count"] == 1
    assert before["boundary_edges"] == 3

    # Pre-fill stats recorded.
    assert stats["pre_simplify_hole_fill_enabled"] is True
    assert stats["pre_simplify_hole_fill_algorithm"] == "reference-clean-boundary-centroid-fan"
    assert stats["pre_simplify_hole_fill_filled_loops"] == 1
    assert stats["pre_simplify_hole_fill_faces_added"] == 3
    assert stats["qem_pre_fill_residual_boundary_loops"] == 0

    # Pre-simplify face count reflects the filled mesh.
    assert stats["pre_simplify_faces"] == faces.shape[0] + 3
    assert stats["source_faces"] == faces.shape[0]
    assert stats["qem_input_faces"] == faces.shape[0] + 3

    # Output is watertight.
    assert after["boundary_loop_count"] == 0
    assert after["boundary_edges"] == 0
    assert after["nonmanifold_edges"] == 0
    assert after["export_blocking_reasons"] == []
    assert mesh.faces.shape[0] <= target


def test_qem_input_prep_noop_on_already_closed_mesh() -> None:
    # S4 determinism: on an already-closed input the fill is a no-op and
    # topology/determinism from slice 2 are preserved.
    vertices, faces = _icosphere(3)
    target = 200

    mesh_a, stats_a = simplify_mesh(
        vertices, faces, target_faces=target, min_component_faces=1, backend="qem",
        small_boundary_loop_fill_max_edges=8,
        small_boundary_loop_fill_max_perimeter=0.03,
    )
    mesh_b, stats_b = simplify_mesh(
        vertices, faces, target_faces=target, min_component_faces=1, backend="qem",
        small_boundary_loop_fill_max_edges=8,
        small_boundary_loop_fill_max_perimeter=0.03,
    )
    after = mesh_metrics(mesh_a.vertices, mesh_a.faces)

    # Fill was enabled but had nothing to fill (closed mesh).
    assert stats_a["pre_simplify_hole_fill_enabled"] is True
    assert stats_a["pre_simplify_hole_fill_filled_loops"] == 0
    assert stats_a["pre_simplify_hole_fill_boundary_edges_before"] == 0
    assert stats_a["qem_pre_fill_residual_boundary_loops"] == 0

    # source and pre_simplify counts are the same (fill was a no-op).
    assert stats_a["source_faces"] == faces.shape[0]
    assert stats_a["pre_simplify_faces"] == faces.shape[0]
    assert stats_a["qem_input_faces"] == faces.shape[0]

    # Still closed, still deterministic.
    assert after["boundary_loop_count"] == 0
    assert after["nonmanifold_edges"] == 0
    np.testing.assert_array_equal(mesh_a.vertices, mesh_b.vertices)
    np.testing.assert_array_equal(mesh_a.faces, mesh_b.faces)


def test_qem_input_prep_disabled_when_fill_max_edges_zero() -> None:
    # S4 opt-out: small_boundary_loop_fill_max_edges=0 disables the pre-fill;
    # the pre_simplify_hole_fill_enabled stat must be False and the small loop
    # remains open in the output.
    vertices, faces = _icosphere_with_small_hole(3)
    target = 200

    mesh, stats = simplify_mesh(
        vertices,
        faces,
        target_faces=target,
        min_component_faces=1,
        backend="qem",
        small_boundary_loop_fill_max_edges=0,
        small_boundary_loop_fill_max_perimeter=0.03,
    )

    assert stats["pre_simplify_hole_fill_enabled"] is False
    assert stats["source_faces"] == faces.shape[0]
    assert stats["pre_simplify_faces"] == faces.shape[0]
    # The small loop was not filled; QEM boundary-locks it so it remains.
    after = mesh_metrics(mesh.vertices, mesh.faces)
    assert after["boundary_loop_count"] >= 1
