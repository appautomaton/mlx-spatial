from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mlx_spatialkit import export_pixal3d_glb, metal_device_available
from mlx_spatialkit.export import (
    _resolve_chart_angle_degrees,
    _export_quality_summary,
    _glb_viewer_compatibility_summary,
    _native_chart_uv_candidate_status,
    _resolve_pixal3d_export_settings,
    _resolve_tile_padding,
    _resolve_pixal3d_uv_backend,
    _simplifier_backend_for_quality_preset,
    _upstream_export_settings_summary,
    _xatlas_chart_parity_summary,
)
from glb_texture_utils import glb_image_payload, png_coverage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _assert_xatlas_parity_measured(diagnostics: dict, uv_stats: dict, texture_stats: dict) -> None:
    parity = diagnostics["quality"]["xatlas_chart_parity"]
    assert parity["status"] == "measured_not_equivalent"
    assert parity["measurement_ready"] is True
    assert parity["parity_ready"] is False
    assert parity["xatlas_chart_parity"] is False
    assert parity["deferred_boundary"] == "not_xatlas_chart_parity"
    assert parity["reference"]["unwrap_backend"] == "xatlas-parallel-spatial"
    assert parity["reference"]["unwrap_chart_count"] == 51_953
    assert parity["reference"]["unwrap_utilization"] == pytest.approx(0.8309683442115784)
    assert parity["native"]["uv_backend"] == "native-chart-atlas"
    assert parity["native"]["chart_count"] == uv_stats["chart_count"]
    assert parity["native"]["chart_rect_fill_ratio"] == pytest.approx(uv_stats["chart_rect_fill_ratio"])
    expected_occupancy = texture_stats["uv_surface_texel_count"] / texture_stats["texture_pixel_count"]
    assert parity["native"]["uv_surface_occupancy_ratio"] == pytest.approx(expected_occupancy)
    assert parity["ratios"]["chart_count_ratio"] == pytest.approx(uv_stats["chart_count"] / 51_953)
    expected_utilization_ratio = expected_occupancy / 0.8309683442115784
    assert parity["ratios"]["uv_surface_occupancy_vs_reference_utilization"] == pytest.approx(expected_utilization_ratio)
    assert parity["deficits"]["reference_utilization_minus_native_uv_surface_occupancy"] == pytest.approx(
        max(0.0, 0.8309683442115784 - expected_occupancy)
    )
    assert parity["deficits"]["uv_surface_occupancy_ratio_gap_to_reference"] == pytest.approx(
        max(0.0, 1.0 - expected_utilization_ratio)
    )
    assert parity["deficits"]["uv_surface_occupancy_ratio_gap_to_equivalence_target"] == pytest.approx(
        max(0.0, 0.95 - expected_utilization_ratio)
    )
    assert parity["deficits"]["equivalence_target_ratio"] == pytest.approx(0.95)
    utilization_check = parity["checks"]["xatlas_utilization_equivalence"]
    assert utilization_check["passed"] is False
    assert utilization_check["actual"] == pytest.approx(expected_utilization_ratio)
    assert utilization_check["required"] == ">=0.95"
    for name, check in parity["checks"].items():
        if name in {"xatlas_backend_equivalence", "xatlas_utilization_equivalence"}:
            assert check["passed"] is False
        else:
            assert check["passed"] is True


@pytest.mark.heavy
def test_export_pixal3d_glb_real_decoded_fixture_writes_glb_and_diagnostics() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-real-pixal3d-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        texture_size=1024,
        target_faces=50_000,
        min_component_faces=32,
    )

    assert result.glb.path == output_dir / "model.glb"
    assert result.diagnostics_path == output_dir / "diagnostics.json"
    assert result.glb.path.read_bytes()[:4] == b"glTF"
    assert result.glb.bytes_written == result.glb.path.stat().st_size
    assert result.glb.bytes_written > 1_000_000

    diagnostics = json.loads(result.diagnostics_path.read_text())
    assert diagnostics["result"]["ready"] is True
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is False
    assert "preview_simplifier_quality_tier" in diagnostics["result"]["quality_warnings"]
    assert diagnostics["quality"]["simplifier_backend"] == "spatial-cluster"
    assert diagnostics["quality"]["simplifier_quality_tier"] == "geometry_aware_preview"
    assert diagnostics["quality"]["production_quality_ready"] is False
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    assert diagnostics["settings"]["texture_size"] == 1024
    assert diagnostics["settings"]["target_faces"] == 50_000
    assert diagnostics["settings"]["grid_size"] == 1024
    assert diagnostics["contracts"]["shape"]["token_count"] == 4_150_336
    assert diagnostics["contracts"]["texture"]["token_count"] == 4_150_336
    assert diagnostics["stages"]["extract_mesh"]["source_faces"] == 8_304_022
    assert diagnostics["stages"]["clean_mesh"]["cleaned_faces"] > 8_000_000
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    assert diagnostics["stages"]["simplify_mesh"]["simplified_faces"] > 0
    assert diagnostics["stages"]["simplify_mesh"]["simplified_faces"] <= 50_000
    assert simplify_stats["backend"] == "spatial-cluster"
    assert simplify_stats["target_reached"] is True
    assert diagnostics["stages"]["uv"]["stats"]["backend"] == "face-atlas"
    assert diagnostics["stages"]["texture_bake"]["stats"]["backend"] == "metal-face-atlas-nearest"
    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    assert texture_stats["sampled_texel_count"] > 0
    assert texture_stats["fallback_filled_texel_count"] > 0
    assert texture_stats["uv_surface_texel_count"] > texture_stats["sampled_texel_count"]
    assert texture_stats["final_visible_coverage_ratio"] > texture_stats["raw_coverage_ratio"]
    assert texture_stats["final_visible_coverage_ratio"] > 0.10
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] > 0.50
    base_color_coverage = png_coverage(glb_image_payload(result.glb.path.read_bytes(), "baseColorTexture"))
    assert base_color_coverage.alpha_coverage_ratio > 0.10
    assert base_color_coverage.rgb_coverage_ratio > 0.10
    assert base_color_coverage.alpha_coverage_ratio == pytest.approx(
        texture_stats["final_visible_coverage_ratio"],
        abs=0.005,
    )
    assert diagnostics["stages"]["write_glb"]["artifact"]["mesh_name"] == "Pixal3D_TexturedMesh"
    write_inspection = diagnostics["stages"]["write_glb"]["inspection"]
    assert write_inspection["primitive_count"] > 1
    assert all(primitive["has_normals"] for primitive in write_inspection["primitives"])
    assert all(primitive["indices_component_type"] == 5123 for primitive in write_inspection["primitives"])
    assert diagnostics["reference"]["final_faces"] == 212_542
    comparison = diagnostics["reference_comparison"]
    assert comparison["spatialkit_simplifier_backend"] == "spatial-cluster"
    assert comparison["reference_unwrap_backend"] == "xatlas-parallel-spatial"
    assert comparison["reference_bake_backend"] == "xatlas-kdtree"
    assert comparison["final_face_count_ratio"] > 0.0
    assert comparison["final_coverage_ratio_vs_reference"] > 0.10
    assert "after_write_glb" in diagnostics["memory_samples"]
    _assert_memory_diagnostics(diagnostics, required_stages=("texture_bake", "write_glb"))


@pytest.mark.heavy
def test_export_pixal3d_glb_native_chart_backend_writes_real_fixture() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-native-chart-pixal3d-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        texture_size=1024,
        target_faces=50_000,
        min_component_faces=32,
        uv_backend="native-chart",
        chart_angle_degrees=45.0,
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())

    assert result.glb.path == output_dir / "model.glb"
    assert result.glb.path.read_bytes()[:4] == b"glTF"
    assert result.glb.bytes_written > 1_000_000
    assert diagnostics["settings"]["requested_uv_backend"] == "native-chart"
    assert diagnostics["settings"]["uv_backend"] == "native-chart"
    assert diagnostics["settings"]["chart_angle_degrees"] == 45.0
    assert diagnostics["settings"]["tile_padding"] == 0.001
    assert diagnostics["settings"]["tile_padding_source"] == "backend_default:native-chart"
    assert diagnostics["result"]["artifact_ready"] is True
    assert "native_chart_uv_candidate_quality_blocked" not in diagnostics["result"]["quality_warnings"]
    uv_stats = diagnostics["stages"]["uv"]["stats"]
    assert uv_stats["backend"] == "native-chart-atlas"
    assert uv_stats["projection"] == "local-frame-pca"
    assert uv_stats["projection_rotation_candidates"] == 19
    assert uv_stats["projection_rotation_step_degrees"] == 5.0
    assert uv_stats["chart_rect_fill_ratio"] > 0.50
    assert uv_stats["source_chart_count"] > 0
    assert uv_stats["chart_count"] >= uv_stats["source_chart_count"]
    assert uv_stats["chart_split_max_faces"] == 512
    assert uv_stats["chart_split_count"] > 0
    assert uv_stats["oversized_source_chart_count"] > 0
    assert uv_stats["pre_low_fill_chart_count"] > 0
    assert uv_stats["low_fill_rect_fill_threshold"] == pytest.approx(0.70)
    assert uv_stats["low_fill_split_min_improvement"] == pytest.approx(0.02)
    assert uv_stats["low_fill_split_min_faces"] == 6
    assert uv_stats["low_fill_split_min_child_faces"] == 3
    assert uv_stats["low_fill_split_max_depth"] == 3
    assert uv_stats["low_fill_split_axis_candidates"] == 2
    assert uv_stats["low_fill_split_position_candidates"] == 3
    assert uv_stats["low_fill_split_candidate_count"] > 0
    assert uv_stats["low_fill_split_axis_candidate_count"] == (
        uv_stats["low_fill_split_candidate_count"] * uv_stats["low_fill_split_axis_candidates"]
    )
    assert uv_stats["low_fill_split_partition_candidate_count"] == (
        uv_stats["low_fill_split_axis_candidate_count"] * uv_stats["low_fill_split_position_candidates"]
    )
    assert uv_stats["low_fill_split_partition_evaluated_count"] > uv_stats["low_fill_split_axis_candidate_count"]
    assert uv_stats["low_fill_split_partition_evaluated_count"] <= uv_stats["low_fill_split_partition_candidate_count"]
    assert uv_stats["low_fill_source_chart_count"] > 0
    assert uv_stats["low_fill_split_accepted_count"] > 0
    assert uv_stats["low_fill_chart_split_count"] == uv_stats["chart_count"] - uv_stats["pre_low_fill_chart_count"]
    assert uv_stats["chart_rect_fill_ratio"] > uv_stats["pre_low_fill_chart_rect_fill_ratio"]
    assert uv_stats["max_chart_faces"] <= uv_stats["chart_split_max_faces"]
    assert uv_stats["packing"] == "aspect-shelf-charts"
    assert uv_stats["chart_count"] > 0
    assert uv_stats["output_faces"] == diagnostics["stages"]["simplify_mesh"]["simplified_faces"]
    assert uv_stats["output_vertices"] <= uv_stats["source_faces"] * 3
    assert uv_stats["duplicated_vertex_ratio"] >= 1.0
    assert uv_stats["shelf_rows"] > 0
    assert uv_stats["shelf_packing_efficiency"] > 0.90
    assert uv_stats["atlas_rect_coverage_ratio"] > 0.90
    assert "atlas_cols" not in uv_stats
    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    assert texture_stats["backend"] == "metal-uv-binned-nearest"
    assert texture_stats["uv_bin_count"] > 0
    assert texture_stats["uv_bin_face_reference_count"] > 0
    assert texture_stats["uv_bin_guard_passed"] is True
    assert texture_stats["sampled_texel_count"] > 0
    assert texture_stats["surface_fill_enabled"] is True
    assert texture_stats["surface_filled_texel_count"] > 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert texture_stats["raw_coverage_ratio"] < texture_stats["final_visible_coverage_ratio"]
    assert texture_stats["uv_surface_exact_coverage_ratio"] < texture_stats["uv_surface_final_visible_coverage_ratio"]
    assert diagnostics["stages"]["write_glb"]["artifact"]["uv_backend"] == "native-chart"
    assert diagnostics["stages"]["write_glb"]["artifact"]["uv_stats_backend"] == "native-chart-atlas"
    candidate = diagnostics["quality"]["native_chart_uv_candidate"]
    assert candidate["status"] == "quality_ready"
    assert candidate["artifact_ready"] is True
    assert candidate["quality_ready"] is True
    assert candidate["uv_backend"] == "native-chart-atlas"
    assert candidate["texture_bake_backend"] == "metal-uv-binned-nearest"
    assert candidate["global_coverage_ratio"] == pytest.approx(texture_stats["final_visible_coverage_ratio"])
    assert candidate["raw_coverage_ratio"] == pytest.approx(texture_stats["raw_coverage_ratio"])
    assert candidate["uv_surface_exact_coverage_ratio"] == pytest.approx(texture_stats["uv_surface_exact_coverage_ratio"])
    assert candidate["surface_filled_texel_count"] == texture_stats["surface_filled_texel_count"]
    assert candidate["surface_unfilled_texel_count"] == 0
    assert candidate["uv_surface_occupancy_ratio"] == pytest.approx(
        texture_stats["uv_surface_texel_count"] / texture_stats["texture_pixel_count"]
    )
    assert uv_stats["chart_rect_fill_ratio"] > 0.5727071617508422
    assert candidate["global_coverage_ratio"] >= 0.50
    assert candidate["uv_surface_occupancy_ratio"] > 0.50
    assert candidate["uv_bin_max_candidate_faces"] <= 512
    assert candidate["checks"]["global_coverage_floor"]["passed"] is True
    assert candidate["checks"]["uv_surface_occupancy_floor"]["passed"] is True
    assert candidate["checks"]["uv_surface_visible_floor"]["passed"] is True
    assert candidate["quality_blockers"] == []
    assert candidate["xatlas_chart_parity"] is False
    assert "not_xatlas_chart_parity" in diagnostics["visual_comparison"]["deferred_parity_boundaries"]
    _assert_memory_diagnostics(diagnostics, required_stages=("uv", "texture_bake", "write_glb"))


@pytest.mark.heavy
def test_export_pixal3d_glb_reference_target_preset_reports_thresholds() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-reference-target-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="reference-target",
        texture_size=1024,
        min_component_faces=32,
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())

    assert diagnostics["settings"]["quality_preset"] == "reference-target"
    assert diagnostics["settings"]["target_faces"] == 212_542
    assert diagnostics["settings"]["target_faces_source"] == "reference_final_faces"
    assert diagnostics["settings"]["requested_simplifier_backend"] == "topology-aware"
    assert diagnostics["settings"]["reference_available"] is True
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is True
    assert diagnostics["result"]["quality_warnings"] == []
    assert diagnostics["quality"]["upstream_export_settings"]["all_passed"] is False
    assert diagnostics["quality"]["upstream_export_settings"]["checks"]["target_faces"]["passed"] is False
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    assert diagnostics["quality"]["native_geometry_candidate"]["status"] == "candidate"
    assert diagnostics["quality"]["native_geometry_candidate"]["reason"] == "native_geometry_candidate_available"
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    assert simplify_stats["requested_backend"] == "topology-aware"
    assert simplify_stats["backend"] == "topology-aware"
    assert simplify_stats["algorithm"] == "native_topology_aware_representative_clustering"
    assert simplify_stats["quality_tier"] == "production"
    assert simplify_stats["production_ready"] is True
    assert simplify_stats["production_blockers"] == []
    assert simplify_stats["target_reached"] is True
    uv_stats = diagnostics["stages"]["uv"]["stats"]
    assert uv_stats["backend"] == "face-atlas"
    assert uv_stats["packing"] == "paired-triangles"
    assert uv_stats["faces_per_tile"] == 2
    assert uv_stats["atlas_tiles"] == (uv_stats["output_faces"] + 1) // 2
    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    assert texture_stats["backend"] == "metal-face-atlas-nearest"
    assert texture_stats["atlas_faces_per_tile"] == 2
    assert texture_stats["final_visible_coverage_ratio"] > 0.269
    thresholds = diagnostics["quality"]["production_thresholds"]["checks"]
    assert thresholds["reference_available"]["passed"] is True
    assert thresholds["quality_preset"]["passed"] is True
    assert thresholds["topology_exportability"]["passed"] is True
    assert thresholds["face_count_ratio"]["passed"] is True
    assert 0.80 <= thresholds["face_count_ratio"]["actual"] <= 1.25
    assert thresholds["backend_tier"]["passed"] is True
    assert thresholds["backend_tier"]["actual"] == "production"
    assert thresholds["final_coverage_ratio"]["actual"] == pytest.approx(
        texture_stats["final_visible_coverage_ratio"]
    )
    assert thresholds["final_coverage_ratio"]["passed"] is True
    assert thresholds["final_coverage_ratio"]["actual"] >= thresholds["final_coverage_ratio"]["required_min"]
    assert thresholds["raw_coverage_ratio"]["passed"] is True
    assert diagnostics["reference_comparison"]["reference_bake_backend"] == "xatlas-kdtree"
    visual = diagnostics["visual_comparison"]
    assert visual["summary"]["all_passed"] is True
    assert 0.80 <= visual["summary"]["face_count_ratio"] <= 1.25
    assert visual["summary"]["texture_resolution_match"] is True
    assert visual["summary"]["base_color_alpha_coverage_ratio"] >= 0.50
    assert visual["summary"]["base_color_rgb_coverage_ratio"] >= 0.50
    assert visual["checks"]["texture_resolution_match"]["passed"] is True
    assert "not_xatlas_chart_parity" in visual["deferred_parity_boundaries"]
    assert "not_1m_face_export_setting_parity" in visual["deferred_parity_boundaries"]
    assert "not_4096_texture_parity" not in visual["deferred_parity_boundaries"]
    assert "not_browser_rendered_visual_proof" not in visual["deferred_parity_boundaries"]
    visual_artifacts = visual["artifacts"]
    assert Path(visual_artifacts["report_json"]).is_file()
    assert Path(visual_artifacts["preview_html"]).is_file()
    assert Path(visual_artifacts["candidate_base_color_png"]).is_file()
    assert Path(visual_artifacts["reference_base_color_png"]).is_file()
    assert Path(visual_artifacts["report_json"]).parent == output_dir / "visual_parity"
    write_inspection = diagnostics["stages"]["write_glb"]["inspection"]
    assert write_inspection["primitive_count"] > 1
    assert all(primitive["has_normals"] for primitive in write_inspection["primitives"])
    assert all(primitive["indices_component_type"] == 5123 for primitive in write_inspection["primitives"])
    assert "after_write_glb" in diagnostics["memory_samples"]
    _assert_memory_diagnostics(diagnostics, required_stages=("texture_bake", "write_glb", "visual_compare"))


@pytest.mark.heavy
def test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-native-chart-reference-target-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="reference-target",
        texture_size=1024,
        min_component_faces=32,
        uv_backend="native-chart",
        chart_angle_degrees=45.0,
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())

    assert result.glb.path == output_dir / "model.glb"
    assert diagnostics["settings"]["quality_preset"] == "reference-target"
    assert diagnostics["settings"]["uv_backend"] == "native-chart"
    assert diagnostics["settings"]["target_faces"] == 212_542
    assert diagnostics["settings"]["tile_padding"] == 0.001
    assert diagnostics["settings"]["tile_padding_source"] == "backend_default:native-chart"
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is True
    assert diagnostics["result"]["quality_warnings"] == []
    assert diagnostics["quality"]["upstream_export_settings"]["all_passed"] is False
    assert diagnostics["quality"]["upstream_export_settings"]["checks"]["target_faces"]["passed"] is False

    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    assert simplify_stats["small_boundary_loop_fill_enabled"] is True
    assert simplify_stats["small_boundary_loop_fill_max_edges"] == 3
    assert simplify_stats["small_boundary_loops_considered"] > 0
    assert simplify_stats["small_boundary_loops_filled"] > 0
    assert simplify_stats["small_boundary_loop_faces_added"] > 0
    assert simplify_stats["small_boundary_loops_budget_limited"] == 0
    assert simplify_stats["final_faces"] <= simplify_stats["target_faces"]

    uv_stats = diagnostics["stages"]["uv"]["stats"]
    assert uv_stats["backend"] == "native-chart-atlas"
    assert uv_stats["chart_count"] > 0
    assert uv_stats["output_faces"] == diagnostics["stages"]["simplify_mesh"]["simplified_faces"]
    assert uv_stats["chart_rect_fill_ratio"] > 0.5670824417746222
    assert uv_stats["low_fill_split_position_candidates"] == 3
    assert uv_stats["low_fill_split_partition_candidate_count"] == (
        uv_stats["low_fill_split_axis_candidate_count"] * uv_stats["low_fill_split_position_candidates"]
    )
    assert uv_stats["low_fill_split_partition_evaluated_count"] > uv_stats["low_fill_split_axis_candidate_count"]
    assert uv_stats["low_fill_split_accepted_count"] > 0

    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    assert texture_stats["backend"] == "metal-uv-binned-nearest"
    assert texture_stats["uv_bin_guard_passed"] is True
    assert texture_stats["surface_fill_enabled"] is True
    assert texture_stats["surface_filled_texel_count"] > 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["raw_coverage_ratio"] < texture_stats["final_visible_coverage_ratio"]
    assert texture_stats["uv_surface_exact_coverage_ratio"] < texture_stats["uv_surface_final_visible_coverage_ratio"]
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)

    export_metrics = diagnostics["stages"]["export_metrics"]["metrics"]
    assert export_metrics["face_count"] == diagnostics["stages"]["simplify_mesh"]["simplified_faces"]
    assert export_metrics["nonmanifold_edges"] == 0
    assert export_metrics["boundary_edges"] > 0
    assert export_metrics["boundary_vertices"] > 0
    assert export_metrics["boundary_loop_count"] > 0
    assert export_metrics["boundary_edges"] < 23822
    assert export_metrics["boundary_loop_count"] < 2594
    assert export_metrics["boundary_small_loop_threshold_edges"] == 32
    assert export_metrics["boundary_small_loop_count"] <= export_metrics["boundary_loop_count"]
    assert export_metrics["boundary_small_loop_edge_count"] <= export_metrics["boundary_edges"]
    assert export_metrics["boundary_max_component_edges"] >= export_metrics["boundary_max_loop_edges"]
    assert export_metrics["export_blocking_reasons"] == []

    candidate = diagnostics["quality"]["native_chart_uv_candidate"]
    assert candidate["status"] == "quality_ready"
    assert candidate["artifact_ready"] is True
    assert candidate["quality_ready"] is True
    assert candidate["uv_backend"] == "native-chart-atlas"
    assert candidate["texture_bake_backend"] == "metal-uv-binned-nearest"
    assert candidate["global_coverage_ratio"] == pytest.approx(texture_stats["final_visible_coverage_ratio"])
    assert candidate["global_coverage_ratio"] >= 0.50
    assert candidate["raw_coverage_ratio"] == pytest.approx(texture_stats["raw_coverage_ratio"])
    assert candidate["uv_surface_exact_coverage_ratio"] == pytest.approx(texture_stats["uv_surface_exact_coverage_ratio"])
    assert candidate["surface_filled_texel_count"] == texture_stats["surface_filled_texel_count"]
    assert candidate["surface_unfilled_texel_count"] == 0
    assert candidate["checks"]["global_coverage_floor"]["passed"] is True
    assert candidate["checks"]["uv_surface_occupancy_floor"]["passed"] is True
    assert candidate["checks"]["uv_surface_visible_floor"]["passed"] is True
    assert candidate["quality_blockers"] == []
    assert candidate["xatlas_chart_parity"] is False
    _assert_xatlas_parity_measured(diagnostics, uv_stats, texture_stats)
    assert diagnostics["quality"]["xatlas_chart_parity"]["ratios"][
        "uv_surface_occupancy_vs_reference_utilization"
    ] > 0.680

    visual = diagnostics["visual_comparison"]
    assert visual["summary"]["all_passed"] is True
    assert 0.80 <= visual["summary"]["face_count_ratio"] <= 1.25
    assert visual["summary"]["texture_resolution_match"] is True
    assert visual["summary"]["base_color_alpha_coverage_ratio"] >= 0.50
    assert "not_xatlas_chart_parity" in visual["deferred_parity_boundaries"]
    assert "not_1m_face_export_setting_parity" in visual["deferred_parity_boundaries"]
    assert "not_4096_texture_parity" not in visual["deferred_parity_boundaries"]
    assert "not_browser_rendered_visual_proof" not in visual["deferred_parity_boundaries"]
    visual_artifacts = visual["artifacts"]
    assert Path(visual_artifacts["report_json"]).is_file()
    assert Path(visual_artifacts["preview_html"]).is_file()
    assert Path(visual_artifacts["candidate_base_color_png"]).is_file()
    assert Path(visual_artifacts["reference_base_color_png"]).is_file()

    write_inspection = diagnostics["stages"]["write_glb"]["inspection"]
    assert write_inspection["primitive_count"] > 1
    assert all(primitive["has_normals"] for primitive in write_inspection["primitives"])
    assert all(primitive["indices_component_type"] == 5123 for primitive in write_inspection["primitives"])
    _assert_memory_diagnostics(diagnostics, required_stages=("uv", "texture_bake", "write_glb", "visual_compare"))


@pytest.mark.heavy
def test_export_pixal3d_glb_reference_target_4096_texture_passes_coverage_gate() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-reference-target-4096-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="reference-target",
        texture_size=4096,
        min_component_faces=32,
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())
    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    thresholds = diagnostics["quality"]["production_thresholds"]["checks"]

    assert diagnostics["settings"]["quality_preset"] == "reference-target"
    assert diagnostics["settings"]["texture_size"] == 4096
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is True
    assert diagnostics["result"]["quality_warnings"] == []
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    assert texture_stats["texture_size"] == 4096
    assert texture_stats["dilation_max_passes"] > 8
    assert texture_stats["dilation_pass_count"] <= texture_stats["dilation_max_passes"]
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] >= 0.50
    assert thresholds["final_coverage_ratio"]["passed"] is True
    assert thresholds["final_coverage_ratio"]["actual"] == pytest.approx(
        texture_stats["final_visible_coverage_ratio"]
    )
    assert thresholds["backend_tier"]["passed"] is True
    assert thresholds["face_count_ratio"]["passed"] is True
    visual = diagnostics["visual_comparison"]
    assert visual["checks"]["texture_resolution_match"]["passed"] is False
    assert visual["checks"]["texture_resolution_match"]["candidate"] == {"height": 4096, "width": 4096}
    assert visual["checks"]["texture_resolution_match"]["reference"] == {"height": 1024, "width": 1024}
    _assert_memory_diagnostics(diagnostics, required_stages=("texture_bake", "write_glb", "visual_compare"))


@pytest.mark.heavy
def test_export_pixal3d_glb_upstream_settings_passes_readiness_gate() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-upstream-settings-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="reference-target",
        target_faces=1_000_000,
        texture_size=4096,
        min_component_faces=32,
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())
    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    upstream = diagnostics["quality"]["upstream_export_settings"]

    assert diagnostics["settings"]["target_faces"] == 1_000_000
    assert diagnostics["settings"]["target_faces_source"] == "explicit"
    assert diagnostics["settings"]["texture_size"] == 4096
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is False
    assert upstream["all_passed"] is True
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    for check in upstream["checks"].values():
        assert check["passed"] is True
    assert upstream["reference"]["decimation_target"] == 1_000_000
    assert upstream["reference"]["texture_size"] == 4096
    assert upstream["reference"]["remesh"] is True
    assert upstream["reference"]["xatlas_chart_parity"] is False
    assert simplify_stats["backend"] == "topology-aware"
    assert simplify_stats["target_faces"] == 1_000_000
    assert simplify_stats["target_reached"] is True
    assert 600_000 <= simplify_stats["final_faces"] <= 1_000_000
    assert texture_stats["texture_size"] == 4096
    assert texture_stats["fallback_radius"] >= 24
    assert texture_stats["dilation_max_passes"] >= 26
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    visual = diagnostics["visual_comparison"]
    assert "not_xatlas_chart_parity" in visual["deferred_parity_boundaries"]
    assert "not_1m_face_export_setting_parity" not in visual["deferred_parity_boundaries"]
    write_inspection = diagnostics["stages"]["write_glb"]["inspection"]
    assert write_inspection["primitive_count"] > 1
    assert all(primitive["has_normals"] for primitive in write_inspection["primitives"])
    assert all(primitive["indices_component_type"] == 5123 for primitive in write_inspection["primitives"])
    _assert_memory_diagnostics(diagnostics, required_stages=("texture_bake", "write_glb", "visual_compare"))


@pytest.mark.heavy
def test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-native-chart-upstream-settings-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="reference-target",
        target_faces=1_000_000,
        texture_size=4096,
        min_component_faces=32,
        uv_backend="native-chart",
        chart_angle_degrees=45.0,
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    uv_stats = diagnostics["stages"]["uv"]["stats"]
    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    upstream = diagnostics["quality"]["upstream_export_settings"]
    candidate = diagnostics["quality"]["native_chart_uv_candidate"]

    assert result.glb.path == output_dir / "model.glb"
    assert diagnostics["settings"]["target_faces"] == 1_000_000
    assert diagnostics["settings"]["target_faces_source"] == "explicit"
    assert diagnostics["settings"]["texture_size"] == 4096
    assert diagnostics["settings"]["uv_backend"] == "native-chart"
    assert diagnostics["settings"]["tile_padding"] == 0.001
    assert diagnostics["settings"]["tile_padding_source"] == "backend_default:native-chart"
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is False
    assert diagnostics["result"]["quality_warnings"] == ["production_thresholds_failed"]

    assert upstream["all_passed"] is True
    for check in upstream["checks"].values():
        assert check["passed"] is True
    assert upstream["reference"]["decimation_target"] == 1_000_000
    assert upstream["reference"]["texture_size"] == 4096
    assert upstream["reference"]["remesh"] is True
    assert upstream["reference"]["xatlas_chart_parity"] is False

    thresholds = diagnostics["quality"]["production_thresholds"]["checks"]
    assert thresholds["face_count_ratio"]["passed"] is False
    assert thresholds["face_count_ratio"]["spatialkit_final_faces"] == simplify_stats["final_faces"]
    assert thresholds["face_count_ratio"]["reference_final_faces"] == 212_542
    assert thresholds["final_coverage_ratio"]["passed"] is True

    assert simplify_stats["backend"] == "topology-aware"
    assert simplify_stats["target_faces"] == 1_000_000
    assert simplify_stats["target_reached"] is True
    assert 600_000 <= simplify_stats["final_faces"] <= 1_000_000

    assert uv_stats["backend"] == "native-chart-atlas"
    assert uv_stats["chart_count"] > 0
    assert uv_stats["output_faces"] == simplify_stats["final_faces"]
    assert uv_stats["chart_rect_fill_ratio"] > 0.50
    assert uv_stats["low_fill_split_position_candidates"] == 3
    assert uv_stats["low_fill_split_partition_candidate_count"] == (
        uv_stats["low_fill_split_axis_candidate_count"] * uv_stats["low_fill_split_position_candidates"]
    )
    assert uv_stats["low_fill_split_partition_evaluated_count"] > uv_stats["low_fill_split_axis_candidate_count"]

    assert texture_stats["backend"] == "metal-uv-binned-nearest"
    assert texture_stats["texture_size"] == 4096
    assert texture_stats["uv_bin_guard_passed"] is True
    assert texture_stats["uv_bin_face_reference_count"] > 0
    assert texture_stats["uv_bin_max_candidate_faces"] > 0
    assert texture_stats["surface_fill_enabled"] is True
    assert texture_stats["surface_filled_texel_count"] > 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)

    assert candidate["status"] == "quality_ready"
    assert candidate["artifact_ready"] is True
    assert candidate["quality_ready"] is True
    assert candidate["global_coverage_ratio"] == pytest.approx(texture_stats["final_visible_coverage_ratio"])
    assert candidate["uv_surface_occupancy_ratio"] >= 0.50
    assert candidate["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert candidate["quality_blockers"] == []
    assert candidate["xatlas_chart_parity"] is False
    _assert_xatlas_parity_measured(diagnostics, uv_stats, texture_stats)

    visual = diagnostics["visual_comparison"]
    assert visual["summary"]["all_passed"] is False
    assert visual["checks"]["face_count_ratio"]["passed"] is False
    assert visual["checks"]["texture_resolution_match"]["passed"] is False
    assert visual["checks"]["texture_resolution_match"]["candidate"] == {"height": 4096, "width": 4096}
    assert visual["checks"]["texture_resolution_match"]["reference"] == {"height": 1024, "width": 1024}
    assert visual["deferred_parity_boundaries"] == ["not_xatlas_chart_parity"]

    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    write_inspection = diagnostics["stages"]["write_glb"]["inspection"]
    assert write_inspection["primitive_count"] > 1
    assert all(primitive["has_normals"] for primitive in write_inspection["primitives"])
    assert all(primitive["indices_component_type"] == 5123 for primitive in write_inspection["primitives"])
    _assert_memory_diagnostics(diagnostics, required_stages=("uv", "texture_bake", "write_glb", "visual_compare"))


def test_export_pixal3d_glb_rejects_invalid_public_guards(tmp_path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    with pytest.raises(ValueError, match="grid_size must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", grid_size=0)

    with pytest.raises(ValueError, match="max_texture_pixels must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", max_texture_pixels=0)


def test_export_pixal3d_uv_backend_settings_contract(tmp_path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    assert _resolve_pixal3d_uv_backend("face-atlas") == "face-atlas"
    assert _resolve_pixal3d_uv_backend("native_chart") == "native-chart"
    assert _resolve_chart_angle_degrees(45.0) == 45.0
    assert _resolve_tile_padding(None, "face-atlas") == (0.08, "backend_default:face-atlas")
    assert _resolve_tile_padding(None, "native-chart") == (0.001, "backend_default:native-chart")
    assert _resolve_tile_padding(0.07, "native-chart") == (0.07, "explicit")

    with pytest.raises(ValueError, match="uv_backend"):
        _resolve_pixal3d_uv_backend("xatlas")
    with pytest.raises(ValueError, match="chart_angle_degrees"):
        _resolve_chart_angle_degrees(float("nan"))
    with pytest.raises(ValueError, match="chart_angle_degrees"):
        _resolve_chart_angle_degrees(181.0)
    with pytest.raises(ValueError, match="tile_padding"):
        _resolve_tile_padding(float("nan"), "native-chart")
    with pytest.raises(ValueError, match="tile_padding"):
        _resolve_tile_padding(0.45, "face-atlas")
    with pytest.raises(ValueError, match="uv_backend"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", uv_backend="xatlas")
    with pytest.raises(ValueError, match="chart_angle_degrees"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", uv_backend="native-chart", chart_angle_degrees=-1.0)


def test_reference_target_preset_resolves_target_faces_from_trace(tmp_path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()
    reference_dir = tmp_path / "pixal3d-1024-cascade-glb-reference"
    reference_dir.mkdir()
    (reference_dir / "trace.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "mesh_export": {
                        "postprocess_stats": {"final_faces": 123_456, "final_vertices": 78_901},
                        "raw_coverage_ratio": 0.25,
                        "coverage_ratio": 1.0,
                        "unwrap_backend": "xatlas-parallel-spatial",
                        "bake_backend": "xatlas-kdtree",
                        "texture_size": 1024,
                        "xatlas_face_guard": 300_000,
                    }
                }
            }
        )
    )

    reference_settings = _resolve_pixal3d_export_settings(decoded_dir, "reference-target", None)
    assert reference_settings["quality_preset"] == "reference-target"
    assert reference_settings["target_faces"] == 123_456
    assert reference_settings["target_faces_source"] == "reference_final_faces"
    assert reference_settings["reference"]["xatlas_face_guard"] == 300_000
    assert _simplifier_backend_for_quality_preset(reference_settings["quality_preset"]) == "topology-aware"

    preview_settings = _resolve_pixal3d_export_settings(decoded_dir, "preview", None)
    assert preview_settings["target_faces"] == 50_000
    assert preview_settings["target_faces_source"] == "preview_default"
    assert _simplifier_backend_for_quality_preset(preview_settings["quality_preset"]) == "spatial-cluster"

    explicit_settings = _resolve_pixal3d_export_settings(decoded_dir, "production", 42_000)
    assert explicit_settings["quality_preset"] == "reference-target"
    assert explicit_settings["target_faces"] == 42_000
    assert explicit_settings["target_faces_source"] == "explicit"

    with pytest.raises(ValueError, match="quality_preset"):
        _resolve_pixal3d_export_settings(decoded_dir, "bad", None)


def test_export_quality_summary_separates_artifact_and_production_readiness() -> None:
    summary = _export_quality_summary(
        {"backend": "spatial-cluster", "quality_tier": "geometry_aware_preview"},
        {"export_blocking_reasons": []},
    )

    assert summary["artifact_ready"] is True
    assert summary["production_quality_ready"] is False
    assert summary["simplifier_backend"] == "spatial-cluster"
    assert summary["simplifier_quality_tier"] == "geometry_aware_preview"
    assert summary["native_geometry_candidate"]["status"] == "not_requested"
    assert "preview_simplifier_quality_tier" in summary["warnings"]

    blocked = _export_quality_summary(
        {"backend": "qem-edge-collapse", "quality_tier": "production"},
        {"export_blocking_reasons": ["nonmanifold_edges_present"]},
    )

    assert blocked["artifact_ready"] is False
    assert blocked["production_quality_ready"] is False
    assert blocked["export_blocking_reasons"] == ("nonmanifold_edges_present",)
    assert "export_blocking_reasons_present" in blocked["warnings"]

    blocked_candidate = _export_quality_summary(
        {
            "backend": "spatial-cluster",
            "quality_tier": "geometry_aware_preview",
            "final_faces": 198_618,
        },
        {"export_blocking_reasons": []},
        {"coverage_ratio": 0.75, "raw_coverage_ratio": 0.20},
        {"final_faces": 212_542, "coverage_ratio": 1.0, "raw_coverage_ratio": 0.40},
        quality_preset="reference-target",
    )
    assert blocked_candidate["artifact_ready"] is True
    assert blocked_candidate["production_quality_ready"] is False
    candidate = blocked_candidate["native_geometry_candidate"]
    assert candidate["status"] == "blocked"
    assert candidate["reason"] == "native_geometry_candidate_blocked"
    assert candidate["detail"] == "reference-target export still uses a preview-tier native simplifier"
    assert candidate["current_backend"] == "spatial-cluster"
    assert candidate["current_quality_tier"] == "geometry_aware_preview"
    assert candidate["face_count_ratio"] == pytest.approx(198_618 / 212_542)
    assert candidate["topology_exportability_passed"] is True

    requested_candidate = _export_quality_summary(
        {
            "requested_backend": "topology-aware",
            "backend": "spatial-cluster",
            "backend_selection_status": "fallback_preview_unimplemented",
            "quality_tier": "geometry_aware_preview",
            "final_faces": 198_618,
        },
        {"export_blocking_reasons": []},
        {"coverage_ratio": 0.75, "raw_coverage_ratio": 0.20},
        {"final_faces": 212_542, "coverage_ratio": 1.0, "raw_coverage_ratio": 0.40},
        quality_preset="reference-target",
    )
    requested_status = requested_candidate["native_geometry_candidate"]
    assert requested_candidate["production_quality_ready"] is False
    assert requested_status["status"] == "blocked"
    assert requested_status["requested_backend"] == "topology-aware"
    assert requested_status["backend_selection_status"] == "fallback_preview_unimplemented"

    production = _export_quality_summary(
        {
            "backend": "qem-edge-collapse",
            "quality_tier": "production",
            "final_faces": 212_542,
        },
        {"export_blocking_reasons": []},
        {"coverage_ratio": 0.75, "raw_coverage_ratio": 0.20},
        {"final_faces": 212_542, "coverage_ratio": 1.0, "raw_coverage_ratio": 0.40},
        quality_preset="reference-target",
    )
    assert production["artifact_ready"] is True
    assert production["production_quality_ready"] is True
    assert production["native_geometry_candidate"]["status"] == "candidate"
    thresholds = production["production_thresholds"]
    assert thresholds["all_passed"] is True
    assert thresholds["checks"]["face_count_ratio"]["actual"] == pytest.approx(1.0)
    assert thresholds["checks"]["final_coverage_ratio"]["actual"] == pytest.approx(0.75)


def test_upstream_export_settings_summary_separates_setting_readiness_from_reference_parity() -> None:
    passing = _upstream_export_settings_summary(
        1_000_000,
        4096,
        {"quality_tier": "production", "target_reached": True, "final_faces": 900_000},
        {"final_visible_coverage_ratio": 0.55},
        {"artifact_ready": True},
    )
    assert passing["all_passed"] is True
    assert passing["reference"]["remesh"] is True
    assert passing["reference"]["xatlas_chart_parity"] is False

    failing = _upstream_export_settings_summary(
        212_542,
        1024,
        {"quality_tier": "production", "target_reached": True, "final_faces": 212_542},
        {"final_visible_coverage_ratio": 0.60},
        {"artifact_ready": True},
    )
    assert failing["all_passed"] is False
    assert failing["checks"]["target_faces"]["passed"] is False
    assert failing["checks"]["texture_size"]["passed"] is False


def test_native_chart_uv_candidate_status_reports_readiness_states() -> None:
    not_requested = _native_chart_uv_candidate_status(
        {"backend": "face-atlas"},
        {"backend": "metal-face-atlas-nearest"},
        "face-atlas",
    )
    assert not_requested["status"] == "not_requested"
    assert not_requested["artifact_ready"] is None
    assert not_requested["quality_ready"] is None

    artifact_blocked = _native_chart_uv_candidate_status(
        {"backend": "native-chart-atlas", "chart_count": 4},
        {"backend": "metal-face-atlas-nearest", "sampled_texel_count": 10},
        "native-chart",
    )
    assert artifact_blocked["status"] == "artifact_blocked"
    assert artifact_blocked["artifact_ready"] is False
    assert artifact_blocked["quality_ready"] is False
    assert "texture_backend" in artifact_blocked["artifact_blockers"]

    quality_blocked = _native_chart_uv_candidate_status(
        {
            "backend": "native-chart-atlas",
            "chart_count": 10,
            "output_vertices": 40,
            "output_faces": 20,
            "duplicated_vertex_ratio": 1.2,
        },
        {
            "backend": "metal-uv-binned-nearest",
            "sampled_texel_count": 64,
            "uv_bin_guard_passed": True,
            "uv_bin_face_reference_count": 128,
            "uv_bin_max_candidate_faces": 8,
            "texture_pixel_count": 100,
            "uv_surface_texel_count": 20,
            "coverage_ratio": 0.14,
            "uv_surface_final_visible_coverage_ratio": 0.62,
        },
        "native-chart",
    )
    assert quality_blocked["status"] == "quality_blocked"
    assert quality_blocked["artifact_ready"] is True
    assert quality_blocked["quality_ready"] is False
    assert quality_blocked["uv_surface_occupancy_ratio"] == pytest.approx(0.20)
    assert quality_blocked["checks"]["global_coverage_floor"]["passed"] is False
    assert quality_blocked["checks"]["uv_surface_occupancy_floor"]["passed"] is False
    assert quality_blocked["checks"]["uv_surface_visible_floor"]["passed"] is True
    assert quality_blocked["quality_blockers"] == ("global_coverage_floor", "uv_surface_occupancy_floor")

    quality_ready = _native_chart_uv_candidate_status(
        {"backend": "native-chart-atlas", "chart_count": 10},
        {
            "backend": "metal-uv-binned-nearest",
            "sampled_texel_count": 64,
            "uv_bin_guard_passed": True,
            "uv_bin_face_reference_count": 128,
            "texture_pixel_count": 100,
            "uv_surface_texel_count": 70,
            "coverage_ratio": 0.60,
            "uv_surface_final_visible_coverage_ratio": 0.90,
        },
        "native-chart",
    )
    assert quality_ready["status"] == "quality_ready"
    assert quality_ready["artifact_ready"] is True
    assert quality_ready["quality_ready"] is True
    assert quality_ready["quality_blockers"] == ()


def test_xatlas_chart_parity_summary_reports_measured_native_chart_gap() -> None:
    summary = _xatlas_chart_parity_summary(
        {
            "unwrap_backend": "xatlas-parallel-spatial",
            "unwrap_chart_count": 100,
            "unwrap_utilization": 0.80,
            "texture_size": 1024,
        },
        {
            "backend": "native-chart-atlas",
            "chart_count": 50,
            "chart_rect_fill_ratio": 0.60,
        },
        {
            "texture_size": 1024,
            "texture_pixel_count": 1000,
            "uv_surface_texel_count": 500,
        },
        "native-chart",
    )

    assert summary["status"] == "measured_not_equivalent"
    assert summary["measurement_ready"] is True
    assert summary["parity_ready"] is False
    assert summary["xatlas_chart_parity"] is False
    assert summary["deferred_boundary"] == "not_xatlas_chart_parity"
    assert summary["reference"]["unwrap_chart_count"] == 100
    assert summary["reference"]["unwrap_utilization"] == pytest.approx(0.80)
    assert summary["native"]["chart_count"] == 50
    assert summary["native"]["uv_surface_occupancy_ratio"] == pytest.approx(0.50)
    assert summary["ratios"]["chart_count_ratio"] == pytest.approx(0.50)
    assert summary["ratios"]["uv_surface_occupancy_vs_reference_utilization"] == pytest.approx(0.625)
    assert summary["deficits"]["reference_utilization_minus_native_uv_surface_occupancy"] == pytest.approx(0.30)
    assert summary["deficits"]["uv_surface_occupancy_ratio_gap_to_reference"] == pytest.approx(0.375)
    assert summary["deficits"]["uv_surface_occupancy_ratio_gap_to_equivalence_target"] == pytest.approx(0.325)
    assert summary["deficits"]["equivalence_target_ratio"] == pytest.approx(0.95)
    assert summary["checks"]["xatlas_backend_equivalence"]["passed"] is False
    assert summary["checks"]["xatlas_utilization_equivalence"]["passed"] is False
    assert summary["checks"]["xatlas_utilization_equivalence"]["actual"] == pytest.approx(0.625)
    assert summary["checks"]["xatlas_utilization_equivalence"]["required"] == ">=0.95"

    not_requested = _xatlas_chart_parity_summary(
        {
            "unwrap_backend": "xatlas-parallel-spatial",
            "unwrap_chart_count": 100,
            "unwrap_utilization": 0.80,
        },
        {"backend": "face-atlas"},
        {},
        "face-atlas",
    )
    assert not_requested["status"] == "not_requested"
    assert not_requested["parity_ready"] is None
    assert not_requested["deficits"] == {}

    missing_reference = _xatlas_chart_parity_summary(
        None,
        {"backend": "native-chart-atlas", "chart_count": 50},
        {"texture_pixel_count": 1000, "uv_surface_texel_count": 500},
        "native-chart",
    )
    assert missing_reference["status"] == "reference_missing"
    assert missing_reference["parity_ready"] is False
    assert missing_reference["deficits"] == {}
    assert missing_reference["checks"]["reference_xatlas_available"]["passed"] is False


def test_glb_viewer_compatibility_summary_checks_normals_and_uint16_chunks() -> None:
    passing = _glb_viewer_compatibility_summary(
        {
            "material_count": 1,
            "texture_count": 2,
            "image_count": 2,
            "primitive_count": 2,
            "total_vertices": 66_000,
            "primitives": [
                {
                    "primitive_index": 0,
                    "vertex_count": 65_535,
                    "normal_count": 65_535,
                    "index_count": 65_535,
                    "indices_component_type": 5123,
                    "indices_min": [0],
                    "indices_max": [65_534],
                    "has_normals": True,
                },
                {
                    "primitive_index": 1,
                    "vertex_count": 465,
                    "normal_count": 465,
                    "index_count": 465,
                    "indices_component_type": 5123,
                    "indices_min": [0],
                    "indices_max": [464],
                    "has_normals": True,
                },
            ],
        }
    )
    assert passing["all_passed"] is True
    assert passing["checks"]["normals"]["passed"] is True
    assert passing["checks"]["uint16_indices"]["passed"] is True
    assert passing["checks"]["chunking_for_large_mesh"]["passed"] is True

    failing = _glb_viewer_compatibility_summary(
        {
            "material_count": 1,
            "texture_count": 2,
            "image_count": 2,
            "primitive_count": 1,
            "total_vertices": 66_000,
            "primitives": [
                {
                    "primitive_index": 0,
                    "vertex_count": 66_000,
                    "normal_count": 0,
                    "index_count": 66_000,
                    "indices_component_type": 5125,
                    "indices_min": [0],
                    "indices_max": [65_999],
                    "has_normals": False,
                }
            ],
        }
    )
    assert failing["all_passed"] is False
    assert failing["checks"]["normals"]["passed"] is False
    assert failing["checks"]["uint16_indices"]["passed"] is False
    assert failing["checks"]["local_index_bounds"]["passed"] is False
    assert failing["checks"]["chunking_for_large_mesh"]["passed"] is False


def _assert_memory_diagnostics(diagnostics: dict, *, required_stages: tuple[str, ...]) -> None:
    memory = diagnostics["memory"]
    assert memory["source"].startswith("process RSS from ps")
    assert memory["poll_interval_sec"] > 0.0
    assert memory["sample_count"] >= len(diagnostics["memory_samples"])
    assert memory["peak_current_rss_bytes"] is not None
    assert memory["peak_current_rss_bytes"] > 0
    assert memory["peak_max_rss_bytes"] is not None
    assert memory["peak_max_rss_bytes"] >= memory["peak_current_rss_bytes"]
    assert memory["last_sample"]["source"].startswith("ps rss")
    assert diagnostics["memory_samples"]["after_write_glb"]["source"].startswith("ps rss")
    stage_peaks = memory["stage_peaks"]
    for stage_name in required_stages:
        stage = stage_peaks[stage_name]
        assert stage["sample_count"] >= 2
        assert stage["start_current_rss_bytes"] is not None
        assert stage["end_current_rss_bytes"] is not None
        assert stage["peak_current_rss_bytes"] >= max(
            stage["start_current_rss_bytes"],
            stage["end_current_rss_bytes"],
        )
