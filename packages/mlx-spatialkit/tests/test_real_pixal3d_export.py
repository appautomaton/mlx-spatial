from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from mlx_spatialkit import export_pixal3d_glb, metal_device_available
from mlx_spatialkit.mesh import (
    clean_mesh,
    extract_flexi_dual_grid,
    mesh_metrics,
    remesh_narrow_band,
    simplify_mesh,
)
from mlx_spatialkit.export import (
    NativeGlbArtifact,
    _build_pixal3d_run_manifest,
    _resolve_chart_angle_degrees,
    _export_quality_summary,
    _glb_viewer_compatibility_summary,
    _load_pixal3d_fixture_manifest,
    _native_chart_uv_candidate_status,
    _production_equivalence_summary,
    _resolve_pixal3d_export_settings,
    _resolve_simplify_backend,
    _resolve_tile_padding,
    _resolve_pixal3d_uv_backend,
    _simplifier_backend_for_quality_preset,
    _topology_blocker_map,
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


def test_pixal3d_fixture_manifests_bind_decoded_and_reference_lineages() -> None:
    root = _repo_root() / "inputs" / "mlx-spatialkit"
    fixtures = (
        (
            root / "pixal3d-1024-cascade-decoded-pbr",
            "pixal3d-1024-cascade:vendors-pixal3d-0-img:1024-cascade",
            "vendor-demo-source",
        ),
        (
            root / "violin-bow" / "pixal3d-1024-cascade-decoded-pbr",
            "pixal3d-1024-cascade:violin-bow:raw",
            "raw",
        ),
        (
            root / "violin-bow-preprocessed-black" / "pixal3d-1024-cascade-decoded-pbr",
            "pixal3d-1024-cascade:violin-bow:upstream-preprocessed-black",
            "upstream-preprocessed-black",
        ),
    )
    for decoded_dir, lineage_id, preprocess_variant in fixtures:
        manifest = _load_pixal3d_fixture_manifest(decoded_dir)
        assert manifest is not None
        assert manifest["lineage_id"] == lineage_id
        assert manifest["roles"]["A"]["lineage_id"] == lineage_id
        assert manifest["roles"]["C"]["lineage_id"] == lineage_id
        assert manifest["source_image"]["preprocess_variant"] == preprocess_variant


def test_pixal3d_fixture_manifest_fails_closed_on_mismatched_or_ambiguous_lineage(tmp_path: Path) -> None:
    decoded_dir = tmp_path / "decoded"
    reference_dir = tmp_path / "reference"
    decoded_dir.mkdir()
    reference_dir.mkdir()
    (decoded_dir / "trace.json").write_text("{}", encoding="utf-8")
    (reference_dir / "trace.json").write_text("{}", encoding="utf-8")
    (reference_dir / "model.glb").write_bytes(b"glTF")
    manifest_dir = tmp_path / "case"
    manifest_dir.mkdir()
    manifest = {
        "manifest_version": 1,
        "case_id": "case",
        "lineage_id": "lineage-a",
        "source_image": {"path": "image.png", "preprocess_variant": "test"},
        "roles": {
            "A": {
                "role": "A",
                "kind": "decoded_model_output",
                "lineage_id": "lineage-a",
                "decoded_dir": "../decoded",
                "trace_path": "../decoded/trace.json",
            },
            "C": {
                "role": "C",
                "kind": "reference_control_glb",
                "lineage_id": "lineage-b",
                "trace_path": "../reference/trace.json",
                "model_glb_path": "../reference/model.glb",
            },
        },
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="lineage mismatch"):
        _load_pixal3d_fixture_manifest(decoded_dir)

    manifest["roles"]["C"]["lineage_id"] = "lineage-a"
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    second_dir = tmp_path / "case-copy"
    second_dir.mkdir()
    (second_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous Pixal3D fixture manifests"):
        _load_pixal3d_fixture_manifest(decoded_dir)


def test_pixal3d_run_manifest_records_abc_roles_and_browser_paths(tmp_path: Path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()
    shape_path = decoded_dir / "shape_decoder_fields.npz"
    texture_path = decoded_dir / "texture_decoder_pbr.npz"
    trace_path = decoded_dir / "trace.json"
    shape_path.write_bytes(b"shape")
    texture_path.write_bytes(b"texture")
    trace_path.write_text("{}", encoding="utf-8")
    glb_path = tmp_path / "model.glb"
    glb_path.write_bytes(b"glTF")
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics = {
        "settings": {"texture_size": 1024, "uv_backend": "native-chart"},
        "result": {"artifact_ready": True, "rendered_visual_ready": False},
        "visual_comparison": {
            "artifacts": {
                "report_json": str(tmp_path / "visual_parity" / "visual_parity.json"),
                "browser_render_report_json": str(tmp_path / "browser_render" / "browser_render_report.json"),
            }
        },
    }
    fixture_manifest = {
        "manifest_path": str(tmp_path / "manifest.json"),
        "manifest_version": 1,
        "case_id": "case",
        "lineage_id": "lineage-a",
        "source_image": {"path": "image.png", "preprocess_variant": "test"},
        "roles": {
            "A": {"lineage_id": "lineage-a"},
            "C": {"lineage_id": "lineage-a"},
        },
    }
    glb = NativeGlbArtifact(path=glb_path, format="glb", bytes_written=4, metadata={})

    manifest = _build_pixal3d_run_manifest(
        decoded_dir=decoded_dir,
        shape_path=shape_path,
        texture_path=texture_path,
        glb=glb,
        diagnostics_path=diagnostics_path,
        diagnostics=diagnostics,
        fixture_manifest=fixture_manifest,
        reference={
            "lineage_id": "lineage-a",
            "model_glb_path": str(tmp_path / "reference.glb"),
            "trace_path": str(tmp_path / "reference-trace.json"),
        },
    )

    assert manifest["lineage_id"] == "lineage-a"
    assert manifest["roles"]["A"]["trace_path"] == str(trace_path)
    assert manifest["roles"]["B"]["model_glb_path"] == str(glb_path)
    assert manifest["roles"]["B"]["diagnostics_path"] == str(diagnostics_path)
    assert manifest["roles"]["B"]["visual_parity_report_path"].endswith("visual_parity.json")
    assert manifest["roles"]["B"]["browser_render_report_path"].endswith("browser_render_report.json")
    assert manifest["roles"]["C"]["lineage_id"] == "lineage-a"


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
    assert diagnostics["result"]["rendered_visual_ready"] is False
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
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] > texture_stats["final_visible_coverage_ratio"]
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert texture_stats["uv_surface_exact_coverage_ratio"] >= 0.99
    assert texture_stats["surface_filled_texel_count"] == 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["coverage_status_histogram"]["missing_surface"] == 0
    assert texture_stats["coverage_status_histogram"]["out_of_grid"] == 0
    assert texture_stats["surface_fill_cross_gap_prevented_count"] > 0
    base_color_coverage = png_coverage(glb_image_payload(result.glb.path.read_bytes(), "baseColorTexture"))
    assert base_color_coverage.alpha_coverage_ratio > 0.10
    assert base_color_coverage.rgb_coverage_ratio > 0.10
    assert base_color_coverage.alpha_coverage_ratio == pytest.approx(texture_stats["render_alpha_coverage_ratio"])
    assert base_color_coverage.rgb_coverage_ratio == pytest.approx(
        texture_stats["render_visible_base_color_texel_count"] / texture_stats["texture_pixel_count"],
        abs=0.005,
    )
    assert texture_stats["visible_base_color_texel_count"] / texture_stats["texture_pixel_count"] == pytest.approx(
        texture_stats["final_visible_coverage_ratio"]
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
    topology = diagnostics["quality"]["topology_blocker_map"]
    assert topology["status"] == "rendered_visual_blocked"
    assert "clean_closed_boundary_loops" in topology["visual_blockers"]
    assert "branched_open_boundary_chains" in topology["visual_blockers"]
    assert "after_write_glb" in diagnostics["memory_samples"]
    _assert_memory_diagnostics(diagnostics, required_stages=("texture_bake", "write_glb"))


@pytest.mark.heavy
def test_export_pixal3d_glb_remesh_closes_topology_and_clears_blocker() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-remesh-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        texture_size=1024,
        target_faces=50_000,
        min_component_faces=32,
        remesh=True,
    )

    assert result.glb.path.read_bytes()[:4] == b"glTF"
    diagnostics = json.loads(result.diagnostics_path.read_text())

    # opt-in settings recorded
    assert diagnostics["settings"]["remesh"] is True
    assert diagnostics["settings"]["remesh_resolution"] == 1024

    # the remesh stage rebuilds watertight topology (all open boundaries closed)
    remesh_stage = diagnostics["stages"]["remesh"]
    assert remesh_stage["stats"]["backend"] == "cpu-narrow-band-dc"
    assert remesh_stage["stats"]["active_voxels"] > 0
    remesh_metrics = remesh_stage["metrics"]
    assert remesh_metrics["boundary_loop_count"] == 0
    assert remesh_metrics["boundary_open_chain_count"] == 0
    assert remesh_metrics["boundary_branched_open_chain_count"] == 0

    # the narrow-band remesh production blocker clears; QEM remains the next gap.
    # (the final post-simplify mesh is not yet fully watertight because the
    # clustering simplifier re-tears it -- that is the QEM follow-on change.)
    topology = diagnostics["quality"]["topology_blocker_map"]
    narrow_band = topology["classes"]["missing_narrow_band_remesh"]
    assert narrow_band["present"] is False
    assert narrow_band["backend"] == "native-narrow-band-dc"
    assert "missing_narrow_band_dc_remesh" not in topology["production_backend_blockers"]
    assert "missing_qem_edge_collapse_simplification" in topology["production_backend_blockers"]


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
    assert diagnostics["settings"]["small_boundary_loop_fill_max_edges"] == 8
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
    assert uv_stats["low_fill_split_min_faces"] == 4
    assert uv_stats["low_fill_split_min_child_faces"] == 2
    assert uv_stats["low_fill_split_max_depth"] == 3
    assert uv_stats["low_fill_split_axis_candidates"] == 2
    assert uv_stats["low_fill_split_position_candidates"] == 5
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
    assert texture_stats["surface_filled_texel_count"] == 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["surface_fill_cross_gap_prevented_count"] > 0
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert texture_stats["uv_surface_exact_coverage_ratio"] >= 0.99
    assert texture_stats["raw_coverage_ratio"] < texture_stats["final_visible_coverage_ratio"]
    assert texture_stats["coverage_status_histogram"]["missing_surface"] == 0
    assert texture_stats["coverage_status_histogram"]["out_of_grid"] == 0
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
    assert candidate["surface_unfilled_texel_count"] == texture_stats["surface_unfilled_texel_count"]
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
def test_export_pixal3d_glb_native_chart_violin_preprocessed_black_fixture() -> None:
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = (
        _repo_root()
        / "inputs"
        / "mlx-spatialkit"
        / "violin-bow-preprocessed-black"
        / "pixal3d-1024-cascade-decoded-pbr"
    )
    if not fixture.exists():
        pytest.skip(f"real Pixal3D violin decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-violin-black-native-chart-export-{os.getpid()}"
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
    assert diagnostics["fixture_manifest"]["lineage_id"] == (
        "pixal3d-1024-cascade:violin-bow:upstream-preprocessed-black"
    )
    assert diagnostics["fixture_manifest"]["source_image"]["preprocess_variant"] == "upstream-preprocessed-black"
    artifact_manifest_path = Path(diagnostics["artifact_manifest"]["path"])
    assert artifact_manifest_path.is_file()
    artifact_manifest = json.loads(artifact_manifest_path.read_text())
    assert artifact_manifest["lineage_id"] == diagnostics["fixture_manifest"]["lineage_id"]
    assert tuple(artifact_manifest["roles"]) == ("A", "B", "C")
    assert artifact_manifest["roles"]["A"]["decoded_dir"] == str(fixture)
    assert artifact_manifest["roles"]["B"]["model_glb_path"] == str(result.glb.path)
    assert Path(artifact_manifest["roles"]["C"]["model_glb_path"]).is_file()

    assert diagnostics["settings"]["uv_backend"] == "native-chart"
    assert diagnostics["settings"]["chart_angle_degrees"] == 45.0
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is False
    assert diagnostics["result"]["production_equivalence_ready"] is False
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True

    uv_stats = diagnostics["stages"]["uv"]["stats"]
    assert uv_stats["backend"] == "native-chart-atlas"
    assert uv_stats["chart_count"] > 0
    assert uv_stats["chart_rect_fill_ratio"] > 0.50
    assert uv_stats["atlas_rect_coverage_ratio"] > 0.90
    assert uv_stats["output_faces"] <= 50_000

    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    assert texture_stats["backend"] == "metal-uv-binned-nearest"
    assert texture_stats["uv_bin_guard_passed"] is True
    assert texture_stats["sampled_texel_count"] > 0
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert texture_stats["uv_surface_exact_coverage_ratio"] >= 0.99
    assert texture_stats["surface_filled_texel_count"] == 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["coverage_status_histogram"]["missing_surface"] == 0
    assert texture_stats["coverage_status_histogram"]["out_of_grid"] == 0

    candidate = diagnostics["quality"]["native_chart_uv_candidate"]
    assert candidate["status"] == "quality_ready"
    assert candidate["artifact_ready"] is True
    assert candidate["quality_ready"] is True
    assert candidate["quality_blockers"] == []
    assert candidate["global_coverage_ratio"] == pytest.approx(texture_stats["final_visible_coverage_ratio"])
    assert candidate["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert candidate["xatlas_chart_parity"] is False

    topology = diagnostics["quality"]["topology_blocker_map"]
    assert topology["status"] == "rendered_visual_blocked"
    assert "clean_closed_boundary_loops" in topology["visual_blockers"]
    assert "branched_open_boundary_chains" in topology["visual_blockers"]
    assert "missing_qem_edge_collapse_simplification" in topology["production_backend_blockers"]
    assert "missing_narrow_band_dc_remesh" in topology["production_backend_blockers"]

    visual = diagnostics["visual_comparison"]
    assert visual["summary"]["all_passed"] is True
    assert visual["summary"]["texture_resolution_match"] is True
    assert visual["rendered_visual_ready"] is False
    assert visual["rendered_visual_blockers"] == ["boundary_open_chains"]
    assert "not_xatlas_chart_parity" in visual["deferred_parity_boundaries"]
    visual_artifacts = visual["artifacts"]
    assert Path(visual_artifacts["report_json"]).is_file()
    assert Path(visual_artifacts["preview_html"]).is_file()
    assert Path(visual_artifacts["candidate_base_color_png"]).is_file()
    assert Path(visual_artifacts["reference_base_color_png"]).is_file()
    _assert_memory_diagnostics(diagnostics, required_stages=("uv", "texture_bake", "write_glb", "visual_compare"))


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
    assert diagnostics["result"]["production_quality_ready"] is False
    assert diagnostics["result"]["production_equivalence_ready"] is False
    assert diagnostics["result"]["remaining_parity_boundaries"] == [
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
        "not_reference_stage_contract",
    ]
    assert diagnostics["result"]["equivalence_blockers"] == [
        "scalar_production_quality_not_ready",
        "reference_stage_contract_not_ready",
        "upstream_export_settings_not_ready",
        "xatlas_chart_parity_not_ready",
        "rendered_visual_not_ready",
        "deferred_parity_boundaries_present",
    ]
    assert diagnostics["result"]["quality_warnings"] == [
        "preview_simplifier_quality_tier",
        "production_thresholds_failed",
        "reference_stage_contract_incomplete",
    ]
    assert diagnostics["quality"]["upstream_export_settings"]["all_passed"] is False
    assert diagnostics["quality"]["upstream_export_settings"]["checks"]["target_faces"]["passed"] is False
    equivalence = diagnostics["quality"]["production_equivalence"]
    assert equivalence["ready"] is False
    assert equivalence["scalar_production_quality_ready"] is False
    assert equivalence["upstream_export_settings_ready"] is False
    assert equivalence["xatlas_chart_parity_ready"] is False
    assert equivalence["visual_comparison_ready"] is False
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    assert diagnostics["quality"]["native_geometry_candidate"]["status"] == "blocked"
    assert diagnostics["quality"]["native_geometry_candidate"]["reason"] == "native_geometry_candidate_blocked"
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    assert simplify_stats["requested_backend"] == "topology-aware"
    assert simplify_stats["backend"] == "topology-aware"
    assert simplify_stats["algorithm"] == "native_topology_aware_quadric_representative_clustering"
    assert simplify_stats["quality_tier"] == "production_candidate_blocked"
    assert simplify_stats["production_ready"] is False
    assert simplify_stats["production_blockers"] == [
        "missing_qem_edge_collapse_simplification",
        "missing_narrow_band_dc_remesh",
    ]
    assert simplify_stats["qem_equivalence_status"] == "qem_scored_not_edge_collapse"
    assert simplify_stats["remesh_equivalence_status"] == "blocked_missing_narrow_band_dc"
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
    assert thresholds["backend_tier"]["passed"] is False
    assert thresholds["backend_tier"]["actual"] == "production_candidate_blocked"
    assert thresholds["final_coverage_ratio"]["actual"] == pytest.approx(
        texture_stats["final_visible_coverage_ratio"]
    )
    assert thresholds["final_coverage_ratio"]["passed"] is True
    assert thresholds["final_coverage_ratio"]["actual"] >= thresholds["final_coverage_ratio"]["required_min"]
    assert thresholds["raw_coverage_ratio"]["passed"] is True
    assert diagnostics["reference_comparison"]["reference_bake_backend"] == "xatlas-kdtree"
    visual = diagnostics["visual_comparison"]
    assert visual["summary"]["all_passed"] is True
    assert visual["rendered_visual_ready"] is False
    assert visual["rendered_visual_blockers"] == ["boundary_open_chains"]
    assert 0.80 <= visual["summary"]["face_count_ratio"] <= 1.25
    assert visual["summary"]["texture_resolution_match"] is True
    assert visual["summary"]["base_color_alpha_coverage_ratio"] >= 0.95
    assert visual["summary"]["base_color_rgb_coverage_ratio"] >= 0.95
    assert visual["checks"]["texture_resolution_match"]["passed"] is True
    assert visual["rendered_visual_checks"]["texture_reference_scalar_gate"]["passed"] is True
    assert visual["rendered_visual_checks"]["surface_unfilled_texels"]["passed"] is True
    assert visual["rendered_visual_checks"]["boundary_open_chains"]["passed"] is False
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
    assert diagnostics["stages"]["write_glb"]["artifact"]["production_equivalence_ready"] is False
    assert diagnostics["settings"]["quality_preset"] == "reference-target"
    assert diagnostics["settings"]["uv_backend"] == "native-chart"
    assert diagnostics["settings"]["target_faces"] == 212_542
    assert diagnostics["settings"]["tile_padding"] == 0.001
    assert diagnostics["settings"]["tile_padding_source"] == "backend_default:native-chart"
    assert diagnostics["settings"]["small_boundary_loop_fill_max_edges"] == 8
    assert diagnostics["result"]["artifact_ready"] is True
    assert diagnostics["result"]["production_quality_ready"] is False
    assert diagnostics["result"]["production_equivalence_ready"] is False
    assert diagnostics["result"]["remaining_parity_boundaries"] == [
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
        "not_reference_stage_contract",
    ]
    assert diagnostics["result"]["equivalence_blockers"] == [
        "scalar_production_quality_not_ready",
        "reference_stage_contract_not_ready",
        "upstream_export_settings_not_ready",
        "xatlas_chart_parity_not_ready",
        "rendered_visual_not_ready",
        "deferred_parity_boundaries_present",
    ]
    assert diagnostics["result"]["quality_warnings"] == [
        "preview_simplifier_quality_tier",
        "production_thresholds_failed",
        "reference_stage_contract_incomplete",
    ]
    assert diagnostics["quality"]["upstream_export_settings"]["all_passed"] is False
    assert diagnostics["quality"]["upstream_export_settings"]["checks"]["target_faces"]["passed"] is False
    equivalence = diagnostics["quality"]["production_equivalence"]
    assert equivalence["ready"] is False
    assert equivalence["scalar_production_quality_ready"] is False
    assert equivalence["upstream_export_settings_ready"] is False
    assert equivalence["xatlas_chart_parity_ready"] is False
    assert equivalence["visual_comparison_ready"] is False

    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    assert simplify_stats["small_boundary_loop_fill_enabled"] is True
    assert simplify_stats["small_boundary_loop_fill_max_edges"] == 8
    assert simplify_stats["small_boundary_loop_fill_algorithm"] == "cumesh-perimeter-centroid-fan"
    assert simplify_stats["small_boundary_loop_fill_max_perimeter"] == pytest.approx(0.03)
    assert simplify_stats["small_boundary_loop_fill_fallback_algorithm"] == "disabled"
    assert simplify_stats["small_boundary_loop_fill_fallback_enabled"] is False
    assert simplify_stats["small_boundary_loop_fill_fallback_max_edges"] == 0
    assert simplify_stats["small_boundary_loop_fill_fallback_policy_max_edges"] == 0
    assert simplify_stats["small_boundary_loop_fill_fallback_effective_max_edges"] == 0
    assert simplify_stats["small_boundary_loop_repair_max_passes"] == 3
    assert 1 <= simplify_stats["small_boundary_loop_repair_pass_count"] <= 3
    assert simplify_stats["small_boundary_branched_cycle_fill_enabled"] is True
    assert simplify_stats["small_boundary_branched_cycle_fill_max_edges"] == 8
    assert simplify_stats["small_boundary_branched_cycle_fill_policy_max_edges"] == 8
    assert simplify_stats["small_boundary_branched_cycle_fill_effective_max_edges"] == 8
    assert simplify_stats["small_boundary_loops_considered"] > 0
    assert simplify_stats["small_boundary_loops_filled_by_ear_clipping"] == 0
    assert simplify_stats["small_boundary_loops_alternative_triangulation_attempted"] == 0
    assert simplify_stats["small_boundary_loops_filled_by_alternative_triangulation"] == 0
    assert simplify_stats["small_boundary_loops_filled_by_centroid_fan"] == simplify_stats["small_boundary_loops_filled"]
    assert simplify_stats["small_boundary_loops_rejected"] == (
        simplify_stats["small_boundary_loops_rejected_ordering"]
        + simplify_stats["small_boundary_loops_rejected_triangulation"]
        + simplify_stats["small_boundary_loops_rejected_perimeter"]
        + simplify_stats["small_boundary_loops_rejected_edge_cap"]
        + simplify_stats["small_boundary_loops_rejected_fallback_cap"]
        + simplify_stats["small_boundary_loops_rejected_degenerate"]
        + simplify_stats["small_boundary_loops_rejected_duplicate"]
        + simplify_stats["small_boundary_loops_rejected_nonmanifold"]
    )
    assert simplify_stats["small_boundary_loop_faces_added"] >= 0
    assert simplify_stats["small_boundary_branched_cycle_candidates"] >= 0
    assert simplify_stats["small_boundary_branched_cycles_filled"] >= 0
    assert simplify_stats["small_boundary_branched_cycles_rejected"] >= 0
    assert simplify_stats["small_boundary_loops_budget_limited"] == 0
    assert simplify_stats["small_boundary_branched_cycles_budget_limited"] == 0
    assert simplify_stats["final_faces"] <= simplify_stats["target_faces"]

    uv_stats = diagnostics["stages"]["uv"]["stats"]
    assert uv_stats["backend"] == "native-chart-atlas"
    assert uv_stats["chart_count"] > 0
    assert uv_stats["output_faces"] == diagnostics["stages"]["simplify_mesh"]["simplified_faces"]
    assert uv_stats["chart_cluster_normal_policy"] == "edge-and-seed-cone"
    assert uv_stats["chart_cone_rejected_adjacency_count"] > 0
    assert uv_stats["chart_rect_fill_ratio"] > 0.55
    assert uv_stats["low_fill_split_min_faces"] == 4
    assert uv_stats["low_fill_split_min_child_faces"] == 2
    assert uv_stats["low_fill_split_position_candidates"] == 5
    assert uv_stats["low_fill_split_partition_candidate_count"] == (
        uv_stats["low_fill_split_axis_candidate_count"] * uv_stats["low_fill_split_position_candidates"]
    )
    assert uv_stats["low_fill_split_partition_evaluated_count"] > uv_stats["low_fill_split_axis_candidate_count"]
    assert uv_stats["low_fill_split_accepted_count"] > 0

    texture_stats = diagnostics["stages"]["texture_bake"]["stats"]
    assert texture_stats["backend"] == "metal-uv-binned-nearest"
    assert texture_stats["uv_bin_guard_passed"] is True
    assert texture_stats["texture_coordinate_order"] == "batch-x-y-z"
    assert texture_stats["surface_fill_enabled"] is True
    assert texture_stats["surface_fill_traversal_policy"] == "uv-surface-only-no-face-gap-blocked"
    assert texture_stats["surface_fill_cross_gap_prevented_count"] > 0
    assert texture_stats["surface_filled_texel_count"] == 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["gutter_fill_enabled"] is True
    assert texture_stats["gutter_fill_max_passes"] == 4
    assert 0 < texture_stats["gutter_fill_pass_count"] <= texture_stats["gutter_fill_max_passes"]
    assert texture_stats["gutter_filled_texel_count"] > 0
    assert texture_stats["trilinear_invalid_texel_count"] == 0
    assert texture_stats["source_projection_nearest_fallback_texel_count"] <= 8
    assert texture_stats["raw_coverage_ratio"] == pytest.approx(texture_stats["final_visible_coverage_ratio"], abs=1e-5)
    assert texture_stats["uv_surface_exact_coverage_ratio"] < texture_stats["uv_surface_final_visible_coverage_ratio"]
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_exact_coverage_ratio"] >= 0.99
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] >= 0.99

    export_metrics = diagnostics["stages"]["export_metrics"]["metrics"]
    assert export_metrics["face_count"] == diagnostics["stages"]["simplify_mesh"]["simplified_faces"]
    assert export_metrics["nonmanifold_edges"] == 0
    assert export_metrics["boundary_edges"] > 0
    assert export_metrics["boundary_vertices"] > 0
    assert export_metrics["boundary_loop_count"] > 0
    assert export_metrics["boundary_small_loop_threshold_edges"] == 32
    assert export_metrics["boundary_small_loop_count"] <= export_metrics["boundary_loop_count"]
    assert export_metrics["boundary_small_loop_edge_count"] <= export_metrics["boundary_edges"]
    assert export_metrics["boundary_open_chain_count"] > 0
    assert export_metrics["boundary_open_chain_edge_count"] > 0
    assert export_metrics["boundary_small_open_chain_count"] <= export_metrics["boundary_open_chain_count"]
    assert export_metrics["boundary_small_open_chain_edge_count"] <= export_metrics["boundary_open_chain_edge_count"]
    assert export_metrics["boundary_simple_open_chain_count"] == 0
    assert export_metrics["boundary_branched_open_chain_count"] == export_metrics["boundary_open_chain_count"]
    assert export_metrics["boundary_open_chain_endpoint_count"] == 0
    assert export_metrics["boundary_open_chain_branch_vertex_count"] > 0
    assert export_metrics["boundary_max_open_chain_edges"] > 0
    assert export_metrics["boundary_max_component_edges"] >= export_metrics["boundary_max_loop_edges"]
    assert export_metrics["boundary_max_component_edges"] >= export_metrics["boundary_max_open_chain_edges"]
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
    assert candidate["surface_unfilled_texel_count"] == texture_stats["surface_unfilled_texel_count"]
    assert candidate["native_behavior_diagnostics"]["requires_xatlas_dependency"] is False
    assert candidate["native_behavior_diagnostics"]["cluster_normal_policy"] == "edge-and-seed-cone"
    assert candidate["checks"]["global_coverage_floor"]["passed"] is True
    assert candidate["checks"]["uv_surface_occupancy_floor"]["passed"] is True
    assert candidate["checks"]["uv_surface_visible_floor"]["passed"] is True
    assert candidate["quality_blockers"] == []
    assert candidate["xatlas_chart_parity"] is False
    _assert_xatlas_parity_measured(diagnostics, uv_stats, texture_stats)
    assert diagnostics["quality"]["xatlas_chart_parity"]["ratios"][
        "uv_surface_occupancy_vs_reference_utilization"
    ] > 0.65

    visual = diagnostics["visual_comparison"]
    assert diagnostics["result"]["rendered_visual_ready"] is False
    assert visual["rendered_visual_ready"] is False
    assert visual["summary"]["all_passed"] is True
    assert visual["rendered_visual_checks"]["texture_reference_scalar_gate"]["passed"] is True
    assert visual["rendered_visual_checks"]["surface_unfilled_texels"]["passed"] is True
    assert visual["rendered_visual_checks"]["boundary_open_chains"]["passed"] is False
    assert visual["rendered_visual_blockers"] == ["boundary_open_chains"]
    assert 0.80 <= visual["summary"]["face_count_ratio"] <= 1.25
    assert visual["summary"]["texture_resolution_match"] is True
    assert visual["summary"]["base_color_alpha_coverage_ratio"] >= 0.50
    assert visual["summary"]["roughness_mean_ratio"] >= 0.75
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
def test_export_pixal3d_glb_reference_target_4096_texture_reports_texture_resolution_gate() -> None:
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
    assert diagnostics["result"]["production_quality_ready"] is False
    assert diagnostics["result"]["quality_warnings"] == [
        "preview_simplifier_quality_tier",
        "production_thresholds_failed",
        "reference_stage_contract_incomplete",
    ]
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    assert texture_stats["texture_size"] == 4096
    assert texture_stats["dilation_max_passes"] > 8
    assert texture_stats["dilation_pass_count"] <= texture_stats["dilation_max_passes"]
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert thresholds["final_coverage_ratio"]["passed"] is True
    assert thresholds["final_coverage_ratio"]["actual"] == pytest.approx(
        texture_stats["final_visible_coverage_ratio"]
    )
    assert thresholds["backend_tier"]["passed"] is False
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
    assert diagnostics["result"]["production_equivalence_ready"] is False
    assert diagnostics["result"]["remaining_parity_boundaries"] == [
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
        "not_reference_stage_contract",
    ]
    assert "scalar_production_quality_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert "reference_stage_contract_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert "upstream_export_settings_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert "xatlas_chart_parity_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert upstream["all_passed"] is False
    assert diagnostics["quality"]["glb_viewer_compatibility"]["all_passed"] is True
    assert upstream["checks"]["backend_tier"]["passed"] is False
    assert upstream["checks"]["final_coverage_ratio"]["passed"] is True
    for name, check in upstream["checks"].items():
        if name == "backend_tier":
            continue
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
    assert texture_stats["surface_fill_traversal_policy"] == "uv-surface-only-no-face-gap-blocked"
    assert texture_stats["surface_fill_cross_gap_prevented_count"] > 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    visual = diagnostics["visual_comparison"]
    assert "not_xatlas_chart_parity" in visual["deferred_parity_boundaries"]
    assert "not_1m_face_export_setting_parity" in visual["deferred_parity_boundaries"]
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
    assert diagnostics["result"]["production_equivalence_ready"] is False
    assert diagnostics["result"]["remaining_parity_boundaries"] == [
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
        "not_reference_stage_contract",
    ]
    assert "scalar_production_quality_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert "reference_stage_contract_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert "upstream_export_settings_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert "xatlas_chart_parity_not_ready" in diagnostics["result"]["equivalence_blockers"]
    assert diagnostics["result"]["quality_warnings"] == [
        "preview_simplifier_quality_tier",
        "production_thresholds_failed",
        "reference_stage_contract_incomplete",
    ]

    assert upstream["all_passed"] is False
    assert upstream["checks"]["backend_tier"]["passed"] is False
    assert upstream["checks"]["final_coverage_ratio"]["passed"] is True
    for name, check in upstream["checks"].items():
        if name == "backend_tier":
            continue
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
    assert uv_stats["low_fill_split_position_candidates"] == 5
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
    assert texture_stats["surface_filled_texel_count"] == 0
    assert texture_stats["surface_unfilled_texel_count"] == 0
    assert texture_stats["surface_fill_cross_gap_prevented_count"] > 0
    assert texture_stats["final_visible_coverage_ratio"] >= 0.50
    assert texture_stats["uv_surface_final_visible_coverage_ratio"] == pytest.approx(1.0)

    assert candidate["status"] == "quality_ready"
    assert candidate["artifact_ready"] is True
    assert candidate["quality_ready"] is True
    assert candidate["global_coverage_ratio"] == pytest.approx(texture_stats["final_visible_coverage_ratio"])
    assert candidate["uv_surface_occupancy_ratio"] >= 0.50
    assert candidate["uv_surface_final_visible_coverage_ratio"] == pytest.approx(
        texture_stats["uv_surface_final_visible_coverage_ratio"]
    )
    assert candidate["surface_unfilled_texel_count"] == texture_stats["surface_unfilled_texel_count"]
    assert candidate["quality_blockers"] == []
    assert candidate["xatlas_chart_parity"] is False
    _assert_xatlas_parity_measured(diagnostics, uv_stats, texture_stats)

    visual = diagnostics["visual_comparison"]
    assert visual["summary"]["all_passed"] is False
    assert visual["checks"]["face_count_ratio"]["passed"] is False
    assert visual["checks"]["texture_resolution_match"]["passed"] is False
    assert visual["checks"]["texture_resolution_match"]["candidate"] == {"height": 4096, "width": 4096}
    assert visual["checks"]["texture_resolution_match"]["reference"] == {"height": 1024, "width": 1024}
    assert visual["deferred_parity_boundaries"] == [
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
    ]

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

    with pytest.raises(ValueError, match="small_boundary_loop_fill_max_edges"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", small_boundary_loop_fill_max_edges=-1)

    with pytest.raises(ValueError, match="source_projection_fallback_mode"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", source_projection_fallback_mode="far-nearest")


def test_export_pixal3d_glb_simplify_backend_validator(tmp_path) -> None:
    # F1: _resolve_simplify_backend accepts None and "qem", rejects anything else.
    assert _resolve_simplify_backend(None) is None
    assert _resolve_simplify_backend("qem") == "qem"
    assert _resolve_simplify_backend("QEM") == "qem"

    with pytest.raises(ValueError, match="simplify_backend"):
        _resolve_simplify_backend("bogus")
    with pytest.raises(ValueError, match="simplify_backend"):
        _resolve_simplify_backend("topology-aware")
    with pytest.raises(ValueError, match="simplify_backend"):
        _resolve_simplify_backend("spatial-cluster")


def test_export_pixal3d_glb_simplify_backend_bogus_raises(tmp_path) -> None:
    # F1 via public API: "bogus" simplify_backend raises ValueError.
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    with pytest.raises(ValueError, match="simplify_backend"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", simplify_backend="bogus")


def test_export_pixal3d_glb_simplify_backend_qem_requires_remesh(tmp_path) -> None:
    # M3: simplify_backend="qem" without remesh=True raises a clear ValueError.
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    # remesh=False (default) raises.
    with pytest.raises(ValueError, match="simplify_backend='qem'.*remesh=True.*remesh_repair_nonmanifold=True"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", simplify_backend="qem")

    # remesh=True but repair=False raises.
    with pytest.raises(ValueError, match="simplify_backend='qem'.*remesh=True.*remesh_repair_nonmanifold=True"):
        export_pixal3d_glb(
            decoded_dir,
            tmp_path / "out",
            simplify_backend="qem",
            remesh=True,
            remesh_repair_nonmanifold=False,
        )


def test_export_pixal3d_glb_simplify_backend_none_preserves_preset_behavior(tmp_path) -> None:
    # simplify_backend=None must leave the preset-derived backend untouched.
    # _simplifier_backend_for_quality_preset is the source of truth; None must
    # not alter the resolved value.
    assert _simplifier_backend_for_quality_preset("preview") == "spatial-cluster"
    assert _simplifier_backend_for_quality_preset("reference-target") == "topology-aware"
    assert _resolve_simplify_backend(None) is None
    # None is the additive identity: preset result is unchanged.
    for preset, expected in [("preview", "spatial-cluster"), ("reference-target", "topology-aware")]:
        preset_backend = _simplifier_backend_for_quality_preset(preset)
        override = _resolve_simplify_backend(None)
        resolved = override if override is not None else preset_backend
        assert resolved == expected


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
    assert blocked["topology_blocker_map"]["status"] == "artifact_blocked"
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
    assert "reference_stage_contract_incomplete" in blocked_candidate["warnings"]
    contract = blocked_candidate["reference_stage_contract"]
    assert contract["status"] == "blocked"
    assert contract["passed"] is False
    assert "qem_simplification" in contract["blockers"]
    assert "xatlas_unwrap" in contract["blockers"]
    assert "trilinear_pbr_sampling" in contract["heuristic_stage_names"]

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
    assert requested_candidate["topology_blocker_map"]["status"] == "topology_clear"
    assert requested_status["status"] == "blocked"
    assert requested_status["requested_backend"] == "topology-aware"
    assert requested_status["backend_selection_status"] == "fallback_preview_unimplemented"

    scalar_thresholds_without_contract = _export_quality_summary(
        {
            "backend": "qem-edge-collapse",
            "algorithm": "qem-edge-collapse",
            "quality_tier": "production",
            "final_faces": 212_542,
        },
        {"export_blocking_reasons": []},
        {"coverage_ratio": 0.75, "raw_coverage_ratio": 0.20},
        {"final_faces": 212_542, "coverage_ratio": 1.0, "raw_coverage_ratio": 0.40},
        quality_preset="reference-target",
    )
    assert scalar_thresholds_without_contract["production_thresholds"]["all_passed"] is True
    assert scalar_thresholds_without_contract["production_quality_ready"] is False
    assert scalar_thresholds_without_contract["reference_stage_contract"]["status"] == "blocked"
    assert "reference_stage_contract_incomplete" in scalar_thresholds_without_contract["warnings"]

    production = _export_quality_summary(
        {
            "backend": "qem-edge-collapse",
            "algorithm": "qem-edge-collapse",
            "quality_tier": "production",
            "final_faces": 212_542,
            "small_boundary_loop_fill_algorithm": "perimeter-centroid-fan",
            "remesh_backend": "narrow-band-dc",
        },
        {"export_blocking_reasons": []},
        {
            "coverage_ratio": 0.75,
            "raw_coverage_ratio": 0.20,
            "uv_raster_interpolate_reference": True,
            "source_projection_used": True,
            "source_projection_detail": "native_bvh",
            "sampling_mode": "trilinear",
            "postprocess_mode": "inpaint-equivalent",
        },
        {"final_faces": 212_542, "coverage_ratio": 1.0, "raw_coverage_ratio": 0.40},
        quality_preset="reference-target",
        uv_stats={"backend": "xatlas"},
    )
    assert production["artifact_ready"] is True
    assert production["production_quality_ready"] is True
    assert production["reference_stage_contract"]["passed"] is True
    assert production["native_geometry_candidate"]["status"] == "candidate"
    thresholds = production["production_thresholds"]
    assert thresholds["all_passed"] is True
    assert thresholds["checks"]["face_count_ratio"]["actual"] == pytest.approx(1.0)
    assert thresholds["checks"]["final_coverage_ratio"]["actual"] == pytest.approx(0.75)


def test_topology_blocker_map_classifies_visual_and_backend_gaps_from_diagnostics() -> None:
    topology = _topology_blocker_map(
        {
            "backend": "topology-aware",
            "quality_tier": "production_candidate_blocked",
            "production_blockers": [
                "missing_qem_edge_collapse_simplification",
                "missing_narrow_band_dc_remesh",
            ],
            "qem_simplification_backend": "not_implemented",
            "qem_equivalence_status": "qem_scored_not_edge_collapse",
            "remesh_backend": "not_implemented",
            "remesh_equivalence_status": "blocked_missing_narrow_band_dc",
        },
        {
            "export_blocking_reasons": [],
            "nonmanifold_edges": 0,
            "boundary_loop_count": 3,
            "boundary_small_loop_edge_count": 18,
            "boundary_simple_open_chain_count": 1,
            "boundary_branched_open_chain_count": 2,
            "boundary_open_chain_edge_count": 40,
            "boundary_open_chain_branch_vertex_count": 5,
        },
    )

    assert topology["status"] == "rendered_visual_blocked"
    assert topology["diagnostic_source"] == "stages.export_metrics.metrics plus stages.simplify_mesh.stats"
    assert topology["artifact_blockers"] == ()
    assert topology["visual_blockers"] == (
        "clean_closed_boundary_loops",
        "simple_open_boundary_chains",
        "branched_open_boundary_chains",
    )
    assert topology["production_backend_blockers"] == (
        "missing_qem_edge_collapse_simplification",
        "missing_narrow_band_dc_remesh",
    )
    classes = topology["classes"]
    assert classes["clean_closed_boundary_loops"]["count"] == 3
    assert classes["simple_open_chains"]["count"] == 1
    assert classes["branched_open_chains"]["count"] == 2
    assert classes["nonmanifold_edges"]["export_blocking"] is False
    assert classes["heuristic_qem"]["production_blocking"] is True
    assert classes["missing_narrow_band_remesh"]["production_blocking"] is True

    nonmanifold = _topology_blocker_map(
        {"quality_tier": "production"},
        {
            "export_blocking_reasons": ["nonmanifold_edges_present"],
            "nonmanifold_edges": 4,
            "boundary_loop_count": 0,
            "boundary_simple_open_chain_count": 0,
            "boundary_branched_open_chain_count": 0,
        },
    )
    assert nonmanifold["status"] == "artifact_blocked"
    assert nonmanifold["classes"]["nonmanifold_edges"]["export_blocking"] is True


def test_topology_blocker_map_qem_backend_clears_qem_blocker() -> None:
    # QEM-04 export side: a qem-shaped stat dict must cause the topology_blocker_map
    # to NOT append missing_qem_edge_collapse_simplification.  The narrow-band remesh
    # blocker must still be present until remesh runs.
    qem_clean_topology = {
        "export_blocking_reasons": [],
        "nonmanifold_edges": 0,
        "boundary_loop_count": 0,
        "boundary_small_loop_edge_count": 0,
        "boundary_simple_open_chain_count": 0,
        "boundary_branched_open_chain_count": 0,
        "boundary_open_chain_edge_count": 0,
        "boundary_open_chain_branch_vertex_count": 0,
    }

    # Main decimation path stat shape (remesh not yet run).
    topology = _topology_blocker_map(
        {
            "backend": "qem",
            "algorithm": "native_qem_edge_collapse",
            "quality_tier": "production_candidate_blocked",
            "production_blockers": ["missing_narrow_band_dc_remesh"],
            "qem_simplification_backend": "native-qem-edge-collapse",
            "qem_equivalence_status": "edge-collapse",
            "remesh_backend": "not_implemented",
            "remesh_equivalence_status": "blocked_missing_narrow_band_dc",
        },
        qem_clean_topology,
    )
    assert "missing_qem_edge_collapse_simplification" not in topology["production_backend_blockers"], (
        "qem backend must not push missing_qem_edge_collapse_simplification"
    )
    assert "missing_narrow_band_dc_remesh" in topology["production_backend_blockers"]
    assert topology["classes"]["heuristic_qem"]["production_blocking"] is False
    assert topology["classes"]["missing_narrow_band_remesh"]["production_blocking"] is True

    # With remesh cleared (export.py:471-478 sets remesh_backend after remesh runs):
    # both blockers gone -> topology_clear -> quality_tier can reach "production".
    topology_remesh_cleared = _topology_blocker_map(
        {
            "backend": "qem",
            "algorithm": "native_qem_edge_collapse",
            "quality_tier": "production",
            "production_blockers": [],
            "qem_simplification_backend": "native-qem-edge-collapse",
            "qem_equivalence_status": "edge-collapse",
            "remesh_backend": "native-narrow-band-dc",
            "remesh_equivalence_status": "narrow_band_dc",
        },
        qem_clean_topology,
    )
    assert topology_remesh_cleared["status"] == "topology_clear"
    assert topology_remesh_cleared["production_backend_blockers"] == ()


def test_reference_stage_contract_qem_simplify_reference_is_true() -> None:
    # QEM-04 export side: the reference_stage_contract logic at export.py:1223-1226
    # must set simplify_reference=True for native_qem_edge_collapse / quality=production.
    # Feeding a qem-shaped stat dict with remesh cleared means simplify_reference
    # and remesh_reference are both True, so that stage passes.
    from mlx_spatialkit.export import _pixal3d_reference_stage_contract

    qem_stats_remesh_cleared = {
        "backend": "qem",
        "algorithm": "native_qem_edge_collapse",
        "quality_tier": "production",
        "production_blockers": [],
        "qem_simplification_backend": "native-qem-edge-collapse",
        "qem_equivalence_status": "edge-collapse",
        "remesh_backend": "native-narrow-band-dc",
        "remesh_equivalence_status": "narrow_band_dc",
        "small_boundary_loop_fill_algorithm": "cumesh-perimeter-centroid-fan",
    }
    uv_stats = {"backend": "xatlas"}
    texture_stats = {
        "backend": "metal-uv-binned-nearest",
        "uv_raster_interpolate_reference": True,
        "source_projection_used": True,
        "source_projection_detail": "native_bvh",
        "sampling_mode": "trilinear",
        "postprocess_mode": "inpaint-equivalent",
    }

    contract = _pixal3d_reference_stage_contract(
        qem_stats_remesh_cleared,
        uv_stats,
        texture_stats,
        None,
        quality_preset="reference-target",
    )
    qem_stage = contract["stages"]["qem_simplification"]
    remesh_stage = contract["stages"]["narrow_band_dc_remesh"]
    assert qem_stage["passed"] is True, (
        f"qem_simplification stage must pass for native_qem_edge_collapse; "
        f"status={qem_stage.get('status')}"
    )
    assert remesh_stage["passed"] is True
    # reference_trace is still a blocker (no live reference fixture in unit test).
    assert "qem_simplification" not in contract["blockers"]
    assert "narrow_band_dc_remesh" not in contract["blockers"]


def test_export_quality_summary_qem_stats_clears_qem_blocker_and_keeps_remesh_blocker() -> None:
    # QEM-04 export side: _export_quality_summary fed a qem stat dict must:
    #   (a) not show missing_qem_edge_collapse_simplification in topology_blocker_map
    #   (b) show missing_narrow_band_dc_remesh still present (remesh not yet run)
    # When remesh is cleared (remesh_backend set to narrow-band-dc and production_blockers
    # empty) -> topology_clear -> quality_tier "production".
    from mlx_spatialkit.export import _export_quality_summary

    qem_stats = {
        "backend": "qem",
        "algorithm": "native_qem_edge_collapse",
        "quality_tier": "production_candidate_blocked",
        "production_blockers": ["missing_narrow_band_dc_remesh"],
        "qem_simplification_backend": "native-qem-edge-collapse",
        "qem_equivalence_status": "edge-collapse",
        "remesh_backend": "not_implemented",
        "remesh_equivalence_status": "blocked_missing_narrow_band_dc",
        "small_boundary_loop_fill_algorithm": "cumesh-perimeter-centroid-fan",
        "final_faces": 120,
    }
    clean_export_metrics = {
        "export_blocking_reasons": [],
        "nonmanifold_edges": 0,
        "boundary_loop_count": 0,
        "boundary_small_loop_edge_count": 0,
        "boundary_simple_open_chain_count": 0,
        "boundary_branched_open_chain_count": 0,
        "boundary_open_chain_edge_count": 0,
        "boundary_open_chain_branch_vertex_count": 0,
    }

    summary = _export_quality_summary(qem_stats, clean_export_metrics)
    topology = summary["topology_blocker_map"]
    assert "missing_qem_edge_collapse_simplification" not in topology["production_backend_blockers"]
    assert "missing_narrow_band_dc_remesh" in topology["production_backend_blockers"]
    assert summary["artifact_ready"] is True

    # With remesh cleared: no production blockers -> topology_clear.
    qem_stats_remesh_cleared = {
        "backend": "qem",
        "algorithm": "native_qem_edge_collapse",
        "quality_tier": "production",
        "production_blockers": [],
        "qem_simplification_backend": "native-qem-edge-collapse",
        "qem_equivalence_status": "edge-collapse",
        "remesh_backend": "native-narrow-band-dc",
        "remesh_equivalence_status": "narrow_band_dc",
        "small_boundary_loop_fill_algorithm": "cumesh-perimeter-centroid-fan",
        "final_faces": 212_542,
    }
    summary_cleared = _export_quality_summary(qem_stats_remesh_cleared, clean_export_metrics)
    topology_cleared = summary_cleared["topology_blocker_map"]
    assert topology_cleared["status"] == "topology_clear"
    assert topology_cleared["production_backend_blockers"] == ()
    assert summary_cleared["simplifier_quality_tier"] == "production"


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


def test_production_equivalence_summary_keeps_parity_boundaries_strict() -> None:
    quality = {
        "artifact_ready": True,
        "production_quality_ready": True,
        "upstream_export_settings": {"all_passed": False},
        "xatlas_chart_parity": {"parity_ready": False},
    }
    visual = {
        "rendered_visual_ready": True,
        "summary": {"all_passed": True},
        "deferred_parity_boundaries": ["not_xatlas_chart_parity", "not_1m_face_export_setting_parity"],
    }

    summary = _production_equivalence_summary(quality, visual)

    assert summary["ready"] is False
    assert summary["artifact_ready"] is True
    assert summary["scalar_production_quality_ready"] is True
    assert summary["reference_stage_contract_ready"] is False
    assert summary["upstream_export_settings_ready"] is False
    assert summary["xatlas_chart_parity_ready"] is False
    assert summary["visual_comparison_available"] is True
    assert summary["visual_comparison_ready"] is True
    assert summary["remaining_parity_boundaries"] == (
        "not_xatlas_chart_parity",
        "not_1m_face_export_setting_parity",
        "not_reference_stage_contract",
    )
    assert summary["blockers"] == (
        "reference_stage_contract_not_ready",
        "upstream_export_settings_not_ready",
        "xatlas_chart_parity_not_ready",
        "deferred_parity_boundaries_present",
    )
    assert summary["checks"]["remaining_parity_boundaries"]["passed"] is False

    missing_visual = _production_equivalence_summary(
        {
            "artifact_ready": True,
            "production_quality_ready": True,
            "reference_stage_contract": {"passed": True},
            "upstream_export_settings": {"all_passed": True},
            "xatlas_chart_parity": {"parity_ready": False},
        },
        None,
    )
    assert missing_visual["ready"] is False
    assert missing_visual["remaining_parity_boundaries"] == ("not_xatlas_chart_parity",)
    assert "visual_comparison_missing" in missing_visual["blockers"]

    scalar_only_visual = _production_equivalence_summary(
        {
            "artifact_ready": True,
            "production_quality_ready": True,
            "reference_stage_contract": {"passed": True},
            "upstream_export_settings": {"all_passed": True},
            "xatlas_chart_parity": {"parity_ready": True},
        },
        {"summary": {"all_passed": True}, "deferred_parity_boundaries": []},
    )
    assert scalar_only_visual["ready"] is False
    assert scalar_only_visual["visual_comparison_ready"] is False
    assert "rendered_visual_not_ready" in scalar_only_visual["blockers"]

    ready = _production_equivalence_summary(
        {
            "artifact_ready": True,
            "production_quality_ready": True,
            "reference_stage_contract": {"passed": True},
            "upstream_export_settings": {"all_passed": True},
            "xatlas_chart_parity": {"parity_ready": True},
        },
        {"rendered_visual_ready": True, "summary": {"all_passed": True}, "deferred_parity_boundaries": []},
    )
    assert ready["ready"] is True
    assert ready["remaining_parity_boundaries"] == ()
    assert ready["blockers"] == ()


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
            "chart_cluster_normal_policy": "edge-and-seed-cone",
            "chart_cone_half_angle_degrees": 45.0,
            "chart_cone_rejected_adjacency_count": 3,
            "chart_edge_rejected_adjacency_count": 1,
            "packing": "aspect-shelf-charts",
            "chart_rect_fill_ratio": 0.72,
            "atlas_rect_coverage_ratio": 0.68,
            "shelf_packing_efficiency": 0.85,
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
    behavior = quality_blocked["native_behavior_diagnostics"]
    assert behavior["requires_xatlas_dependency"] is False
    assert behavior["cluster_normal_policy"] == "edge-and-seed-cone"
    assert behavior["chart_cone_rejected_adjacency_count"] == 3
    assert behavior["chart_rect_fill_ratio"] == pytest.approx(0.72)
    assert behavior["seam_island_risk"]["status"] == "measured"

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


@pytest.mark.heavy
def test_export_pixal3d_glb_qem_input_prep_real_fixture_manifold_and_bounded() -> None:
    """S4 heavy: real fixture export with simplify_backend="qem" produces a fully manifold mesh.

    Contract (QEM-05): the output mesh is FULLY MANIFOLD (nonmanifold_edges==0 and
    nonmanifold_vertices==0) and QEM PRESERVES topology (it adds zero boundary loops).
    Any residual open boundary loops are large-perimeter loops that were already present
    in the bounded pre-QEM fill stage (qem_pre_fill_residual_boundary_loops) and that
    the bounded fill correctly refuses to close (perimeter >> the 0.03 max_perimeter
    policy).  QEM boundary-locks and faithfully preserves those loops -- it does not
    tear them.  Full watertightness vs the reference is a documented follow-on requiring
    non-manifold-tolerant QEM; it is NOT a QEM-05 contract requirement.
    """
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"real Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-qem-input-prep-export-{os.getpid()}"
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="preview",
        remesh=True,
        remesh_resolution=192,
        remesh_repair_nonmanifold=True,
        simplify_backend="qem",
    )
    diagnostics = json.loads(result.diagnostics_path.read_text())
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    export_metrics = diagnostics["stages"]["export_metrics"]["metrics"]

    # 1. QEM backend was selected.
    assert diagnostics["settings"]["simplify_backend"] == "qem"
    assert diagnostics["settings"]["requested_simplifier_backend"] == "qem"
    assert simplify_stats["backend"] == "qem"
    assert simplify_stats["algorithm"] == "native_qem_edge_collapse"
    assert simplify_stats["qem_simplification_backend"] == "native-qem-edge-collapse"

    # Target reached is honest (not forced).
    assert isinstance(simplify_stats["target_reached"], bool)

    # 2. The mesh is FULLY MANIFOLD: QEM's core geometric guarantee.
    #    These MUST be 0; if either is non-zero QEM produced a broken mesh.
    assert export_metrics["nonmanifold_edges"] == 0, (
        f"QEM produced non-manifold edges: nonmanifold_edges={export_metrics['nonmanifold_edges']}"
    )
    assert export_metrics["nonmanifold_vertices"] == 0, (
        f"QEM produced non-manifold vertices: nonmanifold_vertices={export_metrics['nonmanifold_vertices']}"
    )

    # 3. QEM PRESERVED topology (added zero boundary loops).
    #    The only open boundary loops in the output are the large-perimeter residual that
    #    the bounded pre-QEM fill correctly declined to fill.  If QEM tore topology this
    #    assertion fails because boundary_loop_count would exceed the residual.
    pre_fill_residual = simplify_stats["qem_pre_fill_residual_boundary_loops"]
    assert export_metrics["boundary_loop_count"] == pre_fill_residual, (
        f"QEM tore topology: boundary_loop_count={export_metrics['boundary_loop_count']} "
        f"!= qem_pre_fill_residual_boundary_loops={pre_fill_residual}"
    )
    # QEM must not produce any open chains (non-loop boundary components) either.
    assert export_metrics["boundary_open_chain_count"] == 0, (
        f"QEM produced open chains: boundary_open_chain_count={export_metrics['boundary_open_chain_count']}"
    )

    # 4. Residual is recorded AND bounded.
    #    The residual is >= 0 (sanity) and <= 64 (the repair-induced openings from
    #    remesh(repair_nonmanifold=True) splitting non-manifold edges; bounded fill
    #    closes the small ones, correctly refuses the large-perimeter ones).
    #    This bound documents the known behaviour -- a tighter post-remesh repair
    #    path is a follow-on item tracked separately.
    assert pre_fill_residual >= 0
    assert pre_fill_residual <= 64, (
        f"More residual boundary loops than expected from repair-induced openings: "
        f"qem_pre_fill_residual_boundary_loops={pre_fill_residual} (expected <= 64)"
    )

    assert export_metrics["export_blocking_reasons"] == []


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


def _load_remeshed_mesh_from_fixture(
    fixture: Path,
    *,
    grid_size: int = 256,
    remesh_resolution: int = 256,
    min_component_faces: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract, clean, and remesh a Pixal3D fixture; return (vertices, faces).

    Used to produce a single remeshed geometry that can then be fed to both
    simplify_mesh(backend="qem") and simplify_mesh(backend="topology-aware")
    for a fair apples-to-apples clustering contrast without running a full bake.
    """
    shape_path = fixture / "shape_decoder_fields.npz"
    with np.load(shape_path) as payload:
        coordinates = np.asarray(payload["coordinates"])
        fields = np.asarray(payload["fields"])

    raw = extract_flexi_dual_grid(coordinates, fields, grid_size=grid_size)
    cleaned, _ = clean_mesh(raw.vertices, raw.faces, min_component_faces=min_component_faces)
    remeshed, _ = remesh_narrow_band(
        cleaned.vertices,
        cleaned.faces,
        resolution=remesh_resolution,
        repair_nonmanifold=True,
    )
    return remeshed.vertices, remeshed.faces


def _assert_qem_two_fixture_proof(
    fixture: Path,
    output_dir: Path,
    *,
    # Perf budgets: set to 5x the observed value on the calibration run.
    # MAIN fixture (res=256):
    #   observed timings_sec[simplify_mesh] ~ 14.15 s  => BUDGET_S = 70 s
    #   observed peak_current_rss_bytes     ~ 3.65 GB  => BUDGET_B = 18_000_000_000
    # VIOLIN fixture (res=256):
    #   observed timings_sec[simplify_mesh] ~ 2.51 s   => BUDGET_S = 13 s
    #   observed peak_current_rss_bytes     ~ 1.44 GB  => BUDGET_B = 7_500_000_000
    simplify_timing_budget_s: float,
    simplify_memory_budget_bytes: int,
    # Name tag for error messages.
    label: str = "",
) -> dict:
    """Core QEM two-fixture proof assertions (assertions 1-8 from SPEC QEM-05).

    Resolution: remesh_resolution=256 (tractable).
    Reference-scale (res1024 / 1M-face / 4096-texture) is explicitly DEFERRED per SPEC.

    Contract (per SPEC QEM-05 and slice-5 acceptance criteria):
    - QEM selected (backend/algorithm/qem_simplification_backend).
    - FULLY MANIFOLD: nonmanifold_edges==0 AND nonmanifold_vertices==0.
    - TOPOLOGY PRESERVED: boundary_loop_count == qem_pre_fill_residual_boundary_loops,
      boundary_open_chain_count==0.
    - MATERIALLY BETTER THAN CLUSTERING: QEM boundary_loop_count < clustering's on the
      same remeshed input.  NOTE: we do NOT assert strict 0-loop watertightness (that is
      a documented follow-on); we assert QEM is strictly better than clustering.
    - INPUT != TARGET: source_faces >> target_faces.
    - NUMERIC BUDGETS: timings_sec[simplify_mesh] < BUDGET_S,
      memory.stage_peaks[simplify_mesh].peak_current_rss_bytes < BUDGET_B.
    - QUADRIC-ERROR FIDELITY: qem_geometric_error_mean/max present, finite, bounded.
    - PROOF SCOPE: preset/target recorded; non-reference target noted (reference-scale deferred).
    """
    result = export_pixal3d_glb(
        fixture,
        output_dir,
        quality_preset="preview",
        remesh=True,
        remesh_resolution=256,
        remesh_repair_nonmanifold=True,
        simplify_backend="qem",
        min_component_faces=32,
    )

    assert result.glb.path.read_bytes()[:4] == b"glTF", f"[{label}] GLB header corrupt"
    diagnostics = json.loads(result.diagnostics_path.read_text())
    simplify_stats = diagnostics["stages"]["simplify_mesh"]["stats"]
    export_metrics = diagnostics["stages"]["export_metrics"]["metrics"]

    # ------------------------------------------------------------------
    # Assertion 1: QEM selected.
    # ------------------------------------------------------------------
    assert diagnostics["settings"]["simplify_backend"] == "qem", (
        f"[{label}] settings.simplify_backend must be 'qem'"
    )
    assert diagnostics["settings"]["requested_simplifier_backend"] == "qem", (
        f"[{label}] settings.requested_simplifier_backend must be 'qem'"
    )
    assert simplify_stats["backend"] == "qem", (
        f"[{label}] simplify_stats.backend must be 'qem', got {simplify_stats['backend']!r}"
    )
    assert simplify_stats["algorithm"] == "native_qem_edge_collapse", (
        f"[{label}] simplify_stats.algorithm must be 'native_qem_edge_collapse'"
    )
    assert simplify_stats["qem_simplification_backend"] == "native-qem-edge-collapse", (
        f"[{label}] qem_simplification_backend must be 'native-qem-edge-collapse'"
    )
    # target_reached is honest: it reflects whether QEM reached the target, not forced True.
    assert isinstance(simplify_stats["target_reached"], bool), (
        f"[{label}] target_reached must be a bool"
    )

    # ------------------------------------------------------------------
    # Assertion 2: FULLY MANIFOLD — QEM's core geometric guarantee.
    # ------------------------------------------------------------------
    assert export_metrics["nonmanifold_edges"] == 0, (
        f"[{label}] QEM produced nonmanifold edges: {export_metrics['nonmanifold_edges']}"
    )
    assert export_metrics["nonmanifold_vertices"] == 0, (
        f"[{label}] QEM produced nonmanifold vertices: {export_metrics['nonmanifold_vertices']}"
    )

    # ------------------------------------------------------------------
    # Assertion 3: TOPOLOGY PRESERVED (QEM-02 on real data).
    # The output boundary_loop_count must exactly equal the pre-QEM fill residual.
    # If QEM tore topology, boundary_loop_count would exceed the residual.
    # We do NOT assert strict 0-loop watertightness (that is a documented follow-on).
    # ------------------------------------------------------------------
    pre_fill_residual: int = simplify_stats["qem_pre_fill_residual_boundary_loops"]
    assert export_metrics["boundary_loop_count"] == pre_fill_residual, (
        f"[{label}] QEM tore topology: boundary_loop_count={export_metrics['boundary_loop_count']} "
        f"!= qem_pre_fill_residual_boundary_loops={pre_fill_residual}"
    )
    assert export_metrics["boundary_open_chain_count"] == 0, (
        f"[{label}] QEM produced open chains: boundary_open_chain_count="
        f"{export_metrics['boundary_open_chain_count']}"
    )

    # ------------------------------------------------------------------
    # Assertion 4: MATERIALLY BETTER THAN CLUSTERING (R5).
    # Re-extract the same remeshed geometry without a full bake and compare
    # simplify_mesh(topology-aware) boundary_loop_count to the QEM result.
    # ------------------------------------------------------------------
    remesh_v, remesh_f = _load_remeshed_mesh_from_fixture(fixture)
    _cluster_mesh, _cluster_stats = simplify_mesh(
        remesh_v,
        remesh_f,
        target_faces=int(simplify_stats["target_faces"]),
        min_component_faces=32,
        backend="topology-aware",
    )
    cluster_metrics = mesh_metrics(_cluster_mesh.vertices, _cluster_mesh.faces)
    cluster_boundary_loops: int = int(cluster_metrics["boundary_loop_count"])
    qem_boundary_loops: int = int(export_metrics["boundary_loop_count"])

    assert qem_boundary_loops < cluster_boundary_loops, (
        f"[{label}] QEM boundary_loop_count ({qem_boundary_loops}) must be strictly LESS "
        f"than clustering boundary_loop_count ({cluster_boundary_loops}) on the same remeshed input"
    )

    # ------------------------------------------------------------------
    # Assertion 5: INPUT != TARGET (R1).
    # source_faces is what QEM received (the remesh output); it must be >> target_faces.
    # ------------------------------------------------------------------
    source_faces: int = simplify_stats["source_faces"]
    target_faces: int = simplify_stats["target_faces"]
    assert source_faces > target_faces, (
        f"[{label}] source_faces ({source_faces}) must exceed target_faces ({target_faces})"
    )
    # Ensure the remesh actually produced substantial geometry.
    assert source_faces >= 10_000, (
        f"[{label}] source_faces ({source_faces}) unexpectedly low"
    )

    # ------------------------------------------------------------------
    # Assertion 6: NUMERIC BUDGETS (R2).
    # Budgets are anchored at 5x the observed calibration values (recorded above).
    # simplify_mesh timing and per-stage peak RSS must stay within budget.
    # ------------------------------------------------------------------
    observed_timing = diagnostics["timings_sec"]["simplify_mesh"]
    assert observed_timing < simplify_timing_budget_s, (
        f"[{label}] simplify_mesh timing {observed_timing:.2f}s exceeded budget "
        f"{simplify_timing_budget_s:.0f}s"
    )

    _assert_memory_diagnostics(diagnostics, required_stages=("simplify_mesh",))
    _simplify_stage_rss = diagnostics["memory"]["stage_peaks"]["simplify_mesh"]
    simplify_mem_peak = _simplify_stage_rss["peak_current_rss_bytes"]
    simplify_mem_start = _simplify_stage_rss.get("start_current_rss_bytes")
    assert simplify_mem_peak is not None, f"[{label}] simplify_mesh peak RSS not recorded"
    # Budget the stage's OWN allocation as a DELTA (peak - start), not the
    # absolute process peak. Absolute RSS accumulates across the whole heavy
    # suite (prior tests' allocations / fragmentation are not freed), so an
    # absolute-peak budget is contaminated by test order; the delta measures
    # what QEM itself allocated during the stage and is suite-order-independent.
    simplify_mem_delta = (
        simplify_mem_peak - simplify_mem_start
        if simplify_mem_start is not None
        else simplify_mem_peak
    )
    assert simplify_mem_delta < simplify_memory_budget_bytes, (
        f"[{label}] simplify_mesh RSS delta {simplify_mem_delta / 2**30:.2f} GiB exceeded budget "
        f"{simplify_memory_budget_bytes / 2**30:.2f} GiB "
        f"(peak {simplify_mem_peak / 2**30:.2f} GiB, start {(simplify_mem_start or 0) / 2**30:.2f} GiB)"
    )

    # ------------------------------------------------------------------
    # Assertion 7: QUADRIC-ERROR FIDELITY (QEM-03).
    # qem_geometric_error_mean and qem_geometric_error_max must be present,
    # finite, and within the empirically bounded range from calibration runs.
    # ------------------------------------------------------------------
    geo_err_mean = simplify_stats["qem_geometric_error_mean"]
    geo_err_max = simplify_stats["qem_geometric_error_max"]
    assert geo_err_mean is not None, f"[{label}] qem_geometric_error_mean missing"
    assert geo_err_max is not None, f"[{label}] qem_geometric_error_max missing"
    assert math.isfinite(float(geo_err_mean)), (
        f"[{label}] qem_geometric_error_mean is not finite: {geo_err_mean}"
    )
    assert math.isfinite(float(geo_err_max)), (
        f"[{label}] qem_geometric_error_max is not finite: {geo_err_max}"
    )
    assert float(geo_err_mean) >= 0.0, f"[{label}] qem_geometric_error_mean must be >= 0"
    assert float(geo_err_max) >= float(geo_err_mean), (
        f"[{label}] qem_geometric_error_max must be >= mean"
    )
    # Upper bound: must be < 1.0 (normalised mesh units; anything >= 1.0 is pathological).
    assert float(geo_err_max) < 1.0, (
        f"[{label}] qem_geometric_error_max ({geo_err_max}) >= 1.0 (pathological)"
    )

    # ------------------------------------------------------------------
    # Assertion 8: PROOF SCOPE (F2).
    # Record the proof settings so reviewers can confirm the preview/res256 scope.
    # A preview/non-reference target does NOT clear the reference-target production gate.
    # Full reference-target parity (res1024/1M-face/4096-texture) is DEFERRED per SPEC.
    # ------------------------------------------------------------------
    quality_preset = diagnostics["settings"]["quality_preset"]
    proof_target_faces = diagnostics["settings"]["target_faces"]
    proof_target_faces_source = diagnostics["settings"]["target_faces_source"]
    # This proof uses the preview preset and preview default target (50,000 faces).
    # It does not satisfy the reference-target production gate.
    assert quality_preset == "preview", (
        f"[{label}] This proof uses quality_preset='preview'; got {quality_preset!r}"
    )
    assert diagnostics["result"]["production_quality_ready"] is False, (
        f"[{label}] preview/res256 proof must NOT set production_quality_ready=True "
        f"(reference-target parity is deferred)"
    )
    # The proof target must be the preview default, not the reference face count.
    assert proof_target_faces_source == "preview_default", (
        f"[{label}] proof must use preview_default target; got {proof_target_faces_source!r}"
    )

    return {
        "label": label,
        "quality_preset": quality_preset,
        "target_faces": proof_target_faces,
        "target_faces_source": proof_target_faces_source,
        "source_faces": source_faces,
        "qem_boundary_loops": qem_boundary_loops,
        "cluster_boundary_loops": cluster_boundary_loops,
        "pre_fill_residual": pre_fill_residual,
        "simplify_timing_s": observed_timing,
        "simplify_peak_rss_bytes": simplify_mem_peak,
        "qem_geometric_error_mean": float(geo_err_mean),
        "qem_geometric_error_max": float(geo_err_max),
    }


@pytest.mark.heavy
def test_export_pixal3d_glb_qem_two_fixture_main_manifold_and_beats_clustering() -> None:
    """Slice-5 QEM proof: main fixture (pixal3d-1024-cascade-decoded-pbr) at res=256.

    All eight acceptance criteria from SPEC QEM-05 / slice-5:
    1. QEM selected (backend/algorithm/qem_simplification_backend).
    2. Fully manifold: nonmanifold_edges==0 AND nonmanifold_vertices==0.
    3. Topology preserved: boundary_loop_count == qem_pre_fill_residual_boundary_loops,
       boundary_open_chain_count==0.
    4. Materially better than clustering: QEM boundary_loop_count < clustering's on same input.
    5. Input != target: source_faces (remesh output) >> target_faces.
    6. Numeric budgets: simplify_mesh timing < BUDGET_S, peak RSS < BUDGET_B.
       Calibration (observed, 2026-06-04, Apple Silicon, res=256):
         timings_sec[simplify_mesh] ~ 14.15 s   => BUDGET_S = 70 s (5x margin)
         peak_current_rss_bytes     ~ 3.65 GB   => BUDGET_B = 18 GB (5x margin)
    7. Quadric-error fidelity: qem_geometric_error_mean/max present, finite, bounded.
       Calibration: mean ~ 1.11e-4, max ~ 2.85e-2.
    8. Proof scope: preview/res256; reference-scale (res1024/1M/4096) is DEFERRED per SPEC.

    NOTE: strict watertight (boundary_loop_count==0) is NOT asserted.
    That is a documented follow-on requiring a non-manifold-tolerant QEM path.
    """
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"main Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-qem-main-proof-{os.getpid()}"

    # Perf budgets anchored to 5x the observed calibration values (see docstring).
    BUDGET_S = 70.0       # seconds  (observed ~ 14.15 s)
    BUDGET_B = 18_000_000_000  # bytes (~18 GB; observed ~ 3.65 GB = 3,651,977,216)

    proof = _assert_qem_two_fixture_proof(
        fixture,
        output_dir,
        simplify_timing_budget_s=BUDGET_S,
        simplify_memory_budget_bytes=BUDGET_B,
        label="main",
    )

    # Report observed numbers for CI traceability.
    print(
        f"\n[main fixture QEM proof]\n"
        f"  preset={proof['quality_preset']!r}  target={proof['target_faces']}  "
        f"source={proof['target_faces_source']!r}\n"
        f"  source_faces={proof['source_faces']:,}  target_faces={proof['target_faces']:,}\n"
        f"  qem_boundary_loops={proof['qem_boundary_loops']}  "
        f"cluster_boundary_loops={proof['cluster_boundary_loops']}  "
        f"pre_fill_residual={proof['pre_fill_residual']}\n"
        f"  simplify_timing={proof['simplify_timing_s']:.2f}s (budget {BUDGET_S:.0f}s)  "
        f"peak_rss={proof['simplify_peak_rss_bytes'] / 2**30:.2f} GiB "
        f"(budget {BUDGET_B / 2**30:.2f} GiB)\n"
        f"  geo_error_mean={proof['qem_geometric_error_mean']:.3e}  "
        f"geo_error_max={proof['qem_geometric_error_max']:.3e}"
    )


@pytest.mark.heavy
def test_export_pixal3d_glb_qem_two_fixture_violin_manifold_and_beats_clustering() -> None:
    """Slice-5 QEM proof: violin-bow fixture at res=256.

    All eight acceptance criteria from SPEC QEM-05 / slice-5:
    1. QEM selected (backend/algorithm/qem_simplification_backend).
    2. Fully manifold: nonmanifold_edges==0 AND nonmanifold_vertices==0.
    3. Topology preserved: boundary_loop_count == qem_pre_fill_residual_boundary_loops,
       boundary_open_chain_count==0.
    4. Materially better than clustering: QEM boundary_loop_count < clustering's on same input.
    5. Input != target: source_faces (remesh output) >> target_faces.
    6. Numeric budgets: simplify_mesh timing < BUDGET_S, peak RSS < BUDGET_B.
       Calibration (observed, 2026-06-04, Apple Silicon, res=256):
         timings_sec[simplify_mesh] ~ 2.51 s    => BUDGET_S = 13 s (5x margin)
         peak_current_rss_bytes     ~ 1.44 GB   => BUDGET_B = 7.5 GB (5x margin)
    7. Quadric-error fidelity: qem_geometric_error_mean/max present, finite, bounded.
       Calibration: mean ~ 9.54e-6, max ~ 5.86e-3.
    8. Proof scope: preview/res256; reference-scale (res1024/1M/4096) is DEFERRED per SPEC.

    NOTE: strict watertight (boundary_loop_count==0) is NOT asserted.
    That is a documented follow-on requiring a non-manifold-tolerant QEM path.
    """
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "violin-bow" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"violin-bow Pixal3D decoded fixture not present: {fixture}")

    output_dir = Path("/tmp") / f"mlx-spatialkit-qem-violin-proof-{os.getpid()}"

    # Perf budgets anchored to 5x the observed calibration values (see docstring).
    BUDGET_S = 13.0       # seconds  (observed ~ 2.51 s)
    BUDGET_B = 7_500_000_000  # bytes (~7.5 GB; observed ~ 1.44 GB = 1,442,709,504)

    proof = _assert_qem_two_fixture_proof(
        fixture,
        output_dir,
        simplify_timing_budget_s=BUDGET_S,
        simplify_memory_budget_bytes=BUDGET_B,
        label="violin",
    )

    # Report observed numbers for CI traceability.
    print(
        f"\n[violin fixture QEM proof]\n"
        f"  preset={proof['quality_preset']!r}  target={proof['target_faces']}  "
        f"source={proof['target_faces_source']!r}\n"
        f"  source_faces={proof['source_faces']:,}  target_faces={proof['target_faces']:,}\n"
        f"  qem_boundary_loops={proof['qem_boundary_loops']}  "
        f"cluster_boundary_loops={proof['cluster_boundary_loops']}  "
        f"pre_fill_residual={proof['pre_fill_residual']}\n"
        f"  simplify_timing={proof['simplify_timing_s']:.2f}s (budget {BUDGET_S:.0f}s)  "
        f"peak_rss={proof['simplify_peak_rss_bytes'] / 2**30:.2f} GiB "
        f"(budget {BUDGET_B / 2**30:.2f} GiB)\n"
        f"  geo_error_mean={proof['qem_geometric_error_mean']:.3e}  "
        f"geo_error_max={proof['qem_geometric_error_max']:.3e}"
    )


def _assert_chart_growth_parity_against_oracle(fixture: Path, anchor_name: str) -> dict:
    """Slice-4 stage-B parity: native cone-cluster + chart growth vs pip-xatlas anchors.

    Rebuilds the fixture's QEM 50k mesh with the exact anchor recipe
    (tests/tools/gen_uv_oracle_anchors.py), then:
      1. stage A at the pinned production knobs must reproduce the anchors'
         stage_a_cluster_count EXACTLY (same code, deterministic);
      2. stage B at reference defaults must land within the stated chart-count
         band of the per-cluster pip-xatlas composition anchor.

    Band: ours/oracle in [0.60, 1.50]. Measured 2026-06-12 (Apple Silicon):
    main 4007/5166 = 0.776, violin_bow 1677/2527 = 0.664. We sit on the low
    side: the double-precision validity test (flip consistency + boundary
    self-intersection) fails less often than xatlas's float32 one, so growth
    and merging keep slightly larger charts. Chart count is the coarse
    structural parity signal; binding quality gates are overlap/stretch
    (slices 5-6).
    """
    import json
    import math as _math

    from mlx_spatialkit._native import compute_uv_charts, grow_uv_charts

    anchors_path = Path(__file__).resolve().parent / "data" / "uv_oracle_anchors.json"
    anchors = json.loads(anchors_path.read_text())
    oracle = anchors["fixtures"][anchor_name]

    remesh_v, remesh_f = _load_remeshed_mesh_from_fixture(fixture)
    simplified, stats = simplify_mesh(
        remesh_v,
        remesh_f,
        target_faces=50_000,
        min_component_faces=32,
        backend="qem",
    )
    assert stats["backend"] == "qem"
    vertices = np.ascontiguousarray(simplified.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(simplified.faces, dtype=np.int64)

    stage_a = compute_uv_charts(
        vertices,
        faces,
        threshold_cone_half_angle_rad=_math.radians(90.0),
        refine_iterations=0,
        global_iterations=1,
        smooth_strength=1.0,
        area_penalty_weight=0.1,
        perimeter_area_ratio_weight=0.0001,
    )
    assert stage_a["chart_count"] == oracle["stage_a_cluster_count"], (
        f"[{anchor_name}] stage-A cluster count {stage_a['chart_count']} != anchor "
        f"{oracle['stage_a_cluster_count']} (same code+knobs must reproduce exactly)"
    )

    grown = grow_uv_charts(
        vertices,
        faces,
        cluster_ids=np.ascontiguousarray(np.asarray(stage_a["chart_ids"]), dtype=np.int64),
    )
    chart_ids = np.asarray(grown["chart_ids"])
    assert (chart_ids >= 0).all(), f"[{anchor_name}] unassigned faces after growth"
    assert chart_ids.max() == grown["chart_count"] - 1
    assert (
        grown["accepted_chart_count"] + grown["lscm_pending_chart_count"]
        == grown["chart_count"]
    )

    oracle_charts = oracle["per_cluster_composition"]["chart_count"]
    ratio = grown["chart_count"] / oracle_charts
    assert 0.60 <= ratio <= 1.50, (
        f"[{anchor_name}] stage-B chart count {grown['chart_count']} vs oracle "
        f"{oracle_charts} -> ratio {ratio:.3f} outside [0.60, 1.50]"
    )
    return {
        "stage_a_clusters": stage_a["chart_count"],
        "chart_count": grown["chart_count"],
        "oracle_chart_count": oracle_charts,
        "ratio": ratio,
        "accepted": grown["accepted_chart_count"],
        "lscm_pending": grown["lscm_pending_chart_count"],
    }


@pytest.mark.heavy
def test_reference_uv_chart_growth_parity_main_fixture() -> None:
    """Slice-4 heavy: stage-B chart-count parity vs pip-xatlas oracle (main)."""
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"main Pixal3D decoded fixture not present: {fixture}")
    summary = _assert_chart_growth_parity_against_oracle(fixture, "main")
    print(f"\n[main chart-growth parity] {summary}")


@pytest.mark.heavy
def test_reference_uv_chart_growth_parity_violin_fixture() -> None:
    """Slice-4 heavy: stage-B chart-count parity vs pip-xatlas oracle (violin-bow)."""
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = (
        _repo_root() / "inputs" / "mlx-spatialkit" / "violin-bow"
        / "pixal3d-1024-cascade-decoded-pbr"
    )
    if not fixture.exists():
        pytest.skip(f"violin-bow Pixal3D decoded fixture not present: {fixture}")
    summary = _assert_chart_growth_parity_against_oracle(fixture, "violin_bow")
    print(f"\n[violin chart-growth parity] {summary}")


def _assert_parameterization_invariants_and_stretch_parity(
    fixture: Path, anchor_name: str
) -> dict:
    """Slice-5 heavy: zero-overlap/zero-flip invariant + stretch parity.

    Full native pipeline (stage-A production knobs -> stage-B growth ->
    parameterize: projection / LSCM / targeted-split repair), then:
      1. INVARIANT: zero flipped faces (native recount) and zero intra-chart
         UV overlaps, cross-checked per chart via uv_quality_metrics with
         per-chart centering (float32-safe), never trusted from builder
         counters alone (UVU-04).
      2. STRETCH PARITY: scale-free per-chart stretch (l2 * sqrt(Auv/A3d))
         mean and p95 must not exceed 1.25x the pip-xatlas oracle's
         chart_stretch_l2_normalized_summary (UVU-04; measured 2026-06-12:
         ours p95 ~1.1 vs oracle 3.06 (main) / 1.97 (violin) — we are
         substantially better, the band guards regression).
      3. Repair stays bounded: shattered single-face charts < 25% of output
         charts (the targeted-split fix; blind bisection produced >50%).
    """
    import json
    import math as _math

    from mlx_spatialkit._native import (
        compute_uv_charts,
        grow_uv_charts,
        parameterize_uv_charts,
        uv_quality_metrics,
    )

    anchors_path = Path(__file__).resolve().parent / "data" / "uv_oracle_anchors.json"
    oracle = json.loads(anchors_path.read_text())["fixtures"][anchor_name]

    remesh_v, remesh_f = _load_remeshed_mesh_from_fixture(fixture)
    simplified, stats = simplify_mesh(
        remesh_v, remesh_f, target_faces=50_000, min_component_faces=32, backend="qem")
    assert stats["backend"] == "qem"
    vertices = np.ascontiguousarray(simplified.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(simplified.faces, dtype=np.int64)

    stage_a = compute_uv_charts(
        vertices, faces,
        threshold_cone_half_angle_rad=_math.radians(90.0),
        refine_iterations=0, global_iterations=1, smooth_strength=1.0,
        area_penalty_weight=0.1, perimeter_area_ratio_weight=0.0001)
    grown = grow_uv_charts(
        vertices, faces,
        cluster_ids=np.ascontiguousarray(np.asarray(stage_a["chart_ids"]), dtype=np.int64))
    result = parameterize_uv_charts(
        vertices, faces,
        np.ascontiguousarray(np.asarray(grown["chart_ids"]), dtype=np.int64))

    # 1. Invariants.
    assert result["uv_flipped_count"] == 0, f"[{anchor_name}] flipped faces in final UVs"
    assert result["lscm_unconverged_count"] == 0, f"[{anchor_name}] LSCM failed to converge"
    chart_ids = np.asarray(result["chart_ids"])
    assert (chart_ids >= 0).all()
    corner_uvs = np.asarray(result["corner_uvs"]).reshape(faces.shape[0], 3, 2)
    total_overlap = 0
    total_flip = 0
    for chart in range(result["chart_count"]):
        mask = chart_ids == chart
        sub_faces = faces[mask]
        sub_uvs = corner_uvs[mask]
        sub_uvs = sub_uvs - sub_uvs.reshape(-1, 2).mean(axis=0)  # float32-safe
        corner_positions = vertices[sub_faces].reshape(-1, 3).astype(np.float32)
        corner_faces = np.arange(sub_faces.shape[0] * 3, dtype=np.int64).reshape(-1, 3)
        metrics = uv_quality_metrics(
            corner_positions, corner_faces,
            np.ascontiguousarray(sub_uvs.reshape(-1, 2), dtype=np.float32))
        total_overlap += metrics["uv_overlap_count"]
        total_flip += metrics["uv_flipped_count"]
    assert total_overlap == 0, f"[{anchor_name}] intra-chart UV overlaps: {total_overlap}"
    assert total_flip == 0, f"[{anchor_name}] cross-checked flipped faces: {total_flip}"

    # 2. Scale-free stretch parity vs oracle.
    uv_signed = 0.5 * (
        (corner_uvs[:, 1, 0] - corner_uvs[:, 0, 0]) * (corner_uvs[:, 2, 1] - corner_uvs[:, 0, 1])
        - (corner_uvs[:, 2, 0] - corner_uvs[:, 0, 0]) * (corner_uvs[:, 1, 1] - corner_uvs[:, 0, 1])
    )
    tri = vertices[faces].astype(np.float64)
    a3d = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    chart_l2 = np.asarray(result["chart_stretch_l2"])
    measurable = (uv_signed > 1e-12) & (a3d > 0.0)
    normalized = []
    for chart in range(result["chart_count"]):
        in_chart = measurable & (chart_ids == chart)
        auv = float(uv_signed[in_chart].sum())
        a3 = float(a3d[in_chart].sum())
        if chart_l2[chart] > 0.0 and auv > 0.0 and a3 > 0.0:
            normalized.append(chart_l2[chart] * _math.sqrt(auv / a3))
    normalized = np.asarray(normalized)
    oracle_summary = oracle["per_cluster_composition"]["chart_stretch_l2_normalized_summary"]
    ours_mean = float(normalized.mean())
    ours_p95 = float(np.percentile(normalized, 95))
    assert ours_mean <= 1.25 * oracle_summary["mean"], (
        f"[{anchor_name}] normalized stretch mean {ours_mean:.3f} exceeds "
        f"1.25x oracle {oracle_summary['mean']:.3f}"
    )
    assert ours_p95 <= 1.25 * oracle_summary["p95"], (
        f"[{anchor_name}] normalized stretch p95 {ours_p95:.3f} exceeds "
        f"1.25x oracle {oracle_summary['p95']:.3f}"
    )

    # 3. Repair bounded.
    shatter_fraction = result["shattered_face_chart_count"] / max(result["chart_count"], 1)
    assert shatter_fraction < 0.25, (
        f"[{anchor_name}] shattered fraction {shatter_fraction:.3f} >= 0.25"
    )
    return {
        "chart_count": result["chart_count"],
        "projected": result["projected_chart_count"],
        "projection_fallback": result["projection_fallback_chart_count"],
        "lscm": result["lscm_chart_count"],
        "shattered": result["shattered_face_chart_count"],
        "stretch_mean": ours_mean,
        "stretch_p95": ours_p95,
        "oracle_mean": oracle_summary["mean"],
        "oracle_p95": oracle_summary["p95"],
    }


@pytest.mark.heavy
def test_reference_uv_param_overlap_and_stretch_parity_main_fixture() -> None:
    """Slice-5 heavy: zero-overlap invariant + stretch parity (main)."""
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = _repo_root() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-decoded-pbr"
    if not fixture.exists():
        pytest.skip(f"main Pixal3D decoded fixture not present: {fixture}")
    summary = _assert_parameterization_invariants_and_stretch_parity(fixture, "main")
    print(f"\n[main parameterization invariants+parity] {summary}")


@pytest.mark.heavy
def test_reference_uv_param_overlap_and_stretch_parity_violin_fixture() -> None:
    """Slice-5 heavy: zero-overlap invariant + stretch parity (violin-bow)."""
    if not metal_device_available():
        pytest.skip("Metal device unavailable for mlx-spatialkit real Pixal3D export")
    fixture = (
        _repo_root() / "inputs" / "mlx-spatialkit" / "violin-bow"
        / "pixal3d-1024-cascade-decoded-pbr"
    )
    if not fixture.exists():
        pytest.skip(f"violin-bow Pixal3D decoded fixture not present: {fixture}")
    summary = _assert_parameterization_invariants_and_stretch_parity(fixture, "violin_bow")
    print(f"\n[violin parameterization invariants+parity] {summary}")
