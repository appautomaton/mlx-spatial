from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mlx_spatialkit import export_pixal3d_glb, metal_device_available
from mlx_spatialkit.export import (
    _export_quality_summary,
    _resolve_pixal3d_export_settings,
    _simplifier_backend_for_quality_preset,
)
from glb_texture_utils import glb_image_payload, png_coverage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    assert "not_browser_rendered_visual_proof" in visual["deferred_parity_boundaries"]
    visual_artifacts = visual["artifacts"]
    assert Path(visual_artifacts["report_json"]).is_file()
    assert Path(visual_artifacts["preview_html"]).is_file()
    assert Path(visual_artifacts["candidate_base_color_png"]).is_file()
    assert Path(visual_artifacts["reference_base_color_png"]).is_file()
    assert Path(visual_artifacts["report_json"]).parent == output_dir / "visual_parity"
    assert "after_write_glb" in diagnostics["memory_samples"]
    _assert_memory_diagnostics(diagnostics, required_stages=("texture_bake", "write_glb", "visual_compare"))


def test_export_pixal3d_glb_rejects_invalid_public_guards(tmp_path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    with pytest.raises(ValueError, match="grid_size must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", grid_size=0)

    with pytest.raises(ValueError, match="max_texture_pixels must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", max_texture_pixels=0)


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
