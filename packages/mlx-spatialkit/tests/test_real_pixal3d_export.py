from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mlx_spatialkit import export_pixal3d_glb, metal_device_available
from mlx_spatialkit.export import _export_quality_summary
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


def test_export_pixal3d_glb_rejects_invalid_public_guards(tmp_path) -> None:
    decoded_dir = tmp_path / "decoded"
    decoded_dir.mkdir()

    with pytest.raises(ValueError, match="grid_size must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", grid_size=0)

    with pytest.raises(ValueError, match="max_texture_pixels must be positive"):
        export_pixal3d_glb(decoded_dir, tmp_path / "out", max_texture_pixels=0)


def test_export_quality_summary_separates_artifact_and_production_readiness() -> None:
    summary = _export_quality_summary(
        {"backend": "spatial-cluster", "quality_tier": "geometry_aware_preview"},
        {"export_blocking_reasons": []},
    )

    assert summary["artifact_ready"] is True
    assert summary["production_quality_ready"] is False
    assert summary["simplifier_backend"] == "spatial-cluster"
    assert summary["simplifier_quality_tier"] == "geometry_aware_preview"
    assert "preview_simplifier_quality_tier" in summary["warnings"]

    blocked = _export_quality_summary(
        {"backend": "qem-edge-collapse", "quality_tier": "production"},
        {"export_blocking_reasons": ["nonmanifold_edges_present"]},
    )

    assert blocked["artifact_ready"] is False
    assert blocked["production_quality_ready"] is False
    assert blocked["export_blocking_reasons"] == ("nonmanifold_edges_present",)
    assert "export_blocking_reasons_present" in blocked["warnings"]
