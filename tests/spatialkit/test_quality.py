from __future__ import annotations

from pathlib import Path

import pytest

from mlx_spatial.spatialkit import export_decoded_ovoxel_glb
from mlx_spatial.spatialkit.export import _resolve_ovoxel_export_settings
from mlx_spatial.spatialkit.quality import (
    summarize_ovoxel_export_quality,
    unavailable_production_equivalence,
)


def test_trellis2_production_defaults_do_not_use_pixal3d_reference_values(tmp_path: Path) -> None:
    settings = _resolve_ovoxel_export_settings(
        tmp_path,
        "trellis2",
        "reference-target",
        None,
    )

    assert settings["target_faces"] == 200_000
    assert settings["target_faces_source"] == "ovoxel_production_default"
    assert settings["reference"] is None
    assert settings["reference_profile"] is None


def test_model_without_reference_profile_reports_artifact_health_without_pixal3d_parity() -> None:
    summary = summarize_ovoxel_export_quality(
        "trellis2",
        {"backend": "mlx-qem", "quality_tier": "production"},
        {"export_blocking_reasons": ()},
        {"coverage_ratio": 1.0},
        None,
        quality_preset="reference-target",
        uv_stats={"backend": "xatlas-clustered"},
    )

    assert summary["artifact_ready"] is True
    assert summary["production_quality_ready"] is False
    assert summary["reference_profile"]["configured"] is False
    assert "upstream_export_settings" not in summary
    assert "xatlas_chart_parity" not in summary

    equivalence = unavailable_production_equivalence("trellis2", artifact_ready=True)
    assert equivalence["applicable"] is False
    assert equivalence["artifact_ready"] is True
    assert equivalence["blockers"] == ("model_reference_profile_unavailable",)


def test_parallel_xatlas_chunks_are_scoped_to_the_parallel_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires xatlas_parallel_chunks"):
        export_decoded_ovoxel_glb(
            tmp_path,
            tmp_path / "model.glb",
            uv_backend="xatlas-parallel-spatial",
        )

    with pytest.raises(ValueError, match="only applies"):
        export_decoded_ovoxel_glb(
            tmp_path,
            tmp_path / "model.glb",
            uv_backend="xatlas-clustered",
            xatlas_parallel_chunks=4,
        )


def test_model_neutral_quality_warns_about_nonblocking_boundary_gaps() -> None:
    summary = summarize_ovoxel_export_quality(
        "trellis2",
        {"backend": "mlx-qem", "quality_tier": "production"},
        {
            "export_blocking_reasons": (),
            "boundary_loop_count": 3,
            "boundary_small_loop_edge_count": 190,
        },
        {"coverage_ratio": 1.0},
        None,
        quality_preset="reference-target",
        uv_stats={"backend": "xatlas-clustered"},
    )

    assert summary["artifact_ready"] is True
    assert summary["topology_blocker_map"]["status"] == "rendered_visual_blocked"
    assert "topology_visual_gaps_present" in summary["warnings"]
