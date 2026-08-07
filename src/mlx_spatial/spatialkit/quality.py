"""Model-aware quality summaries for decoded O-Voxel exports."""

from __future__ import annotations

from typing import Any

from .pixal3d_quality import (
    _export_quality_summary as _pixal3d_export_quality_summary,
    _normalize_quality_preset,
    _topology_blocker_map,
)


def summarize_ovoxel_export_quality(
    model_family: str,
    simplify_stats: dict[str, Any],
    export_metrics: dict[str, Any],
    texture_stats: dict[str, Any],
    reference: dict[str, Any] | None,
    *,
    quality_preset: str,
    uv_stats: dict[str, Any],
) -> dict[str, Any]:
    """Report artifact health without applying another model's reference profile."""

    if model_family == "pixal3d":
        summary = _pixal3d_export_quality_summary(
            simplify_stats,
            export_metrics,
            texture_stats,
            reference,
            quality_preset=quality_preset,
            uv_stats=uv_stats,
        )
        summary["reference_profile"] = {
            "configured": True,
            "model_family": "pixal3d",
            "profile": "pixal3d-upstream",
            "reference_artifact_available": reference is not None,
        }
        return summary

    preset = _normalize_quality_preset(quality_preset)
    blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    topology_blockers = _topology_blocker_map(simplify_stats, export_metrics)
    warnings: list[str] = []
    if preset == "preview":
        warnings.append("preview_quality_preset")
    if simplifier_quality != "production":
        warnings.append("preview_simplifier_quality_tier")
    if blockers:
        warnings.append("export_blocking_reasons_present")
    if topology_blockers.get("visual_blockers"):
        warnings.append("topology_visual_gaps_present")
    warnings.append("model_reference_profile_unavailable")
    reason = f"no production-equivalence profile is configured for model_family={model_family}"
    return {
        "artifact_ready": not blockers,
        "rendered_visual_ready": False,
        "production_quality_ready": False,
        "quality_preset": preset,
        "simplifier_backend": str(simplify_stats.get("backend", "unknown")),
        "simplifier_quality_tier": simplifier_quality,
        "reference_profile": {
            "configured": False,
            "model_family": model_family,
            "profile": None,
            "reason": reason,
        },
        "reference_stage_contract": {
            "applicable": False,
            "passed": False,
            "status": "unavailable",
            "reason": reason,
        },
        "topology_blocker_map": topology_blockers,
        "export_blocking_reasons": blockers,
        "production_thresholds": {
            "applicable": False,
            "all_passed": False,
            "checks": {},
            "reason": reason,
        },
        "warnings": tuple(warnings),
    }


def unavailable_production_equivalence(
    model_family: str,
    *,
    artifact_ready: bool,
) -> dict[str, Any]:
    """Return an explicit non-verdict when no model reference profile exists."""

    reason = f"no production-equivalence profile is configured for model_family={model_family}"
    blockers = [] if artifact_ready else ["artifact_not_ready"]
    blockers.append("model_reference_profile_unavailable")
    return {
        "applicable": False,
        "profile": None,
        "reason": reason,
        "ready": False,
        "artifact_ready": artifact_ready,
        "scalar_production_quality_ready": False,
        "reference_stage_contract_ready": False,
        "upstream_export_settings_ready": False,
        "xatlas_chart_parity_ready": False,
        "visual_comparison_available": False,
        "visual_comparison_ready": False,
        "remaining_parity_boundaries": (),
        "blockers": tuple(blockers),
        "checks": {},
    }


__all__ = [
    "summarize_ovoxel_export_quality",
    "unavailable_production_equivalence",
]
