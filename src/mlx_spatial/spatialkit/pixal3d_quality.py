"""Measured Pixal3D export quality and parity contracts.

This module contains verdict logic only.  It does not perform model inference,
mesh conversion, texture baking, rendering, or filesystem writes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD = 0.50
PIXAL3D_REFERENCE_FACE_RATIO_MIN = 0.80
PIXAL3D_REFERENCE_FACE_RATIO_MAX = 1.25
PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN = 0.50
PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN = 0.50
PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN = 0.50
PIXAL3D_FACE_ATLAS_TILE_PADDING = 0.08
PIXAL3D_NATIVE_CHART_TILE_PADDING = 0.001
PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN = 0.95
PIXAL3D_XATLAS_CHART_COUNT_RATIO_MIN = 0.80
PIXAL3D_XATLAS_CHART_COUNT_RATIO_MAX = 1.25
PIXAL3D_XATLAS_MAX_OVERLAP_FACE_RATIO = 0.01
PIXAL3D_XATLAS_MAX_UNASSIGNED_SURFACE_RATIO = 1.0e-5
PIXAL3D_XATLAS_MAX_DEGENERATE_SURFACE_RATIO = 1.0e-5

def _normalize_quality_preset(value: str) -> str:
    preset = str(value).strip().lower().replace("_", "-")
    if preset in ("production", "reference", "reference-target"):
        return "reference-target"
    if preset == "preview":
        return "preview"
    raise ValueError("quality_preset must be 'preview' or 'reference-target'")


def _resolve_pixal3d_uv_backend(value: str) -> str:
    backend = str(value).strip().lower().replace("_", "-")
    if backend in (
        "face-atlas",
        "native-chart",
        "xatlas-equivalent-native",
        "xatlas-global",
        "xatlas-clustered",
        "xatlas-parallel-spatial",
    ):
        return backend
    raise ValueError(
        "uv_backend must be 'face-atlas', 'native-chart', 'xatlas-equivalent-native', "
        "'xatlas-global', 'xatlas-clustered', or 'xatlas-parallel-spatial'"
    )


def _resolve_chart_angle_degrees(value: float) -> float:
    angle = float(value)
    if not np.isfinite(angle) or angle < 0.0 or angle > 180.0:
        raise ValueError("chart_angle_degrees must be finite and in [0, 180]")
    return angle


def _resolve_tile_padding(value: float | None, uv_backend: str) -> tuple[float, str]:
    backend = _resolve_pixal3d_uv_backend(uv_backend)
    if value is None:
        if backend == "xatlas-parallel-spatial":
            return 0.02, "backend_default:xatlas-parallel-spatial"
        if backend in ("xatlas-equivalent-native", "xatlas-global", "xatlas-clustered"):
            # The reference unwrap packs with texel gaps (xatlas PackOptions
            # bilinear gutter), not a fractional tile padding.
            return 0.0, f"backend_default:{backend}"
        if backend == "native-chart":
            return PIXAL3D_NATIVE_CHART_TILE_PADDING, "backend_default:native-chart"
        return PIXAL3D_FACE_ATLAS_TILE_PADDING, "backend_default:face-atlas"
    padding = float(value)
    if not np.isfinite(padding) or padding < 0.0 or padding >= 0.45:
        raise ValueError("tile_padding must be finite and in [0, 0.45)")
    return padding, "explicit"


def _resolve_simplify_backend(value: str | None) -> str | None:
    """Validate the explicit simplify_backend opt-in."""
    if value is None:
        return None
    backend = str(value).strip().lower()
    if backend in {"qem", "mlx-qem", "single-layer-qem", "single-layer-mlx-qem"}:
        return backend
    raise ValueError(
        "simplify_backend must be None, 'qem', 'mlx-qem', 'single-layer-qem', "
        f"or 'single-layer-mlx-qem'; got {value!r}"
    )


def _simplifier_backend_for_quality_preset(quality_preset: str) -> str:
    preset = _normalize_quality_preset(quality_preset)
    if preset == "reference-target":
        return "topology-aware"
    return "spatial-cluster"


def _export_quality_summary(
    simplify_stats: dict[str, Any],
    export_metrics: dict[str, Any],
    texture_stats: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    *,
    quality_preset: str = "preview",
    uv_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    simplifier_backend = str(simplify_stats.get("backend", "unknown"))
    preset = _normalize_quality_preset(quality_preset)
    reference_contract = _pixal3d_reference_stage_contract(
        simplify_stats,
        uv_stats or {},
        texture_stats or {},
        reference,
        quality_preset=preset,
    )
    thresholds = _production_thresholds(
        simplify_stats,
        export_metrics,
        texture_stats or {},
        reference,
        quality_preset=preset,
    )
    warnings: list[str] = []
    if preset == "preview":
        warnings.append("preview_quality_preset")
    if simplifier_quality != "production":
        warnings.append("preview_simplifier_quality_tier")
    if blockers:
        warnings.append("export_blocking_reasons_present")
    if not thresholds["all_passed"]:
        warnings.append("production_thresholds_failed")
    if preset == "reference-target" and not bool(reference_contract["passed"]):
        warnings.append("reference_stage_contract_incomplete")
    artifact_ready = len(blockers) == 0
    topology_blockers = _topology_blocker_map(simplify_stats, export_metrics)
    production_quality_ready = (
        artifact_ready
        and bool(thresholds["all_passed"])
        and bool(reference_contract["passed"])
    )
    return {
        "artifact_ready": artifact_ready,
        "rendered_visual_ready": False,
        "production_quality_ready": production_quality_ready,
        "quality_preset": preset,
        "simplifier_backend": simplifier_backend,
        "simplifier_quality_tier": simplifier_quality,
        "reference_stage_contract": reference_contract,
        "native_geometry_candidate": _native_geometry_candidate_status(simplify_stats, thresholds, preset),
        "topology_blocker_map": topology_blockers,
        "export_blocking_reasons": blockers,
        "production_thresholds": thresholds,
        "warnings": tuple(warnings),
    }


def _stage_status(
    *,
    passed: bool,
    status: str,
    source: str,
    spatialkit_backend: Any,
    required: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "status": status,
        "source": source,
        "spatialkit_backend": spatialkit_backend,
        "required": required,
        "detail": detail,
    }


# Maximum sparse-KNN fallback fraction for the trilinear sampling gate to stay
# reference_matched. Measured on real fixtures the fallback is ~2e-5; this bound
# fails the gate if fallback ever dominates the trilinear primary path.
_TRILINEAR_FALLBACK_FRACTION_MAX = 0.05


def _pixal3d_reference_stage_contract(
    simplify_stats: dict[str, Any],
    uv_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    reference: dict[str, Any] | None,
    *,
    quality_preset: str,
) -> dict[str, Any]:
    """Report whether the current export path satisfies the Pixal3D reference stages."""

    preset = _normalize_quality_preset(quality_preset)
    if preset != "reference-target":
        return {
            "status": "not_requested",
            "passed": None,
            "quality_preset": preset,
            "required_stage_names": (),
            "blockers": (),
            "heuristic_stage_names": (),
            "stages": {},
        }

    simplifier_backend = str(simplify_stats.get("backend", "unknown"))
    simplifier_algorithm = str(simplify_stats.get("algorithm", "unknown"))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    hole_fill_algorithm = str(simplify_stats.get("small_boundary_loop_fill_algorithm", "unknown"))
    hole_fill_fallback = str(simplify_stats.get("small_boundary_loop_fill_fallback_algorithm", "unknown"))
    pre_qem_hole_fill_algorithm = str(simplify_stats.get("pre_qem_hole_fill_algorithm", "unknown"))
    post_qem_hole_fill_algorithm = str(simplify_stats.get("post_qem_hole_fill_algorithm", "unknown"))
    uv_backend = str(uv_stats.get("backend", "unknown"))
    texture_backend = str(texture_stats.get("backend", "unknown"))
    sampling_mode = str(texture_stats.get("sampling_mode", "nearest"))
    source_projection_used = texture_stats.get("source_projection_used")
    source_projection_detail = texture_stats.get("source_projection_detail")
    seam_fill_mode = str(texture_stats.get("postprocess_mode", "native-dilation-and-surface-fill"))
    reference_available = reference is not None
    xatlas_backend = str(reference.get("unwrap_backend", "")) if reference is not None else ""

    reference_hole_fill_algorithms = {
        "perimeter-centroid-fan",
        "cumesh-perimeter-centroid-fan",
    }
    mlx_qem_hole_fill_sequence_present = (
        pre_qem_hole_fill_algorithm != "unknown" or post_qem_hole_fill_algorithm != "unknown"
    )
    if mlx_qem_hole_fill_sequence_present:
        hole_fill_reference = (
            pre_qem_hole_fill_algorithm in reference_hole_fill_algorithms
            and post_qem_hole_fill_algorithm in reference_hole_fill_algorithms
            and str(simplify_stats.get("pre_qem_hole_fill_backend", "unknown")) == "native-cpp"
            and str(simplify_stats.get("post_qem_hole_fill_backend", "unknown")) == "native-cpp"
            and abs(float(simplify_stats.get("pre_qem_hole_fill_max_perimeter", 0.0)) - 0.03) <= 1.0e-12
            and abs(float(simplify_stats.get("post_qem_hole_fill_max_perimeter", 0.0)) - 0.03) <= 1.0e-12
        )
    else:
        hole_fill_reference = hole_fill_algorithm in reference_hole_fill_algorithms
    remesh_reference = (
        str(simplify_stats.get("remesh_backend", ""))
        in {
            "native-udf-double-cover-dc",
            "cpu-narrow-band-udf-double-cover-dc",
        }
        and str(simplify_stats.get("remesh_surface_representation", ""))
        == "udf-offset-double-cover"
        and simplify_stats.get("remesh_reference_mechanism_ready") is True
    )
    simplify_reference = (
        simplifier_quality == "production"
        and ("qem" in simplifier_algorithm or "edge-collapse" in simplifier_algorithm)
    )
    actual_xatlas_backend = uv_backend in {
        "xatlas-global",
        "xatlas-clustered",
        "xatlas-parallel-spatial",
    }
    if actual_xatlas_backend:
        unwrap_reference = bool(
            _actual_xatlas_parity_summary(
                reference,
                uv_stats,
                texture_stats,
                uv_backend,
            )["parity_ready"]
        )
    else:
        # The native xatlas-equivalent path has stronger invariants than raw
        # xatlas: its own parameterizer normalizes chart orientation and its
        # packer promises zero positive-area overlap.
        unwrap_reference = (
            uv_backend == "xatlas-equivalent-native"
            and _maybe_int(uv_stats.get("uv_overlap_count")) == 0
            and _maybe_int(uv_stats.get("uv_flipped_count")) == 0
            and _maybe_int(uv_stats.get("lscm_unconverged_count")) == 0
        )
    raster_reference = bool(texture_stats.get("uv_raster_interpolate_reference"))
    projection_reference = source_projection_used is True

    # trilinear_pbr_sampling honest verdict (Codex P1-6): the primary measured
    # path is trilinear AND exact counter conservation holds AND the sparse-KNN
    # fallback fraction is bounded. Relabelling alone cannot flip the gate — a
    # fallback-dominated or non-conserving run fails even with mode "trilinear".
    trilinear_sampled = _maybe_int(texture_stats.get("trilinear_sampled_texel_count"))
    trilinear_fallback = _maybe_int(texture_stats.get("source_projection_nearest_fallback_texel_count"))
    trilinear_invalid = _maybe_int(texture_stats.get("trilinear_invalid_texel_count"))
    trilinear_input = _maybe_int(texture_stats.get("source_projection_input_texel_count"))
    _trilinear_counts = (trilinear_sampled, trilinear_fallback, trilinear_invalid, trilinear_input)
    trilinear_conserved = (
        all(value is not None for value in _trilinear_counts)
        and (trilinear_sampled + trilinear_fallback + trilinear_invalid) == trilinear_input
    )
    trilinear_fallback_fraction = (
        trilinear_fallback / trilinear_input
        if trilinear_conserved and trilinear_input
        else 1.0
    )
    sampling_reference = (
        sampling_mode == "trilinear"
        and projection_reference
        and trilinear_conserved
        and trilinear_sampled is not None
        and trilinear_sampled > 0
        and trilinear_fallback_fraction <= _TRILINEAR_FALLBACK_FRACTION_MAX
    )

    # texture_postprocess honest verdict (Codex P1-6): native Telea mode AND
    # legacy fills disabled and zero AND the reference channel radii AND the raw
    # inverse-coverage mask identity (painted texels == no_face+missing+out_of_grid).
    telea_mask = _maybe_int(texture_stats.get("telea_mask_texel_count"))
    telea_radius_base = _maybe_int(texture_stats.get("telea_radius_base_color_rgb"))
    telea_radius_alpha = _maybe_int(texture_stats.get("telea_radius_alpha"))
    telea_radius_mr = _maybe_int(texture_stats.get("telea_radius_metallic_roughness"))
    legacy_postprocess_applied = texture_stats.get("legacy_postprocess_applied")
    dilation_filled = _maybe_int(texture_stats.get("dilation_filled_texel_count"))
    no_face = _maybe_int(texture_stats.get("no_face_texel_count"))
    missing_surface = _maybe_int(texture_stats.get("missing_texel_count"))
    out_of_grid = _maybe_int(texture_stats.get("out_of_grid_texel_count"))
    _raw_inverse_coverage = (
        no_face + missing_surface + out_of_grid
        if all(value is not None for value in (no_face, missing_surface, out_of_grid))
        else None
    )
    telea_mask_identity = (
        telea_mask is not None
        and _raw_inverse_coverage is not None
        and telea_mask == _raw_inverse_coverage
    )
    postprocess_reference = (
        seam_fill_mode == "native-telea-inpaint"
        and legacy_postprocess_applied is False
        and dilation_filled == 0
        and telea_radius_base == 3
        and telea_radius_alpha == 1
        and telea_radius_mr == 1
        and telea_mask_identity
    )

    stages = {
        "reference_trace": _stage_status(
            passed=reference_available,
            status="available" if reference_available else "missing",
            source="inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json",
            spatialkit_backend=reference.get("trace_path") if reference is not None else None,
            required="reference Pixal3D GLB trace available",
            detail="Reference trace anchors face count, xatlas metrics, texture size, and visual comparison.",
        ),
        "decoded_npz_validation": _stage_status(
            passed=True,
            status="native_contract",
            source="native/spatialkit/cpp/pixal3d_contracts.cpp",
            spatialkit_backend="native_pixal3d_contracts",
            required="decoded shape and texture NPZ native validation",
            detail="Reaching export quality summary requires decoded arrays to pass native contract checks.",
        ),
        "flexi_dual_grid_extract": _stage_status(
            passed=True,
            status="native_port",
            source="vendors/TRELLIS.2/o-voxel/src/convert",
            spatialkit_backend="native_flexi_dual_grid",
            required="o-voxel compatible mesh extraction",
            detail="Mesh extraction is already a native spatialkit boundary.",
        ),
        "cumesh_hole_fill_cleanup": _stage_status(
            passed=hole_fill_reference,
            status="reference_matched" if hole_fill_reference else "heuristic_quarantined",
            source="CuMesh/src/clean_up.cu:450",
            spatialkit_backend={
                "algorithm": hole_fill_algorithm,
                "fallback_algorithm": hole_fill_fallback,
                "pre_qem_algorithm": pre_qem_hole_fill_algorithm,
                "pre_qem_backend": simplify_stats.get("pre_qem_hole_fill_backend"),
                "pre_qem_max_perimeter": simplify_stats.get("pre_qem_hole_fill_max_perimeter"),
                "post_qem_algorithm": post_qem_hole_fill_algorithm,
                "post_qem_backend": simplify_stats.get("post_qem_hole_fill_backend"),
                "post_qem_max_perimeter": simplify_stats.get("post_qem_hole_fill_max_perimeter"),
                "post_qem_residual_clean_boundary_loops": simplify_stats.get(
                    "post_qem_hole_fill_residual_clean_boundary_loops"
                ),
            },
            required="perimeter-limited centroid-fan boundary-loop fill",
            detail=(
                "The MLX-QEM route must match both CuMesh fill stages around final simplification; "
                "residual loops over the 0.03 perimeter limit remain reported as visual evidence."
            ),
        ),
        "narrow_band_dc_remesh": _stage_status(
            passed=remesh_reference,
            status="reference_matched" if remesh_reference else "behavior_matched_quality_blocked",
            source=(
                "vendors/TRELLIS.2/app.py:493 and "
                "vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:174"
            ),
            spatialkit_backend={
                "backend": simplify_stats.get("remesh_backend"),
                "surface_representation": simplify_stats.get("remesh_surface_representation"),
                "single_surface_ready": simplify_stats.get("remesh_single_surface_ready"),
                "reference_mechanism_ready": simplify_stats.get(
                    "remesh_reference_mechanism_ready"
                ),
            },
            required="official UDF-epsilon narrow-band dual-contouring remesh mechanism",
            detail=(
                "The reference application explicitly exports with remesh=True, band=1, and "
                "project_back=0. Its UDF-epsilon offset shell is not single-layer, but that fact "
                "is descriptive and is not itself a reference-parity blocker."
            ),
        ),
        "qem_simplification": _stage_status(
            passed=simplify_reference,
            status="reference_matched" if simplify_reference else "heuristic_quarantined",
            source="CuMesh/src/simplify.cu:531",
            spatialkit_backend={
                "backend": simplifier_backend,
                "algorithm": simplifier_algorithm,
                "quality_tier": simplifier_quality,
            },
            required="QEM-like edge-collapse simplification",
            detail="Topology-aware clustering remains non-reference until the simplifier is QEM-like or explicitly proven equivalent.",
        ),
        "xatlas_unwrap": _stage_status(
            passed=unwrap_reference,
            status="reference_matched" if unwrap_reference else "heuristic_quarantined",
            source="CuMesh/cumesh/cumesh.py:408",
            spatialkit_backend={
                "uv_backend": uv_backend,
                "chart_cluster_normal_policy": uv_stats.get("chart_cluster_normal_policy"),
                "requires_xatlas_dependency": actual_xatlas_backend,
            },
            required="measured xatlas/CuMesh behavior-compatible unwrap",
            detail=(
                f"Reference unwrap backend is {xatlas_backend or 'unknown'}; the selected backend must pass "
                "its backend-specific chart, coverage, seam, and surface-area checks."
            ),
        ),
        "uv_raster_interpolate": _stage_status(
            passed=raster_reference,
            status="reference_matched" if raster_reference else "behavior_gap",
            source="vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:229",
            spatialkit_backend=texture_backend,
            required="UV-space rasterize and barycentric interpolation equivalent to nvdiffrast behavior",
            detail="Metal raster/interpolate must be behavior-equivalent without copying nvdiffrast CUDA code.",
        ),
        "original_mesh_bvh_projection": _stage_status(
            passed=projection_reference,
            status="reference_matched" if projection_reference else "missing_or_deferred",
            source="vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:252",
            spatialkit_backend=source_projection_detail,
            required="project UV-sampled positions back to original high-resolution mesh before voxel sampling",
            detail="This is the main guard against texture smear after simplification or remeshing.",
        ),
        "trilinear_pbr_sampling": _stage_status(
            passed=sampling_reference,
            status="reference_matched" if sampling_reference else "heuristic_quarantined",
            source="vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:258",
            spatialkit_backend={
                "sampling_mode": sampling_mode,
                "sampling_fallback_policy": texture_stats.get("sampling_fallback_policy"),
                "trilinear_sampled_texel_count": trilinear_sampled,
                "nearest_fallback_texel_count": trilinear_fallback,
                "trilinear_invalid_texel_count": trilinear_invalid,
                "counter_conservation": trilinear_conserved,
                "fallback_fraction": trilinear_fallback_fraction,
                "fallback_fraction_max": _TRILINEAR_FALLBACK_FRACTION_MAX,
            },
            required="trilinear sparse-grid PBR voxel sampling (bounded sparse-KNN fallback)",
            detail="reference_matched requires sampling_mode 'trilinear', exact texel counter conservation, and a bounded fallback fraction; nearest-only or fallback-dominated sampling stays quarantined.",
        ),
        "texture_postprocess": _stage_status(
            passed=postprocess_reference,
            status="reference_matched" if postprocess_reference else "heuristic_quarantined",
            source="vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:287",
            spatialkit_backend={
                "postprocess_mode": seam_fill_mode,
                "legacy_postprocess_applied": legacy_postprocess_applied,
                "dilation_filled_texel_count": dilation_filled,
                "telea_mask_texel_count": telea_mask,
                "raw_inverse_coverage_texel_count": _raw_inverse_coverage,
                "telea_mask_identity": telea_mask_identity,
                "telea_radii": {
                    "base_color_rgb": telea_radius_base,
                    "alpha": telea_radius_alpha,
                    "metallic_roughness": telea_radius_mr,
                },
            },
            required="reference Telea inpaint over the raw inverse-coverage mask with disabled legacy fills",
            detail="reference_matched requires postprocess_mode 'native-telea-inpaint', legacy fills disabled (zero), reference channel radii (base 3; alpha/metallic/roughness 1), and painted-texels == raw inverse-coverage mask identity. Legacy dilation/BFS/gutter fill stays quarantined.",
        ),
    }
    required_stage_names = tuple(stages)
    blockers = tuple(name for name in required_stage_names if not bool(stages[name]["passed"]))
    heuristic_stage_names = tuple(
        name
        for name in required_stage_names
        if str(stages[name]["status"]) in {"heuristic_quarantined", "behavior_gap"}
    )
    return {
        "status": "reference_ready" if not blockers else "blocked",
        "passed": not blockers,
        "quality_preset": preset,
        "policy": "Pixal3D production quality requires every reference-critical export stage to pass or be proven equivalent.",
        "required_stage_names": required_stage_names,
        "blockers": blockers,
        "heuristic_stage_names": heuristic_stage_names,
        "stages": stages,
    }


def _native_geometry_candidate_status(
    simplify_stats: dict[str, Any],
    thresholds: dict[str, Any],
    quality_preset: str,
) -> dict[str, Any]:
    checks = thresholds.get("checks", {})
    backend_check = checks.get("backend_tier", {})
    face_check = checks.get("face_count_ratio", {})
    topology_check = checks.get("topology_exportability", {})
    if quality_preset != "reference-target":
        return {
            "status": "not_requested",
            "reason": "quality_preset_is_preview",
            "current_backend": simplify_stats.get("backend"),
            "current_quality_tier": simplify_stats.get("quality_tier"),
            "requested_backend": simplify_stats.get("requested_backend"),
            "backend_selection_status": simplify_stats.get("backend_selection_status"),
        }
    if bool(backend_check.get("passed")):
        return {
            "status": "candidate",
            "reason": "native_geometry_candidate_available",
            "current_backend": simplify_stats.get("backend"),
            "current_quality_tier": simplify_stats.get("quality_tier"),
            "requested_backend": simplify_stats.get("requested_backend"),
            "backend_selection_status": simplify_stats.get("backend_selection_status"),
            "face_count_ratio": face_check.get("actual"),
            "topology_exportability_passed": bool(topology_check.get("passed")),
        }
    return {
        "status": "blocked",
        "reason": "native_geometry_candidate_blocked",
        "detail": "reference-target export still uses a preview-tier native simplifier",
        "current_backend": simplify_stats.get("backend"),
        "current_quality_tier": simplify_stats.get("quality_tier"),
        "requested_backend": simplify_stats.get("requested_backend"),
        "backend_selection_status": simplify_stats.get("backend_selection_status"),
        "face_count_ratio": face_check.get("actual"),
        "topology_exportability_passed": bool(topology_check.get("passed")),
    }


def _topology_blocker_map(simplify_stats: dict[str, Any], export_metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify topology gaps from numeric diagnostics instead of screenshots."""

    artifact_blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    nonmanifold_edges = _maybe_int(export_metrics.get("nonmanifold_edges")) or 0
    closed_loops = _maybe_int(export_metrics.get("boundary_loop_count")) or 0
    closed_loop_edges = _maybe_int(export_metrics.get("boundary_small_loop_edge_count")) or 0
    simple_open_chains = _maybe_int(export_metrics.get("boundary_simple_open_chain_count")) or 0
    simple_open_chain_edges = 0
    if simple_open_chains:
        simple_open_chain_edges = _maybe_int(export_metrics.get("boundary_open_chain_edge_count")) or 0
    branched_open_chains = _maybe_int(export_metrics.get("boundary_branched_open_chain_count")) or 0
    branched_branch_vertices = _maybe_int(export_metrics.get("boundary_open_chain_branch_vertex_count")) or 0
    production_blockers = tuple(str(item) for item in simplify_stats.get("production_blockers", ()))
    qem_missing = (
        "missing_qem_edge_collapse_simplification" in production_blockers
        or str(simplify_stats.get("qem_simplification_backend")) == "not_implemented"
        or str(simplify_stats.get("qem_equivalence_status")) in {"qem_scored_not_edge_collapse", "blocked_missing_qem"}
    )
    narrow_band_missing = (
        "missing_narrow_band_dc_remesh" in production_blockers
        or str(simplify_stats.get("remesh_backend")) == "not_implemented"
        or str(simplify_stats.get("remesh_equivalence_status")) == "blocked_missing_narrow_band_dc"
    )

    visual_blockers = []
    if closed_loops > 0:
        visual_blockers.append("clean_closed_boundary_loops")
    if simple_open_chains > 0:
        visual_blockers.append("simple_open_boundary_chains")
    if branched_open_chains > 0:
        visual_blockers.append("branched_open_boundary_chains")

    production_backend_blockers = []
    if qem_missing:
        production_backend_blockers.append("missing_qem_edge_collapse_simplification")
    if narrow_band_missing:
        production_backend_blockers.append("missing_narrow_band_dc_remesh")

    if artifact_blockers or nonmanifold_edges > 0:
        status = "artifact_blocked"
    elif visual_blockers:
        status = "rendered_visual_blocked"
    elif production_backend_blockers:
        status = "production_backend_blocked"
    else:
        status = "topology_clear"

    return {
        "status": status,
        "diagnostic_source": "stages.export_metrics.metrics plus stages.simplify_mesh.stats",
        "artifact_blockers": artifact_blockers,
        "visual_blockers": tuple(visual_blockers),
        "production_backend_blockers": tuple(production_backend_blockers),
        "classes": {
            "clean_closed_boundary_loops": {
                "present": closed_loops > 0,
                "count": closed_loops,
                "small_loop_edge_count": closed_loop_edges,
                "export_blocking": False,
            },
            "simple_open_chains": {
                "present": simple_open_chains > 0,
                "count": simple_open_chains,
                "edge_count": simple_open_chain_edges,
                "export_blocking": False,
            },
            "branched_open_chains": {
                "present": branched_open_chains > 0,
                "count": branched_open_chains,
                "branch_vertex_count": branched_branch_vertices,
                "export_blocking": False,
            },
            "nonmanifold_edges": {
                "present": nonmanifold_edges > 0,
                "count": nonmanifold_edges,
                "export_blocking": nonmanifold_edges > 0 or "nonmanifold_edges_present" in artifact_blockers,
            },
            "heuristic_qem": {
                "present": qem_missing,
                "backend": simplify_stats.get("qem_simplification_backend"),
                "equivalence_status": simplify_stats.get("qem_equivalence_status"),
                "export_blocking": False,
                "production_blocking": qem_missing,
            },
            "missing_narrow_band_remesh": {
                "present": narrow_band_missing,
                "backend": simplify_stats.get("remesh_backend"),
                "equivalence_status": simplify_stats.get("remesh_equivalence_status"),
                "export_blocking": False,
                "production_blocking": narrow_band_missing,
            },
        },
    }


def _native_chart_uv_candidate_status(
    uv_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    uv_backend: str,
) -> dict[str, Any]:
    uv_stats_backend = str(uv_stats.get("backend", "unknown"))
    texture_backend = str(texture_stats.get("backend", "unknown"))
    if uv_backend != "native-chart":
        return {
            "status": "not_requested",
            "artifact_ready": None,
            "quality_ready": None,
            "requested_uv_backend": uv_backend,
            "uv_backend": uv_stats_backend,
            "texture_bake_backend": texture_backend,
            "checks": {},
            "quality_blockers": (),
            "xatlas_chart_parity": False,
        }
    chart_count = _maybe_int(uv_stats.get("chart_count"))
    sampled_texels = _maybe_int(texture_stats.get("sampled_texel_count"))
    uv_bin_references = _maybe_int(texture_stats.get("uv_bin_face_reference_count"))
    uv_bin_guard_passed = bool(texture_stats.get("uv_bin_guard_passed"))
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    uv_surface_visible = _maybe_float(texture_stats.get("uv_surface_final_visible_coverage_ratio"))
    uv_surface_exact = _maybe_float(texture_stats.get("uv_surface_exact_coverage_ratio"))
    raw_coverage = _maybe_float(texture_stats.get("raw_coverage_ratio"))
    surface_filled_texels = _maybe_int(texture_stats.get("surface_filled_texel_count"))
    surface_unfilled_texels = _maybe_int(texture_stats.get("surface_unfilled_texel_count"))
    texture_pixel_count = _maybe_int(texture_stats.get("texture_pixel_count"))
    uv_surface_texel_count = _maybe_int(texture_stats.get("uv_surface_texel_count"))
    chart_rect_fill_ratio = _maybe_float(uv_stats.get("chart_rect_fill_ratio"))
    atlas_rect_coverage_ratio = _maybe_float(uv_stats.get("atlas_rect_coverage_ratio"))
    shelf_packing_efficiency = _maybe_float(uv_stats.get("shelf_packing_efficiency"))
    duplicated_vertex_ratio = _maybe_float(uv_stats.get("duplicated_vertex_ratio"))
    chart_cluster_normal_policy = str(uv_stats.get("chart_cluster_normal_policy", "unknown"))
    uv_surface_occupancy = None
    if texture_pixel_count not in (None, 0) and uv_surface_texel_count is not None:
        uv_surface_occupancy = float(uv_surface_texel_count) / float(texture_pixel_count)

    checks = {
        "chart_backend": {
            "passed": uv_stats_backend == "native-chart-atlas",
            "actual": uv_stats_backend,
            "required": "native-chart-atlas",
        },
        "texture_backend": {
            "passed": texture_backend == "metal-uv-binned-nearest",
            "actual": texture_backend,
            "required": "metal-uv-binned-nearest",
        },
        "chart_count": {
            "passed": chart_count is not None and chart_count > 0,
            "actual": chart_count,
            "required": ">0",
        },
        "sampled_texels": {
            "passed": sampled_texels is not None and sampled_texels > 0,
            "actual": sampled_texels,
            "required": ">0",
        },
        "uv_bin_guard": {
            "passed": uv_bin_guard_passed,
            "actual": uv_bin_guard_passed,
            "required": True,
        },
        "uv_bin_references": {
            "passed": uv_bin_references is not None and uv_bin_references > 0,
            "actual": uv_bin_references,
            "required": ">0",
        },
        "global_coverage_floor": {
            "passed": final_coverage is not None and final_coverage >= PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN,
            "actual": final_coverage,
            "required_min": PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN,
        },
        "uv_surface_occupancy_floor": {
            "passed": uv_surface_occupancy is not None
            and uv_surface_occupancy >= PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN,
            "actual": uv_surface_occupancy,
            "required_min": PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN,
        },
        "uv_surface_visible_floor": {
            "passed": uv_surface_visible is not None and uv_surface_visible >= PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN,
            "actual": uv_surface_visible,
            "required_min": PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN,
        },
    }
    artifact_check_names = (
        "chart_backend",
        "texture_backend",
        "chart_count",
        "sampled_texels",
        "uv_bin_guard",
        "uv_bin_references",
    )
    quality_check_names = (
        "global_coverage_floor",
        "uv_surface_occupancy_floor",
        "uv_surface_visible_floor",
    )
    artifact_ready = all(bool(checks[name]["passed"]) for name in artifact_check_names)
    quality_ready = artifact_ready and all(bool(checks[name]["passed"]) for name in quality_check_names)
    quality_blockers = tuple(name for name in quality_check_names if not bool(checks[name]["passed"]))
    artifact_blockers = tuple(name for name in artifact_check_names if not bool(checks[name]["passed"]))
    if not artifact_ready:
        status = "artifact_blocked"
    elif quality_ready:
        status = "quality_ready"
    else:
        status = "quality_blocked"
    return {
        "status": status,
        "artifact_ready": artifact_ready,
        "quality_ready": quality_ready,
        "requested_uv_backend": uv_backend,
        "uv_backend": uv_stats_backend,
        "texture_bake_backend": texture_backend,
        "chart_count": _maybe_int(uv_stats.get("chart_count")),
        "output_vertices": _maybe_int(uv_stats.get("output_vertices")),
        "output_faces": _maybe_int(uv_stats.get("output_faces")),
        "duplicated_vertex_ratio": duplicated_vertex_ratio,
        "global_coverage_ratio": final_coverage,
        "raw_coverage_ratio": raw_coverage,
        "uv_surface_occupancy_ratio": uv_surface_occupancy,
        "uv_surface_exact_coverage_ratio": uv_surface_exact,
        "uv_surface_final_visible_coverage_ratio": uv_surface_visible,
        "uv_surface_texel_count": uv_surface_texel_count,
        "surface_filled_texel_count": surface_filled_texels,
        "surface_unfilled_texel_count": surface_unfilled_texels,
        "texture_pixel_count": texture_pixel_count,
        "uv_bin_face_reference_count": _maybe_int(texture_stats.get("uv_bin_face_reference_count")),
        "uv_bin_max_candidate_faces": _maybe_int(texture_stats.get("uv_bin_max_candidate_faces")),
        "native_behavior_diagnostics": {
            "policy": "CuMesh/xatlas behavior metrics without required xatlas dependency",
            "requires_xatlas_dependency": False,
            "cluster_normal_policy": chart_cluster_normal_policy,
            "chart_cone_half_angle_degrees": _maybe_float(uv_stats.get("chart_cone_half_angle_degrees")),
            "chart_edge_rejected_adjacency_count": _maybe_int(uv_stats.get("chart_edge_rejected_adjacency_count")),
            "chart_cone_rejected_adjacency_count": _maybe_int(uv_stats.get("chart_cone_rejected_adjacency_count")),
            "packing": str(uv_stats.get("packing", "unknown")),
            "chart_rect_fill_ratio": chart_rect_fill_ratio,
            "atlas_rect_coverage_ratio": atlas_rect_coverage_ratio,
            "shelf_packing_efficiency": shelf_packing_efficiency,
            "duplicated_vertex_ratio": duplicated_vertex_ratio,
            "seam_island_risk": {
                "status": "measured",
                "seam_proxy": "duplicated seam vertices plus chart count",
                "island_proxy": "chart count, chart rect fill, atlas rect coverage, and UV surface occupancy",
            },
        },
        "checks": checks,
        "artifact_blockers": artifact_blockers,
        "quality_blockers": quality_blockers,
        "xatlas_chart_parity": False,
    }


# Preview-scale parity floors for the reference unwrap backend, anchored to
# the pip-xatlas oracle (tests/data/uv_oracle_anchors.json, xatlas 0.0.11):
# composition uv_bbox_utilization 0.534 (main) / 0.604 (violin); the floor is
# 0.55x the lower anchor (the documented rect-shelf vs rasterized packing
# gap). Stretch floor guards against a degenerate all-sentinel atlas.
PIXAL3D_REFERENCE_UNWRAP_UTILIZATION_FLOOR = 0.29
PIXAL3D_REFERENCE_UNWRAP_STRETCH_MIN = 0.5


def _reference_unwrap_parity_summary(
    reference: dict[str, Any] | None,
    uv_stats: dict[str, Any],
    uv_backend: str,
) -> dict[str, Any]:
    """Computed parity verdict for the xatlas-equivalent-native backend.

    parity_ready is MEASURED, never assumed: the backend must identify
    itself, the packed atlas must be overlap-free and flip-free with a
    converged LSCM pass, and utilization/stretch must clear the recorded
    preview-scale floors. A deliberately bad atlas keeps parity_ready False
    regardless of the backend name (SPEC UVU-06 anti-gaming).
    """

    uv_stats_backend = str(uv_stats.get("backend", "unknown"))
    overlap = _maybe_int(uv_stats.get("uv_overlap_count"))
    flipped = _maybe_int(uv_stats.get("uv_flipped_count"))
    unconverged = _maybe_int(uv_stats.get("lscm_unconverged_count"))
    chart_count = _maybe_int(uv_stats.get("chart_count"))
    cluster_count = _maybe_int(uv_stats.get("stage_a_cluster_count"))
    utilization = _maybe_float(uv_stats.get("uv_bbox_utilization"))
    stretch_l2 = _maybe_float(uv_stats.get("uv_stretch_l2"))
    checks = {
        "reference_unwrap_backend": {
            "passed": uv_stats_backend == "xatlas-equivalent-native",
            "actual": uv_stats_backend,
            "required": "xatlas-equivalent-native",
        },
        "uv_overlap_free": {
            "passed": overlap == 0,
            "actual": overlap,
            "required": "==0",
        },
        "uv_flip_free": {
            "passed": flipped == 0,
            "actual": flipped,
            "required": "==0",
        },
        "lscm_converged": {
            "passed": unconverged == 0,
            "actual": unconverged,
            "required": "==0",
        },
        "charts_present": {
            "passed": chart_count is not None and chart_count > 0
            and cluster_count is not None and cluster_count > 0,
            "actual": {"chart_count": chart_count, "stage_a_cluster_count": cluster_count},
            "required": ">0",
        },
        "utilization_floor": {
            "passed": utilization is not None
            and utilization >= PIXAL3D_REFERENCE_UNWRAP_UTILIZATION_FLOOR,
            "actual": utilization,
            "required": f">={PIXAL3D_REFERENCE_UNWRAP_UTILIZATION_FLOOR}",
        },
        "stretch_measured": {
            "passed": stretch_l2 is not None
            and math.isfinite(stretch_l2)
            and stretch_l2 >= PIXAL3D_REFERENCE_UNWRAP_STRETCH_MIN,
            "actual": stretch_l2,
            "required": f"finite, >={PIXAL3D_REFERENCE_UNWRAP_STRETCH_MIN}",
        },
    }
    parity_ready = all(bool(check["passed"]) for check in checks.values())
    # Reference-trace ratios are informational only: the trace is the
    # full-scale Pixal3D model (212k faces), not the preview-target mesh.
    reference_info = None
    if reference is not None:
        reference_chart_count = _maybe_int(reference.get("unwrap_chart_count"))
        chart_count_ratio = None
        if chart_count is not None and reference_chart_count not in (None, 0):
            chart_count_ratio = float(chart_count) / float(reference_chart_count)
        reference_info = {
            "unwrap_backend": reference.get("unwrap_backend"),
            "unwrap_chart_count": reference_chart_count,
            "chart_count_ratio_informational": chart_count_ratio,
            "scale_note": "reference trace is full-scale; ratios are informational only",
        }
    return {
        "status": "reference_parity_measured" if parity_ready else "reference_parity_failed",
        "reason": "measured_invariants" if parity_ready else "measured_invariant_failed",
        "parity_ready": parity_ready,
        "xatlas_chart_parity": parity_ready,
        "deferred_boundary": None if parity_ready else "not_xatlas_chart_parity",
        "requested_uv_backend": uv_backend,
        "native": {
            "uv_backend": uv_stats_backend,
            "chart_count": chart_count,
            "stage_a_cluster_count": cluster_count,
            "uv_bbox_utilization": utilization,
            "uv_stretch_l2": stretch_l2,
        },
        "reference": reference_info,
        "ratios": {},
        "deficits": {},
        "checks": checks,
    }


def _xatlas_chart_parity_summary(
    reference: dict[str, Any] | None,
    uv_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    uv_backend: str,
) -> dict[str, Any]:
    uv_stats_backend = str(uv_stats.get("backend", "unknown"))
    deferred_boundary = "not_xatlas_chart_parity"
    if uv_backend == "xatlas-equivalent-native":
        return _reference_unwrap_parity_summary(reference, uv_stats, uv_backend)
    if uv_backend in {"xatlas-global", "xatlas-clustered", "xatlas-parallel-spatial"}:
        return _actual_xatlas_parity_summary(reference, uv_stats, texture_stats, uv_backend)
    if uv_backend != "native-chart":
        return {
            "status": "not_requested",
            "reason": "uv_backend_is_not_native_chart",
            "parity_ready": None,
            "xatlas_chart_parity": False,
            "deferred_boundary": deferred_boundary,
            "requested_uv_backend": uv_backend,
            "native": {
                "uv_backend": uv_stats_backend,
            },
            "reference": None,
            "ratios": {},
            "deficits": {},
            "checks": {},
        }

    native_chart_count = _maybe_int(uv_stats.get("chart_count"))
    native_texture_pixels = _maybe_int(texture_stats.get("texture_pixel_count"))
    native_uv_surface_texels = _maybe_int(texture_stats.get("uv_surface_texel_count"))
    native_uv_surface_occupancy = None
    if native_texture_pixels not in (None, 0) and native_uv_surface_texels is not None:
        native_uv_surface_occupancy = float(native_uv_surface_texels) / float(native_texture_pixels)

    if reference is None:
        checks = {
            "reference_xatlas_available": {
                "passed": False,
                "actual": None,
                "required": "reference trace with xatlas unwrap metrics",
            },
            "native_chart_backend": {
                "passed": uv_stats_backend == "native-chart-atlas",
                "actual": uv_stats_backend,
                "required": "native-chart-atlas",
            },
        }
        return {
            "status": "reference_missing",
            "reason": "reference_xatlas_metrics_missing",
            "parity_ready": False,
            "xatlas_chart_parity": False,
            "deferred_boundary": deferred_boundary,
            "requested_uv_backend": uv_backend,
            "native": {
                "uv_backend": uv_stats_backend,
                "chart_count": native_chart_count,
                "uv_surface_occupancy_ratio": native_uv_surface_occupancy,
            },
            "reference": None,
            "ratios": {},
            "deficits": {},
            "checks": checks,
        }

    reference_backend = reference.get("unwrap_backend")
    reference_chart_count = _maybe_int(reference.get("unwrap_chart_count"))
    reference_utilization = _maybe_float(reference.get("unwrap_utilization"))
    reference_texture_size = _maybe_int(reference.get("texture_size"))
    reference_is_xatlas = isinstance(reference_backend, str) and reference_backend.startswith("xatlas")
    chart_count_ratio = None
    if native_chart_count is not None and reference_chart_count not in (None, 0):
        chart_count_ratio = float(native_chart_count) / float(reference_chart_count)
    utilization_ratio = None
    if native_uv_surface_occupancy is not None and reference_utilization not in (None, 0.0):
        utilization_ratio = native_uv_surface_occupancy / float(reference_utilization)
    utilization_gap = None
    utilization_ratio_gap = None
    utilization_equivalence_gap = None
    if native_uv_surface_occupancy is not None and reference_utilization is not None:
        utilization_gap = max(0.0, float(reference_utilization) - native_uv_surface_occupancy)
    if utilization_ratio is not None:
        utilization_ratio_gap = max(0.0, 1.0 - utilization_ratio)
        utilization_equivalence_gap = max(0.0, PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN - utilization_ratio)

    checks = {
        "reference_xatlas_backend": {
            "passed": reference_is_xatlas,
            "actual": reference_backend,
            "required": "xatlas*",
        },
        "reference_chart_count": {
            "passed": reference_chart_count is not None and reference_chart_count > 0,
            "actual": reference_chart_count,
            "required": ">0",
        },
        "reference_utilization": {
            "passed": reference_utilization is not None and reference_utilization > 0.0,
            "actual": reference_utilization,
            "required": ">0",
        },
        "native_chart_backend": {
            "passed": uv_stats_backend == "native-chart-atlas",
            "actual": uv_stats_backend,
            "required": "native-chart-atlas",
        },
        "native_chart_count": {
            "passed": native_chart_count is not None and native_chart_count > 0,
            "actual": native_chart_count,
            "required": ">0",
        },
        "native_uv_surface_occupancy": {
            "passed": native_uv_surface_occupancy is not None and native_uv_surface_occupancy > 0.0,
            "actual": native_uv_surface_occupancy,
            "required": ">0",
        },
        "xatlas_backend_equivalence": {
            "passed": False,
            "actual": uv_stats_backend,
            "required": reference_backend,
        },
        "xatlas_utilization_equivalence": {
            "passed": utilization_ratio is not None
            and utilization_ratio >= PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN,
            "actual": utilization_ratio,
            "required": f">={PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN}",
            "native_uv_surface_occupancy_ratio": native_uv_surface_occupancy,
            "reference_unwrap_utilization": reference_utilization,
        },
    }
    measurement_check_names = (
        "reference_xatlas_backend",
        "reference_chart_count",
        "reference_utilization",
        "native_chart_backend",
        "native_chart_count",
        "native_uv_surface_occupancy",
    )
    measurement_ready = all(bool(checks[name]["passed"]) for name in measurement_check_names)
    return {
        "status": "measured_not_equivalent" if measurement_ready else "measurement_incomplete",
        "reason": "native_chart_backend_is_not_xatlas",
        "measurement_ready": measurement_ready,
        "parity_ready": False,
        "xatlas_chart_parity": False,
        "deferred_boundary": deferred_boundary,
        "requested_uv_backend": uv_backend,
        "reference": {
            "unwrap_backend": reference_backend,
            "unwrap_chart_count": reference_chart_count,
            "unwrap_utilization": reference_utilization,
            "texture_size": reference_texture_size,
        },
        "native": {
            "uv_backend": uv_stats_backend,
            "chart_count": native_chart_count,
            "chart_rect_fill_ratio": _maybe_float(uv_stats.get("chart_rect_fill_ratio")),
            "uv_surface_occupancy_ratio": native_uv_surface_occupancy,
            "texture_size": _maybe_int(texture_stats.get("texture_size")),
        },
        "ratios": {
            "chart_count_ratio": chart_count_ratio,
            "uv_surface_occupancy_vs_reference_utilization": utilization_ratio,
        },
        "deficits": {
            "reference_utilization_minus_native_uv_surface_occupancy": utilization_gap,
            "uv_surface_occupancy_ratio_gap_to_reference": utilization_ratio_gap,
            "uv_surface_occupancy_ratio_gap_to_equivalence_target": utilization_equivalence_gap,
            "equivalence_target_ratio": PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN,
        },
        "checks": checks,
    }


def _actual_xatlas_parity_summary(
    reference: dict[str, Any] | None,
    uv_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    uv_backend: str,
) -> dict[str, Any]:
    """Measure real xatlas output without applying native-packer invariants.

    xatlas can mirror complete charts and, with its reference padding of zero,
    can report a small number of positive-area overlap pairs. The committed
    xatlas 0.0.11 oracle records both behaviors, so correctness is bounded by
    reference ratios and affected surface area instead of a false zero-count
    requirement.
    """

    uv_stats_backend = str(uv_stats.get("backend", "unknown"))
    reference_backend = str(reference.get("unwrap_backend", "")) if reference is not None else ""
    reference_chart_count = _maybe_int(reference.get("unwrap_chart_count")) if reference is not None else None
    reference_utilization = _maybe_float(reference.get("unwrap_utilization")) if reference is not None else None
    chart_count = _maybe_int(uv_stats.get("chart_count"))
    utilization = _maybe_float(uv_stats.get("atlas_utilization"))
    face_count = _maybe_int(uv_stats.get("source_faces"))
    overlap_count = _maybe_int(uv_stats.get("uv_overlap_count"))
    flipped_count = _maybe_int(uv_stats.get("uv_flipped_count"))
    unassigned_surface_ratio = _maybe_float(uv_stats.get("unassigned_surface_area_ratio"))
    degenerate_surface_ratio = _maybe_float(uv_stats.get("uv_degenerate_surface_area_ratio"))
    surface_exact_coverage = _maybe_float(texture_stats.get("uv_surface_exact_coverage_ratio"))
    chart_count_ratio = (
        float(chart_count) / float(reference_chart_count)
        if chart_count is not None and reference_chart_count not in (None, 0)
        else None
    )
    utilization_ratio = (
        float(utilization) / float(reference_utilization)
        if utilization is not None and reference_utilization not in (None, 0.0)
        else None
    )
    overlap_face_ratio = (
        float(overlap_count) / float(face_count)
        if overlap_count is not None and face_count not in (None, 0)
        else None
    )
    flipped_face_ratio = (
        float(flipped_count) / float(face_count)
        if flipped_count is not None and face_count not in (None, 0)
        else None
    )
    partition_shared_edges = _maybe_int(uv_stats.get("spatial_partition_shared_edge_count"))
    partition_cut_edges = _maybe_int(uv_stats.get("spatial_partition_cut_edge_count"))
    partition_cut_ratio = _maybe_float(uv_stats.get("spatial_partition_cut_edge_ratio"))
    partition_cuts_reported = (
        uv_backend != "xatlas-parallel-spatial"
        or (
            partition_shared_edges is not None
            and partition_cut_edges is not None
            and partition_cut_ratio is not None
            and 0 <= partition_cut_edges <= partition_shared_edges
            and 0.0 <= partition_cut_ratio <= 1.0
        )
    )
    checks = {
        "actual_xatlas_backend": {
            "passed": uv_stats_backend == uv_backend,
            "actual": uv_stats_backend,
            "required": uv_backend,
        },
        "reference_xatlas_backend": {
            "passed": reference_backend.startswith("xatlas"),
            "actual": reference_backend or None,
            "required": "xatlas*",
        },
        "reference_xatlas_version": {
            "passed": str(uv_stats.get("xatlas_version", "")) == "0.0.11",
            "actual": uv_stats.get("xatlas_version"),
            "required": "0.0.11 (committed oracle version)",
        },
        "chart_count_ratio": {
            "passed": chart_count_ratio is not None
            and PIXAL3D_XATLAS_CHART_COUNT_RATIO_MIN <= chart_count_ratio <= PIXAL3D_XATLAS_CHART_COUNT_RATIO_MAX,
            "actual": chart_count_ratio,
            "required_min": PIXAL3D_XATLAS_CHART_COUNT_RATIO_MIN,
            "required_max": PIXAL3D_XATLAS_CHART_COUNT_RATIO_MAX,
        },
        "atlas_utilization_ratio": {
            "passed": utilization_ratio is not None
            and utilization_ratio >= PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN,
            "actual": utilization_ratio,
            "required_min": PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN,
        },
        "bounded_overlap_ratio": {
            "passed": overlap_face_ratio is not None
            and overlap_face_ratio <= PIXAL3D_XATLAS_MAX_OVERLAP_FACE_RATIO,
            "actual": overlap_face_ratio,
            "required_max": PIXAL3D_XATLAS_MAX_OVERLAP_FACE_RATIO,
        },
        "bounded_unassigned_surface": {
            "passed": unassigned_surface_ratio is not None
            and unassigned_surface_ratio <= PIXAL3D_XATLAS_MAX_UNASSIGNED_SURFACE_RATIO,
            "actual": unassigned_surface_ratio,
            "required_max": PIXAL3D_XATLAS_MAX_UNASSIGNED_SURFACE_RATIO,
        },
        "bounded_degenerate_surface": {
            "passed": degenerate_surface_ratio is not None
            and degenerate_surface_ratio <= PIXAL3D_XATLAS_MAX_DEGENERATE_SURFACE_RATIO,
            "actual": degenerate_surface_ratio,
            "required_max": PIXAL3D_XATLAS_MAX_DEGENERATE_SURFACE_RATIO,
        },
        "uv_surface_exact_coverage": {
            "passed": surface_exact_coverage is not None and surface_exact_coverage >= 0.999,
            "actual": surface_exact_coverage,
            "required_min": 0.999,
        },
        "mirrored_faces_reported": {
            "passed": flipped_face_ratio is not None and 0.0 <= flipped_face_ratio <= 1.0,
            "actual": flipped_face_ratio,
            "required": "reported; complete chart mirroring is valid xatlas behavior",
        },
        "spatial_partition_cuts_reported": {
            "passed": partition_cuts_reported,
            "actual": {
                "shared_edges": partition_shared_edges,
                "cut_edges": partition_cut_edges,
                "cut_edge_ratio": partition_cut_ratio,
            },
            "required": "measured for xatlas-parallel-spatial",
        },
    }
    integrity_check_names = (
        "actual_xatlas_backend",
        "reference_xatlas_version",
        "bounded_overlap_ratio",
        "bounded_unassigned_surface",
        "bounded_degenerate_surface",
        "uv_surface_exact_coverage",
        "mirrored_faces_reported",
        "spatial_partition_cuts_reported",
    )
    layout_check_names = (
        "reference_xatlas_backend",
        "chart_count_ratio",
        "atlas_utilization_ratio",
    )
    integrity_ready = all(bool(checks[name]["passed"]) for name in integrity_check_names)
    layout_parity_ready = all(bool(checks[name]["passed"]) for name in layout_check_names)
    parity_ready = integrity_ready and layout_parity_ready
    if parity_ready:
        status = "reference_xatlas_measured"
        reason = "measured_reference_ratios"
    elif integrity_ready:
        status = "xatlas_integrity_ready_layout_differs"
        reason = "valid_xatlas_output_with_reference_layout_difference"
    else:
        status = "xatlas_integrity_failed"
        reason = "measured_xatlas_integrity_failed"
    return {
        "status": status,
        "reason": reason,
        "integrity_ready": integrity_ready,
        "layout_parity_ready": layout_parity_ready,
        "parity_ready": parity_ready,
        "xatlas_chart_parity": parity_ready,
        "deferred_boundary": None if parity_ready else "not_xatlas_chart_parity",
        "requested_uv_backend": uv_backend,
        "native": {
            "uv_backend": uv_stats_backend,
            "xatlas_version": uv_stats.get("xatlas_version"),
            "chart_count": chart_count,
            "atlas_utilization": utilization,
            "spatial_partition_shared_edge_count": partition_shared_edges,
            "spatial_partition_cut_edge_count": partition_cut_edges,
            "spatial_partition_cut_edge_ratio": partition_cut_ratio,
            "overlap_face_ratio": overlap_face_ratio,
            "flipped_face_ratio": flipped_face_ratio,
            "unassigned_surface_area_ratio": unassigned_surface_ratio,
            "uv_degenerate_surface_area_ratio": degenerate_surface_ratio,
        },
        "reference": {
            "unwrap_backend": reference_backend or None,
            "unwrap_chart_count": reference_chart_count,
            "unwrap_utilization": reference_utilization,
        },
        "ratios": {
            "chart_count_ratio": chart_count_ratio,
            "atlas_utilization_ratio": utilization_ratio,
        },
        "deficits": {},
        "checks": checks,
    }


def _production_thresholds(
    simplify_stats: dict[str, Any],
    export_metrics: dict[str, Any],
    texture_stats: dict[str, Any],
    reference: dict[str, Any] | None,
    *,
    quality_preset: str,
) -> dict[str, Any]:
    blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    final_faces = _maybe_int(simplify_stats.get("final_faces"))
    reference_faces = _maybe_int(reference.get("final_faces")) if reference is not None else None
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    reference_coverage = _maybe_float(reference.get("coverage_ratio")) if reference is not None else None
    raw_coverage = _maybe_float(texture_stats.get("raw_coverage_ratio"))
    reference_raw_coverage = _maybe_float(reference.get("raw_coverage_ratio")) if reference is not None else None

    face_ratio = None
    face_count_passed = False
    if final_faces is not None and reference_faces not in (None, 0):
        face_ratio = float(final_faces) / float(reference_faces)
        face_count_passed = PIXAL3D_REFERENCE_FACE_RATIO_MIN <= face_ratio <= PIXAL3D_REFERENCE_FACE_RATIO_MAX

    final_coverage_ratio = None
    coverage_passed = False
    if final_coverage is not None and reference_coverage not in (None, 0.0):
        final_coverage_ratio = final_coverage / reference_coverage
        coverage_passed = final_coverage_ratio >= PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD

    raw_coverage_ratio = None
    if raw_coverage is not None and reference_raw_coverage not in (None, 0.0):
        raw_coverage_ratio = raw_coverage / reference_raw_coverage

    checks = {
        "reference_available": {
            "passed": reference is not None,
            "actual": bool(reference is not None),
            "required": True,
        },
        "quality_preset": {
            "passed": quality_preset == "reference-target",
            "actual": quality_preset,
            "required": "reference-target",
        },
        "backend_tier": {
            "passed": simplifier_quality == "production",
            "actual": simplifier_quality,
            "required": "production",
        },
        "topology_exportability": {
            "passed": len(blockers) == 0,
            "actual": blockers,
            "required": [],
        },
        "face_count_ratio": {
            "passed": face_count_passed,
            "actual": face_ratio,
            "required_min": PIXAL3D_REFERENCE_FACE_RATIO_MIN,
            "required_max": PIXAL3D_REFERENCE_FACE_RATIO_MAX,
            "spatialkit_final_faces": final_faces,
            "reference_final_faces": reference_faces,
        },
        "final_coverage_ratio": {
            "passed": coverage_passed,
            "actual": final_coverage_ratio,
            "required_min": PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD,
            "spatialkit_final_coverage_ratio": final_coverage,
            "reference_final_coverage_ratio": reference_coverage,
        },
        "raw_coverage_ratio": {
            "passed": raw_coverage_ratio is not None,
            "actual": raw_coverage_ratio,
            "required": "reported",
            "spatialkit_raw_coverage_ratio": raw_coverage,
            "reference_raw_coverage_ratio": reference_raw_coverage,
        },
    }
    all_passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "all_passed": all_passed,
        "checks": checks,
    }


def _maybe_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return resolved if math.isfinite(resolved) else None
