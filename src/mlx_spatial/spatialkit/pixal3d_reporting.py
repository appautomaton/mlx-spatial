"""Pixal3D fixture lineage, compatibility, and run-report helpers.

This module reads existing manifests and reference traces and computes report
summaries.  It does not perform model inference or mesh conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pixal3d_quality import PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD

PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES = 1_000_000
PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE = 4096
PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN = 0.60
PIXAL3D_RENDERED_VISUAL_MAX_SURFACE_UNFILLED_TEXELS = 0
PIXAL3D_RENDERED_VISUAL_MAX_BOUNDARY_OPEN_CHAINS = 0

def _load_pixal3d_fixture_manifest(decoded_dir: Path) -> dict[str, Any] | None:
    decoded = decoded_dir.resolve()
    candidates: list[Path] = []
    direct_candidates = (
        decoded_dir / "manifest.json",
        decoded_dir.parent / "manifest.json",
    )
    for path in direct_candidates:
        if path.exists():
            candidates.append(path)
    for path in sorted(decoded_dir.parent.glob("*/manifest.json")):
        if path.exists():
            candidates.append(path)

    matching: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in candidates:
        manifest_path = path.resolve()
        if manifest_path in seen:
            continue
        seen.add(manifest_path)
        payload = _read_fixture_manifest(path)
        role_a = _manifest_role(payload, "A")
        role_decoded = _manifest_resolve_path(path, role_a, "decoded_dir", "path")
        if role_decoded is None or role_decoded.resolve() != decoded:
            continue
        _validate_fixture_manifest(payload, path, decoded)
        payload = dict(payload)
        payload["manifest_path"] = str(path)
        matching.append(payload)

    if len(matching) > 1:
        paths = ", ".join(str(item["manifest_path"]) for item in matching)
        raise ValueError(f"ambiguous Pixal3D fixture manifests for {decoded_dir}: {paths}")
    if matching:
        return matching[0]

    fixture_root = (Path.cwd() / "inputs" / "mlx-spatialkit").resolve()
    try:
        decoded.relative_to(fixture_root)
    except ValueError:
        return None
    raise ValueError(f"missing Pixal3D fixture manifest for local fixture: {decoded_dir}")


def _read_fixture_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid Pixal3D fixture manifest JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Pixal3D fixture manifest must be a JSON object: {path}")
    if int(payload.get("manifest_version", 0)) != 1:
        raise ValueError(f"Pixal3D fixture manifest_version must be 1: {path}")
    return payload


def _validate_fixture_manifest(payload: dict[str, Any], path: Path, decoded_dir: Path) -> None:
    lineage_id = str(payload.get("lineage_id") or "").strip()
    if not lineage_id:
        raise ValueError(f"Pixal3D fixture manifest missing lineage_id: {path}")
    role_a = _manifest_role(payload, "A")
    role_c = _manifest_role(payload, "C")
    for role_name, role in (("A", role_a), ("C", role_c)):
        role_lineage = str(role.get("lineage_id") or "").strip()
        if role_lineage != lineage_id:
            raise ValueError(
                f"Pixal3D fixture manifest role {role_name} lineage mismatch in {path}: "
                f"{role_lineage!r} != {lineage_id!r}"
            )
    role_decoded = _manifest_resolve_path(path, role_a, "decoded_dir", "path")
    if role_decoded is None or role_decoded.resolve() != decoded_dir:
        raise ValueError(f"Pixal3D fixture manifest A role does not match decoded dir: {path}")
    role_trace = _manifest_resolve_path(path, role_a, "trace_path")
    if role_trace is not None and not role_trace.exists():
        raise ValueError(f"Pixal3D fixture manifest A trace_path does not exist: {role_trace}")
    reference_trace = _manifest_resolve_path(path, role_c, "trace_path")
    reference_glb = _manifest_resolve_path(path, role_c, "model_glb_path", "path")
    if reference_trace is None or not reference_trace.exists():
        raise ValueError(f"Pixal3D fixture manifest C trace_path does not exist: {reference_trace}")
    if reference_glb is None or not reference_glb.exists():
        raise ValueError(f"Pixal3D fixture manifest C model_glb_path does not exist: {reference_glb}")


def _manifest_role(payload: dict[str, Any], role_name: str) -> dict[str, Any]:
    roles = payload.get("roles")
    if not isinstance(roles, dict):
        raise ValueError("Pixal3D fixture manifest missing roles object")
    role = roles.get(role_name)
    if not isinstance(role, dict):
        raise ValueError(f"Pixal3D fixture manifest missing role {role_name}")
    return role


def _manifest_resolve_path(manifest_path: Path, role: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = role.get(key)
        if value is None:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = manifest_path.parent / path
        return path
    return None


def _fixture_manifest_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_path": payload.get("manifest_path"),
        "lineage_id": payload.get("lineage_id"),
        "case_id": payload.get("case_id"),
        "source_image": payload.get("source_image", {}),
        "roles": tuple(payload.get("roles", {}).keys()),
    }


def _load_pixal3d_reference_trace(
    decoded_dir: Path,
    *,
    fixture_manifest: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if fixture_manifest is not None:
        role_c = _manifest_role(fixture_manifest, "C")
        manifest_path = Path(str(fixture_manifest["manifest_path"]))
        reference_trace = _manifest_resolve_path(manifest_path, role_c, "trace_path")
        candidates = [reference_trace] if reference_trace is not None else []
    else:
        candidates = [
            decoded_dir.parent / "pixal3d-1024-cascade-glb-reference" / "trace.json",
        ]
    for path in candidates:
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            trace = json.load(handle)
        metadata = trace.get("metadata", {})
        mesh_export = metadata.get("mesh_export", {})
        postprocess = mesh_export.get("postprocess_stats", {})
        artifact_metadata = metadata.get("textured_glb_artifact", {}).get("metadata", {})
        reference = {
            "trace_path": str(path),
            "model_glb_path": str(path.with_name("model.glb")) if path.with_name("model.glb").exists() else None,
            "final_faces": _maybe_int(postprocess.get("final_faces")),
            "final_vertices": _maybe_int(postprocess.get("final_vertices")),
            "raw_coverage_ratio": _maybe_float(mesh_export.get("raw_coverage_ratio", artifact_metadata.get("raw_coverage_ratio"))),
            "coverage_ratio": _maybe_float(mesh_export.get("coverage_ratio", artifact_metadata.get("coverage_ratio"))),
            "unwrap_backend": mesh_export.get("unwrap_backend", artifact_metadata.get("unwrap_backend")),
            "unwrap_chunks": _maybe_int(mesh_export.get("unwrap_chunks", artifact_metadata.get("unwrap_chunks"))),
            "unwrap_chart_count": _maybe_int(
                mesh_export.get("unwrap_chart_count", artifact_metadata.get("unwrap_chart_count"))
            ),
            "bake_backend": mesh_export.get("bake_backend", artifact_metadata.get("bake_backend")),
            "texture_size": _maybe_int(mesh_export.get("texture_size", artifact_metadata.get("texture_size"))),
            "xatlas_face_guard": _maybe_int(mesh_export.get("xatlas_face_guard", artifact_metadata.get("xatlas_face_guard"))),
            "unwrap_utilization": _maybe_float(mesh_export.get("unwrap_utilization", artifact_metadata.get("unwrap_utilization"))),
        }
        if fixture_manifest is not None:
            reference["lineage_id"] = fixture_manifest.get("lineage_id")
            reference["manifest_path"] = fixture_manifest.get("manifest_path")
        return reference
    return None


def _upstream_export_settings_summary(
    target_faces: int,
    texture_size: int,
    simplify_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    final_faces = _maybe_int(simplify_stats.get("final_faces"))
    face_retention = None
    if final_faces is not None and PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES > 0:
        face_retention = float(final_faces) / float(PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES)
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    backend_tier = str(simplify_stats.get("quality_tier", "unknown"))
    target_reached = bool(simplify_stats.get("target_reached"))
    artifact_ready = bool(quality.get("artifact_ready"))

    checks = {
        "target_faces": {
            "passed": int(target_faces) == PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES,
            "actual": int(target_faces),
            "required": PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES,
        },
        "texture_size": {
            "passed": int(texture_size) == PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE,
            "actual": int(texture_size),
            "required": PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE,
        },
        "backend_tier": {
            "passed": backend_tier == "production",
            "actual": backend_tier,
            "required": "production",
        },
        "target_reached": {
            "passed": target_reached,
            "actual": target_reached,
            "required": True,
        },
        "face_retention_ratio": {
            "passed": face_retention is not None
            and face_retention >= PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN
            and face_retention <= 1.0,
            "actual": face_retention,
            "required_min": PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN,
            "required_max": 1.0,
            "final_faces": final_faces,
        },
        "artifact_ready": {
            "passed": artifact_ready,
            "actual": artifact_ready,
            "required": True,
        },
        "final_coverage_ratio": {
            "passed": final_coverage is not None and final_coverage >= PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD,
            "actual": final_coverage,
            "required_min": PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD,
        },
    }
    return {
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
        "reference": {
            "source": "vendored_pixal3d_inference_defaults",
            "decimation_target": PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES,
            "texture_size": PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE,
            "remesh": True,
            "remesh_band": 1,
            "remesh_project": 0,
            "xatlas_chart_parity": False,
        },
        "checks": checks,
    }


def _glb_viewer_compatibility_summary(glb_summary: dict[str, Any]) -> dict[str, Any]:
    primitives = list(glb_summary.get("primitives", ()))
    large_mesh_threshold = 65_536
    all_have_normals = bool(primitives) and all(
        bool(primitive.get("has_normals"))
        and int(primitive.get("normal_count", 0)) == int(primitive.get("vertex_count", -1))
        for primitive in primitives
    )
    uint16_only = bool(primitives) and all(
        _maybe_int(primitive.get("indices_component_type")) == 5123 for primitive in primitives
    )
    local_indices_bounded = bool(primitives) and all(
        _primitive_indices_within_uint16(primitive) for primitive in primitives
    )
    triangle_indices = bool(primitives) and all(int(primitive.get("index_count", 0)) % 3 == 0 for primitive in primitives)
    total_vertices = _maybe_int(glb_summary.get("total_vertices")) or 0
    primitive_count = _maybe_int(glb_summary.get("primitive_count")) or 0
    chunking_required = total_vertices > large_mesh_threshold
    checks = {
        "glb_parseable": {
            "passed": bool(primitives),
            "actual": bool(primitives),
            "required": True,
        },
        "textured_material": {
            "passed": glb_summary.get("material_count", 0) >= 1
            and glb_summary.get("texture_count", 0) >= 2
            and glb_summary.get("image_count", 0) >= 2,
            "materials": glb_summary.get("material_count", 0),
            "textures": glb_summary.get("texture_count", 0),
            "images": glb_summary.get("image_count", 0),
            "required": "at_least_one_material_two_textures_two_images",
        },
        "normals": {
            "passed": all_have_normals,
            "actual": [
                {
                    "primitive_index": primitive.get("primitive_index"),
                    "has_normals": primitive.get("has_normals"),
                    "vertex_count": primitive.get("vertex_count"),
                    "normal_count": primitive.get("normal_count"),
                }
                for primitive in primitives
            ],
            "required": "NORMAL attribute with count matching POSITION for every primitive",
        },
        "uint16_indices": {
            "passed": uint16_only,
            "actual": [primitive.get("indices_component_type") for primitive in primitives],
            "required": 5123,
        },
        "local_index_bounds": {
            "passed": local_indices_bounded,
            "actual": [
                {
                    "primitive_index": primitive.get("primitive_index"),
                    "indices_min": primitive.get("indices_min"),
                    "indices_max": primitive.get("indices_max"),
                }
                for primitive in primitives
            ],
            "required_min": 0,
            "required_max": 65_535,
        },
        "triangle_indices": {
            "passed": triangle_indices,
            "actual": [primitive.get("index_count") for primitive in primitives],
            "required": "index_count divisible by 3",
        },
        "chunking_for_large_mesh": {
            "passed": not chunking_required or primitive_count > 1,
            "actual": primitive_count,
            "required": ">1 primitive when total_vertices > 65536",
            "total_vertices": total_vertices,
            "large_mesh_threshold": large_mesh_threshold,
        },
    }
    return {
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def _primitive_indices_within_uint16(primitive: dict[str, Any]) -> bool:
    min_values = primitive.get("indices_min")
    max_values = primitive.get("indices_max")
    if not isinstance(min_values, list) or not min_values:
        return False
    if not isinstance(max_values, list) or not max_values:
        return False
    min_index = _maybe_int(min_values[0])
    max_index = _maybe_int(max_values[0])
    return min_index is not None and max_index is not None and min_index >= 0 and max_index <= 65_535


def _reference_glb_path(reference: dict[str, Any]) -> Path | None:
    path = reference.get("model_glb_path")
    if path is None:
        return None
    reference_glb = Path(path)
    return reference_glb if reference_glb.exists() else None


def _visual_comparison_summary(
    report: dict[str, Any],
    upstream_export_settings: dict[str, Any] | None = None,
    *,
    texture_stats: dict[str, Any] | None = None,
    export_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deferred_boundaries = list(report["deferred_parity_boundaries"])
    if upstream_export_settings is not None and bool(upstream_export_settings.get("all_passed")):
        deferred_boundaries = [
            item for item in deferred_boundaries if item != "not_1m_face_export_setting_parity"
        ]
    texture_scalar_ready = bool(report["summary"].get("all_passed"))
    spatial_proof_ready = report["summary"].get("spatial_proof_ready") is True
    checks: dict[str, dict[str, Any]] = {
        "texture_reference_scalar_gate": {
            "passed": texture_scalar_ready,
            "actual": texture_scalar_ready,
            "required": True,
        },
        "spatial_render_proof": {
            "passed": spatial_proof_ready,
            "actual": spatial_proof_ready,
            "required": True,
            "detail": "A texture-statistics comparison is not a rendered or spatially registered proof.",
        },
    }
    if texture_stats is not None:
        surface_unfilled = _maybe_int(texture_stats.get("surface_unfilled_texel_count"))
        checks["surface_unfilled_texels"] = {
            "passed": surface_unfilled is not None
            and surface_unfilled <= PIXAL3D_RENDERED_VISUAL_MAX_SURFACE_UNFILLED_TEXELS,
            "actual": surface_unfilled,
            "required_max": PIXAL3D_RENDERED_VISUAL_MAX_SURFACE_UNFILLED_TEXELS,
        }
    if export_metrics is not None:
        boundary_open_chains = _maybe_int(export_metrics.get("boundary_open_chain_count"))
        checks["boundary_open_chains"] = {
            "passed": boundary_open_chains is not None
            and boundary_open_chains <= PIXAL3D_RENDERED_VISUAL_MAX_BOUNDARY_OPEN_CHAINS,
            "actual": boundary_open_chains,
            "required_max": PIXAL3D_RENDERED_VISUAL_MAX_BOUNDARY_OPEN_CHAINS,
        }
    rendered_visual_ready = all(bool(check["passed"]) for check in checks.values())
    return {
        "rendered_visual_ready": rendered_visual_ready,
        "texture_scalar_ready": texture_scalar_ready,
        "spatial_proof_ready": spatial_proof_ready,
        "summary": report["summary"],
        "checks": report["checks"],
        "rendered_visual_checks": checks,
        "rendered_visual_blockers": tuple(name for name, check in checks.items() if not bool(check["passed"])),
        "artifacts": report.get("artifacts", {}),
        "deferred_parity_boundaries": deferred_boundaries,
    }


def _production_equivalence_summary(
    quality: dict[str, Any],
    visual_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    artifact_ready = bool(quality.get("artifact_ready"))
    scalar_quality_ready = bool(quality.get("production_quality_ready"))
    reference_contract = quality.get("reference_stage_contract", {})
    reference_contract_ready = bool(reference_contract.get("passed"))
    upstream_settings = quality.get("upstream_export_settings", {})
    upstream_settings_ready = bool(upstream_settings.get("all_passed"))
    xatlas_parity = quality.get("xatlas_chart_parity", {})
    xatlas_chart_parity_ready = xatlas_parity.get("parity_ready") is True
    visual_available = visual_comparison is not None
    visual_comparison_ready = (
        visual_available and visual_comparison.get("rendered_visual_ready") is True
    )

    remaining_boundaries: list[str] = []
    if visual_comparison is not None:
        remaining_boundaries.extend(str(item) for item in visual_comparison.get("deferred_parity_boundaries", ()))
    if xatlas_chart_parity_ready:
        # The GLB comparison report cannot inspect chart provenance and always
        # carries this conservative boundary. A measured UV-stage verdict is
        # the authority that clears it in the aggregate contract.
        remaining_boundaries = [
            item for item in remaining_boundaries if item != "not_xatlas_chart_parity"
        ]
    if not reference_contract_ready:
        remaining_boundaries.append("not_reference_stage_contract")
    if not upstream_settings_ready:
        remaining_boundaries.append("not_1m_face_export_setting_parity")
    if not xatlas_chart_parity_ready:
        remaining_boundaries.append("not_xatlas_chart_parity")
    remaining_boundaries = _unique_strings(remaining_boundaries)

    blockers: list[str] = []
    if not artifact_ready:
        blockers.append("artifact_not_ready")
    if not scalar_quality_ready:
        blockers.append("scalar_production_quality_not_ready")
    if not reference_contract_ready:
        blockers.append("reference_stage_contract_not_ready")
    if not upstream_settings_ready:
        blockers.append("upstream_export_settings_not_ready")
    if not xatlas_chart_parity_ready:
        blockers.append("xatlas_chart_parity_not_ready")
    if not visual_available:
        blockers.append("visual_comparison_missing")
    elif not visual_comparison_ready:
        blockers.append("rendered_visual_not_ready")
    if remaining_boundaries:
        blockers.append("deferred_parity_boundaries_present")
    blockers = _unique_strings(blockers)

    ready = (
        artifact_ready
        and scalar_quality_ready
        and reference_contract_ready
        and upstream_settings_ready
        and xatlas_chart_parity_ready
        and visual_comparison_ready
        and not remaining_boundaries
    )
    return {
        "ready": ready,
        "artifact_ready": artifact_ready,
        "scalar_production_quality_ready": scalar_quality_ready,
        "reference_stage_contract_ready": reference_contract_ready,
        "upstream_export_settings_ready": upstream_settings_ready,
        "xatlas_chart_parity_ready": xatlas_chart_parity_ready,
        "visual_comparison_available": visual_available,
        "visual_comparison_ready": visual_comparison_ready,
        "remaining_parity_boundaries": tuple(remaining_boundaries),
        "blockers": tuple(blockers),
        "checks": {
            "artifact_ready": {"passed": artifact_ready, "actual": artifact_ready, "required": True},
            "scalar_production_quality_ready": {
                "passed": scalar_quality_ready,
                "actual": scalar_quality_ready,
                "required": True,
            },
            "reference_stage_contract_ready": {
                "passed": reference_contract_ready,
                "actual": reference_contract_ready,
                "required": True,
            },
            "upstream_export_settings_ready": {
                "passed": upstream_settings_ready,
                "actual": upstream_settings_ready,
                "required": True,
            },
            "xatlas_chart_parity_ready": {
                "passed": xatlas_chart_parity_ready,
                "actual": xatlas_chart_parity_ready,
                "required": True,
            },
            "visual_comparison_ready": {
                "passed": visual_comparison_ready,
                "actual": visual_comparison_ready,
                "required": True,
            },
            "remaining_parity_boundaries": {
                "passed": not remaining_boundaries,
                "actual": tuple(remaining_boundaries),
                "required": [],
            },
        },
    }


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _reference_comparison(diagnostics: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    simplify_stats = diagnostics.get("stages", {}).get("simplify_mesh", {}).get("stats", {})
    texture_stats = diagnostics.get("stages", {}).get("texture_bake", {}).get("stats", {})
    final_faces = _maybe_int(simplify_stats.get("final_faces"))
    reference_faces = _maybe_int(reference.get("final_faces"))
    raw_coverage = _maybe_float(texture_stats.get("raw_coverage_ratio"))
    final_coverage = _maybe_float(texture_stats.get("coverage_ratio", texture_stats.get("final_visible_coverage_ratio")))
    reference_raw = _maybe_float(reference.get("raw_coverage_ratio"))
    reference_final = _maybe_float(reference.get("coverage_ratio"))
    comparison: dict[str, Any] = {
        "spatialkit_simplifier_backend": simplify_stats.get("backend"),
        "spatialkit_quality_tier": simplify_stats.get("quality_tier"),
        "reference_unwrap_backend": reference.get("unwrap_backend"),
        "reference_bake_backend": reference.get("bake_backend"),
        "spatialkit_final_faces": final_faces,
        "reference_final_faces": reference_faces,
        "spatialkit_raw_coverage_ratio": raw_coverage,
        "reference_raw_coverage_ratio": reference_raw,
        "spatialkit_final_coverage_ratio": final_coverage,
        "reference_final_coverage_ratio": reference_final,
    }
    if final_faces is not None and reference_faces not in (None, 0):
        comparison["final_face_count_ratio"] = float(final_faces) / float(reference_faces)
    if raw_coverage is not None and reference_raw not in (None, 0.0):
        comparison["raw_coverage_ratio_vs_reference"] = raw_coverage / reference_raw
    if final_coverage is not None and reference_final not in (None, 0.0):
        comparison["final_coverage_ratio_vs_reference"] = final_coverage / reference_final
    return comparison


def _build_pixal3d_run_manifest(
    *,
    decoded_dir: Path,
    shape_path: Path,
    texture_path: Path,
    glb: NativeGlbArtifact,
    diagnostics_path: Path,
    diagnostics: dict[str, Any],
    fixture_manifest: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    lineage_id = (
        str(fixture_manifest.get("lineage_id"))
        if fixture_manifest is not None
        else f"unmanifested:{decoded_dir.resolve()}"
    )
    source_image = (
        fixture_manifest.get("source_image", {})
        if fixture_manifest is not None
        else {
            "path": _unmanifested_source_image_path(decoded_dir, diagnostics),
            "preprocess_variant": "unknown",
        }
    )
    roles: dict[str, Any] = {
        "A": {
            "role": "A",
            "kind": "decoded_model_output",
            "lineage_id": lineage_id,
            "decoded_dir": str(decoded_dir),
            "shape_decoder_fields": str(shape_path),
            "texture_decoder_pbr": str(texture_path),
            "trace_path": str(decoded_dir / "trace.json") if (decoded_dir / "trace.json").exists() else None,
        },
        "B": {
            "role": "B",
            "kind": "native_mlx_spatial_spatialkit_glb",
            "lineage_id": lineage_id,
            "model_glb_path": str(glb.path),
            "diagnostics_path": str(diagnostics_path),
            "visual_parity_report_path": _nested_get(
                diagnostics,
                ("visual_comparison", "artifacts", "report_json"),
            ),
            "browser_render_report_path": _nested_get(
                diagnostics,
                ("visual_comparison", "artifacts", "browser_render_report_json"),
            ),
            "settings": diagnostics.get("settings", {}),
        },
    }
    if reference is not None:
        roles["C"] = {
            "role": "C",
            "kind": "reference_control_glb",
            "lineage_id": str(reference.get("lineage_id") or lineage_id),
            "control_kind": "internal-xatlas-control",
            "model_glb_path": reference.get("model_glb_path"),
            "trace_path": reference.get("trace_path"),
        }
    if roles.get("C", {}).get("lineage_id") not in (None, lineage_id):
        raise ValueError("Pixal3D run manifest C lineage does not match decoded lineage")
    return {
        "manifest_version": 1,
        "kind": "pixal3d_glb_export_run",
        "lineage_id": lineage_id,
        "case_id": fixture_manifest.get("case_id") if fixture_manifest is not None else None,
        "fixture_manifest_path": fixture_manifest.get("manifest_path") if fixture_manifest is not None else None,
        "source_image": source_image,
        "roles": roles,
        "readiness": diagnostics.get("result", {}),
    }


def decoded_metadata_value(diagnostics: dict[str, Any], key: str) -> Any:
    source = diagnostics.get("source", {})
    for section in ("shape_decoder", "texture_decoder"):
        metadata = source.get(section, {}).get("metadata", {})
        if key in metadata:
            return metadata[key]
    return None


def _unmanifested_source_image_path(decoded_dir: Path, diagnostics: dict[str, Any]) -> Any:
    metadata_path = decoded_metadata_value(diagnostics, "image_path")
    if metadata_path is not None:
        return metadata_path
    trace_path = decoded_dir / "trace.json"
    if not trace_path.is_file():
        return None
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return trace.get("image_path") if isinstance(trace, dict) else None


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
