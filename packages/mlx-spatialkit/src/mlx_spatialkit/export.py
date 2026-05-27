"""Thin Python entry points for native export functionality."""

from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from ._native import (
    backend_info,
    make_face_atlas_uvs as _make_face_atlas_uvs,
    textured_glb_payload as _textured_glb_payload,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .mesh import NativeMesh, clean_mesh, extract_flexi_dual_grid, mesh_metrics, simplify_mesh

_T = TypeVar("_T")

PIXAL3D_PREVIEW_TARGET_FACES = 50_000
PIXAL3D_REFERENCE_TARGET_FACES = 212_542
PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD = 0.50
PIXAL3D_REFERENCE_FACE_RATIO_MIN = 0.80
PIXAL3D_REFERENCE_FACE_RATIO_MAX = 1.25


@dataclass(frozen=True)
class Pixal3DDecodedInputs:
    """Decoded Pixal3D model-stage arrays validated at the native boundary."""

    shape_coordinates: np.ndarray
    shape_fields: np.ndarray
    texture_coordinates: np.ndarray
    texture_attributes: np.ndarray
    contracts: dict[str, Any]
    shape_metadata: dict[str, Any]
    texture_metadata: dict[str, Any]
    texture_spatial_shape: tuple[int, int, int] | None
    texture_batch_size: int | None
    texture_decode_resolution: int | None
    texture_voxel_size: float | None


@dataclass(frozen=True)
class NativeUvMesh:
    """UV-ready triangle mesh prepared by the native backend."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    stats: dict[str, Any]


@dataclass(frozen=True)
class NativeGlbArtifact:
    """Written native GLB artifact metadata."""

    path: Path
    format: str
    bytes_written: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DGlbExportResult:
    """End-to-end native Pixal3D GLB export result."""

    glb: NativeGlbArtifact
    diagnostics_path: Path
    diagnostics: dict[str, Any]


def validate_pixal3d_decoded(
    shape_coordinates: np.ndarray,
    shape_fields: np.ndarray,
    texture_coordinates: np.ndarray,
    texture_attributes: np.ndarray,
) -> dict[str, Any]:
    """Validate Pixal3D decoded arrays through native contract checks."""

    shape_contract = validate_pixal3d_shape_fields(shape_coordinates, shape_fields)
    texture_contract = validate_pixal3d_texture_attributes(texture_coordinates, texture_attributes)
    return {"shape": shape_contract, "texture": texture_contract}


def make_face_atlas_uvs(vertices: np.ndarray, faces: np.ndarray, *, tile_padding: float = 0.08) -> NativeUvMesh:
    """Create a deterministic native face-atlas UV mesh."""

    result = _make_face_atlas_uvs(vertices, faces, float(tile_padding))
    return NativeUvMesh(
        vertices=np.asarray(result["vertices"]),
        faces=np.asarray(result["faces"]),
        uvs=np.asarray(result["uvs"]),
        stats=dict(result["stats"]),
    )


def textured_glb_payload(
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatialkit",
    mesh_name: str = "TexturedMesh",
    material_name: str = "PBRMaterial",
) -> bytes:
    """Build a native self-contained GLB 2.0 payload."""

    return bytes(
        _textured_glb_payload(
            mesh.vertices,
            mesh.faces,
            mesh.uvs,
            base_color_rgba,
            metallic_roughness,
            str(generator),
            str(mesh_name),
            str(material_name),
        )
    )


def write_textured_glb(
    path: str | Path,
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatialkit",
    mesh_name: str = "TexturedMesh",
    material_name: str = "PBRMaterial",
    metadata: dict[str, Any] | None = None,
) -> NativeGlbArtifact:
    """Write a native GLB payload to disk."""

    output = Path(path)
    if output.suffix.lower() != ".glb":
        raise ValueError("native textured exports require a .glb output path")
    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color_rgba,
        metallic_roughness=metallic_roughness,
        generator=generator,
        mesh_name=mesh_name,
        material_name=material_name,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    payload_metadata = {
        "stage": "textured_glb",
        "format": "glb",
        "bytes_written": int(output.stat().st_size),
        "generator": str(generator),
        "mesh_name": str(mesh_name),
        "material_name": str(material_name),
        **(metadata or {}),
    }
    return NativeGlbArtifact(
        path=output,
        format="glb",
        bytes_written=int(output.stat().st_size),
        metadata=payload_metadata,
    )


def export_pixal3d_glb(
    decoded_dir: str | Path,
    output: str | Path,
    *,
    texture_size: int = 1024,
    target_faces: int | None = None,
    quality_preset: str = "preview",
    grid_size: int | None = None,
    min_component_faces: int = 32,
    tile_padding: float = 0.08,
    max_texture_pixels: int | None = None,
    diagnostics_path: str | Path | None = None,
) -> Pixal3DGlbExportResult:
    """Convert decoded Pixal3D NPZ artifacts into a textured GLB through native hot paths."""

    from .texture import bake_pbr_texture

    source_dir = Path(decoded_dir)
    if not source_dir.is_dir():
        raise ValueError(f"decoded Pixal3D directory does not exist: {source_dir}")
    if texture_size <= 0:
        raise ValueError("texture_size must be positive")
    if grid_size is not None and grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if min_component_faces <= 0:
        raise ValueError("min_component_faces must be positive")
    if max_texture_pixels is not None and max_texture_pixels <= 0:
        raise ValueError("max_texture_pixels must be positive")
    glb_path, resolved_diagnostics_path = _resolve_pixal3d_export_paths(output, diagnostics_path)
    shape_path = source_dir / "shape_decoder_fields.npz"
    texture_path = source_dir / "texture_decoder_pbr.npz"
    if not shape_path.exists():
        raise ValueError(f"missing decoded shape artifact: {shape_path}")
    if not texture_path.exists():
        raise ValueError(f"missing decoded texture artifact: {texture_path}")

    export_settings = _resolve_pixal3d_export_settings(source_dir, quality_preset, target_faces)
    reference = export_settings["reference"]
    resolved_quality_preset = str(export_settings["quality_preset"])
    resolved_target_faces = int(export_settings["target_faces"])

    diagnostics: dict[str, Any] = {
        "stage": "pixal3d_glb_export",
        "source_dir": str(source_dir),
        "output_path": str(glb_path),
        "diagnostics_path": str(resolved_diagnostics_path),
        "settings": {
            "quality_preset": resolved_quality_preset,
            "texture_size": int(texture_size),
            "target_faces": resolved_target_faces,
            "requested_target_faces": int(target_faces) if target_faces is not None else None,
            "target_faces_source": export_settings["target_faces_source"],
            "reference_available": reference is not None,
            "reference_trace_path": str(reference["trace_path"]) if reference is not None else None,
            "reference_target_faces": reference.get("final_faces") if reference is not None else None,
            "reference_texture_size": reference.get("texture_size") if reference is not None else None,
            "reference_xatlas_face_guard": reference.get("xatlas_face_guard") if reference is not None else None,
            "grid_size": int(grid_size) if grid_size is not None else None,
            "min_component_faces": int(min_component_faces),
            "tile_padding": float(tile_padding),
            "max_texture_pixels": int(max_texture_pixels) if max_texture_pixels is not None else None,
        },
        "stages": {},
        "timings_sec": {},
        "memory_samples": {},
    }

    def sample(label: str) -> None:
        diagnostics["memory_samples"][label] = _memory_sample()

    sample("start")
    decoded = _timed_stage(
        diagnostics,
        "load_npz",
        lambda: load_pixal3d_decoded_npz(shape_path, texture_path),
    )
    diagnostics["contracts"] = decoded.contracts
    diagnostics["source"] = {
        "shape_decoder": {
            "path": str(shape_path),
            "metadata": decoded.shape_metadata,
        },
        "texture_decoder": {
            "path": str(texture_path),
            "metadata": decoded.texture_metadata,
            "spatial_shape": decoded.texture_spatial_shape,
            "batch_size": decoded.texture_batch_size,
            "decode_resolution": decoded.texture_decode_resolution,
            "voxel_size": decoded.texture_voxel_size,
        },
    }
    sample("after_load_npz")

    resolved_grid_size = _resolve_positive_int(
        grid_size,
        decoded.texture_decode_resolution,
        decoded.shape_metadata.get("actual_hr_resolution"),
        decoded.texture_metadata.get("decode_resolution"),
        default=1024,
        name="grid_size",
    )
    diagnostics["settings"]["grid_size"] = resolved_grid_size
    resolved_max_texture_pixels = max_texture_pixels if max_texture_pixels is not None else int(texture_size) * int(texture_size)
    diagnostics["settings"]["max_texture_pixels"] = resolved_max_texture_pixels

    shape_coordinates = decoded.shape_coordinates
    shape_fields = decoded.shape_fields
    texture_coordinates = decoded.texture_coordinates
    texture_attributes = decoded.texture_attributes
    texture_decode_resolution = decoded.texture_decode_resolution or resolved_grid_size
    texture_voxel_size = decoded.texture_voxel_size
    del decoded

    mesh = _timed_stage(
        diagnostics,
        "extract_mesh",
        lambda: extract_flexi_dual_grid(shape_coordinates, shape_fields, grid_size=resolved_grid_size),
    )
    diagnostics["stages"]["extract_mesh"].update(_mesh_shape(mesh, "source"))
    del shape_coordinates, shape_fields
    gc.collect()
    sample("after_extract_mesh")

    pre_metrics = _timed_stage(
        diagnostics,
        "source_metrics",
        lambda: mesh_metrics(mesh.vertices, mesh.faces),
    )
    diagnostics["stages"]["source_metrics"]["metrics"] = pre_metrics

    cleaned, clean_stats = _timed_stage(
        diagnostics,
        "clean_mesh",
        lambda: clean_mesh(mesh.vertices, mesh.faces, min_component_faces=min_component_faces),
    )
    diagnostics["stages"]["clean_mesh"].update(_mesh_shape(cleaned, "cleaned"))
    diagnostics["stages"]["clean_mesh"]["stats"] = clean_stats
    del mesh
    gc.collect()
    sample("after_clean_mesh")

    simplified, simplify_stats = _timed_stage(
        diagnostics,
        "simplify_mesh",
        lambda: simplify_mesh(
            cleaned.vertices,
            cleaned.faces,
            target_faces=resolved_target_faces,
            min_component_faces=min_component_faces,
        ),
    )
    diagnostics["stages"]["simplify_mesh"].update(_mesh_shape(simplified, "simplified"))
    diagnostics["stages"]["simplify_mesh"]["stats"] = simplify_stats
    del cleaned
    gc.collect()
    sample("after_simplify_mesh")

    post_metrics = _timed_stage(
        diagnostics,
        "export_metrics",
        lambda: mesh_metrics(simplified.vertices, simplified.faces),
    )
    diagnostics["stages"]["export_metrics"]["metrics"] = post_metrics

    uv_mesh = _timed_stage(
        diagnostics,
        "uv",
        lambda: make_face_atlas_uvs(simplified.vertices, simplified.faces, tile_padding=tile_padding),
    )
    diagnostics["stages"]["uv"].update(_uv_shape(uv_mesh))
    del simplified
    gc.collect()
    sample("after_uv")

    baked = _timed_stage(
        diagnostics,
        "texture_bake",
        lambda: bake_pbr_texture(
            uv_mesh,
            texture_coordinates,
            texture_attributes,
            texture_size=texture_size,
            decode_resolution=texture_decode_resolution,
            voxel_size=texture_voxel_size,
            max_texture_pixels=resolved_max_texture_pixels,
        ),
    )
    diagnostics["stages"]["texture_bake"].update(_texture_shape(baked))
    del texture_coordinates, texture_attributes
    gc.collect()
    sample("after_texture_bake")

    if reference is not None:
        diagnostics["reference"] = reference
        diagnostics["reference_comparison"] = _reference_comparison(diagnostics, reference)

    quality = _export_quality_summary(
        simplify_stats,
        post_metrics,
        baked.stats,
        reference,
        quality_preset=resolved_quality_preset,
    )
    diagnostics["quality"] = quality

    glb = _timed_stage(
        diagnostics,
        "write_glb",
        lambda: write_textured_glb(
            glb_path,
            uv_mesh,
            base_color_rgba=baked.base_color_rgba,
            metallic_roughness=baked.metallic_roughness,
            generator="mlx-spatialkit Pixal3D",
            mesh_name="Pixal3D_TexturedMesh",
            material_name="Pixal3D_PBR",
            metadata={
                "pipeline_type": decoded_metadata_value(diagnostics, "pipeline_type"),
                "shape_decoder_artifact": str(shape_path),
                "texture_decoder_artifact": str(texture_path),
                "texture_size": int(baked.texture_size),
                "target_faces": resolved_target_faces,
                "quality_preset": resolved_quality_preset,
                "bake_backend": str(baked.stats.get("backend")),
                "coverage_ratio": float(baked.stats.get("coverage_ratio", 0.0)),
                "raw_coverage_ratio": float(baked.stats.get("raw_coverage_ratio", 0.0)),
                "simplifier_backend": quality["simplifier_backend"],
                "simplifier_quality_tier": quality["simplifier_quality_tier"],
                "production_quality_ready": bool(quality["production_quality_ready"]),
            },
        ),
    )
    diagnostics["stages"]["write_glb"]["artifact"] = glb.metadata
    sample("after_write_glb")

    diagnostics["result"] = {
        "ready": bool(quality["artifact_ready"]),
        "artifact_ready": bool(quality["artifact_ready"]),
        "production_quality_ready": bool(quality["production_quality_ready"]),
        "quality_warnings": quality["warnings"],
        "model_glb": str(glb.path),
        "diagnostics_json": str(resolved_diagnostics_path),
        "bytes_written": int(glb.bytes_written),
    }
    _write_json_atomic(resolved_diagnostics_path, diagnostics)
    return Pixal3DGlbExportResult(glb=glb, diagnostics_path=resolved_diagnostics_path, diagnostics=diagnostics)


def _load_npz_array(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in payload.files:
        raise ValueError(f"{path} is missing required array {key!r}")
    return np.asarray(payload[key])


def _load_npz_metadata(payload: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    raw = payload["metadata_json"]
    try:
        text = str(raw.item() if raw.shape == () else raw.tolist())
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} contains invalid metadata_json") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} metadata_json must decode to an object")
    return value


def _load_optional_scalar(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> Any:
    if key not in payload.files:
        return None
    value = payload[key]
    if value.shape != ():
        raise ValueError(f"{path} optional scalar {key!r} must be rank 0")
    return value.item()


def load_pixal3d_decoded_npz(
    shape_decoder_path: str | Path,
    texture_decoder_path: str | Path,
) -> Pixal3DDecodedInputs:
    """Load Pixal3D decoded NPZ files and validate their native contracts."""

    shape_path = Path(shape_decoder_path)
    texture_path = Path(texture_decoder_path)
    with np.load(shape_path) as shape_payload:
        shape_coordinates = _load_npz_array(shape_payload, "coordinates", shape_path)
        shape_fields = _load_npz_array(shape_payload, "fields", shape_path)
        shape_metadata = _load_npz_metadata(shape_payload, shape_path)
    with np.load(texture_path) as texture_payload:
        texture_coordinates = _load_npz_array(texture_payload, "coordinates", texture_path)
        texture_attributes = _load_npz_array(texture_payload, "attributes", texture_path)
        texture_metadata = _load_npz_metadata(texture_payload, texture_path)
        texture_spatial_shape = (
            tuple(int(dim) for dim in _load_npz_array(texture_payload, "spatial_shape", texture_path))
            if "spatial_shape" in texture_payload.files
            else None
        )
        texture_batch_size = _load_optional_scalar(texture_payload, "batch_size", texture_path)
        texture_decode_resolution = _load_optional_scalar(texture_payload, "decode_resolution", texture_path)
        texture_voxel_size = _load_optional_scalar(texture_payload, "voxel_size", texture_path)
    contracts = validate_pixal3d_decoded(
        shape_coordinates,
        shape_fields,
        texture_coordinates,
        texture_attributes,
    )
    return Pixal3DDecodedInputs(
        shape_coordinates=shape_coordinates,
        shape_fields=shape_fields,
        texture_coordinates=texture_coordinates,
        texture_attributes=texture_attributes,
        contracts=contracts,
        shape_metadata=shape_metadata,
        texture_metadata=texture_metadata,
        texture_spatial_shape=texture_spatial_shape,
        texture_batch_size=int(texture_batch_size) if texture_batch_size is not None else None,
        texture_decode_resolution=(
            None
            if texture_decode_resolution is None or int(texture_decode_resolution) < 0
            else int(texture_decode_resolution)
        ),
        texture_voxel_size=(
            None
            if texture_voxel_size is None or not np.isfinite(float(texture_voxel_size))
            else float(texture_voxel_size)
        ),
    )


def _timed_stage(diagnostics: dict[str, Any], name: str, fn: Callable[[], _T]) -> _T:
    start = time.perf_counter()
    try:
        return fn()
    finally:
        elapsed = time.perf_counter() - start
        diagnostics["timings_sec"][name] = elapsed
        diagnostics["stages"].setdefault(name, {})["seconds"] = elapsed


def _memory_sample() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
    return {
        "pid": os.getpid(),
        "current_rss_bytes": _current_rss_bytes(),
        "max_rss_bytes": max_rss_bytes,
        "source": "ps rss plus resource.getrusage(RUSAGE_SELF).ru_maxrss",
    }


def _current_rss_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(output.strip()) * 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _mesh_shape(mesh: NativeMesh, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_vertices": int(mesh.vertices.shape[0]),
        f"{prefix}_faces": int(mesh.faces.shape[0]),
    }


def _uv_shape(mesh: NativeUvMesh) -> dict[str, Any]:
    return {
        "vertices_shape": tuple(int(dim) for dim in mesh.vertices.shape),
        "faces_shape": tuple(int(dim) for dim in mesh.faces.shape),
        "uvs_shape": tuple(int(dim) for dim in mesh.uvs.shape),
        "stats": mesh.stats,
    }


def _texture_shape(baked: Any) -> dict[str, Any]:
    return {
        "base_color_shape": tuple(int(dim) for dim in baked.base_color_rgba.shape),
        "metallic_roughness_shape": tuple(int(dim) for dim in baked.metallic_roughness.shape),
        "coverage_status_shape": tuple(int(dim) for dim in baked.coverage_status.shape),
        "stats": baked.stats,
    }


def _resolve_pixal3d_export_paths(output: str | Path, diagnostics_path: str | Path | None) -> tuple[Path, Path]:
    output_path = Path(output)
    glb_path = output_path if output_path.suffix.lower() == ".glb" else output_path / "model.glb"
    if diagnostics_path is None:
        diag_path = glb_path.with_name("diagnostics.json")
    else:
        diag_path = Path(diagnostics_path)
    if diag_path.suffix.lower() != ".json":
        raise ValueError("Pixal3D export diagnostics path must end with .json")
    return glb_path, diag_path


def _resolve_positive_int(*values: Any, default: int, name: str) -> int:
    for value in values:
        if value is None:
            continue
        resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"{name} must be positive")
        return resolved
    return int(default)


def decoded_metadata_value(diagnostics: dict[str, Any], key: str) -> Any:
    source = diagnostics.get("source", {})
    for section in ("shape_decoder", "texture_decoder"):
        metadata = source.get(section, {}).get("metadata", {})
        if key in metadata:
            return metadata[key]
    return None


def _resolve_pixal3d_export_settings(
    decoded_dir: Path,
    quality_preset: str,
    target_faces: int | None,
) -> dict[str, Any]:
    preset = _normalize_quality_preset(quality_preset)
    reference = _load_pixal3d_reference_trace(decoded_dir)
    if target_faces is not None:
        resolved_target_faces = int(target_faces)
        target_source = "explicit"
    elif preset == "reference-target" and reference is not None and reference.get("final_faces") is not None:
        resolved_target_faces = int(reference["final_faces"])
        target_source = "reference_final_faces"
    elif preset == "reference-target":
        resolved_target_faces = PIXAL3D_REFERENCE_TARGET_FACES
        target_source = "reference_default"
    else:
        resolved_target_faces = PIXAL3D_PREVIEW_TARGET_FACES
        target_source = "preview_default"
    if resolved_target_faces <= 0:
        raise ValueError("target_faces must be positive")
    return {
        "quality_preset": preset,
        "target_faces": resolved_target_faces,
        "target_faces_source": target_source,
        "reference": reference,
    }


def _normalize_quality_preset(value: str) -> str:
    preset = str(value).strip().lower().replace("_", "-")
    if preset in ("production", "reference", "reference-target"):
        return "reference-target"
    if preset == "preview":
        return "preview"
    raise ValueError("quality_preset must be 'preview' or 'reference-target'")


def _export_quality_summary(
    simplify_stats: dict[str, Any],
    export_metrics: dict[str, Any],
    texture_stats: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    *,
    quality_preset: str = "preview",
) -> dict[str, Any]:
    blockers = tuple(str(item) for item in export_metrics.get("export_blocking_reasons", ()))
    simplifier_quality = str(simplify_stats.get("quality_tier", "unknown"))
    simplifier_backend = str(simplify_stats.get("backend", "unknown"))
    preset = _normalize_quality_preset(quality_preset)
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
    artifact_ready = len(blockers) == 0
    return {
        "artifact_ready": artifact_ready,
        "production_quality_ready": artifact_ready and bool(thresholds["all_passed"]),
        "quality_preset": preset,
        "simplifier_backend": simplifier_backend,
        "simplifier_quality_tier": simplifier_quality,
        "native_geometry_candidate": _native_geometry_candidate_status(simplify_stats, thresholds, preset),
        "export_blocking_reasons": blockers,
        "production_thresholds": thresholds,
        "warnings": tuple(warnings),
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
        }
    if bool(backend_check.get("passed")):
        return {
            "status": "candidate",
            "reason": "native_geometry_candidate_available",
            "current_backend": simplify_stats.get("backend"),
            "current_quality_tier": simplify_stats.get("quality_tier"),
            "face_count_ratio": face_check.get("actual"),
            "topology_exportability_passed": bool(topology_check.get("passed")),
        }
    return {
        "status": "blocked",
        "reason": "native_geometry_candidate_blocked",
        "detail": "reference-target export still uses a preview-tier native simplifier",
        "current_backend": simplify_stats.get("backend"),
        "current_quality_tier": simplify_stats.get("quality_tier"),
        "face_count_ratio": face_check.get("actual"),
        "topology_exportability_passed": bool(topology_check.get("passed")),
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


def _load_pixal3d_reference_trace(decoded_dir: Path) -> dict[str, Any] | None:
    candidates = [
        decoded_dir.parent / "pixal3d-1024-cascade-glb-reference" / "trace.json",
        Path.cwd() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-glb-reference" / "trace.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            trace = json.load(handle)
        metadata = trace.get("metadata", {})
        mesh_export = metadata.get("mesh_export", {})
        postprocess = mesh_export.get("postprocess_stats", {})
        artifact_metadata = metadata.get("textured_glb_artifact", {}).get("metadata", {})
        return {
            "trace_path": str(path),
            "final_faces": _maybe_int(postprocess.get("final_faces")),
            "final_vertices": _maybe_int(postprocess.get("final_vertices")),
            "raw_coverage_ratio": _maybe_float(mesh_export.get("raw_coverage_ratio", artifact_metadata.get("raw_coverage_ratio"))),
            "coverage_ratio": _maybe_float(mesh_export.get("coverage_ratio", artifact_metadata.get("coverage_ratio"))),
            "unwrap_backend": mesh_export.get("unwrap_backend", artifact_metadata.get("unwrap_backend")),
            "bake_backend": mesh_export.get("bake_backend", artifact_metadata.get("bake_backend")),
            "texture_size": _maybe_int(mesh_export.get("texture_size", artifact_metadata.get("texture_size"))),
            "xatlas_face_guard": _maybe_int(mesh_export.get("xatlas_face_guard", artifact_metadata.get("xatlas_face_guard"))),
            "unwrap_utilization": _maybe_float(mesh_export.get("unwrap_utilization", artifact_metadata.get("unwrap_utilization"))),
        }
    return None


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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

__all__ = [
    "NativeGlbArtifact",
    "NativeUvMesh",
    "Pixal3DGlbExportResult",
    "Pixal3DDecodedInputs",
    "backend_info",
    "export_pixal3d_glb",
    "load_pixal3d_decoded_npz",
    "make_face_atlas_uvs",
    "textured_glb_payload",
    "validate_pixal3d_decoded",
    "write_textured_glb",
]
