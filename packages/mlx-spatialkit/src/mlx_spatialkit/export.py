"""Thin Python entry points for native export functionality."""

from __future__ import annotations

import gc
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

from ._native import (
    backend_info,
    make_face_atlas_uvs as _make_face_atlas_uvs,
    make_native_chart_uvs as _make_native_chart_uvs,
    textured_glb_payload as _textured_glb_payload,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .glb_compare import compare_textured_glbs, inspect_glb
from .mesh import NativeMesh, clean_mesh, extract_flexi_dual_grid, mesh_metrics, remesh_narrow_band, simplify_mesh

_T = TypeVar("_T")

PIXAL3D_PREVIEW_TARGET_FACES = 50_000
PIXAL3D_REFERENCE_TARGET_FACES = 212_542
PIXAL3D_REFERENCE_FINAL_COVERAGE_THRESHOLD = 0.50
PIXAL3D_REFERENCE_FACE_RATIO_MIN = 0.80
PIXAL3D_REFERENCE_FACE_RATIO_MAX = 1.25
PIXAL3D_MEMORY_POLL_INTERVAL_SEC = 0.25
PIXAL3D_UPSTREAM_EXPORT_TARGET_FACES = 1_000_000
PIXAL3D_UPSTREAM_EXPORT_TEXTURE_SIZE = 4096
PIXAL3D_UPSTREAM_EXPORT_FACE_RETENTION_MIN = 0.60
PIXAL3D_CHART_UV_GLOBAL_COVERAGE_MIN = 0.50
PIXAL3D_CHART_UV_SURFACE_OCCUPANCY_MIN = 0.50
PIXAL3D_CHART_UV_SURFACE_VISIBLE_MIN = 0.50
PIXAL3D_FACE_ATLAS_TILE_PADDING = 0.08
PIXAL3D_NATIVE_CHART_TILE_PADDING = 0.001
PIXAL3D_SMALL_BOUNDARY_LOOP_FILL_MAX_EDGES = 8
PIXAL3D_SMALL_BOUNDARY_LOOP_FILL_MAX_PERIMETER = 0.03
PIXAL3D_XATLAS_UTILIZATION_EQUIVALENCE_MIN = 0.95
PIXAL3D_RENDERED_VISUAL_MAX_SURFACE_UNFILLED_TEXELS = 0
PIXAL3D_RENDERED_VISUAL_MAX_BOUNDARY_OPEN_CHAINS = 0


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


def make_native_chart_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chart_angle_degrees: float = 45.0,
    tile_padding: float = 0.04,
) -> NativeUvMesh:
    """Create a deterministic native chart UV mesh."""

    result = _make_native_chart_uvs(vertices, faces, float(chart_angle_degrees), float(tile_padding))
    return NativeUvMesh(
        vertices=np.asarray(result["vertices"]),
        faces=np.asarray(result["faces"]),
        uvs=np.asarray(result["uvs"]),
        stats=dict(result["stats"]),
    )


def make_reference_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    texture_resolution: int = 1024,
    pack_padding_texels: float = 0.0,
) -> NativeUvMesh:
    """Reference-parity UV unwrap (CuMesh cone clustering + xatlas-equivalent
    chart growth, LSCM parameterization, and texel-gap shelf packing).

    Pipeline knobs are pinned to the production reference values
    (o_voxel.postprocess.to_glb -> CuMesh.uv_unwrap with xatlas defaults);
    the atlas is packed at `texture_resolution` with xatlas PackOptions
    semantics (padding + bilinear gutter).
    """

    from ._native import (  # noqa: PLC0415  (lazy: keeps module import light)
        compute_uv_charts,
        grow_uv_charts,
        pack_uv_charts,
        parameterize_uv_charts,
        uv_quality_metrics,
    )

    source_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(faces, dtype=np.int64)

    stage_a = compute_uv_charts(
        source_vertices,
        source_faces,
        threshold_cone_half_angle_rad=math.radians(90.0),
        refine_iterations=0,
        global_iterations=1,
        smooth_strength=1.0,
        area_penalty_weight=0.1,
        perimeter_area_ratio_weight=0.0001,
    )
    grown = grow_uv_charts(
        source_vertices,
        source_faces,
        cluster_ids=np.ascontiguousarray(np.asarray(stage_a["chart_ids"]), dtype=np.int64),
    )
    parameterized = parameterize_uv_charts(
        source_vertices,
        source_faces,
        np.ascontiguousarray(np.asarray(grown["chart_ids"]), dtype=np.int64),
    )
    chart_ids = np.ascontiguousarray(np.asarray(parameterized["chart_ids"]), dtype=np.int64)
    packed = pack_uv_charts(
        source_faces,
        chart_ids,
        np.ascontiguousarray(np.asarray(parameterized["corner_uvs"]), dtype=np.float64),
        resolution=int(texture_resolution),
        padding=float(pack_padding_texels),
    )
    packed_corner_uvs = np.asarray(packed["corner_uvs"])

    # Assemble the duplicated-vertex UV mesh: one output vertex per unique
    # (chart, source vertex) pair, deterministic via sorted unique keys.
    corner_chart = np.repeat(chart_ids, 3)
    corner_source = source_faces.reshape(-1)
    keys = corner_chart * np.int64(source_vertices.shape[0]) + corner_source
    unique_keys, first_index, inverse = np.unique(keys, return_index=True, return_inverse=True)
    vmap = (unique_keys % np.int64(source_vertices.shape[0])).astype(np.int64)
    out_vertices = source_vertices[vmap]
    out_faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int64)
    out_uvs = np.ascontiguousarray(packed_corner_uvs[first_index], dtype=np.float32)

    final_metrics = uv_quality_metrics(
        np.ascontiguousarray(out_vertices, dtype=np.float32),
        out_faces,
        out_uvs,
        chart_ids=chart_ids,
    )

    stats: dict[str, Any] = {
        "backend": "xatlas-equivalent-native",
        "packing": "texel-shelf-pca-rotate",
        "source_vertices": int(source_vertices.shape[0]),
        "source_faces": int(source_faces.shape[0]),
        "output_vertices": int(out_vertices.shape[0]),
        "output_faces": int(out_faces.shape[0]),
        "duplicated_vertex_ratio": float(out_vertices.shape[0] / max(source_vertices.shape[0], 1)),
        "stage_a_cluster_count": int(stage_a["chart_count"]),
        "growth_chart_count": int(grown["chart_count"]),
        "chart_count": int(parameterized["chart_count"]),
        "projected_chart_count": int(parameterized["projected_chart_count"]),
        "projection_fallback_chart_count": int(parameterized["projection_fallback_chart_count"]),
        "lscm_chart_count": int(parameterized["lscm_chart_count"]),
        "shattered_face_chart_count": int(parameterized["shattered_face_chart_count"]),
        "split_event_count": int(parameterized["split_event_count"]),
        "lscm_unconverged_count": int(parameterized["lscm_unconverged_count"]),
        "atlas_resolution": int(packed["atlas_resolution"]),
        "texels_per_unit": float(packed["texels_per_unit"]),
        "packed_height_texels": float(packed["packed_height_texels"]),
        "shelf_count": int(packed["shelf_count"]),
        "gap_texels": float(packed["gap_texels"]),
        "uv_overlap_count": int(final_metrics["uv_overlap_count"]),
        "uv_flipped_count": int(final_metrics["uv_flipped_count"]),
        "uv_degenerate_count": int(final_metrics["uv_degenerate_count"]),
        "uv_stretch_l2": float(final_metrics["uv_stretch_l2"]),
        "uv_stretch_linf": float(final_metrics["uv_stretch_linf"]),
        "uv_bbox_utilization": float(final_metrics["uv_bbox_utilization"]),
        "uv_total_area": float(final_metrics["uv_total_area"]),
    }
    return NativeUvMesh(
        vertices=np.asarray(out_vertices),
        faces=np.asarray(out_faces),
        uvs=np.asarray(out_uvs),
        stats=stats,
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
    uv_backend: str = "face-atlas",
    chart_angle_degrees: float = 45.0,
    tile_padding: float | None = None,
    small_boundary_loop_fill_max_edges: int = PIXAL3D_SMALL_BOUNDARY_LOOP_FILL_MAX_EDGES,
    small_boundary_loop_fill_max_perimeter: float = PIXAL3D_SMALL_BOUNDARY_LOOP_FILL_MAX_PERIMETER,
    max_texture_pixels: int | None = None,
    source_projection: bool = True,
    source_projection_fallback_mode: str = "knn",
    source_projection_fallback_neighbors: int = 8,
    source_projection_fallback_max_distance_voxels: float = 12.0,
    render_padding: bool = True,
    remesh: bool = False,
    remesh_band: float = 1.0,
    remesh_resolution: int | None = None,
    remesh_project_back: float = 0.0,
    remesh_repair_nonmanifold: bool = False,
    simplify_backend: str | None = None,
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
    resolved_small_boundary_loop_fill_max_edges = int(small_boundary_loop_fill_max_edges)
    if resolved_small_boundary_loop_fill_max_edges < 0:
        raise ValueError("small_boundary_loop_fill_max_edges must be non-negative")
    resolved_small_boundary_loop_fill_max_perimeter = float(small_boundary_loop_fill_max_perimeter)
    if (
        not math.isfinite(resolved_small_boundary_loop_fill_max_perimeter)
        or resolved_small_boundary_loop_fill_max_perimeter <= 0
    ):
        raise ValueError("small_boundary_loop_fill_max_perimeter must be positive")
    if max_texture_pixels is not None and max_texture_pixels <= 0:
        raise ValueError("max_texture_pixels must be positive")
    if source_projection_fallback_mode not in {"knn", "disabled"}:
        raise ValueError("source_projection_fallback_mode must be 'knn' or 'disabled'")
    if source_projection_fallback_neighbors <= 0:
        raise ValueError("source_projection_fallback_neighbors must be positive")
    if source_projection_fallback_max_distance_voxels <= 0:
        raise ValueError("source_projection_fallback_max_distance_voxels must be positive")
    if remesh:
        if not math.isfinite(remesh_band) or remesh_band <= 0:
            raise ValueError("remesh_band must be positive and finite")
        if remesh_resolution is not None and remesh_resolution <= 0:
            raise ValueError("remesh_resolution must be positive")
        if not math.isfinite(remesh_project_back) or not (0.0 <= remesh_project_back <= 1.0):
            raise ValueError("remesh_project_back must be in [0, 1]")
    resolved_simplify_backend = _resolve_simplify_backend(simplify_backend)
    if resolved_simplify_backend == "qem" and not (remesh and remesh_repair_nonmanifold):
        raise ValueError(
            "simplify_backend='qem' requires remesh=True and remesh_repair_nonmanifold=True "
            "to guarantee QEM receives a watertight manifold input"
        )
    resolved_uv_backend = _resolve_pixal3d_uv_backend(uv_backend)
    resolved_chart_angle_degrees = _resolve_chart_angle_degrees(chart_angle_degrees)
    resolved_tile_padding, tile_padding_source = _resolve_tile_padding(tile_padding, resolved_uv_backend)
    glb_path, resolved_diagnostics_path = _resolve_pixal3d_export_paths(output, diagnostics_path)
    shape_path = source_dir / "shape_decoder_fields.npz"
    texture_path = source_dir / "texture_decoder_pbr.npz"
    if not shape_path.exists():
        raise ValueError(f"missing decoded shape artifact: {shape_path}")
    if not texture_path.exists():
        raise ValueError(f"missing decoded texture artifact: {texture_path}")

    fixture_manifest = _load_pixal3d_fixture_manifest(source_dir)
    export_settings = _resolve_pixal3d_export_settings(
        source_dir,
        quality_preset,
        target_faces,
        fixture_manifest=fixture_manifest,
    )
    reference = export_settings["reference"]
    resolved_quality_preset = str(export_settings["quality_preset"])
    resolved_target_faces = int(export_settings["target_faces"])
    requested_simplifier_backend = _simplifier_backend_for_quality_preset(resolved_quality_preset)
    if resolved_simplify_backend is not None:
        requested_simplifier_backend = resolved_simplify_backend
    diagnostics: dict[str, Any] = {
        "stage": "pixal3d_glb_export",
        "source_dir": str(source_dir),
        "output_path": str(glb_path),
        "diagnostics_path": str(resolved_diagnostics_path),
        "settings": {
            "quality_preset": resolved_quality_preset,
            "texture_size": int(texture_size),
            "target_faces": resolved_target_faces,
            "requested_simplifier_backend": requested_simplifier_backend,
            "requested_target_faces": int(target_faces) if target_faces is not None else None,
            "target_faces_source": export_settings["target_faces_source"],
            "reference_available": reference is not None,
            "reference_trace_path": str(reference["trace_path"]) if reference is not None else None,
            "reference_target_faces": reference.get("final_faces") if reference is not None else None,
            "reference_texture_size": reference.get("texture_size") if reference is not None else None,
            "reference_xatlas_face_guard": reference.get("xatlas_face_guard") if reference is not None else None,
            "grid_size": int(grid_size) if grid_size is not None else None,
            "min_component_faces": int(min_component_faces),
            "small_boundary_loop_fill_max_edges": resolved_small_boundary_loop_fill_max_edges,
            "small_boundary_loop_fill_max_perimeter": resolved_small_boundary_loop_fill_max_perimeter,
            "requested_uv_backend": str(uv_backend),
            "uv_backend": resolved_uv_backend,
            "chart_angle_degrees": resolved_chart_angle_degrees,
            "tile_padding": resolved_tile_padding,
            "tile_padding_source": tile_padding_source,
            "max_texture_pixels": int(max_texture_pixels) if max_texture_pixels is not None else None,
            "source_projection": bool(source_projection),
            "source_projection_fallback_mode": source_projection_fallback_mode,
            "source_projection_fallback_neighbors": int(source_projection_fallback_neighbors),
            "source_projection_fallback_max_distance_voxels": float(source_projection_fallback_max_distance_voxels),
            "render_padding": bool(render_padding),
            "remesh": bool(remesh),
            "remesh_band": float(remesh_band),
            "remesh_resolution": int(remesh_resolution) if remesh_resolution is not None else None,
            "remesh_project_back": float(remesh_project_back),
            "remesh_repair_nonmanifold": bool(remesh_repair_nonmanifold),
            "simplify_backend": resolved_simplify_backend,
        },
        "stages": {},
        "timings_sec": {},
        "memory_samples": {},
    }
    if fixture_manifest is not None:
        diagnostics["fixture_manifest"] = _fixture_manifest_summary(fixture_manifest)

    memory_monitor = _ProcessMemoryMonitor()

    def sample(label: str) -> None:
        diagnostics["memory_samples"][label] = memory_monitor.sample(label)

    memory_monitor.start()
    sample("start")
    decoded = _timed_stage(
        diagnostics,
        "load_npz",
        lambda: load_pixal3d_decoded_npz(shape_path, texture_path),
        memory_monitor=memory_monitor,
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
    resolved_remesh_resolution = int(remesh_resolution) if remesh_resolution is not None else resolved_grid_size
    if remesh:
        diagnostics["settings"]["remesh_resolution"] = resolved_remesh_resolution
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
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["extract_mesh"].update(_mesh_shape(mesh, "source"))
    del shape_coordinates, shape_fields
    gc.collect()
    sample("after_extract_mesh")

    pre_metrics = _timed_stage(
        diagnostics,
        "source_metrics",
        lambda: mesh_metrics(mesh.vertices, mesh.faces),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["source_metrics"]["metrics"] = pre_metrics

    cleaned, clean_stats = _timed_stage(
        diagnostics,
        "clean_mesh",
        lambda: clean_mesh(mesh.vertices, mesh.faces, min_component_faces=min_component_faces),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["clean_mesh"].update(_mesh_shape(cleaned, "cleaned"))
    diagnostics["stages"]["clean_mesh"]["stats"] = clean_stats
    del mesh
    gc.collect()
    sample("after_clean_mesh")
    source_projection_vertices = cleaned.vertices
    source_projection_faces = cleaned.faces

    simplify_source = cleaned
    if remesh:
        remeshed, remesh_stats = _timed_stage(
            diagnostics,
            "remesh",
            lambda: remesh_narrow_band(
                cleaned.vertices,
                cleaned.faces,
                resolution=resolved_remesh_resolution,
                band=remesh_band,
                project_back=remesh_project_back,
                repair_nonmanifold=remesh_repair_nonmanifold,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["remesh"].update(_mesh_shape(remeshed, "remeshed"))
        diagnostics["stages"]["remesh"]["stats"] = remesh_stats
        diagnostics["stages"]["remesh"]["metrics"] = mesh_metrics(remeshed.vertices, remeshed.faces)
        simplify_source = remeshed
        sample("after_remesh")

    simplified, simplify_stats = _timed_stage(
        diagnostics,
        "simplify_mesh",
        lambda: simplify_mesh(
            simplify_source.vertices,
            simplify_source.faces,
            target_faces=resolved_target_faces,
            min_component_faces=min_component_faces,
            backend=requested_simplifier_backend,
            small_boundary_loop_fill_max_edges=resolved_small_boundary_loop_fill_max_edges,
            small_boundary_loop_fill_max_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["simplify_mesh"].update(_mesh_shape(simplified, "simplified"))
    diagnostics["stages"]["simplify_mesh"]["stats"] = simplify_stats
    if remesh:
        simplify_stats["remesh_backend"] = "native-narrow-band-dc"
        simplify_stats["remesh_equivalence_status"] = "native-narrow-band-dc-measured"
        simplify_stats["production_blockers"] = tuple(
            blocker
            for blocker in simplify_stats.get("production_blockers", ())
            if blocker != "missing_narrow_band_dc_remesh"
        )
    del cleaned
    gc.collect()
    sample("after_simplify_mesh")

    post_metrics = _timed_stage(
        diagnostics,
        "export_metrics",
        lambda: mesh_metrics(simplified.vertices, simplified.faces),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["export_metrics"]["metrics"] = post_metrics

    def build_uv_mesh() -> NativeUvMesh:
        if resolved_uv_backend == "xatlas-equivalent-native":
            return make_reference_uvs(
                simplified.vertices,
                simplified.faces,
                texture_resolution=texture_size,
            )
        if resolved_uv_backend == "native-chart":
            return make_native_chart_uvs(
                simplified.vertices,
                simplified.faces,
                chart_angle_degrees=resolved_chart_angle_degrees,
                tile_padding=resolved_tile_padding,
            )
        return make_face_atlas_uvs(simplified.vertices, simplified.faces, tile_padding=resolved_tile_padding)

    uv_mesh = _timed_stage(diagnostics, "uv", build_uv_mesh, memory_monitor=memory_monitor)
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
            source_vertices=source_projection_vertices,
            source_faces=source_projection_faces,
            source_projection=source_projection,
            source_projection_fallback_mode=source_projection_fallback_mode,
            source_projection_fallback_neighbors=source_projection_fallback_neighbors,
            source_projection_fallback_max_distance_voxels=source_projection_fallback_max_distance_voxels,
            render_padding=render_padding,
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["texture_bake"].update(_texture_shape(baked))
    del texture_coordinates, texture_attributes, source_projection_vertices, source_projection_faces
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
        uv_stats=uv_mesh.stats,
    )
    chart_uv_candidate = _native_chart_uv_candidate_status(
        uv_mesh.stats,
        baked.stats,
        resolved_uv_backend,
    )
    quality["native_chart_uv_candidate"] = chart_uv_candidate
    quality["xatlas_chart_parity"] = _xatlas_chart_parity_summary(
        reference,
        uv_mesh.stats,
        baked.stats,
        resolved_uv_backend,
    )
    if chart_uv_candidate.get("status") == "quality_blocked":
        quality["warnings"] = tuple([*quality["warnings"], "native_chart_uv_candidate_quality_blocked"])
    quality["upstream_export_settings"] = _upstream_export_settings_summary(
        resolved_target_faces,
        texture_size,
        simplify_stats,
        baked.stats,
        quality,
    )
    quality["production_equivalence"] = _production_equivalence_summary(quality, None)
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
                "uv_backend": resolved_uv_backend,
                "uv_stats_backend": str(uv_mesh.stats.get("backend")),
                "chart_angle_degrees": resolved_chart_angle_degrees,
                "bake_backend": str(baked.stats.get("backend")),
                "coverage_ratio": float(baked.stats.get("coverage_ratio", 0.0)),
                "raw_coverage_ratio": float(baked.stats.get("raw_coverage_ratio", 0.0)),
                "simplifier_backend": quality["simplifier_backend"],
                "simplifier_quality_tier": quality["simplifier_quality_tier"],
                "production_quality_ready": bool(quality["production_quality_ready"]),
                "production_equivalence_ready": bool(quality["production_equivalence"]["ready"]),
            },
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["write_glb"]["artifact"] = glb.metadata
    glb_inspection = inspect_glb(glb.path)
    diagnostics["stages"]["write_glb"]["inspection"] = glb_inspection
    quality["glb_viewer_compatibility"] = _glb_viewer_compatibility_summary(glb_inspection)
    diagnostics["quality"] = quality
    sample("after_write_glb")

    if reference is not None:
        reference_glb = _reference_glb_path(reference)
        if reference_glb is not None:
            visual_report = _timed_stage(
                diagnostics,
                "visual_compare",
                lambda: compare_textured_glbs(
                    glb.path,
                    reference_glb,
                    output_dir=glb.path.parent / "visual_parity",
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["visual_comparison"] = _visual_comparison_summary(
                visual_report,
                quality.get("upstream_export_settings"),
                texture_stats=baked.stats,
                export_metrics=post_metrics,
            )
            quality["rendered_visual_ready"] = bool(diagnostics["visual_comparison"]["rendered_visual_ready"])
            quality["production_equivalence"] = _production_equivalence_summary(
                quality,
                diagnostics["visual_comparison"],
            )
            diagnostics["quality"] = quality

    diagnostics["result"] = {
        "ready": bool(quality["artifact_ready"]),
        "artifact_ready": bool(quality["artifact_ready"]),
        "rendered_visual_ready": bool(quality.get("rendered_visual_ready", False)),
        "production_quality_ready": bool(quality["production_quality_ready"]),
        "production_equivalence_ready": bool(quality["production_equivalence"]["ready"]),
        "remaining_parity_boundaries": quality["production_equivalence"]["remaining_parity_boundaries"],
        "equivalence_blockers": quality["production_equivalence"]["blockers"],
        "quality_warnings": quality["warnings"],
        "model_glb": str(glb.path),
        "diagnostics_json": str(resolved_diagnostics_path),
        "bytes_written": int(glb.bytes_written),
    }
    manifest_path = glb.path.parent / "artifact-manifest.json"
    run_manifest = _build_pixal3d_run_manifest(
        decoded_dir=source_dir,
        shape_path=shape_path,
        texture_path=texture_path,
        glb=glb,
        diagnostics_path=resolved_diagnostics_path,
        diagnostics=diagnostics,
        fixture_manifest=fixture_manifest,
        reference=reference,
    )
    diagnostics["artifact_manifest"] = {
        "path": str(manifest_path),
        "lineage_id": run_manifest["lineage_id"],
        "roles": tuple(run_manifest["roles"]),
    }
    _write_json_atomic(manifest_path, run_manifest)
    memory_monitor.stop()
    diagnostics["memory"] = memory_monitor.summary()
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


def _timed_stage(
    diagnostics: dict[str, Any],
    name: str,
    fn: Callable[[], _T],
    *,
    memory_monitor: _ProcessMemoryMonitor | None = None,
) -> _T:
    start = time.perf_counter()
    try:
        if memory_monitor is None:
            return fn()
        with memory_monitor.track_stage(name):
            return fn()
    except BaseException:
        if memory_monitor is not None:
            memory_monitor.stop()
        raise
    finally:
        elapsed = time.perf_counter() - start
        diagnostics["timings_sec"][name] = elapsed
        diagnostics["stages"].setdefault(name, {})["seconds"] = elapsed


class _MemoryStageScope:
    def __init__(self, monitor: _ProcessMemoryMonitor, stage: str):
        self._monitor = monitor
        self._stage = stage
        self._prior_stage = "idle"

    def __enter__(self) -> None:
        self._prior_stage = self._monitor._set_active_stage(self._stage)
        self._monitor._set_stage_boundary(self._stage, "start", self._monitor.sample(f"{self._stage}:start"))

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._monitor._set_stage_boundary(self._stage, "end", self._monitor.sample(f"{self._stage}:end"))
        self._monitor._set_active_stage(self._prior_stage)


class _ProcessMemoryMonitor:
    def __init__(
        self,
        *,
        poll_interval_sec: float = PIXAL3D_MEMORY_POLL_INTERVAL_SEC,
        sample_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        self._poll_interval_sec = float(poll_interval_sec)
        self._sample_fn = sample_fn or _memory_sample
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._active_stage = "idle"
        self._sample_count = 0
        self._peak_current_rss_bytes: int | None = None
        self._peak_current_rss_label: str | None = None
        self._peak_current_rss_stage: str | None = None
        self._peak_max_rss_bytes: int | None = None
        self._peak_max_rss_label: str | None = None
        self._peak_max_rss_stage: str | None = None
        self._last_sample: dict[str, Any] | None = None
        self._stage_peaks: dict[str, dict[str, Any]] = {}

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval_sec

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.sample("monitor_start")
        thread = threading.Thread(
            target=self._poll_loop,
            name="mlx-spatialkit-memory-monitor",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._poll_interval_sec * 4.0))
        self.sample("monitor_stop")

    def sample(self, label: str) -> dict[str, Any]:
        sample = self._sample_fn()
        self._record(label, sample)
        return sample

    def track_stage(self, stage: str) -> _MemoryStageScope:
        return _MemoryStageScope(self, stage)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stage_peaks = {stage: dict(values) for stage, values in sorted(self._stage_peaks.items())}
            return {
                "source": "process RSS from ps; high-water RSS from resource.getrusage(RUSAGE_SELF).ru_maxrss",
                "poll_interval_sec": self._poll_interval_sec,
                "sample_count": self._sample_count,
                "peak_current_rss_bytes": self._peak_current_rss_bytes,
                "peak_current_rss_label": self._peak_current_rss_label,
                "peak_current_rss_stage": self._peak_current_rss_stage,
                "peak_max_rss_bytes": self._peak_max_rss_bytes,
                "peak_max_rss_label": self._peak_max_rss_label,
                "peak_max_rss_stage": self._peak_max_rss_stage,
                "last_sample": dict(self._last_sample) if self._last_sample is not None else None,
                "stage_peaks": stage_peaks,
            }

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_sec):
            self.sample("poll")

    def _set_active_stage(self, stage: str) -> str:
        with self._lock:
            prior = self._active_stage
            self._active_stage = stage
            if stage != "idle":
                self._stage_peaks.setdefault(stage, self._empty_stage_record())
            return prior

    def _set_stage_boundary(self, stage: str, boundary: str, sample: dict[str, Any]) -> None:
        current_rss = _sample_int(sample, "current_rss_bytes")
        max_rss = _sample_int(sample, "max_rss_bytes")
        with self._lock:
            record = self._stage_peaks.setdefault(stage, self._empty_stage_record())
            record[f"{boundary}_current_rss_bytes"] = current_rss
            record[f"{boundary}_max_rss_bytes"] = max_rss

    def _record(self, label: str, sample: dict[str, Any]) -> None:
        current_rss = _sample_int(sample, "current_rss_bytes")
        max_rss = _sample_int(sample, "max_rss_bytes")
        with self._lock:
            self._sample_count += 1
            self._last_sample = dict(sample)
            stage = self._active_stage
            if current_rss is not None and (
                self._peak_current_rss_bytes is None or current_rss > self._peak_current_rss_bytes
            ):
                self._peak_current_rss_bytes = current_rss
                self._peak_current_rss_label = label
                self._peak_current_rss_stage = stage
            if max_rss is not None and (self._peak_max_rss_bytes is None or max_rss > self._peak_max_rss_bytes):
                self._peak_max_rss_bytes = max_rss
                self._peak_max_rss_label = label
                self._peak_max_rss_stage = stage
            if stage == "idle":
                return
            record = self._stage_peaks.setdefault(stage, self._empty_stage_record())
            record["sample_count"] += 1
            if current_rss is not None and (
                record["peak_current_rss_bytes"] is None or current_rss > record["peak_current_rss_bytes"]
            ):
                record["peak_current_rss_bytes"] = current_rss
                record["peak_current_rss_label"] = label
            if max_rss is not None and (
                record["peak_max_rss_bytes"] is None or max_rss > record["peak_max_rss_bytes"]
            ):
                record["peak_max_rss_bytes"] = max_rss
                record["peak_max_rss_label"] = label

    @staticmethod
    def _empty_stage_record() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "start_current_rss_bytes": None,
            "end_current_rss_bytes": None,
            "peak_current_rss_bytes": None,
            "peak_current_rss_label": None,
            "start_max_rss_bytes": None,
            "end_max_rss_bytes": None,
            "peak_max_rss_bytes": None,
            "peak_max_rss_label": None,
        }


def _sample_int(sample: dict[str, Any], key: str) -> int | None:
    value = sample.get(key)
    return None if value is None else int(value)


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
    *,
    fixture_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset = _normalize_quality_preset(quality_preset)
    reference = _load_pixal3d_reference_trace(decoded_dir, fixture_manifest=fixture_manifest)
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


def _resolve_pixal3d_uv_backend(value: str) -> str:
    backend = str(value).strip().lower().replace("_", "-")
    if backend in ("face-atlas", "native-chart", "xatlas-equivalent-native"):
        return backend
    raise ValueError(
        "uv_backend must be 'face-atlas', 'native-chart', or 'xatlas-equivalent-native'"
    )


def _resolve_chart_angle_degrees(value: float) -> float:
    angle = float(value)
    if not np.isfinite(angle) or angle < 0.0 or angle > 180.0:
        raise ValueError("chart_angle_degrees must be finite and in [0, 180]")
    return angle


def _resolve_tile_padding(value: float | None, uv_backend: str) -> tuple[float, str]:
    backend = _resolve_pixal3d_uv_backend(uv_backend)
    if value is None:
        if backend == "xatlas-equivalent-native":
            # The reference unwrap packs with texel gaps (xatlas PackOptions
            # bilinear gutter), not a fractional tile padding.
            return 0.0, "backend_default:xatlas-equivalent-native"
        if backend == "native-chart":
            return PIXAL3D_NATIVE_CHART_TILE_PADDING, "backend_default:native-chart"
        return PIXAL3D_FACE_ATLAS_TILE_PADDING, "backend_default:face-atlas"
    padding = float(value)
    if not np.isfinite(padding) or padding < 0.0 or padding >= 0.45:
        raise ValueError("tile_padding must be finite and in [0, 0.45)")
    return padding, "explicit"


def _resolve_simplify_backend(value: str | None) -> str | None:
    """Validate the explicit simplify_backend opt-in; return None or 'qem'."""
    if value is None:
        return None
    backend = str(value).strip().lower()
    if backend == "qem":
        return "qem"
    raise ValueError(
        f"simplify_backend must be None or 'qem'; got {value!r}"
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
    uv_backend = str(uv_stats.get("backend", "unknown"))
    texture_backend = str(texture_stats.get("backend", "unknown"))
    sampling_mode = str(texture_stats.get("sampling_mode", "nearest"))
    source_projection_used = texture_stats.get("source_projection_used")
    source_projection_detail = texture_stats.get("source_projection_detail")
    seam_fill_mode = str(texture_stats.get("postprocess_mode", "native-dilation-and-surface-fill"))
    reference_available = reference is not None
    xatlas_backend = str(reference.get("unwrap_backend", "")) if reference is not None else ""

    hole_fill_reference = hole_fill_algorithm in {
        "perimeter-centroid-fan",
        "cumesh-perimeter-centroid-fan",
    }
    remesh_reference = str(simplify_stats.get("remesh_backend", "")) in {
        "narrow-band-dc",
        "cumesh-narrow-band-dc",
        "native-narrow-band-dc",
    }
    simplify_reference = (
        simplifier_quality == "production"
        and ("qem" in simplifier_algorithm or "edge-collapse" in simplifier_algorithm)
    )
    unwrap_reference = uv_backend.startswith("xatlas")
    raster_reference = bool(texture_stats.get("uv_raster_interpolate_reference"))
    projection_reference = source_projection_used is True
    sampling_reference = sampling_mode == "trilinear"
    postprocess_reference = seam_fill_mode in {
        "telea-inpaint",
        "inpaint-equivalent",
        "reference-inpaint-equivalent",
    }

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
            source="packages/mlx-spatialkit/cpp/pixal3d_contracts.cpp",
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
            source="/tmp/CuMesh/src/clean_up.cu:450",
            spatialkit_backend={
                "algorithm": hole_fill_algorithm,
                "fallback_algorithm": hole_fill_fallback,
            },
            required="perimeter-limited centroid-fan boundary-loop fill",
            detail="Current projected ear-clipping or branch-cycle repair cannot claim CuMesh hole-fill parity.",
        ),
        "narrow_band_dc_remesh": _stage_status(
            passed=remesh_reference,
            status="reference_matched" if remesh_reference else "missing_or_deferred",
            source="/tmp/CuMesh/cumesh/remeshing.py:24",
            spatialkit_backend=simplify_stats.get("remesh_backend"),
            required="narrow-band dual contour remesh or measured equivalent",
            detail="Pixal3D's export path uses remesh=True with remesh_band=1.",
        ),
        "qem_simplification": _stage_status(
            passed=simplify_reference,
            status="reference_matched" if simplify_reference else "heuristic_quarantined",
            source="/tmp/CuMesh/src/simplify.cu:531",
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
            source="/tmp/CuMesh/cumesh/cumesh.py:408",
            spatialkit_backend={
                "uv_backend": uv_backend,
                "chart_cluster_normal_policy": uv_stats.get("chart_cluster_normal_policy"),
                "requires_xatlas_dependency": False,
            },
            required="xatlas/CuMesh behavior-compatible unwrap without an unapproved required xatlas dependency",
            detail=(
                f"Reference unwrap backend is {xatlas_backend or 'unknown'}; native chart remains a measured "
                "behavior candidate until chart, coverage, seam, and visual checks prove equivalence."
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
            spatialkit_backend=sampling_mode,
            required="trilinear sparse-grid PBR voxel sampling",
            detail="Nearest-voxel sampling and broad fallback fill cannot claim Pixal3D sampling parity.",
        ),
        "texture_postprocess": _stage_status(
            passed=postprocess_reference,
            status="reference_matched" if postprocess_reference else "heuristic_quarantined",
            source="vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:287",
            spatialkit_backend={
                "postprocess_mode": seam_fill_mode,
                "surface_fill_enabled": texture_stats.get("surface_fill_enabled"),
                "gutter_fill_enabled": texture_stats.get("gutter_fill_enabled"),
            },
            required="reference-like texture inpaint/fill with sample/fill diagnostics",
            detail="Native dilation, BFS surface fill, and gutter fill are useful diagnostics but remain quarantined until reference-equivalent.",
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


def _xatlas_chart_parity_summary(
    reference: dict[str, Any] | None,
    uv_stats: dict[str, Any],
    texture_stats: dict[str, Any],
    uv_backend: str,
) -> dict[str, Any]:
    uv_stats_backend = str(uv_stats.get("backend", "unknown"))
    deferred_boundary = "not_xatlas_chart_parity"
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
            Path.cwd() / "inputs" / "mlx-spatialkit" / "pixal3d-1024-cascade-glb-reference" / "trace.json",
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
    checks: dict[str, dict[str, Any]] = {
        "texture_reference_scalar_gate": {
            "passed": bool(report["summary"].get("all_passed")),
            "actual": bool(report["summary"].get("all_passed")),
            "required": True,
        }
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
        else {"path": decoded_metadata_value(diagnostics, "image_path"), "preprocess_variant": "unknown"}
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
            "kind": "native_mlx_spatialkit_glb",
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
    "make_native_chart_uvs",
    "textured_glb_payload",
    "validate_pixal3d_decoded",
    "write_textured_glb",
]
