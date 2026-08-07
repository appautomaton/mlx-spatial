"""Thin Python entry points for native export functionality."""

from __future__ import annotations

import gc
import json
import math
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
from .mesh import (
    NativeMesh,
    clean_mesh,
    extract_flexi_dual_grid,
    fill_holes,
    mesh_metrics,
    remesh_narrow_band,
    repair_nonmanifold_mesh,
    simplify_mesh,
    simplify_mesh_mlx_parallel_qem,
    unify_face_orientations,
)
from .monitoring import (
    PIXAL3D_MEMORY_POLL_INTERVAL_SEC,
    _ProcessMemoryMonitor,
    _observed_substage,
    _timed_stage,
)
from .pixal3d_quality import (
    _export_quality_summary,
    _native_chart_uv_candidate_status,
    _normalize_quality_preset,
    _pixal3d_reference_stage_contract as _pixal3d_reference_stage_contract,
    _resolve_chart_angle_degrees,
    _resolve_pixal3d_uv_backend,
    _resolve_simplify_backend,
    _resolve_tile_padding,
    _simplifier_backend_for_quality_preset,
    _topology_blocker_map as _topology_blocker_map,
    _xatlas_chart_parity_summary,
)
from .pixal3d_reporting import (
    _build_pixal3d_run_manifest as _build_pixal3d_run_manifest,
    _fixture_manifest_summary,
    _glb_viewer_compatibility_summary as _glb_viewer_compatibility_summary,
    _load_pixal3d_fixture_manifest as _load_pixal3d_fixture_manifest,
    _load_pixal3d_reference_trace,
    _production_equivalence_summary as _production_equivalence_summary,
    _reference_comparison,
    _reference_glb_path,
    _upstream_export_settings_summary as _upstream_export_settings_summary,
    _visual_comparison_summary,
)

_T = TypeVar("_T")

PIXAL3D_PREVIEW_TARGET_FACES = 50_000
PIXAL3D_REFERENCE_TARGET_FACES = 212_542
PIXAL3D_SMALL_BOUNDARY_LOOP_FILL_MAX_EDGES = 8
PIXAL3D_SMALL_BOUNDARY_LOOP_FILL_MAX_PERIMETER = 0.03


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
    # Populated only when export_pixal3d_glb(..., expose_raw_postprocess_inputs=True):
    # the raw pre-postprocess bake channels + coverage status (Slice-2 oracle contract).
    raw_texture_inputs: dict[str, np.ndarray] | None = None


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

    substage_timings: dict[str, float] = {}
    stage_a = _observed_substage(
        "uv.compute_charts",
        lambda: compute_uv_charts(
            source_vertices,
            source_faces,
            threshold_cone_half_angle_rad=math.radians(90.0),
            refine_iterations=0,
            global_iterations=1,
            smooth_strength=1.0,
            area_penalty_weight=0.1,
            perimeter_area_ratio_weight=0.0001,
        ),
        substage_timings,
    )
    grown = _observed_substage(
        "uv.grow_charts",
        lambda: grow_uv_charts(
            source_vertices,
            source_faces,
            cluster_ids=np.ascontiguousarray(np.asarray(stage_a["chart_ids"]), dtype=np.int64),
        ),
        substage_timings,
    )
    parameterized = _observed_substage(
        "uv.parameterize",
        lambda: parameterize_uv_charts(
            source_vertices,
            source_faces,
            np.ascontiguousarray(np.asarray(grown["chart_ids"]), dtype=np.int64),
        ),
        substage_timings,
    )
    chart_ids = np.ascontiguousarray(np.asarray(parameterized["chart_ids"]), dtype=np.int64)
    packed = _observed_substage(
        "uv.pack",
        lambda: pack_uv_charts(
            source_faces,
            chart_ids,
            np.ascontiguousarray(np.asarray(parameterized["corner_uvs"]), dtype=np.float64),
            resolution=int(texture_resolution),
            padding=float(pack_padding_texels),
        ),
        substage_timings,
    )
    packed_corner_uvs = np.asarray(packed["corner_uvs"])

    # Assemble the duplicated-vertex UV mesh: one output vertex per unique
    # (chart, source vertex) pair, deterministic via sorted unique keys.
    def assemble_uv_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        corner_chart = np.repeat(chart_ids, 3)
        corner_source = source_faces.reshape(-1)
        keys = corner_chart * np.int64(source_vertices.shape[0]) + corner_source
        unique_keys, first_index, inverse = np.unique(keys, return_index=True, return_inverse=True)
        vmap = (unique_keys % np.int64(source_vertices.shape[0])).astype(np.int64)
        return (
            source_vertices[vmap],
            np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int64),
            np.ascontiguousarray(packed_corner_uvs[first_index], dtype=np.float32),
        )

    out_vertices, out_faces, out_uvs = _observed_substage(
        "uv.assemble",
        assemble_uv_mesh,
        substage_timings,
    )

    final_metrics = _observed_substage(
        "uv.metrics",
        lambda: uv_quality_metrics(
            np.ascontiguousarray(out_vertices, dtype=np.float32),
            out_faces,
            out_uvs,
            chart_ids=chart_ids,
        ),
        substage_timings,
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
        "timings_sec": substage_timings,
    }
    return NativeUvMesh(
        vertices=np.asarray(out_vertices),
        faces=np.asarray(out_faces),
        uvs=np.asarray(out_uvs),
        stats=stats,
    )


def make_xatlas_uvs(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    clustered: bool = False,
    parallel_chunks: int = 1,
    spatial_tile_padding: float = 0.02,
) -> NativeUvMesh:
    """Create a measured Apple Silicon xatlas candidate UV mesh."""

    from ._native import compute_uv_charts, uv_quality_metrics  # noqa: PLC0415
    from .xatlas import unwrap_xatlas, unwrap_xatlas_spatial  # noqa: PLC0415

    source_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(faces, dtype=np.int64)
    if parallel_chunks <= 0:
        raise ValueError(f"parallel_chunks must be positive, got {parallel_chunks}")
    if clustered and parallel_chunks != 1:
        raise ValueError("clustered and parallel spatial xatlas modes are mutually exclusive")
    if (
        not math.isfinite(spatial_tile_padding)
        or spatial_tile_padding < 0.0
        or spatial_tile_padding >= 0.5
    ):
        raise ValueError("spatial_tile_padding must be finite and in [0, 0.5)")
    substage_timings: dict[str, float] = {}

    def observe(name: str, fn: Callable[[], _T]) -> _T:
        return _observed_substage(name, fn, substage_timings)

    clusters = None
    cluster_ids = None
    if clustered:
        clusters = observe(
            "uv.xatlas-clustered.compute_charts",
            lambda: compute_uv_charts(
                source_vertices,
                source_faces,
                threshold_cone_half_angle_rad=math.radians(90.0),
                refine_iterations=0,
                global_iterations=1,
                smooth_strength=1.0,
                area_penalty_weight=0.1,
                perimeter_area_ratio_weight=0.0001,
            ),
        )
        cluster_ids = np.ascontiguousarray(np.asarray(clusters["chart_ids"]), dtype=np.int64)

    if parallel_chunks > 1:
        result = unwrap_xatlas_spatial(
            source_vertices,
            source_faces,
            chunks=parallel_chunks,
            tile_padding=spatial_tile_padding,
            observe=observe,
        )
    else:
        result = unwrap_xatlas(
            source_vertices,
            source_faces,
            cluster_ids=cluster_ids,
            observe=observe,
        )
    metrics = observe(
        f"uv.{result.stats['backend']}.metrics",
        lambda: uv_quality_metrics(
            result.vertices,
            result.faces,
            result.uvs,
            chart_ids=result.chart_ids if np.all(result.chart_ids >= 0) else None,
        ),
    )
    source_triangles = source_vertices[source_faces].astype(np.float64, copy=False)
    source_face_areas = 0.5 * np.linalg.norm(
        np.cross(
            source_triangles[:, 1] - source_triangles[:, 0],
            source_triangles[:, 2] - source_triangles[:, 0],
        ),
        axis=1,
    )
    if not np.array_equal(
        np.sort(result.source_face_ids),
        np.arange(source_faces.shape[0], dtype=np.int64),
    ):
        raise ValueError("xatlas source_face_ids must be a permutation of all source faces")
    ordered_source_face_areas = source_face_areas[result.source_face_ids]
    total_surface_area = float(source_face_areas.sum())
    unassigned_mask = result.chart_ids < 0
    output_triangle_uvs = result.uvs[result.faces].astype(np.float64, copy=False)
    output_uv_double_area = np.abs(
        (output_triangle_uvs[:, 1, 0] - output_triangle_uvs[:, 0, 0])
        * (output_triangle_uvs[:, 2, 1] - output_triangle_uvs[:, 0, 1])
        - (output_triangle_uvs[:, 2, 0] - output_triangle_uvs[:, 0, 0])
        * (output_triangle_uvs[:, 1, 1] - output_triangle_uvs[:, 0, 1])
    )
    uv_degenerate_mask = output_uv_double_area <= 2.0e-14
    unassigned_surface_area = float(ordered_source_face_areas[unassigned_mask].sum())
    uv_degenerate_surface_area = float(ordered_source_face_areas[uv_degenerate_mask].sum())
    stats = {
        **result.stats,
        "stage_a_cluster_count": (
            int(clusters["chart_count"]) if clusters is not None else None
        ),
        "uv_overlap_count": int(metrics["uv_overlap_count"]),
        "uv_flipped_count": int(metrics["uv_flipped_count"]),
        "uv_degenerate_count": int(metrics["uv_degenerate_count"]),
        "uv_stretch_l2": float(metrics["uv_stretch_l2"]),
        "uv_stretch_linf": float(metrics["uv_stretch_linf"]),
        "uv_bbox_utilization": float(metrics["uv_bbox_utilization"]),
        "uv_total_area": float(metrics["uv_total_area"]),
        "source_surface_area": total_surface_area,
        "unassigned_surface_area": unassigned_surface_area,
        "unassigned_surface_area_ratio": (
            unassigned_surface_area / total_surface_area if total_surface_area > 0.0 else 0.0
        ),
        "uv_degenerate_surface_area": uv_degenerate_surface_area,
        "uv_degenerate_surface_area_ratio": (
            uv_degenerate_surface_area / total_surface_area if total_surface_area > 0.0 else 0.0
        ),
        "uv_flipped_face_ratio": float(metrics["uv_flipped_count"]) / max(source_faces.shape[0], 1),
        "timings_sec": substage_timings,
    }
    return NativeUvMesh(
        vertices=result.vertices,
        faces=result.faces,
        uvs=result.uvs,
        stats=stats,
    )


def textured_glb_payload(
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatial SpatialKit",
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
    generator: str = "mlx-spatial SpatialKit",
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
    xatlas_parallel_chunks: int = 4,
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
    texture_postprocess: str = "legacy-dilation",
    expose_raw_postprocess_inputs: bool = False,
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
    if resolved_simplify_backend == "mlx-qem" and not remesh:
        raise ValueError("simplify_backend='mlx-qem' requires remesh=True")
    if resolved_simplify_backend in {"single-layer-qem", "single-layer-mlx-qem"} and remesh:
        raise ValueError(
            f"simplify_backend={resolved_simplify_backend!r} requires remesh=False because it preserves "
            "the original FlexiDualGrid single-layer surface"
        )
    resolved_uv_backend = _resolve_pixal3d_uv_backend(uv_backend)
    if xatlas_parallel_chunks <= 0:
        raise ValueError("xatlas_parallel_chunks must be positive")
    if resolved_uv_backend == "xatlas-parallel-spatial" and xatlas_parallel_chunks <= 1:
        raise ValueError(
            "uv_backend='xatlas-parallel-spatial' requires xatlas_parallel_chunks > 1"
        )
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
            "xatlas_parallel_chunks": int(xatlas_parallel_chunks),
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
        remesh_metrics = _timed_stage(
            diagnostics,
            "remesh_metrics",
            lambda: mesh_metrics(remeshed.vertices, remeshed.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["remesh_metrics"]["metrics"] = remesh_metrics
        remesh_cleaned, remesh_clean_stats = _timed_stage(
            diagnostics,
            "clean_remeshed_mesh",
            lambda: clean_mesh(
                remeshed.vertices,
                remeshed.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_remeshed_mesh"].update(
            _mesh_shape(remesh_cleaned, "cleaned_remeshed")
        )
        diagnostics["stages"]["clean_remeshed_mesh"]["stats"] = remesh_clean_stats
        clean_remeshed_metrics = _timed_stage(
            diagnostics,
            "clean_remeshed_metrics",
            lambda: mesh_metrics(remesh_cleaned.vertices, remesh_cleaned.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_remeshed_metrics"]["metrics"] = clean_remeshed_metrics
        simplify_source = remesh_cleaned
        del remeshed
        sample("after_remesh")

    if requested_simplifier_backend in {"single-layer-qem", "single-layer-mlx-qem"}:
        coarse_target_faces = max(resolved_target_faces * 3, resolved_target_faces)
        if requested_simplifier_backend == "single-layer-mlx-qem":
            coarse_mlx_input, coarse_mlx_fill_stats = _timed_stage(
                diagnostics,
                "fill_pre_coarse_mlx_qem_holes",
                lambda: fill_holes(
                    simplify_source.vertices,
                    simplify_source.faces,
                    max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["fill_pre_coarse_mlx_qem_holes"].update(
                _mesh_shape(coarse_mlx_input, "filled_pre_coarse_mlx_qem")
            )
            diagnostics["stages"]["fill_pre_coarse_mlx_qem_holes"]["stats"] = coarse_mlx_fill_stats
            coarse_simplified, coarse_simplify_stats = _timed_stage(
                diagnostics,
                "coarse_simplify_mesh",
                lambda: simplify_mesh_mlx_parallel_qem(
                    coarse_mlx_input.vertices,
                    coarse_mlx_input.faces,
                    target_faces=coarse_target_faces,
                    max_rounds=128,
                    topology_policy="cumesh-reference",
                ),
                memory_monitor=memory_monitor,
            )
            coarse_simplify_stats["pipeline"] = "coarse-mlx-parallel-qem"
            coarse_simplify_stats["pre_simplify_hole_fill_backend"] = str(
                coarse_mlx_fill_stats.get("backend", "unknown")
            )
            coarse_simplify_stats["pre_simplify_hole_fill_algorithm"] = str(
                coarse_mlx_fill_stats.get("algorithm", "unknown")
            )
            coarse_simplify_stats["pre_simplify_hole_fill_filled_loops"] = int(
                coarse_mlx_fill_stats.get("filled_loops", 0)
            )
            coarse_simplify_stats["pre_simplify_hole_fill_residual_clean_boundary_loops"] = int(
                coarse_mlx_fill_stats.get("residual_clean_boundary_loops", 0)
            )
            del coarse_mlx_input
        else:
            coarse_simplified, coarse_simplify_stats = _timed_stage(
                diagnostics,
                "coarse_simplify_mesh",
                lambda: simplify_mesh(
                    simplify_source.vertices,
                    simplify_source.faces,
                    target_faces=coarse_target_faces,
                    min_component_faces=min_component_faces,
                    backend="topology-aware",
                    small_boundary_loop_fill_max_edges=resolved_small_boundary_loop_fill_max_edges,
                    small_boundary_loop_fill_max_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
        diagnostics["stages"]["coarse_simplify_mesh"].update(
            _mesh_shape(coarse_simplified, "coarse_simplified")
        )
        diagnostics["stages"]["coarse_simplify_mesh"]["stats"] = coarse_simplify_stats

        repaired, repair_stats = _timed_stage(
            diagnostics,
            "repair_nonmanifold_mesh",
            lambda: repair_nonmanifold_mesh(coarse_simplified.vertices, coarse_simplified.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["repair_nonmanifold_mesh"].update(_mesh_shape(repaired, "repaired"))
        diagnostics["stages"]["repair_nonmanifold_mesh"]["stats"] = repair_stats
        repair_metrics = _timed_stage(
            diagnostics,
            "repair_nonmanifold_metrics",
            lambda: mesh_metrics(repaired.vertices, repaired.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["repair_nonmanifold_metrics"]["metrics"] = repair_metrics

        repaired_cleaned, repaired_clean_stats = _timed_stage(
            diagnostics,
            "clean_repaired_mesh",
            lambda: clean_mesh(
                repaired.vertices,
                repaired.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_repaired_mesh"].update(
            _mesh_shape(repaired_cleaned, "cleaned_repaired")
        )
        diagnostics["stages"]["clean_repaired_mesh"]["stats"] = repaired_clean_stats
        repaired_clean_metrics = _timed_stage(
            diagnostics,
            "clean_repaired_metrics",
            lambda: mesh_metrics(repaired_cleaned.vertices, repaired_cleaned.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_repaired_metrics"]["metrics"] = repaired_clean_metrics

        if requested_simplifier_backend == "single-layer-mlx-qem":
            mlx_qem_input, mlx_qem_pre_fill_stats = _timed_stage(
                diagnostics,
                "fill_pre_mlx_qem_holes",
                lambda: fill_holes(
                    repaired_cleaned.vertices,
                    repaired_cleaned.faces,
                    max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["fill_pre_mlx_qem_holes"].update(
                _mesh_shape(mlx_qem_input, "filled_pre_mlx_qem")
            )
            diagnostics["stages"]["fill_pre_mlx_qem_holes"]["stats"] = mlx_qem_pre_fill_stats
            simplified, simplify_stats = _timed_stage(
                diagnostics,
                "simplify_mesh",
                lambda: simplify_mesh_mlx_parallel_qem(
                    mlx_qem_input.vertices,
                    mlx_qem_input.faces,
                    target_faces=resolved_target_faces,
                    max_rounds=128,
                    topology_policy="cumesh-reference",
                ),
                memory_monitor=memory_monitor,
            )
            simplify_stats["pipeline"] = "single-layer-coarse-repair-mlx-parallel-qem"
            mlx_qem_raw = simplified
            mlx_qem_raw_metrics = _timed_stage(
                diagnostics,
                "mlx_qem_raw_metrics",
                lambda: mesh_metrics(mlx_qem_raw.vertices, mlx_qem_raw.faces),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["mlx_qem_raw_metrics"]["metrics"] = mlx_qem_raw_metrics
            mlx_qem_cleaned, mlx_qem_clean_stats = _timed_stage(
                diagnostics,
                "clean_mlx_qem_mesh",
                lambda: clean_mesh(
                    mlx_qem_raw.vertices,
                    mlx_qem_raw.faces,
                    min_component_faces=min_component_faces,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["clean_mlx_qem_mesh"].update(
                _mesh_shape(mlx_qem_cleaned, "cleaned_mlx_qem")
            )
            diagnostics["stages"]["clean_mlx_qem_mesh"]["stats"] = mlx_qem_clean_stats
            mlx_qem_repaired, mlx_qem_repair_stats = _timed_stage(
                diagnostics,
                "repair_mlx_qem_mesh",
                lambda: repair_nonmanifold_mesh(mlx_qem_cleaned.vertices, mlx_qem_cleaned.faces),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["repair_mlx_qem_mesh"].update(
                _mesh_shape(mlx_qem_repaired, "repaired_mlx_qem")
            )
            diagnostics["stages"]["repair_mlx_qem_mesh"]["stats"] = mlx_qem_repair_stats
            simplified, mlx_qem_final_clean_stats = _timed_stage(
                diagnostics,
                "clean_repaired_mlx_qem_mesh",
                lambda: clean_mesh(
                    mlx_qem_repaired.vertices,
                    mlx_qem_repaired.faces,
                    min_component_faces=min_component_faces,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["clean_repaired_mlx_qem_mesh"].update(
                _mesh_shape(simplified, "cleaned_repaired_mlx_qem")
            )
            diagnostics["stages"]["clean_repaired_mlx_qem_mesh"]["stats"] = mlx_qem_final_clean_stats
            simplified, mlx_qem_hole_fill_stats = _timed_stage(
                diagnostics,
                "fill_repaired_mlx_qem_holes",
                lambda: fill_holes(
                    simplified.vertices,
                    simplified.faces,
                    max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["fill_repaired_mlx_qem_holes"].update(
                _mesh_shape(simplified, "filled_repaired_mlx_qem")
            )
            diagnostics["stages"]["fill_repaired_mlx_qem_holes"]["stats"] = mlx_qem_hole_fill_stats
            mlx_qem_repaired_metrics = _timed_stage(
                diagnostics,
                "mlx_qem_repaired_metrics",
                lambda: mesh_metrics(simplified.vertices, simplified.faces),
                memory_monitor=memory_monitor,
            )
            diagnostics["stages"]["mlx_qem_repaired_metrics"]["metrics"] = mlx_qem_repaired_metrics
            simplify_stats["qem_raw_final_vertices"] = int(mlx_qem_raw.vertices.shape[0])
            simplify_stats["qem_raw_final_faces"] = int(mlx_qem_raw.faces.shape[0])
            simplify_stats["pre_qem_hole_fill_backend"] = str(
                mlx_qem_pre_fill_stats.get("backend", "unknown")
            )
            simplify_stats["pre_qem_hole_fill_algorithm"] = str(
                mlx_qem_pre_fill_stats.get("algorithm", "unknown")
            )
            simplify_stats["pre_qem_hole_fill_filled_loops"] = int(
                mlx_qem_pre_fill_stats.get("filled_loops", 0)
            )
            simplify_stats["pre_qem_hole_fill_residual_clean_boundary_loops"] = int(
                mlx_qem_pre_fill_stats.get("residual_clean_boundary_loops", 0)
            )
            simplify_stats["pre_qem_hole_fill_max_perimeter"] = float(
                mlx_qem_pre_fill_stats.get("max_hole_perimeter", 0.0)
            )
            simplify_stats["post_qem_repair_backend"] = str(mlx_qem_repair_stats.get("backend", "unknown"))
            simplify_stats["post_qem_hole_fill_backend"] = str(
                mlx_qem_hole_fill_stats.get("backend", "unknown")
            )
            simplify_stats["post_qem_hole_fill_algorithm"] = str(
                mlx_qem_hole_fill_stats.get("algorithm", "unknown")
            )
            simplify_stats["post_qem_hole_fill_filled_loops"] = int(
                mlx_qem_hole_fill_stats.get("filled_loops", 0)
            )
            simplify_stats["post_qem_hole_fill_faces_added"] = int(
                mlx_qem_hole_fill_stats.get("faces_added", 0)
            )
            simplify_stats["post_qem_hole_fill_residual_clean_boundary_loops"] = int(
                mlx_qem_hole_fill_stats.get("residual_clean_boundary_loops", 0)
            )
            simplify_stats["post_qem_hole_fill_skipped_large_loops"] = int(
                mlx_qem_hole_fill_stats.get("skipped_large_loops", 0)
            )
            simplify_stats["post_qem_hole_fill_skipped_complex_components"] = int(
                mlx_qem_hole_fill_stats.get("skipped_complex_components", 0)
            )
            simplify_stats["post_qem_hole_fill_max_perimeter"] = float(
                mlx_qem_hole_fill_stats.get("max_hole_perimeter", 0.0)
            )
            simplify_stats["post_qem_final_vertices"] = int(simplified.vertices.shape[0])
            simplify_stats["post_qem_final_faces"] = int(simplified.faces.shape[0])
            simplify_stats["final_vertices"] = int(simplified.vertices.shape[0])
            simplify_stats["final_faces"] = int(simplified.faces.shape[0])
            simplify_stats["target_reached"] = int(mlx_qem_raw.faces.shape[0]) <= resolved_target_faces
            simplify_stats["final_cleanup_exceeds_target"] = (
                int(simplified.faces.shape[0]) > resolved_target_faces
            )
            simplify_stats["post_qem_topology_ready"] = not bool(
                mlx_qem_repaired_metrics.get("export_blocking_reasons")
            )
            del mlx_qem_raw
            del mlx_qem_input
            del mlx_qem_cleaned
            del mlx_qem_repaired
        else:
            simplified, simplify_stats = _timed_stage(
                diagnostics,
                "simplify_mesh",
                lambda: simplify_mesh(
                    repaired_cleaned.vertices,
                    repaired_cleaned.faces,
                    target_faces=resolved_target_faces,
                    min_component_faces=min_component_faces,
                    backend="qem",
                    small_boundary_loop_fill_max_edges=resolved_small_boundary_loop_fill_max_edges,
                    small_boundary_loop_fill_max_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
                ),
                memory_monitor=memory_monitor,
            )
            simplify_stats["pipeline"] = "single-layer-coarse-repair-qem"
        simplify_stats["requested_backend"] = requested_simplifier_backend
        simplify_stats["coarse_backend"] = "topology-aware"
        simplify_stats["coarse_target_faces"] = coarse_target_faces
        simplify_stats["coarse_final_faces"] = int(coarse_simplified.faces.shape[0])
        simplify_stats["repair_backend"] = str(repair_stats.get("backend", "unknown"))
        simplify_stats["remesh_backend"] = "single-layer-flexi-dual-grid"
        simplify_stats["remesh_equivalence_status"] = "single-layer-alternative-measurement-pending"
        simplify_stats["remesh_surface_representation"] = "original-flexi-dual-grid-surface"
        simplify_stats["remesh_single_surface_ready"] = False
        del coarse_simplified
        del repaired
        del repaired_cleaned
    elif requested_simplifier_backend == "mlx-qem":
        mlx_qem_raw, simplify_stats = _timed_stage(
            diagnostics,
            "simplify_mesh",
            lambda: simplify_mesh_mlx_parallel_qem(
                simplify_source.vertices,
                simplify_source.faces,
                target_faces=resolved_target_faces,
                max_rounds=128,
                topology_policy="manifold-preserving",
            ),
            memory_monitor=memory_monitor,
        )
        simplify_stats["pipeline"] = "narrow-band-remesh-mlx-parallel-qem"
        mlx_qem_raw_metrics = _timed_stage(
            diagnostics,
            "mlx_qem_raw_metrics",
            lambda: mesh_metrics(mlx_qem_raw.vertices, mlx_qem_raw.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["mlx_qem_raw_metrics"]["metrics"] = mlx_qem_raw_metrics
        mlx_qem_cleaned, mlx_qem_clean_stats = _timed_stage(
            diagnostics,
            "clean_mlx_qem_mesh",
            lambda: clean_mesh(
                mlx_qem_raw.vertices,
                mlx_qem_raw.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_mlx_qem_mesh"].update(
            _mesh_shape(mlx_qem_cleaned, "cleaned_mlx_qem")
        )
        diagnostics["stages"]["clean_mlx_qem_mesh"]["stats"] = mlx_qem_clean_stats
        mlx_qem_repaired, mlx_qem_repair_stats = _timed_stage(
            diagnostics,
            "repair_mlx_qem_mesh",
            lambda: repair_nonmanifold_mesh(mlx_qem_cleaned.vertices, mlx_qem_cleaned.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["repair_mlx_qem_mesh"].update(
            _mesh_shape(mlx_qem_repaired, "repaired_mlx_qem")
        )
        diagnostics["stages"]["repair_mlx_qem_mesh"]["stats"] = mlx_qem_repair_stats
        simplified, mlx_qem_final_clean_stats = _timed_stage(
            diagnostics,
            "clean_repaired_mlx_qem_mesh",
            lambda: clean_mesh(
                mlx_qem_repaired.vertices,
                mlx_qem_repaired.faces,
                min_component_faces=min_component_faces,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["clean_repaired_mlx_qem_mesh"].update(
            _mesh_shape(simplified, "cleaned_repaired_mlx_qem")
        )
        diagnostics["stages"]["clean_repaired_mlx_qem_mesh"]["stats"] = mlx_qem_final_clean_stats
        simplified, mlx_qem_hole_fill_stats = _timed_stage(
            diagnostics,
            "fill_repaired_mlx_qem_holes",
            lambda: fill_holes(
                simplified.vertices,
                simplified.faces,
                max_hole_perimeter=resolved_small_boundary_loop_fill_max_perimeter,
            ),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["fill_repaired_mlx_qem_holes"].update(
            _mesh_shape(simplified, "filled_repaired_mlx_qem")
        )
        diagnostics["stages"]["fill_repaired_mlx_qem_holes"]["stats"] = mlx_qem_hole_fill_stats
        simplify_stats["qem_raw_final_vertices"] = int(mlx_qem_raw.vertices.shape[0])
        simplify_stats["qem_raw_final_faces"] = int(mlx_qem_raw.faces.shape[0])
        simplify_stats["post_qem_repair_backend"] = str(mlx_qem_repair_stats.get("backend", "unknown"))
        simplify_stats["post_qem_hole_fill_backend"] = str(
            mlx_qem_hole_fill_stats.get("backend", "unknown")
        )
        simplify_stats["post_qem_hole_fill_filled_loops"] = int(
            mlx_qem_hole_fill_stats.get("filled_loops", 0)
        )
        simplify_stats["post_qem_hole_fill_residual_clean_boundary_loops"] = int(
            mlx_qem_hole_fill_stats.get("residual_clean_boundary_loops", 0)
        )
        simplify_stats["post_qem_final_vertices"] = int(simplified.vertices.shape[0])
        simplify_stats["post_qem_final_faces"] = int(simplified.faces.shape[0])
        simplify_stats["final_vertices"] = int(simplified.vertices.shape[0])
        simplify_stats["final_faces"] = int(simplified.faces.shape[0])
        simplify_stats["target_reached"] = int(mlx_qem_raw.faces.shape[0]) <= resolved_target_faces
        simplify_stats["final_cleanup_exceeds_target"] = int(simplified.faces.shape[0]) > resolved_target_faces
        del mlx_qem_raw
        del mlx_qem_cleaned
        del mlx_qem_repaired
    else:
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
    if requested_simplifier_backend in {"mlx-qem", "single-layer-qem", "single-layer-mlx-qem"}:
        simplified, orientation_stats = _timed_stage(
            diagnostics,
            "unify_face_orientations",
            lambda: unify_face_orientations(simplified.vertices, simplified.faces),
            memory_monitor=memory_monitor,
        )
        diagnostics["stages"]["unify_face_orientations"].update(
            _mesh_shape(simplified, "orientation_unified")
        )
        diagnostics["stages"]["unify_face_orientations"]["stats"] = orientation_stats
        simplify_stats["orientation_backend"] = str(orientation_stats.get("backend", "unknown"))
        simplify_stats["orientation_algorithm"] = str(orientation_stats.get("algorithm", "unknown"))
        simplify_stats["orientation_faces_flipped"] = int(orientation_stats.get("faces_flipped", 0))
        simplify_stats["orientation_consistent"] = bool(
            orientation_stats.get("orientation_consistent", False)
        )
    diagnostics["stages"]["simplify_mesh"].update(_mesh_shape(simplified, "simplified"))
    diagnostics["stages"]["simplify_mesh"]["stats"] = simplify_stats
    if remesh:
        simplify_stats["remesh_backend"] = "native-udf-double-cover-dc"
        simplify_stats["remesh_equivalence_status"] = "cumesh-reference-mechanism-matched"
        simplify_stats["remesh_surface_representation"] = "udf-offset-double-cover"
        simplify_stats["remesh_single_surface_ready"] = False
        simplify_stats["remesh_reference_mechanism_ready"] = True
    del simplify_source
    if remesh:
        del remesh_cleaned
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
        if resolved_uv_backend == "xatlas-global":
            return make_xatlas_uvs(simplified.vertices, simplified.faces, clustered=False)
        if resolved_uv_backend == "xatlas-clustered":
            return make_xatlas_uvs(simplified.vertices, simplified.faces, clustered=True)
        if resolved_uv_backend == "xatlas-parallel-spatial":
            return make_xatlas_uvs(
                simplified.vertices,
                simplified.faces,
                parallel_chunks=xatlas_parallel_chunks,
                spatial_tile_padding=resolved_tile_padding,
            )
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
            postprocess=texture_postprocess,
            expose_raw_postprocess_inputs=expose_raw_postprocess_inputs,
        ),
        memory_monitor=memory_monitor,
    )
    diagnostics["stages"]["texture_bake"].update(_texture_shape(baked))
    raw_texture_inputs: dict[str, np.ndarray] | None = None
    if expose_raw_postprocess_inputs and baked.raw_base_color_rgba is not None:
        raw_texture_inputs = {
            "raw_base_color_rgba": baked.raw_base_color_rgba,
            "raw_metallic_roughness": baked.raw_metallic_roughness,
            "raw_coverage_status": baked.raw_coverage_status,
        }
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
            generator="mlx-spatial SpatialKit Pixal3D",
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
                "xatlas_parallel_chunks": int(xatlas_parallel_chunks),
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
    return Pixal3DGlbExportResult(
        glb=glb,
        diagnostics_path=resolved_diagnostics_path,
        diagnostics=diagnostics,
        raw_texture_inputs=raw_texture_inputs,
    )


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
    "make_reference_uvs",
    "make_xatlas_uvs",
    "textured_glb_payload",
    "validate_pixal3d_decoded",
    "write_textured_glb",
]
