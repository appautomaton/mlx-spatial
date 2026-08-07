"""UV generation for SpatialKit mesh exports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np

from ._native import (
    make_face_atlas_uvs as _make_face_atlas_uvs,
    make_native_chart_uvs as _make_native_chart_uvs,
)
from .monitoring import _observed_substage

_T = TypeVar("_T")


@dataclass(frozen=True)
class NativeUvMesh:
    """UV-ready triangle mesh prepared by the native backend."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    stats: dict[str, Any]


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


__all__ = [
    "NativeUvMesh",
    "make_face_atlas_uvs",
    "make_native_chart_uvs",
    "make_reference_uvs",
    "make_xatlas_uvs",
]
