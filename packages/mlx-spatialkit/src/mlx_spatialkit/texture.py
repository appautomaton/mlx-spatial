"""Thin Python entry points for native texture functionality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._native import bake_pbr_texture_metal as _bake_pbr_texture_metal
from ._native import metal_device_available, validate_pixal3d_texture_attributes
from .export import NativeUvMesh


COVERAGE_STATUS_LABELS: dict[int, str] = {
    0: "no_face",
    1: "exact_sampled",
    2: "missing_surface",
    3: "out_of_grid",
    4: "fallback_filled",
    5: "surface_filled",
}


@dataclass(frozen=True)
class NativeTextureBakeResult:
    """PBR texture buffers produced by the native Metal bake backend."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    base_color_rgba: np.ndarray
    metallic_roughness: np.ndarray
    coverage_mask: np.ndarray
    coverage_status: np.ndarray
    texture_size: int
    voxel_count: int
    origin: tuple[float, float, float]
    voxel_size: float
    stats: dict[str, Any]


def bake_pbr_texture(
    mesh: NativeUvMesh,
    texture_coordinates: np.ndarray,
    texture_attributes: np.ndarray,
    *,
    texture_size: int,
    origin: tuple[float, float, float] = (-0.5, -0.5, -0.5),
    voxel_size: float | None = None,
    decode_resolution: int | None = None,
    max_texture_pixels: int = 1_048_576,
    source_vertices: np.ndarray | None = None,
    source_faces: np.ndarray | None = None,
    source_projection: bool = True,
    source_projection_fallback_mode: str = "knn",
    source_projection_fallback_neighbors: int = 8,
    source_projection_fallback_max_distance_voxels: float = 12.0,
    surface_fill: bool = True,
    render_padding: bool = True,
) -> NativeTextureBakeResult:
    """Bake Pixal3D/TRELLIS-style PBR textures through the native Metal backend."""

    if source_projection_fallback_mode not in {"knn", "disabled"}:
        raise ValueError("source_projection_fallback_mode must be 'knn' or 'disabled'")
    if source_projection_fallback_neighbors <= 0:
        raise ValueError("source_projection_fallback_neighbors must be positive")
    if source_projection_fallback_max_distance_voxels <= 0:
        raise ValueError("source_projection_fallback_max_distance_voxels must be positive")
    if not source_projection:
        source_vertices = None
        source_faces = None
    atlas_cols = int(mesh.stats.get("atlas_cols", 0))
    atlas_rows = int(mesh.stats.get("atlas_rows", 0))
    atlas_faces_per_tile = int(mesh.stats.get("faces_per_tile", 0))
    tile_padding = float(mesh.stats.get("tile_padding", 0.08))
    result = _bake_pbr_texture_metal(
        mesh.vertices,
        mesh.faces,
        mesh.uvs,
        texture_coordinates,
        texture_attributes,
        int(texture_size),
        tuple(float(value) for value in origin),
        float("nan") if voxel_size is None else float(voxel_size),
        -1 if decode_resolution is None else int(decode_resolution),
        atlas_cols,
        atlas_rows,
        atlas_faces_per_tile,
        tile_padding,
        int(max_texture_pixels),
        source_vertices,
        source_faces,
        source_projection_fallback_mode,
        int(source_projection_fallback_neighbors),
        float(source_projection_fallback_max_distance_voxels),
        bool(render_padding),
        bool(surface_fill),
    )
    stats = dict(result["stats"])
    coverage_status = np.asarray(result["coverage_mask"], dtype=np.uint8)
    stats["coverage_status_legend"] = dict(COVERAGE_STATUS_LABELS)
    stats["coverage_status_histogram"] = coverage_status_histogram(coverage_status)
    return NativeTextureBakeResult(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=mesh.uvs,
        base_color_rgba=np.asarray(result["base_color_rgba"], dtype=np.uint8),
        metallic_roughness=np.asarray(result["metallic_roughness"], dtype=np.uint8),
        coverage_mask=coverage_status == 1,
        coverage_status=coverage_status,
        texture_size=int(stats["texture_size"]),
        voxel_count=int(stats["voxel_count"]),
        origin=tuple(float(value) for value in stats["origin"]),
        voxel_size=float(stats["voxel_size"]),
        stats=stats,
    )


def coverage_status_histogram(coverage_status: np.ndarray) -> dict[str, int]:
    """Count every known texture coverage status plus unknown native values."""

    status = np.asarray(coverage_status, dtype=np.uint8)
    histogram = {label: 0 for label in COVERAGE_STATUS_LABELS.values()}
    histogram["unknown"] = 0
    values, counts = np.unique(status, return_counts=True)
    for value, count in zip(values.tolist(), counts.tolist(), strict=True):
        label = COVERAGE_STATUS_LABELS.get(int(value))
        if label is None:
            histogram["unknown"] += int(count)
        else:
            histogram[label] = int(count)
    return histogram


__all__ = [
    "COVERAGE_STATUS_LABELS",
    "NativeTextureBakeResult",
    "bake_pbr_texture",
    "coverage_status_histogram",
    "metal_device_available",
    "validate_pixal3d_texture_attributes",
]
