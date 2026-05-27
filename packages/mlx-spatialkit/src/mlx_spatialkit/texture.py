"""Thin Python entry points for native texture functionality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._native import bake_pbr_texture_metal as _bake_pbr_texture_metal
from ._native import metal_device_available, validate_pixal3d_texture_attributes
from .export import NativeUvMesh


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
) -> NativeTextureBakeResult:
    """Bake Pixal3D/TRELLIS-style PBR textures through the native Metal backend."""

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
    )
    stats = dict(result["stats"])
    coverage_status = np.asarray(result["coverage_mask"], dtype=np.uint8)
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


__all__ = [
    "NativeTextureBakeResult",
    "bake_pbr_texture",
    "metal_device_available",
    "validate_pixal3d_texture_attributes",
]
