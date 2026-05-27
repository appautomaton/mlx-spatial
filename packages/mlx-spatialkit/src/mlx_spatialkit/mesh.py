"""Thin Python entry points for native mesh functionality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._native import clean_mesh as _clean_mesh
from ._native import backend_info, extract_flexi_dual_grid as _extract_flexi_dual_grid
from ._native import mesh_metrics as _mesh_metrics
from ._native import simplify_mesh as _simplify_mesh
from ._native import validate_pixal3d_shape_fields


@dataclass(frozen=True)
class NativeMesh:
    """Triangle mesh returned by native mlx-spatialkit extraction."""

    vertices: np.ndarray
    faces: np.ndarray


def extract_flexi_dual_grid(
    coordinates: np.ndarray,
    fields: np.ndarray,
    *,
    grid_size: int,
) -> NativeMesh:
    """Extract a FlexiDualGrid triangle mesh through the native C++ backend."""

    result = _extract_flexi_dual_grid(coordinates, fields, int(grid_size))
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"]))


def mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    """Return native mesh diagnostics."""

    return dict(_mesh_metrics(vertices, faces))


def clean_mesh(vertices: np.ndarray, faces: np.ndarray, *, min_component_faces: int = 32) -> tuple[NativeMesh, dict[str, object]]:
    """Clean mesh data through the native backend."""

    result = _clean_mesh(vertices, faces, int(min_component_faces))
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_faces: int,
    min_component_faces: int = 32,
) -> tuple[NativeMesh, dict[str, object]]:
    """Simplify mesh data through the native-owned first-pass interface."""

    result = _simplify_mesh(vertices, faces, int(target_faces), int(min_component_faces))
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


__all__ = [
    "NativeMesh",
    "backend_info",
    "clean_mesh",
    "extract_flexi_dual_grid",
    "mesh_metrics",
    "simplify_mesh",
    "validate_pixal3d_shape_fields",
]
