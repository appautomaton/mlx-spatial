"""Thin Python entry points for native mesh functionality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._native import backend_info, extract_flexi_dual_grid as _extract_flexi_dual_grid
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


__all__ = [
    "NativeMesh",
    "backend_info",
    "extract_flexi_dual_grid",
    "validate_pixal3d_shape_fields",
]
