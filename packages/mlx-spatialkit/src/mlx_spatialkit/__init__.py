"""Native spatial export primitives for mlx-spatial."""

from __future__ import annotations

from ._native import (
    backend_info,
    metal_device_available,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .export import Pixal3DDecodedInputs, load_pixal3d_decoded_npz, validate_pixal3d_decoded
from .mesh import NativeMesh, clean_mesh, extract_flexi_dual_grid, mesh_metrics, simplify_mesh

__all__ = [
    "NativeMesh",
    "Pixal3DDecodedInputs",
    "backend_info",
    "clean_mesh",
    "extract_flexi_dual_grid",
    "load_pixal3d_decoded_npz",
    "mesh_metrics",
    "metal_device_available",
    "simplify_mesh",
    "validate_pixal3d_decoded",
    "validate_pixal3d_shape_fields",
    "validate_pixal3d_texture_attributes",
]
