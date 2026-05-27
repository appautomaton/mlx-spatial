"""Native spatial export primitives for mlx-spatial."""

from __future__ import annotations

from ._native import (
    backend_info,
    metal_device_available,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .export import Pixal3DDecodedInputs, load_pixal3d_decoded_npz, validate_pixal3d_decoded
from .mesh import NativeMesh, extract_flexi_dual_grid

__all__ = [
    "NativeMesh",
    "Pixal3DDecodedInputs",
    "backend_info",
    "extract_flexi_dual_grid",
    "load_pixal3d_decoded_npz",
    "metal_device_available",
    "validate_pixal3d_decoded",
    "validate_pixal3d_shape_fields",
    "validate_pixal3d_texture_attributes",
]
