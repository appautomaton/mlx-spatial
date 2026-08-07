"""Coordinate-system conversions shared by model export adapters."""

from __future__ import annotations

import numpy as np


GLTF_Y_UP = "gltf-y-up"
TRELLIS_Z_UP = "trellis-z-up"
SUPPORTED_SOURCE_COORDINATE_SYSTEMS = (GLTF_Y_UP, TRELLIS_Z_UP)


def vertices_to_gltf_y_up(
    vertices: np.ndarray,
    *,
    source_coordinate_system: str,
) -> np.ndarray:
    """Return vertices expressed in glTF's right-handed Y-up convention.

    TRELLIS uses a right-handed Z-up model space. Its reference exporter maps
    ``(x, y, z)`` to ``(x, z, -y)`` before serializing a GLB. The transform is
    a proper rotation, so face winding and UV coordinates remain unchanged.
    """

    values = np.asarray(vertices, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"vertices must have shape (num_vertices, 3), got {values.shape}")
    if source_coordinate_system not in SUPPORTED_SOURCE_COORDINATE_SYSTEMS:
        raise ValueError(
            "source_coordinate_system must be one of "
            f"{SUPPORTED_SOURCE_COORDINATE_SYSTEMS}, got {source_coordinate_system!r}"
        )
    if source_coordinate_system == GLTF_Y_UP:
        return np.ascontiguousarray(values)

    converted = np.empty_like(values)
    converted[:, 0] = values[:, 0]
    converted[:, 1] = values[:, 2]
    converted[:, 2] = -values[:, 1]
    return np.ascontiguousarray(converted)
