"""Native spatial export primitives for mlx-spatial."""

from __future__ import annotations

from ._native import (
    backend_info,
    metal_device_available,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .export import (
    NativeGlbArtifact,
    NativeUvMesh,
    Pixal3DDecodedInputs,
    load_pixal3d_decoded_npz,
    make_face_atlas_uvs,
    textured_glb_payload,
    validate_pixal3d_decoded,
    write_textured_glb,
)
from .mesh import NativeMesh, clean_mesh, extract_flexi_dual_grid, mesh_metrics, simplify_mesh

__all__ = [
    "NativeMesh",
    "NativeGlbArtifact",
    "NativeUvMesh",
    "Pixal3DDecodedInputs",
    "backend_info",
    "clean_mesh",
    "extract_flexi_dual_grid",
    "load_pixal3d_decoded_npz",
    "make_face_atlas_uvs",
    "mesh_metrics",
    "metal_device_available",
    "simplify_mesh",
    "textured_glb_payload",
    "validate_pixal3d_decoded",
    "validate_pixal3d_shape_fields",
    "validate_pixal3d_texture_attributes",
    "write_textured_glb",
]
