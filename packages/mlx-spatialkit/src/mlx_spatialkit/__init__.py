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
    Pixal3DGlbExportResult,
    export_pixal3d_glb,
    load_pixal3d_decoded_npz,
    make_face_atlas_uvs,
    textured_glb_payload,
    validate_pixal3d_decoded,
    write_textured_glb,
)
from .glb_compare import compare_textured_glbs, inspect_glb, parse_glb, png_coverage
from .mesh import NativeMesh, clean_mesh, extract_flexi_dual_grid, mesh_metrics, simplify_mesh
from .texture import NativeTextureBakeResult, bake_pbr_texture

__all__ = [
    "NativeMesh",
    "NativeGlbArtifact",
    "NativeTextureBakeResult",
    "NativeUvMesh",
    "Pixal3DDecodedInputs",
    "Pixal3DGlbExportResult",
    "backend_info",
    "bake_pbr_texture",
    "clean_mesh",
    "compare_textured_glbs",
    "export_pixal3d_glb",
    "extract_flexi_dual_grid",
    "inspect_glb",
    "load_pixal3d_decoded_npz",
    "make_face_atlas_uvs",
    "mesh_metrics",
    "metal_device_available",
    "parse_glb",
    "png_coverage",
    "simplify_mesh",
    "textured_glb_payload",
    "validate_pixal3d_decoded",
    "validate_pixal3d_shape_fields",
    "validate_pixal3d_texture_attributes",
    "write_textured_glb",
]
