"""Native spatial export primitives for mlx-spatial."""

from __future__ import annotations

from ._native import (
    backend_info,
    metal_device_available,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)
from .contracts import (
    DecodedOVoxelInputs,
    load_decoded_ovoxel_npz,
    validate_decoded_ovoxel,
    validate_ovoxel_shape_fields,
    validate_ovoxel_texture_attributes,
)
from .export import (
    OVoxelGlbExportResult,
    Pixal3DDecodedInputs,
    Pixal3DGlbExportResult,
    export_decoded_ovoxel_glb,
    export_pixal3d_glb,
    load_pixal3d_decoded_npz,
    validate_pixal3d_decoded,
)
from .glb import NativeGlbArtifact, textured_glb_payload, write_textured_glb
from .glb_compare import compare_textured_glbs, inspect_glb, parse_glb, png_coverage
from .mesh import (
    NativeMesh,
    bidirectional_surface_distance_metrics,
    clean_mesh,
    extract_flexi_dual_grid,
    fill_holes,
    mesh_metrics,
    point_to_mesh_distances,
    repair_nonmanifold_mesh,
    sampled_surface_to_mesh_distance_metrics,
    simplify_mesh,
    simplify_mesh_mlx_parallel_qem,
    unify_face_orientations,
)
from .texture import (
    COVERAGE_STATUS_LABELS,
    NativeTextureBakeResult,
    bake_pbr_texture,
    coverage_status_histogram,
    telea_inpaint,
)
from .uv import NativeUvMesh, make_face_atlas_uvs, make_native_chart_uvs, make_xatlas_uvs
from .xatlas import (
    XAtlasUvResult,
    resolve_xatlas_parallel_chunks,
    unwrap_xatlas,
    unwrap_xatlas_spatial,
)

__all__ = [
    "DecodedOVoxelInputs",
    "NativeMesh",
    "NativeGlbArtifact",
    "NativeTextureBakeResult",
    "NativeUvMesh",
    "OVoxelGlbExportResult",
    "Pixal3DDecodedInputs",
    "Pixal3DGlbExportResult",
    "XAtlasUvResult",
    "COVERAGE_STATUS_LABELS",
    "backend_info",
    "bake_pbr_texture",
    "bidirectional_surface_distance_metrics",
    "clean_mesh",
    "compare_textured_glbs",
    "coverage_status_histogram",
    "export_decoded_ovoxel_glb",
    "export_pixal3d_glb",
    "extract_flexi_dual_grid",
    "fill_holes",
    "inspect_glb",
    "load_decoded_ovoxel_npz",
    "load_pixal3d_decoded_npz",
    "make_face_atlas_uvs",
    "make_native_chart_uvs",
    "make_xatlas_uvs",
    "mesh_metrics",
    "metal_device_available",
    "parse_glb",
    "png_coverage",
    "point_to_mesh_distances",
    "repair_nonmanifold_mesh",
    "resolve_xatlas_parallel_chunks",
    "sampled_surface_to_mesh_distance_metrics",
    "simplify_mesh",
    "simplify_mesh_mlx_parallel_qem",
    "unify_face_orientations",
    "unwrap_xatlas",
    "unwrap_xatlas_spatial",
    "telea_inpaint",
    "textured_glb_payload",
    "validate_decoded_ovoxel",
    "validate_ovoxel_shape_fields",
    "validate_ovoxel_texture_attributes",
    "validate_pixal3d_decoded",
    "validate_pixal3d_shape_fields",
    "validate_pixal3d_texture_attributes",
    "write_textured_glb",
]
