"""Native SAM 3D Objects gaussian PLY export helpers."""

from __future__ import annotations

import json
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .export_utils import fill_texture_holes_ndimage, pad_glb_buffer, rasterize_uv_positions, texture_png_payload
from .ovoxel import FlexibleDualGridMesh, MeshHoleFillStats, fill_flexible_dual_grid_mesh_holes


SAM3D_SH_C0 = 0.28209479177387814
SAM3D_GLB_DEFAULT_TARGET_FACES = 300_000
SAM3D_XATLAS_FACE_GUARD = 400_000
SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACES = 256
SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACE_FRACTION = 5e-4


SAM3D_GAUSSIAN_PLY_FIELDS = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


@dataclass(frozen=True)
class Sam3dGaussianPlyStats:
    """Metadata for a written SAM3D gaussian splat PLY."""

    path: Path
    vertex_count: int
    bytes_written: int
    format: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class Sam3dGlbStats:
    """Metadata for a written SAM3D basic mesh GLB."""

    path: Path
    vertex_count: int
    face_count: int
    bytes_written: int
    has_vertex_color: bool
    has_normals: bool = True
    has_texture: bool = False
    format: str = "glb"


@dataclass(frozen=True)
class Sam3dMeshPostprocessStats:
    """Metadata for SAM3D preview mesh cleanup before GLB export."""

    mode: str
    original_vertices: int
    original_faces: int
    invalid_faces_removed: int
    duplicate_faces_removed: int
    degenerate_faces_removed: int
    unreferenced_vertices_removed: int
    min_component_faces: int
    min_component_face_fraction: float
    components_before: int
    components_removed: int
    component_faces_removed: int
    post_simplify_components_removed: int
    post_simplify_component_faces_removed: int
    components_after: int
    hole_fill: MeshHoleFillStats
    cleaned_vertices: int
    cleaned_faces: int
    smoothed: bool
    smoothing_iterations: int
    simplified: bool
    simplification_target_faces: int
    final_vertices: int
    final_faces: int
    boundary_edges: int
    nonmanifold_edges: int
    nonmanifold_edges_including_boundary: int
    has_vertex_color: bool
    has_normals: bool


@dataclass(frozen=True)
class Sam3dMeshPostprocessResult:
    """Cleaned SAM3D preview mesh payload ready for GLB writing."""

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray | None
    normals: np.ndarray
    stats: Sam3dMeshPostprocessStats


@dataclass(frozen=True)
class Sam3dGaussianTextureBakeStats:
    """Metadata for SAM3D Gaussian-to-mesh texture baking."""

    backend: str
    texture_size: int
    gaussian_count: int
    k_neighbors: int
    texel_chunk_size: int
    sampled_texel_count: int
    raster_texel_count: int
    raw_coverage_ratio: float
    final_coverage_ratio: float
    unwrap_backend: str
    xatlas_face_guard: int
    unwrap_seconds: float | None
    unwrap_chunks: int
    unwrap_chart_count: int | None
    unwrap_utilization: float | None
    elapsed_seconds: float


@dataclass(frozen=True)
class Sam3dGaussianTextureBakeResult:
    """Textured SAM3D preview mesh payload ready for GLB writing."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    base_color_rgba: np.ndarray
    coverage_mask: np.ndarray
    stats: Sam3dGaussianTextureBakeStats


def write_sam3d_gaussians_ply(
    path: str | Path,
    *,
    xyz: np.ndarray,
    features_dc: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    binary: bool = True,
) -> Sam3dGaussianPlyStats:
    """Write official-field SAM3D gaussian splats as PLY without torch/plyfile."""

    rows = pack_sam3d_gaussian_rows(
        xyz=xyz,
        features_dc=features_dc,
        opacity=opacity,
        scale=scale,
        rotation=rotation,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    if binary:
        payload = _binary_ply_payload(rows)
        fmt = "binary_little_endian"
    else:
        payload = _ascii_ply_payload(rows)
        fmt = "ascii"
    try:
        if binary:
            tmp_path.write_bytes(payload)
        else:
            tmp_path.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(tmp_path, output_path)
    except OSError:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return Sam3dGaussianPlyStats(
        path=output_path,
        vertex_count=int(rows.shape[0]),
        bytes_written=output_path.stat().st_size,
        format=fmt,
        fields=SAM3D_GAUSSIAN_PLY_FIELDS,
    )


def pack_sam3d_gaussian_rows(
    *,
    xyz: np.ndarray,
    features_dc: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
) -> np.ndarray:
    """Pack gaussian arrays into official SAM3D PLY column order."""

    xyz_np = _as_float_matrix(xyz, 3, "xyz")
    features_np = _coerce_features_dc(features_dc, xyz_np.shape[0])
    opacity_np = _as_float_matrix(opacity, 1, "opacity")
    scale_np = _as_float_matrix(scale, 3, "scale")
    rotation_np = _as_float_matrix(rotation, 4, "rotation")
    expected = xyz_np.shape[0]
    for label, array in (
        ("features_dc", features_np),
        ("opacity", opacity_np),
        ("scale", scale_np),
        ("rotation", rotation_np),
    ):
        if array.shape[0] != expected:
            raise ValueError(f"{label} row count {array.shape[0]} does not match xyz row count {expected}")

    normals = np.zeros_like(xyz_np, dtype=np.float32)
    return np.concatenate(
        (xyz_np, normals, features_np, opacity_np, scale_np, rotation_np),
        axis=1,
    ).astype(np.float32, copy=False)


def _coerce_features_dc(features: np.ndarray, rows: int) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.shape == (rows, 1, 3):
        return array[:, 0, :]
    if array.shape == (rows, 3, 1):
        return array[:, :, 0]
    return _as_float_matrix(array, 3, "features_dc")


def _as_float_matrix(value: np.ndarray, width: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and width == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (N, {width}); got {array.shape}")
    return array


def _ply_header(vertex_count: int, fmt: str) -> str:
    lines = [
        "ply",
        f"format {fmt} 1.0",
        f"element vertex {vertex_count}",
    ]
    lines.extend(f"property float {field}" for field in SAM3D_GAUSSIAN_PLY_FIELDS)
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _binary_ply_payload(rows: np.ndarray) -> bytes:
    header = _ply_header(int(rows.shape[0]), "binary_little_endian").encode("ascii")
    return header + rows.astype("<f4", copy=False).tobytes(order="C")


def _ascii_ply_payload(rows: np.ndarray) -> str:
    header = _ply_header(int(rows.shape[0]), "ascii")
    lines = [" ".join(f"{value:.8g}" for value in row) for row in rows]
    return header + "\n".join(lines) + ("\n" if lines else "")


def read_sam3d_gaussian_ply_vertex_count(path: str | Path) -> int:
    """Read only the vertex count from an official-field gaussian PLY header."""

    with Path(path).open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY header ended before element vertex was found")
            decoded = line.decode("ascii", errors="replace").strip()
            if decoded.startswith("element vertex "):
                return int(decoded.split()[-1])
            if decoded == "end_header":
                raise ValueError("PLY header did not contain element vertex")


def sam3d_binary_row_size() -> int:
    """Return the byte width of one binary SAM3D gaussian vertex row."""

    return struct.calcsize("<" + "f" * len(SAM3D_GAUSSIAN_PLY_FIELDS))


def sam3d_basic_glb_payload(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> bytes:
    """Build a self-contained GLB 2.0 payload for the basic SAM3D mesh path."""

    vertices_np, faces_np, colors_np, normals_np = _validate_glb_mesh_arrays(vertices, faces, colors, normals)
    indices = faces_np.reshape(-1)
    if int(indices.max()) <= np.iinfo(np.uint16).max:
        index_array = indices.astype("<u2", copy=False)
        index_component_type = 5123
    else:
        index_array = indices.astype("<u4", copy=False)
        index_component_type = 5125

    bin_blob = bytearray()
    buffer_views: list[dict[str, object]] = []

    def add_buffer_view(payload: bytes, *, target: int | None = None) -> int:
        pad_glb_buffer(bin_blob, pad_byte=0)
        offset = len(bin_blob)
        bin_blob.extend(payload)
        view: dict[str, object] = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    position_view = add_buffer_view(np.ascontiguousarray(vertices_np, dtype="<f4").tobytes(), target=34962)
    normal_view = add_buffer_view(np.ascontiguousarray(normals_np, dtype="<f4").tobytes(), target=34962)
    color_view = None
    if colors_np is not None:
        color_view = add_buffer_view(np.ascontiguousarray(colors_np, dtype="<f4").tobytes(), target=34962)
    index_view = add_buffer_view(np.ascontiguousarray(index_array).tobytes(), target=34963)
    pad_glb_buffer(bin_blob, pad_byte=0)

    attributes: dict[str, int] = {"POSITION": 0, "NORMAL": 1}
    accessors: list[dict[str, object]] = [
        {
            "bufferView": position_view,
            "byteOffset": 0,
            "componentType": 5126,
            "count": int(vertices_np.shape[0]),
            "type": "VEC3",
            "min": [float(value) for value in vertices_np.min(axis=0)],
            "max": [float(value) for value in vertices_np.max(axis=0)],
        }
    ]
    accessors.append(
        {
            "bufferView": normal_view,
            "byteOffset": 0,
            "componentType": 5126,
            "count": int(normals_np.shape[0]),
            "type": "VEC3",
            "min": [float(value) for value in normals_np.min(axis=0)],
            "max": [float(value) for value in normals_np.max(axis=0)],
        }
    )
    if color_view is not None and colors_np is not None:
        attributes["COLOR_0"] = len(accessors)
        accessors.append(
            {
                "bufferView": color_view,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(colors_np.shape[0]),
                "type": "VEC3",
                "min": [float(value) for value in colors_np.min(axis=0)],
                "max": [float(value) for value in colors_np.max(axis=0)],
            }
        )
    index_accessor = len(accessors)
    accessors.append(
        {
            "bufferView": index_view,
            "byteOffset": 0,
            "componentType": index_component_type,
            "count": int(index_array.size),
            "type": "SCALAR",
            "min": [int(indices.min())],
            "max": [int(indices.max())],
        }
    )

    document = {
        "asset": {"version": "2.0", "generator": "mlx-spatial SAM3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "SAM3D_BasicMesh"}],
        "meshes": [
            {
                "name": "SAM3D_BasicMesh",
                "primitives": [{"attributes": attributes, "indices": index_accessor, "material": 0}],
            }
        ],
        "materials": [
            {
                "name": "SAM3D_BasicMaterial",
                "doubleSided": True,
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.8,
                },
            }
        ],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((4 - len(json_payload) % 4) % 4)
    binary_payload = bytes(bin_blob)
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total_length),
            struct.pack("<I4s", len(json_payload), b"JSON"),
            json_payload,
            struct.pack("<I4s", len(binary_payload), b"BIN\x00"),
            binary_payload,
        ]
    )


def write_sam3d_basic_glb(
    path: str | Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> Sam3dGlbStats:
    """Write a Blender-readable basic mesh GLB for SAM3D mesh decoder output."""

    vertices_np, faces_np, colors_np, normals_np = _validate_glb_mesh_arrays(vertices, faces, colors, normals)
    output_path = Path(path)
    if output_path.suffix.lower() != ".glb":
        raise ValueError("SAM3D basic mesh output must use a .glb path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sam3d_basic_glb_payload(vertices=vertices_np, faces=faces_np, colors=colors_np, normals=normals_np)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, output_path)
    except OSError:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return Sam3dGlbStats(
        path=output_path,
        vertex_count=int(vertices_np.shape[0]),
        face_count=int(faces_np.shape[0]),
        bytes_written=output_path.stat().st_size,
        has_vertex_color=colors_np is not None,
        has_normals=True,
        has_texture=False,
    )


def bake_sam3d_gaussian_texture_for_glb(
    mesh: Sam3dMeshPostprocessResult | FlexibleDualGridMesh,
    *,
    gaussian_xyz: np.ndarray,
    gaussian_features_dc: np.ndarray,
    gaussian_opacity: np.ndarray,
    gaussian_scale: np.ndarray,
    texture_size: int = 1024,
    k_neighbors: int = 8,
    texel_chunk_size: int = 262_144,
    xatlas_face_guard: int = SAM3D_XATLAS_FACE_GUARD,
    xatlas_parallel_chunks: int = 0,
) -> Sam3dGaussianTextureBakeResult:
    """Bake SAM3D Gaussian DC colors onto a cleaned preview mesh."""

    if texture_size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size}")
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
    if texel_chunk_size <= 0:
        raise ValueError(f"texel_chunk_size must be positive, got {texel_chunk_size}")
    if xatlas_face_guard <= 0:
        raise ValueError(f"xatlas_face_guard must be positive, got {xatlas_face_guard}")
    if xatlas_parallel_chunks < 0:
        raise ValueError(f"xatlas_parallel_chunks must be non-negative, got {xatlas_parallel_chunks}")

    start = time.perf_counter()
    vertices, faces, normals = _mesh_arrays_for_texture_bake(mesh)
    from .trellis2_export import unwrap_trellis2_mesh_xatlas_with_stats

    unwrap = unwrap_trellis2_mesh_xatlas_with_stats(
        FlexibleDualGridMesh(vertices=vertices, faces=faces),
        face_guard=int(xatlas_face_guard),
        parallel_chunks=xatlas_parallel_chunks,
    )
    atlas_vertices = unwrap.vertices.astype(np.float32, copy=False)
    atlas_faces = unwrap.faces.astype(np.int64, copy=False)
    atlas_uvs = unwrap.uvs.astype(np.float32, copy=False)
    atlas_normals = _transfer_normals_to_unwrapped_vertices(vertices, normals, atlas_vertices)
    positions, raster_mask = rasterize_uv_positions(atlas_vertices, atlas_faces, atlas_uvs, texture_size)
    raster_texel_count = int(np.count_nonzero(raster_mask))
    if raster_texel_count == 0:
        raise ValueError("SAM3D Gaussian texture bake produced no rasterized UV texels")

    gaussian_points, gaussian_colors, gaussian_weights, gaussian_radius = _prepare_gaussian_texture_sources(
        gaussian_xyz,
        gaussian_features_dc,
        gaussian_opacity,
        gaussian_scale,
    )
    query_points = positions[raster_mask]
    sampled = _sample_gaussian_colors_chunked(
        query_points,
        gaussian_points,
        gaussian_colors,
        gaussian_weights,
        gaussian_radius,
        k_neighbors=min(k_neighbors, gaussian_points.shape[0]),
        texel_chunk_size=texel_chunk_size,
    )

    base_color = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    valid_mask = np.zeros((texture_size, texture_size), dtype=bool)
    ys, xs = np.where(raster_mask)
    base_color[ys, xs] = sampled
    valid_mask[ys, xs] = True
    raw_coverage_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)
    base_color, final_mask = fill_texture_holes_ndimage(base_color, valid_mask)
    base_color = np.power(np.clip(base_color, 0.0, 1.0), 1.0 / 2.2)
    alpha = np.ones((texture_size, texture_size, 1), dtype=np.float32)
    base_color_rgba = (np.concatenate([base_color, alpha], axis=-1) * 255.0 + 0.5).astype(np.uint8)
    elapsed = time.perf_counter() - start

    return Sam3dGaussianTextureBakeResult(
        vertices=atlas_vertices,
        faces=atlas_faces,
        normals=atlas_normals,
        uvs=atlas_uvs,
        base_color_rgba=base_color_rgba,
        coverage_mask=final_mask,
        stats=Sam3dGaussianTextureBakeStats(
            backend="gaussian-kdtree",
            texture_size=int(texture_size),
            gaussian_count=int(gaussian_points.shape[0]),
            k_neighbors=int(min(k_neighbors, gaussian_points.shape[0])),
            texel_chunk_size=int(texel_chunk_size),
            sampled_texel_count=raster_texel_count,
            raster_texel_count=raster_texel_count,
            raw_coverage_ratio=raw_coverage_ratio,
            final_coverage_ratio=float(np.count_nonzero(final_mask) / final_mask.size),
            unwrap_backend=unwrap.stats.backend,
            xatlas_face_guard=int(xatlas_face_guard),
            unwrap_seconds=unwrap.stats.elapsed_seconds,
            unwrap_chunks=unwrap.stats.chunks,
            unwrap_chart_count=unwrap.stats.chart_count,
            unwrap_utilization=unwrap.stats.utilization,
            elapsed_seconds=float(elapsed),
        ),
    )


def sam3d_gaussian_dc_to_rgb(features_dc: np.ndarray) -> np.ndarray:
    """Convert SAM3D Gaussian DC SH coefficients to preview RGB."""

    features = _coerce_features_dc(features_dc, np.asarray(features_dc).shape[0])
    return np.clip(features * np.float32(SAM3D_SH_C0) + 0.5, 0.0, 1.0).astype(np.float32, copy=False)


def sam3d_textured_glb_payload(baked_texture: Sam3dGaussianTextureBakeResult) -> bytes:
    """Build a self-contained textured GLB 2.0 payload for SAM3D preview meshes."""

    vertices = np.asarray(baked_texture.vertices, dtype=np.float32)
    faces = np.asarray(baked_texture.faces, dtype=np.int64)
    normals = np.asarray(baked_texture.normals, dtype=np.float32)
    uvs = np.asarray(baked_texture.uvs, dtype=np.float32)
    _validate_textured_glb_arrays(vertices, faces, normals, uvs, baked_texture.base_color_rgba)
    indices = faces.reshape(-1)
    if int(indices.max()) <= np.iinfo(np.uint16).max:
        index_array = indices.astype("<u2", copy=False)
        index_component_type = 5123
    else:
        index_array = indices.astype("<u4", copy=False)
        index_component_type = 5125

    bin_blob = bytearray()
    buffer_views: list[dict[str, object]] = []

    def add_buffer_view(payload: bytes, *, target: int | None = None) -> int:
        pad_glb_buffer(bin_blob, pad_byte=0)
        offset = len(bin_blob)
        bin_blob.extend(payload)
        view: dict[str, object] = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    position_view = add_buffer_view(np.ascontiguousarray(vertices, dtype="<f4").tobytes(), target=34962)
    normal_view = add_buffer_view(np.ascontiguousarray(normals, dtype="<f4").tobytes(), target=34962)
    uv_view = add_buffer_view(np.ascontiguousarray(uvs, dtype="<f4").tobytes(), target=34962)
    index_view = add_buffer_view(np.ascontiguousarray(index_array).tobytes(), target=34963)
    base_color_view = add_buffer_view(texture_png_payload(baked_texture.base_color_rgba))
    pad_glb_buffer(bin_blob, pad_byte=0)

    document = {
        "asset": {"version": "2.0", "generator": "mlx-spatial SAM3D"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "SAM3D_TexturedPreviewMesh"}],
        "meshes": [
            {
                "name": "SAM3D_TexturedPreviewMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                        "indices": 3,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "SAM3D_GaussianPreviewMaterial",
                "doubleSided": True,
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
            }
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"bufferView": base_color_view, "mimeType": "image/png", "name": "SAM3DGaussianBaseColor"}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": position_view,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(vertices.shape[0]),
                "type": "VEC3",
                "min": [float(value) for value in vertices.min(axis=0)],
                "max": [float(value) for value in vertices.max(axis=0)],
            },
            {
                "bufferView": normal_view,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(normals.shape[0]),
                "type": "VEC3",
                "min": [float(value) for value in normals.min(axis=0)],
                "max": [float(value) for value in normals.max(axis=0)],
            },
            {
                "bufferView": uv_view,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(uvs.shape[0]),
                "type": "VEC2",
                "min": [float(value) for value in uvs.min(axis=0)],
                "max": [float(value) for value in uvs.max(axis=0)],
            },
            {
                "bufferView": index_view,
                "byteOffset": 0,
                "componentType": index_component_type,
                "count": int(index_array.size),
                "type": "SCALAR",
                "min": [int(indices.min())],
                "max": [int(indices.max())],
            },
        ],
    }

    json_payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_payload += b" " * ((4 - len(json_payload) % 4) % 4)
    binary_payload = bytes(bin_blob)
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total_length),
            struct.pack("<I4s", len(json_payload), b"JSON"),
            json_payload,
            struct.pack("<I4s", len(binary_payload), b"BIN\x00"),
            binary_payload,
        ]
    )


def write_sam3d_textured_glb(path: str | Path, baked_texture: Sam3dGaussianTextureBakeResult) -> Sam3dGlbStats:
    """Write a Blender-readable textured GLB for SAM3D Gaussian preview baking."""

    output_path = Path(path)
    if output_path.suffix.lower() != ".glb":
        raise ValueError("SAM3D textured mesh output must use a .glb path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sam3d_textured_glb_payload(baked_texture)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        tmp_path.write_bytes(payload)
        os.replace(tmp_path, output_path)
    except OSError:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return Sam3dGlbStats(
        path=output_path,
        vertex_count=int(baked_texture.vertices.shape[0]),
        face_count=int(baked_texture.faces.shape[0]),
        bytes_written=output_path.stat().st_size,
        has_vertex_color=False,
        has_normals=True,
        has_texture=True,
    )


def postprocess_sam3d_mesh_for_glb(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    target_faces: int = SAM3D_GLB_DEFAULT_TARGET_FACES,
    simplify: bool = True,
    smooth_iterations: int = 0,
    min_component_faces: int = SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACES,
    min_component_face_fraction: float = SAM3D_GLB_DEFAULT_MIN_COMPONENT_FACE_FRACTION,
) -> Sam3dMeshPostprocessResult:
    """Clean SAM3D mesh-decoder output before preview GLB export."""

    if target_faces < 0:
        raise ValueError(f"target_faces must be non-negative, got {target_faces}")
    if smooth_iterations < 0:
        raise ValueError(f"smooth_iterations must be non-negative, got {smooth_iterations}")
    if min_component_faces <= 0:
        raise ValueError(f"min_component_faces must be positive, got {min_component_faces}")
    if min_component_face_fraction < 0:
        raise ValueError(f"min_component_face_fraction must be non-negative, got {min_component_face_fraction}")

    original_vertices = np.asarray(vertices, dtype=np.float32)
    original_faces = np.asarray(faces, dtype=np.int64)
    colors_np = None if colors is None else np.asarray(colors, dtype=np.float32)
    _validate_postprocess_input(original_vertices, original_faces, colors_np)

    work_vertices = original_vertices
    work_faces, invalid_removed = _remove_invalid_faces(original_faces, work_vertices.shape[0])
    if work_faces.shape[0] == 0:
        raise ValueError("SAM3D mesh cleanup removed every face as invalid")

    work_vertices, work_faces, colors_np, degenerate_removed = _remove_degenerate_faces_with_colors(
        work_vertices,
        work_faces,
        colors_np,
    )
    work_faces, duplicate_removed = _remove_duplicate_faces(work_faces)
    work_vertices, work_faces, colors_np, unreferenced_removed = _compact_mesh_with_colors(
        work_vertices,
        work_faces,
        colors_np,
    )

    components_before = _mesh_component_count(work_vertices.shape[0], work_faces)
    min_faces = max(min_component_faces, int(work_faces.shape[0] * min_component_face_fraction))
    work_vertices, work_faces, colors_np, components_removed, component_faces_removed = _remove_small_connected_components(
        work_vertices,
        work_faces,
        colors_np,
        min_component_faces=min_faces,
    )
    work_vertices, work_faces, colors_np, unreferenced_after_components = _compact_mesh_with_colors(
        work_vertices,
        work_faces,
        colors_np,
    )
    unreferenced_removed += unreferenced_after_components

    filled, hole_stats = fill_flexible_dual_grid_mesh_holes(
        FlexibleDualGridMesh(vertices=work_vertices, faces=work_faces)
    )
    if colors_np is not None and filled.vertices.shape[0] > work_vertices.shape[0]:
        colors_np = _extend_colors_to_new_vertices(work_vertices, filled.vertices, colors_np)
    work_vertices, work_faces = filled.vertices, filled.faces

    work_vertices, work_faces, colors_np, degenerate_after_fill = _remove_degenerate_faces_with_colors(
        work_vertices,
        work_faces,
        colors_np,
    )
    work_faces, duplicate_after_fill = _remove_duplicate_faces(work_faces)
    work_vertices, work_faces, colors_np, unreferenced_after_fill = _compact_mesh_with_colors(
        work_vertices,
        work_faces,
        colors_np,
    )
    degenerate_removed += degenerate_after_fill
    duplicate_removed += duplicate_after_fill
    unreferenced_removed += unreferenced_after_fill

    cleaned_vertices = int(work_vertices.shape[0])
    cleaned_faces = int(work_faces.shape[0])

    smoothed = False
    if smooth_iterations > 0:
        work_vertices = _laplacian_smooth_vertices(work_vertices, work_faces, iterations=smooth_iterations)
        smoothed = True

    simplified = False
    if simplify and target_faces > 0 and work_faces.shape[0] > target_faces:
        try:
            import fast_simplification
        except ImportError as error:
            raise ValueError("SAM3D cleaned GLB simplification requires fast-simplification") from error

        source_vertices = work_vertices.astype(np.float32, copy=True)
        source_colors = None if colors_np is None else colors_np.astype(np.float32, copy=True)
        work_vertices, work_faces = fast_simplification.simplify(
            work_vertices.astype(np.float64, copy=False),
            work_faces.astype(np.int64, copy=False),
            target_count=int(target_faces),
            agg=7.0,
        )
        work_vertices = np.asarray(work_vertices, dtype=np.float32)
        work_faces = np.asarray(work_faces, dtype=np.int64)
        work_vertices, work_faces, colors_np, degenerate_after_simplify = _remove_degenerate_faces_with_colors(
            work_vertices,
            work_faces,
            None,
        )
        work_faces, duplicate_after_simplify = _remove_duplicate_faces(work_faces)
        work_vertices, work_faces, colors_np, unreferenced_after_simplify = _compact_mesh_with_colors(
            work_vertices,
            work_faces,
            colors_np,
        )
        if source_colors is not None:
            colors_np = _nearest_vertex_colors(source_vertices, source_colors, work_vertices)
        degenerate_removed += degenerate_after_simplify
        duplicate_removed += duplicate_after_simplify
        unreferenced_removed += unreferenced_after_simplify
        simplified = True

    post_simplify_components_removed = 0
    post_simplify_component_faces_removed = 0
    if work_faces.shape[0] > 0:
        post_min_faces = max(min_component_faces, int(work_faces.shape[0] * min_component_face_fraction))
        work_vertices, work_faces, colors_np, post_simplify_components_removed, post_simplify_component_faces_removed = (
            _remove_small_connected_components(
                work_vertices,
                work_faces,
                colors_np,
                min_component_faces=post_min_faces,
            )
        )
        work_vertices, work_faces, colors_np, unreferenced_after_final_components = _compact_mesh_with_colors(
            work_vertices,
            work_faces,
            colors_np,
        )
        unreferenced_removed += unreferenced_after_final_components

    if colors_np is not None:
        colors_np = np.clip(colors_np, 0.0, 1.0).astype(np.float32, copy=False)
    if work_vertices.shape[0] == 0 or work_faces.shape[0] == 0:
        raise ValueError("SAM3D mesh postprocess produced an empty mesh")
    if np.any(work_faces < 0) or np.any(work_faces >= work_vertices.shape[0]):
        raise ValueError("SAM3D mesh postprocess produced invalid face indices")

    boundary_edges, nonmanifold_edges, nonmanifold_edges_including_boundary = _mesh_edge_stats(work_faces)
    components_after = _mesh_component_count(work_vertices.shape[0], work_faces)
    normals = compute_sam3d_vertex_normals(work_vertices, work_faces)
    stats = Sam3dMeshPostprocessStats(
        mode="cleaned",
        original_vertices=int(original_vertices.shape[0]),
        original_faces=int(original_faces.shape[0]),
        invalid_faces_removed=int(invalid_removed),
        duplicate_faces_removed=int(duplicate_removed),
        degenerate_faces_removed=int(degenerate_removed),
        unreferenced_vertices_removed=int(unreferenced_removed),
        min_component_faces=int(min_component_faces),
        min_component_face_fraction=float(min_component_face_fraction),
        components_before=int(components_before),
        components_removed=int(components_removed),
        component_faces_removed=int(component_faces_removed),
        post_simplify_components_removed=int(post_simplify_components_removed),
        post_simplify_component_faces_removed=int(post_simplify_component_faces_removed),
        components_after=int(components_after),
        hole_fill=hole_stats,
        cleaned_vertices=cleaned_vertices,
        cleaned_faces=cleaned_faces,
        smoothed=smoothed,
        smoothing_iterations=int(smooth_iterations),
        simplified=simplified,
        simplification_target_faces=int(target_faces),
        final_vertices=int(work_vertices.shape[0]),
        final_faces=int(work_faces.shape[0]),
        boundary_edges=int(boundary_edges),
        nonmanifold_edges=int(nonmanifold_edges),
        nonmanifold_edges_including_boundary=int(nonmanifold_edges_including_boundary),
        has_vertex_color=colors_np is not None,
        has_normals=True,
    )
    return Sam3dMeshPostprocessResult(
        vertices=work_vertices.astype(np.float32, copy=False),
        faces=work_faces.astype(np.int64, copy=False),
        colors=colors_np,
        normals=normals,
        stats=stats,
    )


def compute_sam3d_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals for SAM3D GLB export."""

    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(f"normal vertices must have shape (N, 3), got {vertices_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"normal faces must have shape (M, 3), got {faces_np.shape}")
    if vertices_np.shape[0] == 0 or faces_np.shape[0] == 0:
        raise ValueError("normal computation needs non-empty vertices and faces")
    if np.any(faces_np < 0) or np.any(faces_np >= vertices_np.shape[0]):
        raise ValueError("normal faces contain vertex indices outside the vertex array")

    tri = vertices_np[faces_np]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices_np, dtype=np.float32)
    for axis in range(3):
        np.add.at(normals, faces_np[:, axis], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 1e-12
    normals[valid] /= lengths[valid]
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normals.astype(np.float32, copy=False)


def _mesh_arrays_for_texture_bake(
    mesh: Sam3dMeshPostprocessResult | FlexibleDualGridMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(mesh, Sam3dMeshPostprocessResult):
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        normals = np.asarray(mesh.normals, dtype=np.float32)
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        normals = compute_sam3d_vertex_normals(vertices, faces)
    _validate_textured_glb_arrays(vertices, faces, normals, np.zeros((vertices.shape[0], 2), dtype=np.float32), np.ones((1, 1, 4), dtype=np.uint8))
    return vertices, faces, normals


def _prepare_gaussian_texture_sources(
    gaussian_xyz: np.ndarray,
    gaussian_features_dc: np.ndarray,
    gaussian_opacity: np.ndarray,
    gaussian_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = _as_float_matrix(gaussian_xyz, 3, "gaussian_xyz")
    colors = sam3d_gaussian_dc_to_rgb(gaussian_features_dc)
    opacity = _sigmoid(_as_float_matrix(gaussian_opacity, 1, "gaussian_opacity")).reshape(-1)
    scale = _as_float_matrix(gaussian_scale, 3, "gaussian_scale")
    if colors.shape[0] != xyz.shape[0] or opacity.shape[0] != xyz.shape[0] or scale.shape[0] != xyz.shape[0]:
        raise ValueError("SAM3D Gaussian texture sources must have matching row counts")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(colors)) or not np.all(np.isfinite(scale)):
        raise ValueError("SAM3D Gaussian texture sources must contain only finite values")
    if xyz.shape[0] == 0:
        raise ValueError("SAM3D Gaussian texture bake requires at least one Gaussian")
    radius = np.clip(np.exp(np.mean(scale, axis=1)), 1e-4, 0.25).astype(np.float32, copy=False)
    weights = np.clip(opacity, 0.0, 1.0).astype(np.float32, copy=False) * radius
    return xyz.astype(np.float32, copy=False), colors.astype(np.float32, copy=False), weights, radius


def _sample_gaussian_colors_chunked(
    query_points: np.ndarray,
    gaussian_points: np.ndarray,
    gaussian_colors: np.ndarray,
    gaussian_weights: np.ndarray,
    gaussian_radius: np.ndarray,
    *,
    k_neighbors: int,
    texel_chunk_size: int,
) -> np.ndarray:
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(f"SAM3D texture query points must have shape (N,3), got {query_points.shape}")
    if query_points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
    if texel_chunk_size <= 0:
        raise ValueError(f"texel_chunk_size must be positive, got {texel_chunk_size}")

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(gaussian_points, dtype=np.float32))
    sampled = np.empty((query_points.shape[0], 3), dtype=np.float32)
    for start in range(0, query_points.shape[0], texel_chunk_size):
        stop = min(start + texel_chunk_size, query_points.shape[0])
        distances, indices = tree.query(query_points[start:stop], k=k_neighbors, workers=-1)
        if np.ndim(indices) == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        idx = np.asarray(indices, dtype=np.int64)
        dist = np.asarray(distances, dtype=np.float32)
        local_radius = gaussian_radius[idx]
        local_weight = gaussian_weights[idx] * (local_radius / (dist + local_radius + 1e-8))
        weight_sum = local_weight.sum(axis=1, keepdims=True)
        fallback = weight_sum[:, 0] <= 1e-12
        if np.any(fallback):
            local_weight[fallback] = 1.0
            weight_sum = local_weight.sum(axis=1, keepdims=True)
        normalized = local_weight / np.maximum(weight_sum, 1e-12)
        sampled[start:stop] = np.sum(gaussian_colors[idx] * normalized[..., None], axis=1)
    return np.clip(sampled, 0.0, 1.0).astype(np.float32, copy=False)


def _transfer_normals_to_unwrapped_vertices(
    source_vertices: np.ndarray,
    source_normals: np.ndarray,
    unwrapped_vertices: np.ndarray,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(source_vertices, dtype=np.float32))
    _, indices = tree.query(np.asarray(unwrapped_vertices, dtype=np.float32), k=1, workers=-1)
    normals = np.asarray(source_normals, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    return normals.astype(np.float32, copy=False)


def _validate_textured_glb_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
    base_color_rgba: np.ndarray,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"textured GLB vertices must have shape (N,3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"textured GLB faces must have shape (M,3), got {faces.shape}")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("textured GLB faces contain vertex indices outside the vertex array")
    if normals.shape != vertices.shape:
        raise ValueError(f"textured GLB normals must have shape {vertices.shape}, got {normals.shape}")
    if uvs.ndim != 2 or uvs.shape != (vertices.shape[0], 2):
        raise ValueError(f"textured GLB UVs must have shape ({vertices.shape[0]},2), got {uvs.shape}")
    if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)) or not np.all(np.isfinite(uvs)):
        raise ValueError("textured GLB arrays must contain only finite values")
    if np.any(uvs < 0.0) or np.any(uvs > 1.0):
        raise ValueError("textured GLB UVs must stay in [0,1]")
    if base_color_rgba.ndim != 3 or base_color_rgba.shape[2] != 4:
        raise ValueError(f"textured GLB base color must have shape (H,W,4), got {base_color_rgba.shape}")
    if base_color_rgba.dtype != np.uint8:
        raise ValueError(f"textured GLB base color must use uint8 pixels, got {base_color_rgba.dtype}")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _validate_glb_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    normals: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)
    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3 or vertices_np.shape[0] == 0:
        raise ValueError(f"GLB vertices must have shape (N, 3), got {vertices_np.shape}")
    if not np.all(np.isfinite(vertices_np)):
        raise ValueError("GLB vertices must contain only finite values")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3 or faces_np.shape[0] == 0:
        raise ValueError(f"GLB faces must have shape (M, 3), got {faces_np.shape}")
    if np.any(faces_np < 0) or np.any(faces_np >= vertices_np.shape[0]):
        raise ValueError("GLB faces contain vertex indices outside the vertex array")
    colors_np = None if colors is None else np.asarray(colors, dtype=np.float32)
    if colors_np is not None:
        if colors_np.ndim != 2 or colors_np.shape != vertices_np.shape:
            raise ValueError(f"GLB colors must have shape {vertices_np.shape}, got {colors_np.shape}")
        if not np.all(np.isfinite(colors_np)):
            raise ValueError("GLB colors must contain only finite values")
        if np.any(colors_np < 0.0) or np.any(colors_np > 1.0):
            raise ValueError("GLB colors must stay in [0, 1]")
    normals_np = compute_sam3d_vertex_normals(vertices_np, faces_np) if normals is None else np.asarray(normals, dtype=np.float32)
    if normals_np.shape != vertices_np.shape:
        raise ValueError(f"GLB normals must have shape {vertices_np.shape}, got {normals_np.shape}")
    if not np.all(np.isfinite(normals_np)):
        raise ValueError("GLB normals must contain only finite values")
    normal_lengths = np.linalg.norm(normals_np, axis=1)
    if np.any(normal_lengths <= 1e-12):
        raise ValueError("GLB normals must be nonzero")
    normals_np = normals_np / normal_lengths[:, None]
    return vertices_np, faces_np, colors_np, normals_np.astype(np.float32, copy=False)


def _validate_postprocess_input(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"SAM3D postprocess vertices must have shape (N, 3), got {vertices.shape}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("SAM3D postprocess vertices must contain only finite values")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"SAM3D postprocess faces must have shape (M, 3), got {faces.shape}")
    if colors is not None:
        if colors.ndim != 2 or colors.shape != vertices.shape:
            raise ValueError(f"SAM3D postprocess colors must have shape {vertices.shape}, got {colors.shape}")
        if not np.all(np.isfinite(colors)):
            raise ValueError("SAM3D postprocess colors must contain only finite values")


def _remove_invalid_faces(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int]:
    keep = np.all((faces >= 0) & (faces < vertex_count), axis=1)
    return faces[keep], int(np.count_nonzero(~keep))


def _remove_degenerate_faces_with_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    *,
    area_epsilon: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = distinct & np.isfinite(area2) & (area2 > area_epsilon)
    if not np.any(keep):
        raise ValueError("SAM3D mesh cleanup removed every face as degenerate")
    return vertices, faces[keep], colors, int(np.count_nonzero(~keep))


def _remove_duplicate_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    if faces.shape[0] == 0:
        return faces, 0
    canonical = np.sort(faces, axis=1)
    _, first_indices = np.unique(canonical, axis=0, return_index=True)
    first_indices.sort()
    return faces[first_indices], int(faces.shape[0] - first_indices.shape[0])


def _compact_mesh_with_colors(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    if faces.shape[0] == 0:
        raise ValueError("SAM3D mesh cleanup produced no faces")
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_vertices = vertices[used]
    compact_faces = inverse.reshape(faces.shape).astype(np.int64, copy=False)
    compact_colors = None if colors is None else colors[used]
    return compact_vertices, compact_faces, compact_colors, int(vertices.shape[0] - used.shape[0])


def _mesh_component_count(vertex_count: int, faces: np.ndarray) -> int:
    parent, find, union = _mesh_union_find(vertex_count)
    for a, b, c in faces:
        union(int(a), int(b))
        union(int(a), int(c))
    face_roots = np.fromiter((find(int(face[0])) for face in faces), dtype=np.int64, count=faces.shape[0])
    return int(np.unique(face_roots).shape[0])


def _remove_small_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
    *,
    min_component_faces: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, int]:
    if faces.shape[0] == 0:
        raise ValueError("SAM3D mesh cleanup produced no faces")

    parent, find, union = _mesh_union_find(vertices.shape[0])
    for a, b, c in faces:
        union(int(a), int(b))
        union(int(a), int(c))

    face_roots = np.fromiter((find(int(face[0])) for face in faces), dtype=np.int64, count=faces.shape[0])
    roots, counts = np.unique(face_roots, return_counts=True)
    largest_root = roots[int(np.argmax(counts))]
    keep_roots = {int(root) for root, count in zip(roots, counts, strict=True) if count >= min_component_faces}
    keep_roots.add(int(largest_root))
    keep = np.fromiter((int(root) in keep_roots for root in face_roots), dtype=np.bool_, count=face_roots.shape[0])
    removed_faces = int(np.count_nonzero(~keep))
    if removed_faces == 0:
        return vertices, faces, colors, 0, 0
    return vertices, faces[keep], colors, int(len(roots) - len(keep_roots)), removed_faces


def _mesh_union_find(vertex_count: int):
    parent = np.arange(vertex_count, dtype=np.int64)
    rank = np.zeros(vertex_count, dtype=np.int8)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            parent[left_root] = right_root
        elif rank[left_root] > rank[right_root]:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] += 1

    return parent, find, union


def _extend_colors_to_new_vertices(
    old_vertices: np.ndarray,
    new_vertices: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    added_vertices = new_vertices[old_vertices.shape[0] :]
    if added_vertices.shape[0] == 0:
        return colors
    added_colors = _nearest_vertex_colors(old_vertices, colors, added_vertices)
    return np.concatenate([colors, added_colors], axis=0).astype(np.float32, copy=False)


def _nearest_vertex_colors(source_vertices: np.ndarray, source_colors: np.ndarray, query_vertices: np.ndarray) -> np.ndarray:
    if query_vertices.shape[0] == 0:
        return np.empty((0, source_colors.shape[1]), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise ValueError("SAM3D cleaned GLB color transfer requires scipy") from error
    tree = cKDTree(np.asarray(source_vertices, dtype=np.float32))
    _, indices = tree.query(np.asarray(query_vertices, dtype=np.float32), k=1)
    return np.asarray(source_colors, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]


def _laplacian_smooth_vertices(vertices: np.ndarray, faces: np.ndarray, *, iterations: int) -> np.ndarray:
    smoothed = vertices.astype(np.float32, copy=True)
    neighbors: list[set[int]] = [set() for _ in range(smoothed.shape[0])]
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        neighbors[ia].update((ib, ic))
        neighbors[ib].update((ia, ic))
        neighbors[ic].update((ia, ib))
    for _ in range(iterations):
        next_vertices = smoothed.copy()
        for index, adjacent in enumerate(neighbors):
            if adjacent:
                next_vertices[index] = 0.75 * smoothed[index] + 0.25 * smoothed[np.fromiter(adjacent, dtype=np.int64)].mean(axis=0)
        smoothed = next_vertices
    return smoothed.astype(np.float32, copy=False)


def _mesh_edge_stats(faces: np.ndarray) -> tuple[int, int, int]:
    directed = np.empty((faces.shape[0] * 3, 2), dtype=np.int64)
    directed[0::3] = faces[:, [0, 1]]
    directed[1::3] = faces[:, [1, 2]]
    directed[2::3] = faces[:, [2, 0]]
    keys = np.sort(directed, axis=1)
    key_view = np.ascontiguousarray(keys).view([("a", keys.dtype), ("b", keys.dtype)]).reshape(-1)
    _, counts = np.unique(key_view, return_counts=True)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    return boundary_edges, nonmanifold_edges, boundary_edges + nonmanifold_edges
