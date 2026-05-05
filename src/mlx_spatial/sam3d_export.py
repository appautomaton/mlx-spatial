"""Native SAM 3D Objects gaussian PLY export helpers."""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
    format: str = "glb"


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
    if binary:
        payload = _binary_ply_payload(rows)
        output_path.write_bytes(payload)
        fmt = "binary_little_endian"
    else:
        payload = _ascii_ply_payload(rows)
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        fmt = "ascii"
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
) -> bytes:
    """Build a self-contained GLB 2.0 payload for the basic SAM3D mesh path."""

    vertices_np, faces_np, colors_np = _validate_glb_mesh_arrays(vertices, faces, colors)
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
        _pad_glb_buffer(bin_blob, pad_byte=0)
        offset = len(bin_blob)
        bin_blob.extend(payload)
        view: dict[str, object] = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    position_view = add_buffer_view(np.ascontiguousarray(vertices_np, dtype="<f4").tobytes(), target=34962)
    color_view = None
    if colors_np is not None:
        color_view = add_buffer_view(np.ascontiguousarray(colors_np, dtype="<f4").tobytes(), target=34962)
    index_view = add_buffer_view(np.ascontiguousarray(index_array).tobytes(), target=34963)
    _pad_glb_buffer(bin_blob, pad_byte=0)

    attributes: dict[str, int] = {"POSITION": 0}
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
) -> Sam3dGlbStats:
    """Write a Blender-readable basic mesh GLB for SAM3D mesh decoder output."""

    vertices_np, faces_np, colors_np = _validate_glb_mesh_arrays(vertices, faces, colors)
    output_path = Path(path)
    if output_path.suffix.lower() != ".glb":
        raise ValueError("SAM3D basic mesh output must use a .glb path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sam3d_basic_glb_payload(vertices=vertices_np, faces=faces_np, colors=colors_np)
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
    )


def _validate_glb_mesh_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
    return vertices_np, faces_np, colors_np


def _pad_glb_buffer(buffer: bytearray, *, pad_byte: int) -> None:
    padding = (4 - len(buffer) % 4) % 4
    if padding:
        buffer.extend(bytes([pad_byte]) * padding)
