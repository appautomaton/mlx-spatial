"""Shape-only export helpers for TRELLIS.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .ovoxel import FlexibleDualGridMesh


SUPPORTED_TRELLIS2_EXPORT_SUFFIXES = (".glb", ".obj")


@dataclass(frozen=True)
class Trellis2ExportArtifact:
    """Written TRELLIS.2 shape artifact metadata."""

    path: Path
    format: str
    bytes_written: int
    detail: str


def validate_trellis2_export_path(
    output_path: str | Path,
    *,
    outputs_root: str | Path | None = None,
    suffixes: tuple[str, ...] = SUPPORTED_TRELLIS2_EXPORT_SUFFIXES,
) -> Path:
    """Validate a TRELLIS.2 export suffix and optionally constrain its root."""

    path = Path(output_path)
    normalized_suffixes = tuple(suffix.lower() for suffix in suffixes)
    if path.suffix.lower() not in normalized_suffixes:
        raise ValueError(
            f"unsupported TRELLIS.2 export format: {path.suffix or '<none>'}; "
            f"supported suffixes are {normalized_suffixes}"
        )

    resolved_path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    if outputs_root is None:
        return resolved_path

    resolved_root = Path(outputs_root).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"TRELLIS.2 export path must stay under {outputs_root}") from error
    return resolved_path


def sparse_coordinates_to_obj_payload(
    coordinates: mx.array,
    *,
    grid_size: int | None = None,
) -> bytes:
    """Convert sparse occupancy coordinates to a coarse OBJ preview."""

    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"sparse coordinates must have shape (num_tokens, 4), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("sparse coordinates must contain at least one token")
    if np.any(coords[:, 0] != 0):
        raise ValueError("OBJ preview currently supports only batch index 0")
    spatial = coords[:, 1:].astype(np.int32)
    size = int(grid_size or (spatial.max() + 1))
    if size <= 0:
        raise ValueError("grid_size must be positive")

    occupied = {tuple(int(value) for value in row) for row in spatial}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int, int]] = []
    for z, y, x in sorted(occupied):
        for normal, corners in _VOXEL_FACES:
            neighbor = (z + normal[0], y + normal[1], x + normal[2])
            if neighbor in occupied:
                continue
            face_indices = []
            for dz, dy, dx in corners:
                vertices.append(
                    (
                        (x + dx) / size - 0.5,
                        (y + dy) / size - 0.5,
                        (z + dz) / size - 0.5,
                    )
                )
                face_indices.append(len(vertices))
            faces.append(tuple(face_indices))

    lines = [
        "# mlx-spatial TRELLIS.2 sparse-structure occupancy preview",
        "# This is a coarse voxel OBJ, not the final FlexiDualGrid mesh.",
    ]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(f"f {a} {b} {c} {d}" for a, b, c, d in faces)
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def write_sparse_coordinate_preview_obj(
    coordinates: mx.array,
    output_path: str | Path,
    *,
    outputs_root: str | Path | None = None,
    grid_size: int | None = None,
) -> Trellis2ExportArtifact:
    """Write a coarse sparse-occupancy OBJ for inspection."""

    path = Path(output_path)
    if path.suffix.lower() != ".obj":
        raise ValueError("sparse coordinate preview exports require a .obj output path")
    payload = sparse_coordinates_to_obj_payload(coordinates, grid_size=grid_size)
    path = validate_trellis2_export_path(path, outputs_root=outputs_root, suffixes=(".obj",))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return Trellis2ExportArtifact(
        path=path,
        format="obj",
        bytes_written=len(payload),
        detail="wrote coarse TRELLIS.2 sparse-structure occupancy OBJ preview",
    )


def write_flexible_dual_grid_obj(
    mesh: FlexibleDualGridMesh,
    output_path: str | Path,
    *,
    outputs_root: str | Path | None = None,
) -> Trellis2ExportArtifact:
    """Write a decoded FlexiDualGrid shape mesh as OBJ."""

    path = Path(output_path)
    if path.suffix.lower() != ".obj":
        raise ValueError("shape mesh exports require a .obj output path")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mesh vertices must have shape (num_vertices, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh faces must have shape (num_faces, 3), got {faces.shape}")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("FlexiDualGrid mesh must contain vertices and faces")

    path = validate_trellis2_export_path(
        path,
        outputs_root=outputs_root,
        suffixes=(".obj",),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# mlx-spatial TRELLIS.2 FlexiDualGrid shape mesh\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return Trellis2ExportArtifact(
        path=path,
        format="obj",
        bytes_written=path.stat().st_size,
        detail="wrote TRELLIS.2 FlexiDualGrid shape OBJ",
    )


_VOXEL_FACES = (
    ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    ((1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0))),
    ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
    ((0, 0, 1), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
)


__all__ = [
    "SUPPORTED_TRELLIS2_EXPORT_SUFFIXES",
    "Trellis2ExportArtifact",
    "sparse_coordinates_to_obj_payload",
    "validate_trellis2_export_path",
    "write_flexible_dual_grid_obj",
    "write_sparse_coordinate_preview_obj",
]
