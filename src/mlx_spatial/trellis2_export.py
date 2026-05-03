"""Export boundary helpers for TRELLIS.2 forward tracing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .trellis2_forward import Trellis2ForwardBlocker, Trellis2ForwardTraceResult
from .ovoxel import FlexibleDualGridMesh


SUPPORTED_TRELLIS2_EXPORT_SUFFIXES = (".glb", ".obj")


@dataclass(frozen=True)
class Trellis2ExportArtifact:
    path: Path
    format: str
    bytes_written: int
    detail: str


@dataclass(frozen=True)
class Trellis2ExportResult:
    artifact: Trellis2ExportArtifact | None = None
    blocker: Trellis2ForwardBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.artifact is not None and self.blocker is None


def validate_trellis2_export_path(
    output_path: str | Path,
    *,
    outputs_root: str | Path = "outputs",
) -> Path:
    """Validate that a mesh export target stays inside the ignored outputs tree."""

    path = Path(output_path)
    root = Path(outputs_root)
    if path.suffix.lower() not in SUPPORTED_TRELLIS2_EXPORT_SUFFIXES:
        raise ValueError(
            f"unsupported TRELLIS.2 export format: {path.suffix or '<none>'}; "
            f"supported suffixes are {SUPPORTED_TRELLIS2_EXPORT_SUFFIXES}"
        )

    resolved_root = root.resolve()
    resolved_path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"TRELLIS.2 export path must stay under {root}") from error
    return resolved_path


def write_trellis2_export_artifact(
    payload: bytes,
    output_path: str | Path,
    *,
    outputs_root: str | Path = "outputs",
) -> Trellis2ExportArtifact:
    if not payload:
        raise ValueError("TRELLIS.2 export payload must not be empty")
    path = validate_trellis2_export_path(output_path, outputs_root=outputs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return Trellis2ExportArtifact(
        path=path,
        format=path.suffix.lower().lstrip("."),
        bytes_written=len(payload),
        detail="wrote TRELLIS.2 mesh export artifact under ignored outputs tree",
    )


def sparse_coordinates_to_obj_payload(
    coordinates: mx.array,
    *,
    grid_size: int | None = None,
) -> bytes:
    """Convert sparse `(batch, z, y, x)` occupancy coordinates to a coarse OBJ preview."""

    coords = np.array(coordinates)
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
                vx = (x + dx) / size - 0.5
                vy = (y + dy) / size - 0.5
                vz = (z + dz) / size - 0.5
                vertices.append((vx, vy, vz))
                face_indices.append(len(vertices))
            faces.append(tuple(face_indices))

    lines = [
        "# mlx-spatial TRELLIS.2 sparse-structure occupancy preview",
        "# This is a coarse voxel OBJ, not the final FlexiDualGrid TRELLIS mesh.",
    ]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(f"f {a} {b} {c} {d}" for a, b, c, d in faces)
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def write_sparse_coordinate_preview_obj(
    coordinates: mx.array,
    output_path: str | Path,
    *,
    outputs_root: str | Path = "outputs",
    grid_size: int | None = None,
) -> Trellis2ExportArtifact:
    path = Path(output_path)
    if path.suffix.lower() != ".obj":
        raise ValueError("sparse coordinate preview exports require a .obj output path")
    payload = sparse_coordinates_to_obj_payload(coordinates, grid_size=grid_size)
    artifact = write_trellis2_export_artifact(payload, path, outputs_root=outputs_root)
    return Trellis2ExportArtifact(
        path=artifact.path,
        format=artifact.format,
        bytes_written=artifact.bytes_written,
        detail="wrote coarse TRELLIS.2 sparse-structure occupancy OBJ preview",
    )


def write_flexible_dual_grid_obj(
    mesh: FlexibleDualGridMesh,
    output_path: str | Path,
    *,
    outputs_root: str | Path = "outputs",
) -> Trellis2ExportArtifact:
    path = Path(output_path)
    if path.suffix.lower() != ".obj":
        raise ValueError("shape mesh exports require a .obj output path")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError("FlexiDualGrid mesh must contain vertices and faces")
    path = validate_trellis2_export_path(path, outputs_root=outputs_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# mlx-spatial TRELLIS.2 FlexiDualGrid shape mesh\n")
        for x, y, z in mesh.vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in mesh.faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")
    bytes_written = path.stat().st_size
    return Trellis2ExportArtifact(
        path=path,
        format="obj",
        bytes_written=bytes_written,
        detail="wrote TRELLIS.2 FlexiDualGrid shape OBJ",
    )


def assess_trellis2_export_boundary(
    trace: Trellis2ForwardTraceResult,
    *,
    output_path: str | Path | None = None,
    outputs_root: str | Path = "outputs",
) -> Trellis2ExportResult:
    """Return export readiness or a precise blocker for the current forward trace."""

    if output_path is not None:
        try:
            validate_trellis2_export_path(output_path, outputs_root=outputs_root)
        except ValueError as error:
            return Trellis2ExportResult(
                blocker=Trellis2ForwardBlocker(
                    stage="mesh-export",
                    operation="TRELLIS.2 export path validation",
                    reference=str(output_path),
                    reason=str(error),
                    next_slice="choose a .glb or .obj path under outputs/ for TRELLIS.2 exports",
                )
            )

    if trace.blocker is not None:
        return Trellis2ExportResult(
            blocker=Trellis2ForwardBlocker(
                stage="mesh-export",
                operation="upstream inference completion before export",
                reference=trace.blocker.reference,
                reason=(
                    f"export requires decoded mesh/texture payload, but forward trace is blocked at "
                    f"{trace.blocker.stage} / {trace.blocker.operation}: {trace.blocker.reason}"
                ),
                next_slice=trace.blocker.next_slice,
            )
        )

    return Trellis2ExportResult(
        blocker=Trellis2ForwardBlocker(
            stage="mesh-export",
            operation="decoded mesh payload availability",
            reference=str(trace.root),
            reason="forward trace completed without a decoded mesh/texture payload to export",
            next_slice="attach decoded mesh/texture payload metadata before GLB/OBJ export",
        )
    )


_VOXEL_FACES = (
    ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    ((1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0))),
    ((0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
    ((0, 0, 1), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
)
