"""Shared O-Voxel to GLB export policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .spatialkit import OVoxelGlbExportResult


SpatialKitExporter = Callable[..., "OVoxelGlbExportResult"]
OVOXEL_DEFAULT_TARGET_FACES = 200_000
OVOXEL_PREVIEW_TARGET_FACES = 50_000


def load_spatialkit_exporter() -> tuple[SpatialKitExporter | None, str | None]:
    """Load the integrated SpatialKit exporter without hiding import failures."""

    try:
        from .spatialkit import export_decoded_ovoxel_glb
    except ImportError as error:
        return None, f"mlx_spatial.spatialkit is not importable: {error}"
    return export_decoded_ovoxel_glb, None


def export_ovoxel_glb(
    decoded_dir: str | Path,
    output: str | Path,
    *,
    texture_size: int,
    target_faces: int,
    grid_size: int,
    diagnostics_path: str | Path | None = None,
    exporter: SpatialKitExporter | None = None,
) -> "OVoxelGlbExportResult":
    """Export decoded O-Voxel artifacts with the production SpatialKit policy."""

    for name, value in (
        ("texture_size", texture_size),
        ("target_faces", target_faces),
        ("grid_size", grid_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    resolved_exporter = exporter
    if resolved_exporter is None:
        resolved_exporter, import_error = load_spatialkit_exporter()
        if resolved_exporter is None:
            raise ImportError(import_error or "mlx_spatial.spatialkit exporter is unavailable")
    return resolved_exporter(
        decoded_dir,
        output,
        texture_size=texture_size,
        target_faces=target_faces,
        quality_preset="reference-target",
        grid_size=grid_size,
        uv_backend="xatlas-clustered",
        remesh=True,
        remesh_resolution=grid_size,
        simplify_backend="mlx-qem",
        texture_postprocess="telea",
        diagnostics_path=diagnostics_path,
    )


__all__ = [
    "OVOXEL_DEFAULT_TARGET_FACES",
    "OVOXEL_PREVIEW_TARGET_FACES",
    "SpatialKitExporter",
    "export_ovoxel_glb",
    "load_spatialkit_exporter",
]
