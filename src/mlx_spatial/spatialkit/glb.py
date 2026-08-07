"""GLB serialization for SpatialKit textured meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._native import textured_glb_payload as _textured_glb_payload
from .uv import NativeUvMesh


@dataclass(frozen=True)
class NativeGlbArtifact:
    """Written native GLB artifact metadata."""

    path: Path
    format: str
    bytes_written: int
    metadata: dict[str, Any]


def textured_glb_payload(
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatial SpatialKit",
    mesh_name: str = "TexturedMesh",
    material_name: str = "PBRMaterial",
) -> bytes:
    """Build a native self-contained GLB 2.0 payload."""

    return bytes(
        _textured_glb_payload(
            mesh.vertices,
            mesh.faces,
            mesh.uvs,
            base_color_rgba,
            metallic_roughness,
            str(generator),
            str(mesh_name),
            str(material_name),
        )
    )


def write_textured_glb(
    path: str | Path,
    mesh: NativeUvMesh,
    *,
    base_color_rgba: np.ndarray,
    metallic_roughness: np.ndarray,
    generator: str = "mlx-spatial SpatialKit",
    mesh_name: str = "TexturedMesh",
    material_name: str = "PBRMaterial",
    metadata: dict[str, Any] | None = None,
) -> NativeGlbArtifact:
    """Write a native GLB payload to disk."""

    output = Path(path)
    if output.suffix.lower() != ".glb":
        raise ValueError("native textured exports require a .glb output path")
    payload = textured_glb_payload(
        mesh,
        base_color_rgba=base_color_rgba,
        metallic_roughness=metallic_roughness,
        generator=generator,
        mesh_name=mesh_name,
        material_name=material_name,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(f".{output.name}.tmp")
    try:
        tmp_path.write_bytes(payload)
        tmp_path.replace(output)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    payload_metadata = {
        "stage": "textured_glb",
        "format": "glb",
        "bytes_written": int(output.stat().st_size),
        "generator": str(generator),
        "mesh_name": str(mesh_name),
        "material_name": str(material_name),
        **(metadata or {}),
    }
    return NativeGlbArtifact(
        path=output,
        format="glb",
        bytes_written=int(output.stat().st_size),
        metadata=payload_metadata,
    )


__all__ = [
    "NativeGlbArtifact",
    "textured_glb_payload",
    "write_textured_glb",
]
