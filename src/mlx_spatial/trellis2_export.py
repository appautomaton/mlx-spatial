"""Export boundary helpers for TRELLIS.2 forward tracing."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from .export_utils import (
    fill_texture_holes,
    fill_texture_holes_ndimage,
    pad_glb_buffer,
    rasterize_uv_positions,
    texture_png_payload,
)
from .trellis2_forward import Trellis2ForwardBlocker, Trellis2ForwardTraceResult
from .ovoxel import FlexibleDualGridMesh, MeshHoleFillStats, fill_flexible_dual_grid_mesh_holes


SUPPORTED_TRELLIS2_EXPORT_SUFFIXES = (".glb", ".obj")
TRELLIS2_GLB_DEFAULT_FACE_TARGET = 50_000
TRELLIS2_XATLAS_FACE_GUARD = 125_000
TRELLIS2_XATLAS_AUTO_FACE_GUARD = "auto"
TRELLIS2_XATLAS_AUTO_FACE_GUARD_HEADROOM = 1.5
TRELLIS2_XATLAS_MAX_AUTO_FACE_GUARD = 300_000
TRELLIS2_XATLAS_PARALLEL_FACE_TARGET = 50_000
TRELLIS2_XATLAS_MAX_AUTO_PARALLEL_CHUNKS = 8
TRELLIS2_MAC_EXPORT_IMPORTS = ("xatlas", "scipy", "fast_simplification")
TRELLIS2_TEXTURE_BAKE_BACKENDS = ("trilinear", "kdtree")
TRELLIS2_PBR_ATTR_LAYOUT = {
    "base_color": slice(0, 3),
    "metallic": slice(3, 4),
    "roughness": slice(4, 5),
    "alpha": slice(5, 6),
}


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


@dataclass(frozen=True)
class Trellis2TextureBakeResult:
    """Deterministic UV texture payload produced from TRELLIS.2 texture voxels."""

    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    base_color_rgba: np.ndarray
    metallic_roughness: np.ndarray
    coverage_mask: np.ndarray
    texture_size: int
    voxel_count: int
    k_neighbors: int
    origin: tuple[float, float, float]
    voxel_size: float
    backend: str = "face-atlas"
    raw_coverage_ratio: float | None = None
    unwrap_backend: str = "face-atlas"
    unwrap_seconds: float | None = None
    unwrap_chunks: int = 1
    unwrap_chart_count: int | None = None
    unwrap_utilization: float | None = None
    xatlas_face_guard: int | None = None
    xatlas_face_guard_mode: str = "manual"
    sampled_texel_count: int = 0
    missing_texel_count: int = 0
    out_of_grid_texel_count: int = 0
    source_projection_used: bool = False
    source_projection_detail: str = "not requested"

    @property
    def coverage_ratio(self) -> float:
        return float(np.count_nonzero(self.coverage_mask) / self.coverage_mask.size)


@dataclass(frozen=True)
class Trellis2MeshPostprocessStats:
    original_vertices: int
    original_faces: int
    duplicate_faces_removed: int
    degenerate_faces_removed: int
    unreferenced_vertices_removed: int
    components_removed: int
    component_faces_removed: int
    hole_fill: MeshHoleFillStats
    cleaned_vertices: int
    cleaned_faces: int
    simplified: bool
    simplification_target_faces: int
    final_vertices: int
    final_faces: int
    boundary_edges: int
    nonmanifold_edges: int


@dataclass(frozen=True)
class Trellis2MeshPostprocessResult:
    mesh: FlexibleDualGridMesh
    stats: Trellis2MeshPostprocessStats
    source_mesh: FlexibleDualGridMesh | None = None


@dataclass(frozen=True)
class Trellis2SparseTrilinearSampleResult:
    attributes: np.ndarray
    valid_mask: np.ndarray
    sampled_texel_count: int
    missing_texel_count: int
    out_of_grid_texel_count: int


@dataclass(frozen=True)
class Trellis2PostprocessParityItem:
    stage: str
    official: str
    mlx_spatial: str
    parity: str
    next_action: str


@dataclass(frozen=True)
class Trellis2XAtlasUnwrapStats:
    backend: str
    input_vertices: int
    input_faces: int
    output_vertices: int
    output_faces: int
    elapsed_seconds: float
    chunks: int
    chunk_faces: tuple[int, ...]
    chart_count: int | None = None
    atlas_width: int | None = None
    atlas_height: int | None = None
    utilization: float | None = None


@dataclass(frozen=True)
class Trellis2XAtlasUnwrapResult:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    stats: Trellis2XAtlasUnwrapStats


def validate_trellis2_export_path(
    output_path: str | Path,
    *,
    outputs_root: str | Path = "outputs",
    suffixes: tuple[str, ...] = SUPPORTED_TRELLIS2_EXPORT_SUFFIXES,
) -> Path:
    """Validate that a mesh export target stays inside the ignored outputs tree."""

    path = Path(output_path)
    root = Path(outputs_root)
    normalized_suffixes = tuple(suffix.lower() for suffix in suffixes)
    if path.suffix.lower() not in normalized_suffixes:
        raise ValueError(
            f"unsupported TRELLIS.2 export format: {path.suffix or '<none>'}; "
            f"supported suffixes are {normalized_suffixes}"
        )

    resolved_root = root.resolve()
    resolved_path = path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"TRELLIS.2 export path must stay under {root}") from error
    return resolved_path


def resolve_trellis2_xatlas_face_guard(
    face_count: int,
    face_guard: int | str = TRELLIS2_XATLAS_AUTO_FACE_GUARD,
    *,
    min_face_guard: int = TRELLIS2_XATLAS_FACE_GUARD,
    max_auto_face_guard: int = TRELLIS2_XATLAS_MAX_AUTO_FACE_GUARD,
    auto_headroom: float = TRELLIS2_XATLAS_AUTO_FACE_GUARD_HEADROOM,
) -> int:
    """Resolve an explicit or adaptive xatlas face guard for a postprocessed mesh."""

    face_count = int(face_count)
    if face_count <= 0:
        raise ValueError(f"xatlas face guard resolution needs a positive face count, got {face_count}")
    if isinstance(face_guard, str):
        normalized = face_guard.strip().lower()
        if normalized == TRELLIS2_XATLAS_AUTO_FACE_GUARD:
            if min_face_guard <= 0:
                raise ValueError(f"min_face_guard must be positive, got {min_face_guard}")
            if max_auto_face_guard <= 0:
                raise ValueError(f"max_auto_face_guard must be positive, got {max_auto_face_guard}")
            if auto_headroom < 1.0:
                raise ValueError(f"auto_headroom must be >= 1.0, got {auto_headroom}")
            adaptive_guard = int(math.ceil(face_count * float(auto_headroom)))
            return min(max(int(min_face_guard), adaptive_guard), int(max_auto_face_guard))
        try:
            face_guard = int(normalized)
        except ValueError as error:
            raise ValueError(
                f"xatlas-face-guard must be '{TRELLIS2_XATLAS_AUTO_FACE_GUARD}' or a positive integer, got {face_guard!r}"
            ) from error
    try:
        explicit_guard = int(face_guard)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"xatlas-face-guard must be '{TRELLIS2_XATLAS_AUTO_FACE_GUARD}' or a positive integer, got {face_guard!r}"
        ) from error
    if isinstance(face_guard, bool) or explicit_guard <= 0:
        raise ValueError(
            f"xatlas-face-guard must be '{TRELLIS2_XATLAS_AUTO_FACE_GUARD}' or a positive integer, got {face_guard!r}"
        )
    return explicit_guard


def trellis2_postprocess_parity_audit() -> tuple[Trellis2PostprocessParityItem, ...]:
    """Return the current Mac-native export parity map against upstream TRELLIS.2."""

    return (
        Trellis2PostprocessParityItem(
            stage="FlexiDualGrid mesh extraction",
            official="o_voxel.convert.flexible_dual_grid_to_mesh inference path",
            mlx_spatial="NumPy inference-mode FlexiDualGrid conversion",
            parity="close",
            next_action="keep fixture coverage against upstream channel semantics",
        ),
        Trellis2PostprocessParityItem(
            stage="mesh cleanup",
            official="cumesh fill_holes, non-manifold repair, component cleanup, simplify, orientation unification",
            mlx_spatial="NumPy bad-face cleanup, small component removal, bounded hole fill, fast_simplification",
            parity="partial",
            next_action="measure topology metrics before attempting remeshing parity",
        ),
        Trellis2PostprocessParityItem(
            stage="remeshing",
            official="cumesh.remeshing.remesh_narrow_band_dc with BVH project-back",
            mlx_spatial="not implemented",
            parity="missing",
            next_action="defer until texture bake parity is measurable",
        ),
        Trellis2PostprocessParityItem(
            stage="UV unwrap",
            official="cumesh.CuMesh.uv_unwrap",
            mlx_spatial="xatlas global or spatial chunked unwrap",
            parity="functional replacement",
            next_action="track unwrap backend, chart count, and utilization in trace",
        ),
        Trellis2PostprocessParityItem(
            stage="texture sampling",
            official="nvdiffrast UV raster plus flex_gemm grid_sample_3d trilinear sparse volume sampling",
            mlx_spatial="NumPy UV raster plus sparse-grid trilinear sampling; KDTree kept as explicit debug backend",
            parity="closer replacement",
            next_action="compare live trilinear bake against previous KDTree bake",
        ),
        Trellis2PostprocessParityItem(
            stage="GLB material packing",
            official="trimesh PBRMaterial with baseColor and metallicRoughness textures",
            mlx_spatial="direct GLB 2.0 writer with embedded baseColor RGBA and metallicRoughness PNGs",
            parity="functional replacement",
            next_action="keep Blender import and texture shape tests",
        ),
    )


def missing_trellis2_mac_export_dependencies() -> tuple[str, ...]:
    """Return missing Mac-native export dependency import names."""

    missing = []
    for module_name in TRELLIS2_MAC_EXPORT_IMPORTS:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return tuple(missing)


def ensure_trellis2_mac_export_dependencies() -> None:
    missing = missing_trellis2_mac_export_dependencies()
    if missing:
        raise ValueError(
            "Mac-native TRELLIS.2 GLB export requires "
            f"{', '.join(TRELLIS2_MAC_EXPORT_IMPORTS)}; missing {', '.join(missing)}"
        )


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


def make_trellis2_face_atlas_uvs(
    mesh: FlexibleDualGridMesh,
    *,
    tile_padding: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic per-face UV atlas by duplicating triangle vertices.

    This is a small Mac-native fixture unwrap surface. It intentionally avoids
    CUDA-only upstream UV tooling while preserving a GLB-compatible mesh/UV
    contract for the subsequent writer slice.
    """

    vertices, faces = _validate_mesh_np(mesh)
    if not 0 <= tile_padding < 0.45:
        raise ValueError(f"tile_padding must be in [0, 0.45), got {tile_padding}")

    face_count = int(faces.shape[0])
    cols = int(np.ceil(np.sqrt(face_count)))
    rows = int(np.ceil(face_count / cols))
    local_uv = np.array(
        [
            [tile_padding, tile_padding],
            [1.0 - tile_padding, tile_padding],
            [tile_padding, 1.0 - tile_padding],
        ],
        dtype=np.float32,
    )

    atlas_vertices = np.empty((face_count * 3, 3), dtype=np.float32)
    atlas_faces = np.arange(face_count * 3, dtype=np.int64).reshape(face_count, 3)
    atlas_uvs = np.empty((face_count * 3, 2), dtype=np.float32)
    for face_index, face in enumerate(faces):
        start = face_index * 3
        atlas_vertices[start : start + 3] = vertices[face]
        col = face_index % cols
        row = face_index // cols
        atlas_uvs[start : start + 3, 0] = (col + local_uv[:, 0]) / cols
        atlas_uvs[start : start + 3, 1] = (row + local_uv[:, 1]) / rows
    return atlas_vertices, atlas_faces, atlas_uvs


def postprocess_trellis2_mesh_for_glb(
    mesh: FlexibleDualGridMesh,
    *,
    target_faces: int = TRELLIS2_GLB_DEFAULT_FACE_TARGET,
    simplify: bool = True,
    min_component_faces: int = 32,
    min_component_face_fraction: float = 1e-5,
) -> Trellis2MeshPostprocessResult:
    """Clean and simplify a FlexiDualGrid mesh for previewable GLB export."""

    if target_faces <= 0:
        raise ValueError(f"target_faces must be positive, got {target_faces}")
    if min_component_faces <= 0:
        raise ValueError(f"min_component_faces must be positive, got {min_component_faces}")
    if min_component_face_fraction < 0:
        raise ValueError(f"min_component_face_fraction must be non-negative, got {min_component_face_fraction}")

    original_vertices, original_faces = _validate_mesh_np(mesh)
    vertices, faces, degenerate_removed = _remove_degenerate_faces(original_vertices, original_faces)
    faces, duplicate_removed = _remove_duplicate_faces(faces)
    vertices, faces, unreferenced_removed = _compact_mesh(vertices, faces)
    vertices, faces, components_removed, component_faces_removed = _remove_small_connected_components(
        vertices,
        faces,
        min_component_faces=max(min_component_faces, int(faces.shape[0] * min_component_face_fraction)),
    )
    vertices, faces, unreferenced_after_components = _compact_mesh(vertices, faces)
    unreferenced_removed += unreferenced_after_components

    filled, first_hole_stats = fill_flexible_dual_grid_mesh_holes(FlexibleDualGridMesh(vertices=vertices, faces=faces))
    vertices, faces = filled.vertices, filled.faces
    vertices, faces, degenerate_after_fill = _remove_degenerate_faces(vertices, faces)
    faces, duplicate_after_fill = _remove_duplicate_faces(faces)
    vertices, faces, unreferenced_after_fill = _compact_mesh(vertices, faces)
    degenerate_removed += degenerate_after_fill
    duplicate_removed += duplicate_after_fill
    unreferenced_removed += unreferenced_after_fill
    cleaned_vertices = int(vertices.shape[0])
    cleaned_faces = int(faces.shape[0])
    source_mesh = FlexibleDualGridMesh(
        vertices=vertices.astype(np.float32, copy=True),
        faces=faces.astype(np.int64, copy=True),
    )

    simplified = False
    if simplify and faces.shape[0] > target_faces:
        ensure_trellis2_mac_export_dependencies()
        import fast_simplification

        for agg in (7.0, 10.0, 15.0):
            if faces.shape[0] <= target_faces:
                break
            previous_face_count = int(faces.shape[0])
            vertices, faces = fast_simplification.simplify(
                vertices.astype(np.float64, copy=False),
                faces.astype(np.int64, copy=False),
                target_count=int(target_faces),
                agg=agg,
            )
            vertices = np.asarray(vertices, dtype=np.float32)
            faces = np.asarray(faces, dtype=np.int64)
            vertices, faces, degenerate_after_simplify = _remove_degenerate_faces(vertices, faces)
            faces, duplicate_after_simplify = _remove_duplicate_faces(faces)
            vertices, faces, unreferenced_after_simplify = _compact_mesh(vertices, faces)
            filled, second_hole_stats = fill_flexible_dual_grid_mesh_holes(
                FlexibleDualGridMesh(vertices=vertices, faces=faces)
            )
            vertices, faces = filled.vertices, filled.faces
            degenerate_removed += degenerate_after_simplify
            duplicate_removed += duplicate_after_simplify
            unreferenced_removed += unreferenced_after_simplify
            first_hole_stats = _combine_hole_fill_stats(first_hole_stats, second_hole_stats)
            simplified = True
            if faces.shape[0] >= previous_face_count:
                break

    boundary_edges, nonmanifold_edges = _mesh_edge_stats(faces)
    stats = Trellis2MeshPostprocessStats(
        original_vertices=int(original_vertices.shape[0]),
        original_faces=int(original_faces.shape[0]),
        duplicate_faces_removed=int(duplicate_removed),
        degenerate_faces_removed=int(degenerate_removed),
        unreferenced_vertices_removed=int(unreferenced_removed),
        components_removed=int(components_removed),
        component_faces_removed=int(component_faces_removed),
        hole_fill=first_hole_stats,
        cleaned_vertices=cleaned_vertices,
        cleaned_faces=cleaned_faces,
        simplified=simplified,
        simplification_target_faces=int(target_faces),
        final_vertices=int(vertices.shape[0]),
        final_faces=int(faces.shape[0]),
        boundary_edges=int(boundary_edges),
        nonmanifold_edges=int(nonmanifold_edges),
    )
    return Trellis2MeshPostprocessResult(
        mesh=FlexibleDualGridMesh(vertices=vertices.astype(np.float32, copy=False), faces=faces.astype(np.int64, copy=False)),
        stats=stats,
        source_mesh=source_mesh,
    )


def unwrap_trellis2_mesh_xatlas(
    mesh: FlexibleDualGridMesh,
    *,
    face_guard: int = TRELLIS2_XATLAS_FACE_GUARD,
    parallel_chunks: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = unwrap_trellis2_mesh_xatlas_with_stats(
        mesh,
        face_guard=face_guard,
        parallel_chunks=parallel_chunks,
    )
    return result.vertices, result.faces, result.uvs


def unwrap_trellis2_mesh_xatlas_with_stats(
    mesh: FlexibleDualGridMesh,
    *,
    face_guard: int = TRELLIS2_XATLAS_FACE_GUARD,
    parallel_chunks: int = 0,
) -> Trellis2XAtlasUnwrapResult:
    ensure_trellis2_mac_export_dependencies()

    if face_guard <= 0:
        raise ValueError(f"face_guard must be positive, got {face_guard}")
    if parallel_chunks < 0:
        raise ValueError(f"parallel_chunks must be non-negative, got {parallel_chunks}")
    vertices, faces = _validate_mesh_np(mesh)
    if faces.shape[0] > face_guard:
        raise ValueError(
            f"xatlas unwrap face count {faces.shape[0]} exceeds guard {face_guard}; "
            "simplify the GLB mesh before UV unwrapping"
        )
    resolved_chunks = _resolve_xatlas_parallel_chunks(int(faces.shape[0]), parallel_chunks)
    start = time.perf_counter()
    if resolved_chunks <= 1:
        unwrapped_vertices, unwrapped_faces, unwrapped_uvs, stats = _xatlas_parametrize_arrays(
            vertices,
            faces,
            backend="xatlas-global",
            chunks=1,
            chunk_faces=(int(faces.shape[0]),),
            elapsed_seconds=0.0,
        )
    else:
        unwrapped_vertices, unwrapped_faces, unwrapped_uvs, stats = _xatlas_parametrize_spatial_chunks(
            vertices,
            faces,
            chunks=resolved_chunks,
        )
    elapsed = time.perf_counter() - start
    stats = Trellis2XAtlasUnwrapStats(
        backend=stats.backend,
        input_vertices=stats.input_vertices,
        input_faces=stats.input_faces,
        output_vertices=stats.output_vertices,
        output_faces=stats.output_faces,
        elapsed_seconds=elapsed,
        chunks=stats.chunks,
        chunk_faces=stats.chunk_faces,
        chart_count=stats.chart_count,
        atlas_width=stats.atlas_width,
        atlas_height=stats.atlas_height,
        utilization=stats.utilization,
    )
    if unwrapped_uvs.shape != (unwrapped_vertices.shape[0], 2):
        raise ValueError(
            f"xatlas returned UV shape {unwrapped_uvs.shape}, expected ({unwrapped_vertices.shape[0]}, 2)"
        )
    return Trellis2XAtlasUnwrapResult(
        vertices=unwrapped_vertices.astype(np.float32, copy=False),
        faces=unwrapped_faces.astype(np.int64, copy=False),
        uvs=np.clip(unwrapped_uvs.astype(np.float32, copy=False), 0.0, 1.0),
        stats=stats,
    )


def _resolve_xatlas_parallel_chunks(face_count: int, requested_chunks: int) -> int:
    if requested_chunks > 0:
        return min(int(requested_chunks), max(face_count, 1))
    if face_count <= TRELLIS2_XATLAS_PARALLEL_FACE_TARGET:
        return 1
    cpu_count = max(1, os.cpu_count() or 1)
    face_chunks = int(math.ceil(face_count / TRELLIS2_XATLAS_PARALLEL_FACE_TARGET))
    return max(1, min(face_chunks, cpu_count, TRELLIS2_XATLAS_MAX_AUTO_PARALLEL_CHUNKS))


def _xatlas_parametrize_arrays(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    backend: str,
    chunks: int,
    chunk_faces: tuple[int, ...],
    elapsed_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Trellis2XAtlasUnwrapStats]:
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(vertices.astype(np.float32, copy=False)),
        np.ascontiguousarray(faces.astype(np.uint32, copy=False)),
    )
    atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions(), False)
    vmapping, indices, uvs = atlas.get_mesh(0)
    out_vertices = vertices[np.asarray(vmapping, dtype=np.int64)]
    out_faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    out_uvs = np.asarray(uvs, dtype=np.float32)
    stats = Trellis2XAtlasUnwrapStats(
        backend=backend,
        input_vertices=int(vertices.shape[0]),
        input_faces=int(faces.shape[0]),
        output_vertices=int(out_vertices.shape[0]),
        output_faces=int(out_faces.shape[0]),
        elapsed_seconds=float(elapsed_seconds),
        chunks=int(chunks),
        chunk_faces=chunk_faces,
        chart_count=int(atlas.chart_count),
        atlas_width=int(atlas.width),
        atlas_height=int(atlas.height),
        utilization=float(atlas.utilization),
    )
    return out_vertices, out_faces, out_uvs, stats


def _xatlas_parametrize_spatial_chunks(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chunks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Trellis2XAtlasUnwrapStats]:
    face_partitions = _partition_faces_for_parallel_xatlas(vertices, faces, chunks)
    chunk_inputs = []
    for face_indices in face_partitions:
        local_vertices, local_faces = _submesh_for_faces(vertices, faces[face_indices])
        chunk_inputs.append((local_vertices, local_faces))

    def unwrap_chunk(item: tuple[np.ndarray, np.ndarray]):
        local_vertices, local_faces = item
        out_vertices, out_faces, out_uvs, stats = _xatlas_parametrize_arrays(
            local_vertices,
            local_faces,
            backend="xatlas-parallel-chunk",
            chunks=1,
            chunk_faces=(int(local_faces.shape[0]),),
            elapsed_seconds=0.0,
        )
        return out_vertices, out_faces, out_uvs, stats

    max_workers = min(len(chunk_inputs), max(1, os.cpu_count() or 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="xatlas") as executor:
        chunk_results = list(executor.map(unwrap_chunk, chunk_inputs))

    packed_vertices: list[np.ndarray] = []
    packed_faces: list[np.ndarray] = []
    packed_uvs: list[np.ndarray] = []
    vertex_offset = 0
    cols = int(math.ceil(math.sqrt(len(chunk_results))))
    rows = int(math.ceil(len(chunk_results) / cols))
    tile_padding = 0.02
    for chunk_index, (chunk_vertices, chunk_faces, chunk_uvs, _stats) in enumerate(chunk_results):
        row = chunk_index // cols
        col = chunk_index % cols
        normalized_uvs = _normalize_chunk_uvs(chunk_uvs)
        normalized_uvs = tile_padding + normalized_uvs * (1.0 - 2.0 * tile_padding)
        normalized_uvs[:, 0] = (col + normalized_uvs[:, 0]) / cols
        normalized_uvs[:, 1] = (row + normalized_uvs[:, 1]) / rows
        packed_vertices.append(chunk_vertices.astype(np.float32, copy=False))
        packed_faces.append(chunk_faces.astype(np.int64, copy=False) + vertex_offset)
        packed_uvs.append(normalized_uvs.astype(np.float32, copy=False))
        vertex_offset += int(chunk_vertices.shape[0])

    out_vertices = np.concatenate(packed_vertices, axis=0)
    out_faces = np.concatenate(packed_faces, axis=0)
    out_uvs = np.concatenate(packed_uvs, axis=0)
    chart_counts = []
    utilizations = []
    for _chunk_vertices, _chunk_faces, _chunk_uvs, chunk_stats in chunk_results:
        if chunk_stats.chart_count is not None:
            chart_counts.append(chunk_stats.chart_count)
        if chunk_stats.utilization is not None:
            utilizations.append(chunk_stats.utilization)
    stats = Trellis2XAtlasUnwrapStats(
        backend="xatlas-parallel-spatial",
        input_vertices=int(vertices.shape[0]),
        input_faces=int(faces.shape[0]),
        output_vertices=int(out_vertices.shape[0]),
        output_faces=int(out_faces.shape[0]),
        elapsed_seconds=0.0,
        chunks=len(chunk_results),
        chunk_faces=tuple(int(local_faces.shape[0]) for _local_vertices, local_faces in chunk_inputs),
        chart_count=int(sum(chart_counts)) if chart_counts else None,
        atlas_width=None,
        atlas_height=None,
        utilization=float(np.mean(utilizations)) if utilizations else None,
    )
    return out_vertices, out_faces, out_uvs, stats


def _partition_faces_for_parallel_xatlas(vertices: np.ndarray, faces: np.ndarray, chunks: int) -> list[np.ndarray]:
    centroids = vertices[faces].mean(axis=1)
    partitions = [np.arange(faces.shape[0], dtype=np.int64)]
    while len(partitions) < chunks:
        split_index = max(range(len(partitions)), key=lambda index: partitions[index].shape[0])
        face_indices = partitions.pop(split_index)
        if face_indices.shape[0] <= 1:
            partitions.append(face_indices)
            break
        spans = np.ptp(centroids[face_indices], axis=0)
        axis = int(np.argmax(spans))
        order = face_indices[np.argsort(centroids[face_indices, axis], kind="mergesort")]
        midpoint = order.shape[0] // 2
        partitions.append(order[:midpoint])
        partitions.append(order[midpoint:])
    return [partition for partition in partitions if partition.size]


def _submesh_for_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return vertices[unique_vertices].astype(np.float32, copy=False), inverse.reshape(-1, 3).astype(np.int64, copy=False)


def _normalize_chunk_uvs(uvs: np.ndarray) -> np.ndarray:
    normalized = np.asarray(uvs, dtype=np.float32).copy()
    uv_min = normalized.min(axis=0)
    uv_max = normalized.max(axis=0)
    span = np.maximum(uv_max - uv_min, 1e-6)
    normalized = (normalized - uv_min) / span
    return np.clip(normalized, 0.0, 1.0)


def bake_trellis2_texture_fields(
    mesh: FlexibleDualGridMesh,
    texture_coordinates: mx.array | np.ndarray,
    texture_attributes: mx.array | np.ndarray,
    *,
    decode_resolution: int | None = None,
    texture_size: int = 256,
    origin: tuple[float, float, float] = (-0.5, -0.5, -0.5),
    voxel_size: float | None = None,
    k_neighbors: int = 4,
    max_query_voxel_pairs: int = 8_000_000,
    max_texture_pixels: int = 1_048_576,
) -> Trellis2TextureBakeResult:
    """Bake 6-channel texture decoder voxels into a UV texture payload.

    The attribute layout follows upstream TRELLIS.2 PBR fields:
    base color channels 0:3, metallic 3:4, roughness 4:5, alpha 5:6.
    """

    if texture_size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size}")
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
    if max_query_voxel_pairs <= 0:
        raise ValueError(f"max_query_voxel_pairs must be positive, got {max_query_voxel_pairs}")
    if max_texture_pixels <= 0:
        raise ValueError(f"max_texture_pixels must be positive, got {max_texture_pixels}")
    texture_pixels = int(texture_size * texture_size)
    if texture_pixels > max_texture_pixels:
        raise ValueError(
            f"texture bake would allocate {texture_pixels} pixels, above guard {max_texture_pixels}"
        )

    coords = np.array(texture_coordinates, dtype=np.int32)
    attrs = np.array(texture_attributes, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"texture coordinates must have shape (num_voxels, 4), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("texture coordinates must contain at least one voxel")
    if np.any(coords[:, 0] != 0):
        raise ValueError("texture baking currently supports only batch index 0")
    if attrs.ndim != 2 or attrs.shape[0] != coords.shape[0] or attrs.shape[1] < 6:
        raise ValueError(
            f"texture attributes must have shape ({coords.shape[0]}, at least 6), got {attrs.shape}"
        )
    if not np.all(np.isfinite(attrs[:, :6])):
        raise ValueError("texture attributes must contain only finite values")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("texture coordinates must be unique")

    spatial_coords = coords[:, 1:]
    if np.any(spatial_coords < 0):
        raise ValueError("texture spatial coordinates must be non-negative")
    if decode_resolution is not None and np.any(spatial_coords >= decode_resolution):
        raise ValueError(
            f"texture spatial coordinates must be < decode_resolution ({decode_resolution})"
        )
    order = np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))
    coords = coords[order]
    attrs = attrs[order]

    atlas_vertices, atlas_faces, atlas_uvs = make_trellis2_face_atlas_uvs(mesh)
    positions, mask = rasterize_uv_positions(atlas_vertices, atlas_faces, atlas_uvs, texture_size)
    if not np.any(mask):
        raise ValueError("UV atlas rasterization produced no covered texels")

    resolved_voxel_size = _resolve_voxel_size(coords[:, 1:], decode_resolution, voxel_size)
    origin_np = np.asarray(origin, dtype=np.float32)
    if origin_np.shape != (3,):
        raise ValueError(f"origin must contain three values, got {origin}")
    voxel_centers = coords[:, 1:].astype(np.float32) * resolved_voxel_size + origin_np + resolved_voxel_size * 0.5

    query_points = positions[mask]
    sampled = _sample_voxel_attributes(
        query_points,
        voxel_centers,
        coords[:, 1:],
        attrs[:, :6],
        k_neighbors=min(k_neighbors, coords.shape[0]),
        max_query_voxel_pairs=max_query_voxel_pairs,
        voxel_size=resolved_voxel_size,
        origin=origin_np,
    )

    base_color = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    alpha = np.zeros((texture_size, texture_size, 1), dtype=np.float32)
    metallic = np.zeros((texture_size, texture_size, 1), dtype=np.float32)
    roughness = np.ones((texture_size, texture_size, 1), dtype=np.float32)

    ys, xs = np.where(mask)
    base_color[ys, xs] = np.clip(sampled[:, TRELLIS2_PBR_ATTR_LAYOUT["base_color"]], 0.0, 1.0)
    metallic[ys, xs, 0] = np.clip(sampled[:, TRELLIS2_PBR_ATTR_LAYOUT["metallic"]].reshape(-1), 0.0, 1.0)
    roughness[ys, xs, 0] = np.clip(sampled[:, TRELLIS2_PBR_ATTR_LAYOUT["roughness"]].reshape(-1), 0.0, 1.0)
    alpha[ys, xs, 0] = np.clip(sampled[:, TRELLIS2_PBR_ATTR_LAYOUT["alpha"]].reshape(-1), 0.0, 1.0)

    base_color, _ = fill_texture_holes(base_color, mask)
    metallic, _ = fill_texture_holes(metallic, mask)
    roughness, _ = fill_texture_holes(roughness, mask)
    alpha, _ = fill_texture_holes(alpha, mask)
    base_color = np.power(np.clip(base_color, 0.0, 1.0), 1.0 / 2.2)

    base_color_rgba = (np.concatenate([base_color, alpha], axis=-1) * 255.0 + 0.5).astype(np.uint8)
    metallic_roughness = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    metallic_roughness[:, :, 1] = (roughness[:, :, 0] * 255.0 + 0.5).astype(np.uint8)
    metallic_roughness[:, :, 2] = (metallic[:, :, 0] * 255.0 + 0.5).astype(np.uint8)

    return Trellis2TextureBakeResult(
        vertices=atlas_vertices,
        faces=atlas_faces,
        uvs=atlas_uvs,
        base_color_rgba=base_color_rgba,
        metallic_roughness=metallic_roughness,
        coverage_mask=mask,
        texture_size=texture_size,
        voxel_count=int(coords.shape[0]),
        k_neighbors=min(k_neighbors, coords.shape[0]),
        origin=tuple(float(value) for value in origin_np),
        voxel_size=float(resolved_voxel_size),
    )


def bake_trellis2_texture_fields_mac_native(
    mesh: FlexibleDualGridMesh,
    texture_coordinates: mx.array | np.ndarray,
    texture_attributes: mx.array | np.ndarray,
    *,
    decode_resolution: int | None = None,
    texture_size: int = 1024,
    origin: tuple[float, float, float] = (-0.5, -0.5, -0.5),
    voxel_size: float | None = None,
    k_neighbors: int = 8,
    max_texture_pixels: int = 1_048_576,
    xatlas_face_guard: int | str = TRELLIS2_XATLAS_AUTO_FACE_GUARD,
    xatlas_parallel_chunks: int = 0,
    texture_bake_backend: str = "trilinear",
    projection_source_mesh: FlexibleDualGridMesh | None = None,
) -> Trellis2TextureBakeResult:
    """Bake TRELLIS.2 texture fields with xatlas UVs and Mac-native sampling."""

    ensure_trellis2_mac_export_dependencies()

    if texture_size <= 0:
        raise ValueError(f"texture_size must be positive, got {texture_size}")
    if k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be positive, got {k_neighbors}")
    if max_texture_pixels <= 0:
        raise ValueError(f"max_texture_pixels must be positive, got {max_texture_pixels}")
    texture_pixels = int(texture_size * texture_size)
    if texture_pixels > max_texture_pixels:
        raise ValueError(
            f"texture bake would allocate {texture_pixels} pixels, above guard {max_texture_pixels}"
        )

    if texture_bake_backend not in TRELLIS2_TEXTURE_BAKE_BACKENDS:
        raise ValueError(
            f"texture_bake_backend must be one of {TRELLIS2_TEXTURE_BAKE_BACKENDS}, got {texture_bake_backend}"
        )

    coords, attrs = _validate_texture_fields(texture_coordinates, texture_attributes, decode_resolution=decode_resolution)
    export_vertices, export_faces = _validate_mesh_np(mesh)
    resolved_xatlas_face_guard = resolve_trellis2_xatlas_face_guard(export_faces.shape[0], xatlas_face_guard)
    xatlas_face_guard_mode = (
        "auto"
        if isinstance(xatlas_face_guard, str) and xatlas_face_guard.strip().lower() == TRELLIS2_XATLAS_AUTO_FACE_GUARD
        else "manual"
    )
    if xatlas_parallel_chunks < 0:
        raise ValueError(f"xatlas_parallel_chunks must be non-negative, got {xatlas_parallel_chunks}")
    unwrap_result = unwrap_trellis2_mesh_xatlas_with_stats(
        mesh,
        face_guard=resolved_xatlas_face_guard,
        parallel_chunks=xatlas_parallel_chunks,
    )
    atlas_vertices, atlas_faces, atlas_uvs = unwrap_result.vertices, unwrap_result.faces, unwrap_result.uvs
    positions, raster_mask = rasterize_uv_positions(atlas_vertices, atlas_faces, atlas_uvs, texture_size)
    if not np.any(raster_mask):
        raise ValueError("xatlas UV rasterization produced no covered texels")

    resolved_voxel_size = _resolve_voxel_size(coords[:, 1:], decode_resolution, voxel_size)
    origin_np = np.asarray(origin, dtype=np.float32)
    if origin_np.shape != (3,):
        raise ValueError(f"origin must contain three values, got {origin}")

    query_points = positions[raster_mask]
    source_projection_used = False
    source_projection_detail = "not requested"
    if projection_source_mesh is not None:
        source_vertices, source_faces = _validate_mesh_np(projection_source_mesh)
        if (
            source_vertices.shape[0] != export_vertices.shape[0]
            or source_faces.shape[0] != export_faces.shape[0]
        ):
            source_projection_detail = (
                "source mesh preserved, but exact CPU BVH projection is not implemented; "
                "sampling export-surface positions"
            )
        else:
            source_projection_detail = "source mesh matches export mesh; projection not needed"

    missing_texel_count = 0
    out_of_grid_texel_count = 0
    if texture_bake_backend == "trilinear":
        sample_result = sample_trellis2_sparse_trilinear_attributes(
            query_points,
            coords,
            attrs[:, :6],
            origin=origin_np,
            voxel_size=resolved_voxel_size,
            decode_resolution=decode_resolution,
        )
        sampled = sample_result.attributes
        valid = sample_result.valid_mask
        sampled_texel_count = sample_result.sampled_texel_count
        missing_texel_count = sample_result.missing_texel_count
        out_of_grid_texel_count = sample_result.out_of_grid_texel_count
        result_backend = "xatlas-trilinear"
    else:
        from scipy.spatial import cKDTree

        voxel_centers = coords[:, 1:].astype(np.float32) * resolved_voxel_size + origin_np + resolved_voxel_size * 0.5
        tree = cKDTree(voxel_centers)
        distances, indices = tree.query(query_points, k=min(k_neighbors, coords.shape[0]), workers=-1)
        if indices.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        max_dist = float(resolved_voxel_size) * 2.0
        eps = float(resolved_voxel_size) * 0.1
        weights = 1.0 / (distances + eps)
        weights[distances > max_dist] = 0.0
        weight_sums = weights.sum(axis=1, keepdims=True)
        valid = (weight_sums[:, 0] > 0.0)
        weights = np.where(weight_sums > 0.0, weights / np.maximum(weight_sums, 1e-12), 0.0)
        sampled = np.sum(attrs[indices, :6] * weights[..., None], axis=1)
        sampled_texel_count = int(np.count_nonzero(valid))
        result_backend = "xatlas-kdtree"

    base_color = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    alpha = np.zeros((texture_size, texture_size, 1), dtype=np.float32)
    metallic = np.zeros((texture_size, texture_size, 1), dtype=np.float32)
    roughness = np.ones((texture_size, texture_size, 1), dtype=np.float32)
    valid_mask = np.zeros((texture_size, texture_size), dtype=bool)

    ys, xs = np.where(raster_mask)
    valid_y = ys[valid]
    valid_x = xs[valid]
    if valid_y.size == 0:
        raise ValueError(f"{texture_bake_backend} texture bake found no texels with sampleable texture voxels")
    base_color[valid_y, valid_x] = np.clip(sampled[valid, TRELLIS2_PBR_ATTR_LAYOUT["base_color"]], 0.0, 1.0)
    metallic[valid_y, valid_x, 0] = np.clip(sampled[valid, TRELLIS2_PBR_ATTR_LAYOUT["metallic"]].reshape(-1), 0.0, 1.0)
    roughness[valid_y, valid_x, 0] = np.clip(sampled[valid, TRELLIS2_PBR_ATTR_LAYOUT["roughness"]].reshape(-1), 0.0, 1.0)
    alpha[valid_y, valid_x, 0] = np.clip(sampled[valid, TRELLIS2_PBR_ATTR_LAYOUT["alpha"]].reshape(-1), 0.0, 1.0)
    valid_mask[valid_y, valid_x] = True
    raw_coverage_ratio = float(np.count_nonzero(valid_mask) / valid_mask.size)

    base_color, final_mask = fill_texture_holes_ndimage(base_color, valid_mask)
    metallic, _ = fill_texture_holes_ndimage(metallic, valid_mask)
    roughness, _ = fill_texture_holes_ndimage(roughness, valid_mask)
    alpha, _ = fill_texture_holes_ndimage(alpha, valid_mask)
    base_color = np.power(np.clip(base_color, 0.0, 1.0), 1.0 / 2.2)

    base_color_rgba = (np.concatenate([base_color, alpha], axis=-1) * 255.0 + 0.5).astype(np.uint8)
    metallic_roughness = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    metallic_roughness[:, :, 1] = (roughness[:, :, 0] * 255.0 + 0.5).astype(np.uint8)
    metallic_roughness[:, :, 2] = (metallic[:, :, 0] * 255.0 + 0.5).astype(np.uint8)

    return Trellis2TextureBakeResult(
        vertices=atlas_vertices,
        faces=atlas_faces,
        uvs=atlas_uvs,
        base_color_rgba=base_color_rgba,
        metallic_roughness=metallic_roughness,
        coverage_mask=final_mask,
        texture_size=texture_size,
        voxel_count=int(coords.shape[0]),
        k_neighbors=min(k_neighbors, coords.shape[0]),
        origin=tuple(float(value) for value in origin_np),
        voxel_size=float(resolved_voxel_size),
        backend=result_backend,
        raw_coverage_ratio=raw_coverage_ratio,
        unwrap_backend=unwrap_result.stats.backend,
        unwrap_seconds=unwrap_result.stats.elapsed_seconds,
        unwrap_chunks=unwrap_result.stats.chunks,
        unwrap_chart_count=unwrap_result.stats.chart_count,
        unwrap_utilization=unwrap_result.stats.utilization,
        xatlas_face_guard=resolved_xatlas_face_guard,
        xatlas_face_guard_mode=xatlas_face_guard_mode,
        sampled_texel_count=sampled_texel_count,
        missing_texel_count=missing_texel_count,
        out_of_grid_texel_count=out_of_grid_texel_count,
        source_projection_used=source_projection_used,
        source_projection_detail=source_projection_detail,
    )


def trellis2_texture_png_payload(image: np.ndarray) -> bytes:
    """Encode a baked texture image as a PNG payload for tests and previews."""

    return texture_png_payload(image)


def trellis2_textured_glb_payload(baked_texture: Trellis2TextureBakeResult) -> bytes:
    """Build a self-contained GLB 2.0 payload from a baked TRELLIS.2 texture."""

    vertices = np.asarray(baked_texture.vertices, dtype=np.float32)
    faces = np.asarray(baked_texture.faces, dtype=np.int64)
    uvs = np.asarray(baked_texture.uvs, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"GLB vertices must have shape (num_vertices, 3), got {vertices.shape}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("GLB vertices must contain only finite values")
    if uvs.ndim != 2 or uvs.shape != (vertices.shape[0], 2):
        raise ValueError(f"GLB UVs must have shape ({vertices.shape[0]}, 2), got {uvs.shape}")
    if not np.all(np.isfinite(uvs)):
        raise ValueError("GLB UVs must contain only finite values")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"GLB faces must have shape (num_faces, 3), got {faces.shape}")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("GLB faces contain vertex indices outside the vertex array")
    if np.any(uvs < 0.0) or np.any(uvs > 1.0):
        raise ValueError("GLB UVs must stay in [0, 1]")
    if baked_texture.base_color_rgba.ndim != 3 or baked_texture.base_color_rgba.shape[2] != 4:
        raise ValueError(
            f"base color texture must have shape (height, width, 4), got {baked_texture.base_color_rgba.shape}"
        )
    if baked_texture.base_color_rgba.dtype != np.uint8:
        raise ValueError(f"base color texture must use uint8 pixels, got {baked_texture.base_color_rgba.dtype}")
    if baked_texture.metallic_roughness.ndim != 3 or baked_texture.metallic_roughness.shape[2] != 3:
        raise ValueError(
            "metallic-roughness texture must have shape "
            f"(height, width, 3), got {baked_texture.metallic_roughness.shape}"
        )
    if baked_texture.metallic_roughness.dtype != np.uint8:
        raise ValueError(
            f"metallic-roughness texture must use uint8 pixels, got {baked_texture.metallic_roughness.dtype}"
        )

    indices = faces.reshape(-1)
    if int(indices.max()) <= np.iinfo(np.uint16).max:
        index_array = indices.astype("<u2", copy=False)
        index_component_type = 5123
    else:
        index_array = indices.astype("<u4", copy=False)
        index_component_type = 5125

    bin_blob = bytearray()
    buffer_views: list[dict] = []

    def add_buffer_view(payload: bytes, *, target: int | None = None) -> int:
        pad_glb_buffer(bin_blob, pad_byte=0)
        offset = len(bin_blob)
        bin_blob.extend(payload)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        buffer_views.append(view)
        return len(buffer_views) - 1

    position_view = add_buffer_view(np.ascontiguousarray(vertices, dtype="<f4").tobytes(), target=34962)
    uv_view = add_buffer_view(np.ascontiguousarray(uvs, dtype="<f4").tobytes(), target=34962)
    index_view = add_buffer_view(np.ascontiguousarray(index_array).tobytes(), target=34963)
    base_color_view = add_buffer_view(trellis2_texture_png_payload(baked_texture.base_color_rgba))
    metallic_roughness_view = add_buffer_view(trellis2_texture_png_payload(baked_texture.metallic_roughness))
    pad_glb_buffer(bin_blob, pad_byte=0)

    document = {
        "asset": {"version": "2.0", "generator": "mlx-spatial TRELLIS.2"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "TRELLIS2_TexturedMesh"}],
        "meshes": [
            {
                "name": "TRELLIS2_TexturedMesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "TRELLIS2_PBR",
                "doubleSided": True,
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                    "metallicFactor": 1.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}],
        "images": [
            {"bufferView": base_color_view, "mimeType": "image/png", "name": "baseColorTexture"},
            {"bufferView": metallic_roughness_view, "mimeType": "image/png", "name": "metallicRoughnessTexture"},
        ],
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


def write_trellis2_textured_glb(
    baked_texture: Trellis2TextureBakeResult,
    output_path: str | Path,
    *,
    outputs_root: str | Path = "outputs",
) -> Trellis2ExportArtifact:
    path = Path(output_path)
    if path.suffix.lower() != ".glb":
        raise ValueError("textured TRELLIS.2 exports require a .glb output path")
    path = validate_trellis2_export_path(path, outputs_root=outputs_root, suffixes=(".glb",))
    payload = trellis2_textured_glb_payload(baked_texture)
    artifact = write_trellis2_export_artifact(payload, path, outputs_root=outputs_root)
    return Trellis2ExportArtifact(
        path=artifact.path,
        format=artifact.format,
        bytes_written=artifact.bytes_written,
        detail="wrote TRELLIS.2 textured GLB",
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


def _validate_mesh_np(mesh: FlexibleDualGridMesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"mesh vertices must have shape (num_vertices, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"mesh faces must have shape (num_faces, 3), got {faces.shape}")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("mesh must contain vertices and faces")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("mesh faces contain vertex indices outside the vertex array")
    return vertices, faces


def _validate_texture_fields(
    texture_coordinates: mx.array | np.ndarray,
    texture_attributes: mx.array | np.ndarray,
    *,
    decode_resolution: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.array(texture_coordinates, dtype=np.int32)
    attrs = np.array(texture_attributes, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"texture coordinates must have shape (num_voxels, 4), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("texture coordinates must contain at least one voxel")
    if np.any(coords[:, 0] != 0):
        raise ValueError("texture baking currently supports only batch index 0")
    if attrs.ndim != 2 or attrs.shape[0] != coords.shape[0] or attrs.shape[1] < 6:
        raise ValueError(
            f"texture attributes must have shape ({coords.shape[0]}, at least 6), got {attrs.shape}"
        )
    if not np.all(np.isfinite(attrs[:, :6])):
        raise ValueError("texture attributes must contain only finite values")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("texture coordinates must be unique")

    spatial_coords = coords[:, 1:]
    if np.any(spatial_coords < 0):
        raise ValueError("texture spatial coordinates must be non-negative")
    if decode_resolution is not None and np.any(spatial_coords >= decode_resolution):
        raise ValueError(
            f"texture spatial coordinates must be < decode_resolution ({decode_resolution})"
        )
    order = np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1], coords[:, 0]))
    return coords[order], attrs[order]


def _remove_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    area_epsilon: float = 1e-14,
) -> tuple[np.ndarray, np.ndarray, int]:
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = distinct & np.isfinite(area2) & (area2 > area_epsilon)
    return vertices, faces[keep], int(np.count_nonzero(~keep))


def _remove_duplicate_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    if faces.shape[0] == 0:
        return faces, 0
    canonical = np.sort(faces, axis=1)
    _, first_indices = np.unique(canonical, axis=0, return_index=True)
    first_indices.sort()
    return faces[first_indices], int(faces.shape[0] - first_indices.shape[0])


def _compact_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_vertices = vertices[used]
    compact_faces = inverse.reshape(faces.shape).astype(np.int64, copy=False)
    return compact_vertices, compact_faces, int(vertices.shape[0] - used.shape[0])


def _remove_small_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_component_faces: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    if faces.shape[0] == 0:
        raise ValueError("mesh cleanup produced no faces")

    parent = np.arange(vertices.shape[0], dtype=np.int64)
    rank = np.zeros(vertices.shape[0], dtype=np.int8)

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
        return vertices, faces, 0, 0
    return vertices, faces[keep], int(len(roots) - len(keep_roots)), removed_faces


def _mesh_edge_stats(faces: np.ndarray) -> tuple[int, int]:
    directed = np.empty((faces.shape[0] * 3, 2), dtype=np.int64)
    directed[0::3] = faces[:, [0, 1]]
    directed[1::3] = faces[:, [1, 2]]
    directed[2::3] = faces[:, [2, 0]]
    keys = np.sort(directed, axis=1)
    key_view = np.ascontiguousarray(keys).view([("a", keys.dtype), ("b", keys.dtype)]).reshape(-1)
    _, counts = np.unique(key_view, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts != 2))


def _combine_hole_fill_stats(first: MeshHoleFillStats, second: MeshHoleFillStats) -> MeshHoleFillStats:
    return MeshHoleFillStats(
        boundary_edges_before=first.boundary_edges_before,
        clean_boundary_loops=first.clean_boundary_loops + second.clean_boundary_loops,
        filled_loops=first.filled_loops + second.filled_loops,
        skipped_large_loops=first.skipped_large_loops + second.skipped_large_loops,
        skipped_complex_components=first.skipped_complex_components + second.skipped_complex_components,
        vertices_added=first.vertices_added + second.vertices_added,
        faces_added=first.faces_added + second.faces_added,
    )


def _resolve_voxel_size(
    spatial_coordinates: np.ndarray,
    decode_resolution: int | None,
    voxel_size: float | None,
) -> float:
    if voxel_size is not None:
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive, got {voxel_size}")
        return float(voxel_size)
    if decode_resolution is not None:
        if decode_resolution <= 0:
            raise ValueError(f"decode_resolution must be positive, got {decode_resolution}")
        return 1.0 / float(decode_resolution)
    inferred_resolution = int(np.max(spatial_coordinates)) + 1
    if inferred_resolution <= 0:
        raise ValueError("could not infer a positive texture voxel resolution")
    return 1.0 / float(inferred_resolution)


def sample_trellis2_sparse_trilinear_attributes(
    query_points: np.ndarray,
    texture_coordinates: mx.array | np.ndarray,
    texture_attributes: mx.array | np.ndarray,
    *,
    origin: tuple[float, float, float] | np.ndarray = (-0.5, -0.5, -0.5),
    voxel_size: float | None = None,
    decode_resolution: int | None = None,
    require_all_corners: bool = False,
) -> Trellis2SparseTrilinearSampleResult:
    """Sample sparse TRELLIS.2 voxel attributes with dense-grid trilinear semantics."""

    coords = np.asarray(texture_coordinates, dtype=np.int32)
    attrs = np.asarray(texture_attributes, dtype=np.float32)
    queries = np.asarray(query_points, dtype=np.float32)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError(f"query_points must have shape (num_queries, 3), got {queries.shape}")
    if not np.all(np.isfinite(queries)):
        raise ValueError("query_points must contain only finite values")
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"texture coordinates must have shape (num_voxels, 4), got {coords.shape}")
    if coords.shape[0] == 0:
        raise ValueError("texture coordinates must contain at least one voxel")
    if np.any(coords[:, 0] != 0):
        raise ValueError("sparse trilinear sampling currently supports only batch index 0")
    if attrs.ndim != 2 or attrs.shape[0] != coords.shape[0]:
        raise ValueError(f"texture attributes must have shape ({coords.shape[0]}, channels), got {attrs.shape}")
    if attrs.shape[1] == 0:
        raise ValueError("texture attributes must contain at least one channel")
    if not np.all(np.isfinite(attrs)):
        raise ValueError("texture attributes must contain only finite values")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("texture coordinates must be unique")

    spatial_coords = coords[:, 1:]
    if np.any(spatial_coords < 0):
        raise ValueError("texture spatial coordinates must be non-negative")
    if decode_resolution is not None and np.any(spatial_coords >= decode_resolution):
        raise ValueError(f"texture spatial coordinates must be < decode_resolution ({decode_resolution})")
    resolved_voxel_size = _resolve_voxel_size(spatial_coords, decode_resolution, voxel_size)
    if decode_resolution is None:
        grid_shape = spatial_coords.max(axis=0).astype(np.int64) + 1
    else:
        grid_shape = np.array([decode_resolution, decode_resolution, decode_resolution], dtype=np.int64)
    if np.any(grid_shape <= 0):
        raise ValueError(f"sparse trilinear grid shape must be positive, got {grid_shape}")

    origin_np = np.asarray(origin, dtype=np.float32)
    if origin_np.shape != (3,):
        raise ValueError(f"origin must contain three values, got {origin}")
    grid_points = (queries - origin_np.reshape(1, 3)) / float(resolved_voxel_size)
    base = np.floor(grid_points).astype(np.int64)
    frac = grid_points - base.astype(np.float32)
    strides = np.array([grid_shape[1] * grid_shape[2], grid_shape[2], 1], dtype=np.int64)
    coord_keys = spatial_coords.astype(np.int64) @ strides
    order = np.argsort(coord_keys, kind="mergesort")
    sorted_keys = coord_keys[order]
    sorted_indices = order.astype(np.int64, copy=False)

    sampled = np.zeros((queries.shape[0], attrs.shape[1]), dtype=np.float32)
    present_weight = np.zeros((queries.shape[0],), dtype=np.float32)
    missing = np.zeros((queries.shape[0],), dtype=bool)
    out_of_grid = np.zeros((queries.shape[0],), dtype=bool)
    eps = 1e-6
    for oz in (0, 1):
        wz = frac[:, 0] if oz else 1.0 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            for ox in (0, 1):
                wx = frac[:, 2] if ox else 1.0 - frac[:, 2]
                weights = (wz * wy * wx).astype(np.float32, copy=False)
                active = weights > eps
                if not np.any(active):
                    continue
                corners = base + np.array([oz, oy, ox], dtype=np.int64).reshape(1, 3)
                inside = np.all((corners >= 0) & (corners < grid_shape.reshape(1, 3)), axis=1)
                out_of_grid |= active & ~inside
                candidates = active & inside
                if not np.any(candidates):
                    continue
                keys = corners[candidates] @ strides
                positions = np.searchsorted(sorted_keys, keys)
                found = positions < sorted_keys.shape[0]
                safe_positions = np.minimum(positions, sorted_keys.shape[0] - 1)
                found &= sorted_keys[safe_positions] == keys
                candidate_rows = np.flatnonzero(candidates)
                found_rows = candidate_rows[found]
                missing[candidate_rows[~found]] = True
                if found_rows.size:
                    attr_indices = sorted_indices[safe_positions[found]]
                    corner_weights = weights[found_rows]
                    sampled[found_rows] += attrs[attr_indices] * corner_weights[:, None]
                    present_weight[found_rows] += corner_weights

    valid = present_weight > eps
    if require_all_corners:
        invalid = missing | out_of_grid | ~valid
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0])
            raise ValueError(
                "sparse trilinear sampling requires all non-zero interpolation corners; "
                f"query {first} is missing required sparse voxel data"
            )
    return Trellis2SparseTrilinearSampleResult(
        attributes=sampled,
        valid_mask=valid,
        sampled_texel_count=int(np.count_nonzero(valid)),
        missing_texel_count=int(np.count_nonzero(missing)),
        out_of_grid_texel_count=int(np.count_nonzero(out_of_grid)),
    )


def _sample_voxel_attributes(
    query_points: np.ndarray,
    voxel_centers: np.ndarray,
    voxel_coordinates: np.ndarray,
    attributes: np.ndarray,
    *,
    k_neighbors: int,
    max_query_voxel_pairs: int,
    voxel_size: float,
    origin: np.ndarray,
) -> np.ndarray:
    pair_count = int(query_points.shape[0] * voxel_centers.shape[0])
    if pair_count > max_query_voxel_pairs:
        return _sample_voxel_attributes_spatial_hash(
            query_points,
            voxel_coordinates,
            attributes,
            k_neighbors=k_neighbors,
            voxel_size=voxel_size,
            origin=origin,
        )

    return _sample_voxel_attributes_dense(
        query_points,
        voxel_centers,
        attributes,
        k_neighbors=k_neighbors,
    )


def _sample_voxel_attributes_dense(
    query_points: np.ndarray,
    voxel_centers: np.ndarray,
    attributes: np.ndarray,
    *,
    k_neighbors: int,
) -> np.ndarray:
    distances_sq = np.sum((query_points[:, None, :] - voxel_centers[None, :, :]) ** 2, axis=-1)
    if k_neighbors == voxel_centers.shape[0]:
        candidate_indices = np.broadcast_to(np.arange(voxel_centers.shape[0]), distances_sq.shape)
        neighbor_indices = np.lexsort((candidate_indices, distances_sq), axis=1)
    else:
        partitioned = np.argpartition(distances_sq, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]
        neighbor_distances = np.take_along_axis(distances_sq, partitioned, axis=1)
        order = np.lexsort((partitioned, neighbor_distances), axis=1)
        neighbor_indices = np.take_along_axis(partitioned, order, axis=1)
    neighbor_distances = np.sqrt(np.take_along_axis(distances_sq, neighbor_indices, axis=1))
    eps = 1e-6
    weights = 1.0 / (neighbor_distances + eps)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return np.sum(attributes[neighbor_indices] * weights[..., None], axis=1)


def _sample_voxel_attributes_spatial_hash(
    query_points: np.ndarray,
    voxel_coordinates: np.ndarray,
    attributes: np.ndarray,
    *,
    k_neighbors: int,
    voxel_size: float,
    origin: np.ndarray,
    max_search_radius: int = 64,
) -> np.ndarray:
    if max_search_radius <= 0:
        raise ValueError(f"max_search_radius must be positive, got {max_search_radius}")
    coords = np.asarray(voxel_coordinates, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"voxel coordinates must have shape (num_voxels, 3), got {coords.shape}")
    spatial_shape = coords.max(axis=0).astype(np.int64) + 1
    strides = np.array([spatial_shape[1] * spatial_shape[2], spatial_shape[2], 1], dtype=np.int64)
    coord_keys = coords.astype(np.int64) @ strides
    lookup = {int(key): int(index) for index, key in enumerate(coord_keys)}
    shells = [_cube_shell_offsets(radius) for radius in range(max_search_radius + 1)]
    query_grid = (np.asarray(query_points, dtype=np.float32) - origin.astype(np.float32)) / float(voxel_size) - 0.5
    sampled = np.empty((query_grid.shape[0], attributes.shape[1]), dtype=np.float32)

    for query_index, query in enumerate(query_grid):
        base = np.floor(query + 0.5).astype(np.int32)
        fractional = query - base.astype(np.float32)
        candidates: list[int] = []
        selected: np.ndarray | None = None
        for radius, offsets in enumerate(shells):
            shell_coords = base[None, :] + offsets
            inside = np.all((shell_coords >= 0) & (shell_coords < spatial_shape[None, :]), axis=1)
            if np.any(inside):
                keys = shell_coords[inside].astype(np.int64) @ strides
                for key in keys:
                    candidate_index = lookup.get(int(key))
                    if candidate_index is not None:
                        candidates.append(candidate_index)
            if len(candidates) >= k_neighbors:
                candidate_array = np.array(candidates, dtype=np.int64)
                distances_sq = np.sum((coords[candidate_array].astype(np.float32) - query[None, :]) ** 2, axis=1)
                order = np.lexsort((candidate_array, distances_sq))
                kth_distance_sq = float(distances_sq[order[k_neighbors - 1]])
                outside_lower_bound_sq = float(np.min(np.square(radius + 1 - np.abs(fractional))))
                if kth_distance_sq < outside_lower_bound_sq:
                    selected = candidate_array[order[:k_neighbors]]
                    break
        if selected is None:
            raise ValueError(
                f"texture bake spatial hash could not find {k_neighbors} voxel neighbor(s) "
                f"within radius {max_search_radius} for query {query_index}"
            )
        neighbor_distances = np.sqrt(np.sum((coords[selected].astype(np.float32) - query[None, :]) ** 2, axis=1))
        eps = 1e-6
        weights = 1.0 / (neighbor_distances + eps)
        weights /= np.sum(weights)
        sampled[query_index] = np.sum(attributes[selected] * weights[:, None], axis=0)

    return sampled


def _cube_shell_offsets(radius: int) -> np.ndarray:
    values = np.arange(-radius, radius + 1, dtype=np.int32)
    zz, yy, xx = np.meshgrid(values, values, values, indexing="ij")
    offsets = np.stack([zz.reshape(-1), yy.reshape(-1), xx.reshape(-1)], axis=1)
    if radius == 0:
        return offsets
    return offsets[np.max(np.abs(offsets), axis=1) == radius]
