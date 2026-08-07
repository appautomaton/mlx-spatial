"""Thin Python entry points for native mesh functionality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._native import clean_mesh as _clean_mesh
from ._native import backend_info, extract_flexi_dual_grid as _extract_flexi_dual_grid
from ._native import fill_holes as _fill_holes
from ._native import mesh_metrics as _mesh_metrics
from ._native import point_to_mesh_distances as _point_to_mesh_distances
from ._native import repair_nonmanifold_mesh as _repair_nonmanifold_mesh
from ._native import remesh_narrow_band as _remesh_narrow_band
from ._native import simplify_mesh as _simplify_mesh
from ._native import unify_face_orientations as _unify_face_orientations
from ._native import validate_pixal3d_shape_fields


@dataclass(frozen=True)
class NativeMesh:
    """Triangle mesh returned by native mlx-spatial SpatialKit extraction."""

    vertices: np.ndarray
    faces: np.ndarray


def extract_flexi_dual_grid(
    coordinates: np.ndarray,
    fields: np.ndarray,
    *,
    grid_size: int,
) -> NativeMesh:
    """Extract a FlexiDualGrid triangle mesh through the native C++ backend."""

    result = _extract_flexi_dual_grid(coordinates, fields, int(grid_size))
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"]))


def mesh_metrics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    """Return native mesh diagnostics."""

    return dict(_mesh_metrics(vertices, faces))


def point_to_mesh_distances(
    query_points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    return_closest: bool = False,
) -> tuple[np.ndarray, dict[str, object]] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return exact unsigned distances from points to a triangle mesh."""

    result = _point_to_mesh_distances(
        np.ascontiguousarray(query_points, dtype=np.float32),
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int64),
    )
    distances = np.asarray(result["distances"])
    stats = dict(result["stats"])
    if return_closest:
        return (
            distances,
            np.asarray(result["closest_points"]),
            np.asarray(result["closest_faces"]),
            stats,
        )
    return distances, stats


def bidirectional_surface_distance_metrics(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    candidate_vertices: np.ndarray,
    candidate_faces: np.ndarray,
    *,
    max_samples_per_mesh: int = 200_000,
    voxel_size: float | None = None,
) -> dict[str, object]:
    """Measure sampled bidirectional distance against exact triangle surfaces.

    Samples are deterministic mesh vertices plus triangle centroids. Each
    query is measured against the other mesh's exact triangle surface by the
    shared native BVH. The resulting maximum is therefore a sampled Hausdorff
    estimate, not an unsupported claim of continuous-mesh Hausdorff distance.
    """

    if max_samples_per_mesh <= 0:
        raise ValueError("max_samples_per_mesh must be positive")
    if voxel_size is not None and (not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("voxel_size must be positive and finite")

    source_vertices = np.ascontiguousarray(source_vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(source_faces, dtype=np.int64)
    candidate_vertices = np.ascontiguousarray(candidate_vertices, dtype=np.float32)
    candidate_faces = np.ascontiguousarray(candidate_faces, dtype=np.int64)
    source_samples, source_sample_stats = _deterministic_surface_samples(
        source_vertices,
        source_faces,
        max_samples=max_samples_per_mesh,
    )
    candidate_samples, candidate_sample_stats = _deterministic_surface_samples(
        candidate_vertices,
        candidate_faces,
        max_samples=max_samples_per_mesh,
    )

    candidate_to_source, candidate_closest, candidate_closest_faces, candidate_query_stats = point_to_mesh_distances(
        candidate_samples,
        source_vertices,
        source_faces,
        return_closest=True,
    )
    source_to_candidate, source_closest, source_closest_faces, source_query_stats = point_to_mesh_distances(
        source_samples,
        candidate_vertices,
        candidate_faces,
        return_closest=True,
    )

    source_extent = np.ptp(source_vertices, axis=0)
    source_bbox_diagonal = float(np.linalg.norm(source_extent.astype(np.float64)))
    if not np.isfinite(source_bbox_diagonal) or source_bbox_diagonal <= 0:
        raise ValueError("source mesh bounding-box diagonal must be positive and finite")
    candidate_summary = _distance_summary(
        candidate_to_source,
        query_points=candidate_samples,
        closest_points=candidate_closest,
        closest_faces=candidate_closest_faces,
        bbox_diagonal=source_bbox_diagonal,
        voxel_size=voxel_size,
    )
    source_summary = _distance_summary(
        source_to_candidate,
        query_points=source_samples,
        closest_points=source_closest,
        closest_faces=source_closest_faces,
        bbox_diagonal=source_bbox_diagonal,
        voxel_size=voxel_size,
    )
    symmetric = {
        "sampled_chamfer_l1": float((candidate_summary["mean"] + source_summary["mean"]) * 0.5),
        "sampled_p95_max": float(max(candidate_summary["p95"], source_summary["p95"])),
        "sampled_p99_max": float(max(candidate_summary["p99"], source_summary["p99"])),
        "sampled_hausdorff": float(max(candidate_summary["max"], source_summary["max"])),
    }
    symmetric.update(
        {
            f"{name}_bbox_diagonal_ratio": float(value / source_bbox_diagonal)
            for name, value in tuple(symmetric.items())
        }
    )
    if voxel_size is not None:
        symmetric.update(
            {
                f"{name}_voxels": float(value / voxel_size)
                for name, value in tuple(symmetric.items())
                if not name.endswith("_ratio")
            }
        )

    return {
        "method": "deterministic-vertices-and-face-centroids-to-exact-triangle-bvh",
        "distance_kind": "unsigned-bidirectional-sampled-surface-distance",
        "source_bbox_diagonal": source_bbox_diagonal,
        "voxel_size": voxel_size,
        "source_samples": source_sample_stats,
        "candidate_samples": candidate_sample_stats,
        "candidate_to_source": {**candidate_summary, "native": candidate_query_stats},
        "source_to_candidate": {**source_summary, "native": source_query_stats},
        "symmetric": symmetric,
    }


def sampled_surface_to_mesh_distance_metrics(
    query_vertices: np.ndarray,
    query_faces: np.ndarray,
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    *,
    max_samples: int = 200_000,
    normalization_vertices: np.ndarray | None = None,
    voxel_size: float | None = None,
) -> dict[str, object]:
    """Measure one sampled mesh surface against an exact triangle surface."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if voxel_size is not None and (not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("voxel_size must be positive and finite")
    query_vertices = np.ascontiguousarray(query_vertices, dtype=np.float32)
    query_faces = np.ascontiguousarray(query_faces, dtype=np.int64)
    reference_vertices = np.ascontiguousarray(reference_vertices, dtype=np.float32)
    reference_faces = np.ascontiguousarray(reference_faces, dtype=np.int64)
    samples, sample_stats = _deterministic_surface_samples(
        query_vertices,
        query_faces,
        max_samples=max_samples,
    )
    distances, closest, closest_faces, native_stats = point_to_mesh_distances(
        samples,
        reference_vertices,
        reference_faces,
        return_closest=True,
    )
    normalization = (
        reference_vertices
        if normalization_vertices is None
        else np.ascontiguousarray(normalization_vertices, dtype=np.float32)
    )
    bbox_diagonal = float(np.linalg.norm(np.ptp(normalization, axis=0).astype(np.float64)))
    if not np.isfinite(bbox_diagonal) or bbox_diagonal <= 0:
        raise ValueError("normalization bounding-box diagonal must be positive and finite")
    return {
        "method": "deterministic-vertices-and-face-centroids-to-exact-triangle-bvh",
        "distance_kind": "unsigned-one-way-sampled-surface-distance",
        "normalization_bbox_diagonal": bbox_diagonal,
        "voxel_size": voxel_size,
        "query_samples": sample_stats,
        **_distance_summary(
            distances,
            query_points=samples,
            closest_points=closest,
            closest_faces=closest_faces,
            bbox_diagonal=bbox_diagonal,
            voxel_size=voxel_size,
        ),
        "native": native_stats,
    }


def _deterministic_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_samples: int,
) -> tuple[np.ndarray, dict[str, int | str]]:
    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])
    if vertex_count <= 0 or face_count <= 0:
        raise ValueError("surface sampling requires non-empty vertices and faces")

    if vertex_count + face_count <= max_samples:
        vertex_budget = vertex_count
        face_budget = face_count
    else:
        vertex_budget = min(vertex_count, max_samples // 2)
        face_budget = min(face_count, max_samples - vertex_budget)
        vertex_budget = min(vertex_count, max_samples - face_budget)

    vertex_indices = _evenly_spaced_indices(vertex_count, vertex_budget)
    face_indices = _evenly_spaced_indices(face_count, face_budget)
    sampled_vertices = vertices[vertex_indices]
    sampled_faces = faces[face_indices]
    sampled_centroids = vertices[sampled_faces].mean(axis=1, dtype=np.float32)
    samples = np.ascontiguousarray(
        np.concatenate((sampled_vertices, sampled_centroids), axis=0),
        dtype=np.float32,
    )
    return samples, {
        "policy": "deterministic-even-index-vertices-and-face-centroids",
        "mesh_vertices": vertex_count,
        "mesh_faces": face_count,
        "sampled_vertices": int(vertex_budget),
        "sampled_face_centroids": int(face_budget),
        "total": int(samples.shape[0]),
    }


def _evenly_spaced_indices(count: int, budget: int) -> np.ndarray:
    if budget <= 0:
        return np.empty((0,), dtype=np.int64)
    if budget >= count:
        return np.arange(count, dtype=np.int64)
    return np.floor(np.arange(budget, dtype=np.float64) * count / budget).astype(np.int64)


def _distance_summary(
    distances: np.ndarray,
    *,
    query_points: np.ndarray,
    closest_points: np.ndarray,
    closest_faces: np.ndarray,
    bbox_diagonal: float,
    voxel_size: float | None,
) -> dict[str, object]:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or values.size <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("native surface distances must be a non-empty finite vector")
    if query_points.shape != (values.size, 3) or closest_points.shape != (values.size, 3):
        raise ValueError("native closest-point arrays do not match the distance vector")
    if closest_faces.shape != (values.size,):
        raise ValueError("native closest-face array does not match the distance vector")
    max_index = int(np.argmax(values))
    summary: dict[str, object] = {
        "sample_count": int(values.size),
        "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "max_sample_index": max_index,
        "max_query_point": [float(value) for value in query_points[max_index]],
        "max_closest_point": [float(value) for value in closest_points[max_index]],
        "max_closest_face": int(closest_faces[max_index]),
        "max_displacement": [
            float(value)
            for value in (query_points[max_index].astype(np.float64) - closest_points[max_index])
        ],
    }
    for name in ("mean", "rms", "p50", "p95", "p99", "max"):
        value = float(summary[name])
        summary[f"{name}_bbox_diagonal_ratio"] = float(value / bbox_diagonal)
        if voxel_size is not None:
            summary[f"{name}_voxels"] = float(value / voxel_size)
    return summary


def clean_mesh(vertices: np.ndarray, faces: np.ndarray, *, min_component_faces: int = 32) -> tuple[NativeMesh, dict[str, object]]:
    """Clean mesh data through the native backend."""

    result = _clean_mesh(vertices, faces, int(min_component_faces))
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


def repair_nonmanifold_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[NativeMesh, dict[str, object]]:
    """Split disconnected incident face fans without deleting mesh faces."""

    result = _repair_nonmanifold_mesh(vertices, faces)
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


def fill_holes(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_hole_perimeter: float = 0.03,
) -> tuple[NativeMesh, dict[str, object]]:
    """Fill clean boundary loops with the CuMesh perimeter centroid-fan contract."""

    result = _fill_holes(vertices, faces, float(max_hole_perimeter))
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


def unify_face_orientations(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[NativeMesh, dict[str, object]]:
    """Unify face winding across manifold adjacency with CuMesh parity semantics."""

    result = _unify_face_orientations(vertices, faces)
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_faces: int,
    min_component_faces: int = 32,
    backend: str = "spatial-cluster",
    small_boundary_loop_fill_max_edges: int = 8,
    small_boundary_loop_fill_max_perimeter: float = 0.03,
) -> tuple[NativeMesh, dict[str, object]]:
    """Simplify mesh data through the native-owned first-pass interface."""

    result = _simplify_mesh(
        vertices,
        faces,
        int(target_faces),
        int(min_component_faces),
        str(backend),
        int(small_boundary_loop_fill_max_edges),
        float(small_boundary_loop_fill_max_perimeter),
    )
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


def simplify_mesh_mlx_parallel_qem(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_faces: int,
    max_rounds: int = 48,
    topology_policy: str = "manifold-preserving",
) -> tuple[NativeMesh, dict[str, object]]:
    """Run the experimental CuMesh-style batched QEM path on MLX GPU."""

    from .mlx_qem import simplify_mesh_mlx_parallel_qem as _simplify_mesh_mlx_parallel_qem

    out_vertices, out_faces, stats = _simplify_mesh_mlx_parallel_qem(
        vertices,
        faces,
        target_faces=int(target_faces),
        max_rounds=int(max_rounds),
        topology_policy=str(topology_policy),
    )
    return NativeMesh(vertices=out_vertices, faces=out_faces), dict(stats)


def remesh_narrow_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    resolution: int,
    band: float = 1.0,
    project_back: float = 0.0,
    repair_nonmanifold: bool = False,
) -> tuple[NativeMesh, dict[str, object]]:
    """Build the official upstream-style watertight UDF offset shell.

    This mirrors CuMesh's ``UDF - eps`` behavior. The resulting inner/outer
    offset shell is intentionally not a single-layer copy of the input, but
    that representation is the reference application's remesh mechanism and
    must not be treated as a parity failure by itself.
    """

    result = _remesh_narrow_band(
        vertices, faces, int(resolution), float(band), float(project_back), bool(repair_nonmanifold)
    )
    return NativeMesh(vertices=np.asarray(result["vertices"]), faces=np.asarray(result["faces"])), dict(result["stats"])


__all__ = [
    "NativeMesh",
    "backend_info",
    "clean_mesh",
    "extract_flexi_dual_grid",
    "fill_holes",
    "mesh_metrics",
    "bidirectional_surface_distance_metrics",
    "point_to_mesh_distances",
    "sampled_surface_to_mesh_distance_metrics",
    "repair_nonmanifold_mesh",
    "remesh_narrow_band",
    "simplify_mesh",
    "simplify_mesh_mlx_parallel_qem",
    "unify_face_orientations",
    "validate_pixal3d_shape_fields",
]
