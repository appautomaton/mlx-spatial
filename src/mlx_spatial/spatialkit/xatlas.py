"""Generic Apple Silicon xatlas UV unwrap helpers.

The CUDA/CuMesh implementation is never imported or executed here.  The
clustered mode mirrors the repository's pinned oracle composition: native
cone clusters become independent xatlas meshes inside one shared atlas.
The spatial mode unwraps independent spatial partitions concurrently and
packs their local atlases into deterministic tiles.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np

_T = TypeVar("_T")
_Observer = Callable[[str, Callable[[], Any]], Any]

XATLAS_PARALLEL_FACE_TARGET = 50_000
XATLAS_MAX_AUTO_PARALLEL_CHUNKS = 8


@dataclass(frozen=True)
class XAtlasUvResult:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    chart_ids: np.ndarray
    source_face_ids: np.ndarray
    stats: dict[str, Any]


def unwrap_xatlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    cluster_ids: np.ndarray | None = None,
    observe: _Observer | None = None,
) -> XAtlasUvResult:
    """Unwrap one mesh globally or a fixed set of clusters in one atlas."""

    import xatlas  # noqa: PLC0415 (optional work is intentionally lazy)

    source_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(faces, dtype=np.int64)
    _validate_mesh(source_vertices, source_faces)

    if cluster_ids is None:
        dense_cluster_ids = np.zeros(source_faces.shape[0], dtype=np.int64)
        backend = "xatlas-global"
    else:
        dense_cluster_ids = np.ascontiguousarray(cluster_ids, dtype=np.int64)
        if dense_cluster_ids.shape != (source_faces.shape[0],):
            raise ValueError(
                f"cluster_ids must have shape ({source_faces.shape[0]},), "
                f"got {dense_cluster_ids.shape}"
            )
        if dense_cluster_ids.size and int(dense_cluster_ids.min()) < 0:
            raise ValueError("cluster_ids must be non-negative")
        unique_ids = np.unique(dense_cluster_ids)
        if not np.array_equal(unique_ids, np.arange(unique_ids.size, dtype=np.int64)):
            raise ValueError("cluster_ids must be dense and start at zero")
        backend = "xatlas-clustered"

    run = _make_observer(observe)
    atlas = xatlas.Atlas()

    def add_meshes() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        cluster_count = int(dense_cluster_ids.max()) + 1 if dense_cluster_ids.size else 0
        # Build cluster membership once. Re-running a full boolean scan for
        # every cluster is O(cluster_count * face_count) and dominated real
        # Pixal3D UV time. Stable grouping preserves the ascending source-face
        # order produced by np.flatnonzero in the original implementation.
        face_order = np.argsort(dense_cluster_ids, kind="stable")
        cluster_counts = np.bincount(dense_cluster_ids, minlength=cluster_count)
        cluster_offsets = np.empty(cluster_count + 1, dtype=np.int64)
        cluster_offsets[0] = 0
        np.cumsum(cluster_counts, out=cluster_offsets[1:])
        for cluster_index in range(cluster_count):
            source_face_ids = face_order[
                cluster_offsets[cluster_index] : cluster_offsets[cluster_index + 1]
            ]
            local_source_vertices, inverse = np.unique(
                source_faces[source_face_ids].reshape(-1),
                return_inverse=True,
            )
            local_faces = np.ascontiguousarray(
                inverse.reshape(-1, 3),
                dtype=np.uint32,
            )
            local_vertices = np.ascontiguousarray(
                source_vertices[local_source_vertices],
                dtype=np.float32,
            )
            atlas.add_mesh(local_vertices, local_faces)
            inputs.append((local_vertices, local_source_vertices, source_face_ids))
        return inputs

    inputs = run(f"uv.{backend}.add_meshes", add_meshes)

    def generate() -> None:
        chart_options = xatlas.ChartOptions()
        chart_options.max_cost = 2.0
        chart_options.normal_deviation_weight = 2.0
        chart_options.roundness_weight = 0.01
        chart_options.straightness_weight = 6.0
        chart_options.normal_seam_weight = 4.0
        chart_options.texture_seam_weight = 0.5
        chart_options.max_iterations = 1
        pack_options = xatlas.PackOptions()
        pack_options.padding = 0
        pack_options.bilinear = True
        pack_options.rotate_charts = True
        pack_options.bruteForce = False
        atlas.generate(chart_options, pack_options, False)

    run(f"uv.{backend}.generate", generate)

    def assemble() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        out_vertices: list[np.ndarray] = []
        out_uvs: list[np.ndarray] = []
        out_faces = np.empty_like(source_faces)
        out_chart_ids = np.full(source_faces.shape[0], -1, dtype=np.int64)
        vertex_offset = 0
        chart_offset = 0
        for mesh_index, (local_vertices, _local_source_vertices, source_face_ids) in enumerate(inputs):
            vmapping, indices, uvs = atlas.get_mesh(mesh_index)
            mesh_vertices = local_vertices[np.asarray(vmapping, dtype=np.int64)]
            mesh_faces = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
            mesh_uvs = np.asarray(uvs, dtype=np.float32)
            mesh_chart_count = int(atlas.get_mesh_chart_count(mesh_index))
            local_chart_ids = np.full(source_face_ids.shape[0], -1, dtype=np.int64)
            for chart_index in range(mesh_chart_count):
                chart = atlas.get_mesh_chart(mesh_index, chart_index)
                local_chart_ids[np.asarray(chart.faces, dtype=np.int64)] = chart_offset + chart_index
            out_faces[source_face_ids] = mesh_faces + vertex_offset
            out_chart_ids[source_face_ids] = local_chart_ids
            out_vertices.append(mesh_vertices)
            out_uvs.append(mesh_uvs)
            vertex_offset += int(mesh_vertices.shape[0])
            chart_offset += mesh_chart_count
        return (
            np.ascontiguousarray(np.concatenate(out_vertices, axis=0), dtype=np.float32),
            np.ascontiguousarray(out_faces, dtype=np.int64),
            np.ascontiguousarray(np.concatenate(out_uvs, axis=0), dtype=np.float32),
            out_chart_ids,
        )

    out_vertices, out_faces, out_uvs, chart_ids = run(f"uv.{backend}.assemble", assemble)
    stats = {
        "backend": backend,
        "xatlas_version": str(xatlas.__version__),
        "source_vertices": int(source_vertices.shape[0]),
        "source_faces": int(source_faces.shape[0]),
        "output_vertices": int(out_vertices.shape[0]),
        "output_faces": int(out_faces.shape[0]),
        "duplicated_vertex_ratio": float(out_vertices.shape[0] / max(source_vertices.shape[0], 1)),
        "input_cluster_count": len(inputs),
        "chart_count": int(atlas.chart_count),
        "atlas_count": int(atlas.atlas_count),
        "atlas_width": int(atlas.width),
        "atlas_height": int(atlas.height),
        "atlas_utilization": float(atlas.utilization),
        "unassigned_face_count": int(np.count_nonzero(chart_ids < 0)),
        "spatial_partition_shared_edge_count": 0,
        "spatial_partition_cut_edge_count": 0,
        "spatial_partition_cut_edge_ratio": 0.0,
    }
    return XAtlasUvResult(
        vertices=out_vertices,
        faces=out_faces,
        uvs=np.clip(out_uvs, 0.0, 1.0),
        chart_ids=chart_ids,
        source_face_ids=np.arange(source_faces.shape[0], dtype=np.int64),
        stats=stats,
    )


def resolve_xatlas_parallel_chunks(
    face_count: int,
    requested_chunks: int,
    *,
    face_target: int = XATLAS_PARALLEL_FACE_TARGET,
    max_auto_chunks: int = XATLAS_MAX_AUTO_PARALLEL_CHUNKS,
) -> int:
    """Resolve an explicit or automatic spatial xatlas worker count."""

    if face_count < 0:
        raise ValueError(f"face_count must be non-negative, got {face_count}")
    if requested_chunks < 0:
        raise ValueError(f"requested_chunks must be non-negative, got {requested_chunks}")
    if face_target <= 0:
        raise ValueError(f"face_target must be positive, got {face_target}")
    if max_auto_chunks <= 0:
        raise ValueError(f"max_auto_chunks must be positive, got {max_auto_chunks}")
    if requested_chunks > 0:
        return min(int(requested_chunks), max(face_count, 1))
    if face_count <= face_target:
        return 1
    cpu_count = max(1, os.cpu_count() or 1)
    face_chunks = int(math.ceil(face_count / face_target))
    return max(1, min(face_chunks, cpu_count, max_auto_chunks))


def unwrap_xatlas_spatial(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    chunks: int,
    tile_padding: float = 0.02,
    observe: _Observer | None = None,
) -> XAtlasUvResult:
    """Unwrap one mesh globally or as deterministic concurrent spatial chunks.

    This uses xatlas's default chart and pack options to preserve the existing
    TRELLIS.2/SAM3D conversion contract.  ``source_face_ids`` records how each
    output face maps back to the input because spatial packing intentionally
    groups faces by partition instead of restoring the original face order.
    """

    source_vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(faces, dtype=np.int64)
    _validate_mesh(source_vertices, source_faces)
    if chunks <= 0:
        raise ValueError(f"chunks must be positive, got {chunks}")
    if not math.isfinite(tile_padding) or tile_padding < 0.0 or tile_padding >= 0.5:
        raise ValueError("tile_padding must be finite and in [0, 0.5)")
    resolved_chunks = min(int(chunks), int(source_faces.shape[0]))
    run = _make_observer(observe)

    if resolved_chunks == 1:
        return run(
            "uv.xatlas-global.generate",
            lambda: _unwrap_xatlas_default(
                source_vertices,
                source_faces,
                backend="xatlas-global",
                chunks=1,
                chunk_faces=(int(source_faces.shape[0]),),
            ),
        )

    face_partitions = run(
        "uv.xatlas-parallel-spatial.partition",
        lambda: _partition_faces_spatially(source_vertices, source_faces, resolved_chunks),
    )
    partition_cut_stats = run(
        "uv.xatlas-parallel-spatial.measure_partition_cuts",
        lambda: _measure_spatial_partition_cuts(source_faces, face_partitions),
    )
    chunk_inputs = run(
        "uv.xatlas-parallel-spatial.build_submeshes",
        lambda: [
            (*_submesh_for_faces(source_vertices, source_faces[face_ids]), face_ids)
            for face_ids in face_partitions
        ],
    )

    def unwrap_chunks() -> list[XAtlasUvResult]:
        def unwrap_chunk(item: tuple[np.ndarray, np.ndarray, np.ndarray]) -> XAtlasUvResult:
            local_vertices, local_faces, _source_face_ids = item
            return _unwrap_xatlas_default(
                local_vertices,
                local_faces,
                backend="xatlas-parallel-chunk",
                chunks=1,
                chunk_faces=(int(local_faces.shape[0]),),
            )

        max_workers = min(len(chunk_inputs), max(1, os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="xatlas",
        ) as executor:
            return list(executor.map(unwrap_chunk, chunk_inputs))

    chunk_results = run("uv.xatlas-parallel-spatial.generate", unwrap_chunks)

    def pack_chunks() -> XAtlasUvResult:
        packed_vertices: list[np.ndarray] = []
        packed_faces: list[np.ndarray] = []
        packed_uvs: list[np.ndarray] = []
        packed_chart_ids: list[np.ndarray] = []
        packed_source_face_ids: list[np.ndarray] = []
        vertex_offset = 0
        chart_offset = 0
        cols = int(math.ceil(math.sqrt(len(chunk_results))))
        rows = int(math.ceil(len(chunk_results) / cols))
        for chunk_index, (chunk_result, chunk_input) in enumerate(
            zip(chunk_results, chunk_inputs, strict=True)
        ):
            _local_vertices, _local_faces, source_face_ids = chunk_input
            row = chunk_index // cols
            col = chunk_index % cols
            normalized_uvs = _normalize_chunk_uvs(chunk_result.uvs)
            normalized_uvs = tile_padding + normalized_uvs * (1.0 - 2.0 * tile_padding)
            normalized_uvs[:, 0] = (col + normalized_uvs[:, 0]) / cols
            normalized_uvs[:, 1] = (row + normalized_uvs[:, 1]) / rows
            packed_vertices.append(chunk_result.vertices.astype(np.float32, copy=False))
            packed_faces.append(chunk_result.faces.astype(np.int64, copy=False) + vertex_offset)
            packed_uvs.append(normalized_uvs.astype(np.float32, copy=False))
            packed_chart_ids.append(chunk_result.chart_ids + chart_offset)
            packed_source_face_ids.append(source_face_ids)
            vertex_offset += int(chunk_result.vertices.shape[0])
            chart_offset += int(chunk_result.stats["chart_count"])

        out_vertices = np.ascontiguousarray(np.concatenate(packed_vertices), dtype=np.float32)
        out_faces = np.ascontiguousarray(np.concatenate(packed_faces), dtype=np.int64)
        out_uvs = np.ascontiguousarray(np.concatenate(packed_uvs), dtype=np.float32)
        chart_ids = np.ascontiguousarray(np.concatenate(packed_chart_ids), dtype=np.int64)
        source_face_ids = np.ascontiguousarray(
            np.concatenate(packed_source_face_ids),
            dtype=np.int64,
        )
        chunk_utilizations = [float(result.stats["atlas_utilization"]) for result in chunk_results]
        stats = {
            "backend": "xatlas-parallel-spatial",
            "xatlas_version": str(chunk_results[0].stats["xatlas_version"]),
            "source_vertices": int(source_vertices.shape[0]),
            "source_faces": int(source_faces.shape[0]),
            "output_vertices": int(out_vertices.shape[0]),
            "output_faces": int(out_faces.shape[0]),
            "duplicated_vertex_ratio": float(
                out_vertices.shape[0] / max(source_vertices.shape[0], 1)
            ),
            "input_cluster_count": len(chunk_results),
            "chunks": len(chunk_results),
            "chunk_faces": tuple(int(item[1].shape[0]) for item in chunk_inputs),
            "chart_count": chart_offset,
            "atlas_count": len(chunk_results),
            "atlas_width": None,
            "atlas_height": None,
            # Retain the historical TRELLIS.2 statistic.  The actual packed
            # output utilization is measured later from the combined UVs.
            "atlas_utilization": float(np.mean(chunk_utilizations)),
            "spatial_tile_padding": float(tile_padding),
            "unassigned_face_count": int(np.count_nonzero(chart_ids < 0)),
            **partition_cut_stats,
        }
        return XAtlasUvResult(
            vertices=out_vertices,
            faces=out_faces,
            uvs=np.clip(out_uvs, 0.0, 1.0),
            chart_ids=chart_ids,
            source_face_ids=source_face_ids,
            stats=stats,
        )

    return run("uv.xatlas-parallel-spatial.pack", pack_chunks)


def _unwrap_xatlas_default(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    backend: str,
    chunks: int,
    chunk_faces: tuple[int, ...],
) -> XAtlasUvResult:
    import xatlas  # noqa: PLC0415 (optional work is intentionally lazy)

    atlas = xatlas.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
    )
    atlas.generate(xatlas.ChartOptions(), xatlas.PackOptions(), False)
    vmapping, indices, uvs = atlas.get_mesh(0)
    out_vertices = np.ascontiguousarray(
        vertices[np.asarray(vmapping, dtype=np.int64)],
        dtype=np.float32,
    )
    out_faces = np.ascontiguousarray(
        np.asarray(indices, dtype=np.int64).reshape(-1, 3),
        dtype=np.int64,
    )
    out_uvs = np.ascontiguousarray(np.asarray(uvs, dtype=np.float32), dtype=np.float32)
    chart_count = int(atlas.chart_count)
    chart_ids = np.full(faces.shape[0], -1, dtype=np.int64)
    for chart_index in range(int(atlas.get_mesh_chart_count(0))):
        chart = atlas.get_mesh_chart(0, chart_index)
        chart_ids[np.asarray(chart.faces, dtype=np.int64)] = chart_index
    stats = {
        "backend": backend,
        "xatlas_version": str(xatlas.__version__),
        "source_vertices": int(vertices.shape[0]),
        "source_faces": int(faces.shape[0]),
        "output_vertices": int(out_vertices.shape[0]),
        "output_faces": int(out_faces.shape[0]),
        "duplicated_vertex_ratio": float(out_vertices.shape[0] / max(vertices.shape[0], 1)),
        "input_cluster_count": 1,
        "chunks": int(chunks),
        "chunk_faces": chunk_faces,
        "chart_count": chart_count,
        "atlas_count": int(atlas.atlas_count),
        "atlas_width": int(atlas.width),
        "atlas_height": int(atlas.height),
        "atlas_utilization": float(atlas.utilization),
        "unassigned_face_count": int(np.count_nonzero(chart_ids < 0)),
        "spatial_partition_shared_edge_count": 0,
        "spatial_partition_cut_edge_count": 0,
        "spatial_partition_cut_edge_ratio": 0.0,
    }
    return XAtlasUvResult(
        vertices=out_vertices,
        faces=out_faces,
        uvs=out_uvs,
        chart_ids=chart_ids,
        source_face_ids=np.arange(faces.shape[0], dtype=np.int64),
        stats=stats,
    )


def _partition_faces_spatially(
    vertices: np.ndarray,
    faces: np.ndarray,
    chunks: int,
) -> list[np.ndarray]:
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


def _submesh_for_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return (
        vertices[unique_vertices].astype(np.float32, copy=False),
        inverse.reshape(-1, 3).astype(np.int64, copy=False),
    )


def _measure_spatial_partition_cuts(
    faces: np.ndarray,
    face_partitions: list[np.ndarray],
) -> dict[str, int | float]:
    partition_ids = np.empty(faces.shape[0], dtype=np.int32)
    for partition_index, face_ids in enumerate(face_partitions):
        partition_ids[face_ids] = partition_index

    edges = np.concatenate(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edge_face_ids = np.tile(np.arange(faces.shape[0], dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    sorted_face_ids = edge_face_ids[order]
    group_start = np.empty(sorted_edges.shape[0], dtype=bool)
    group_start[0] = True
    group_start[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(group_start)
    ends = np.append(starts[1:], sorted_edges.shape[0])
    pair_mask = ends - starts == 2
    pair_starts = starts[pair_mask]
    first_faces = sorted_face_ids[pair_starts]
    second_faces = sorted_face_ids[pair_starts + 1]
    shared_edge_count = int(pair_starts.shape[0])
    cut_edge_count = int(
        np.count_nonzero(partition_ids[first_faces] != partition_ids[second_faces])
    )
    return {
        "spatial_partition_shared_edge_count": shared_edge_count,
        "spatial_partition_cut_edge_count": cut_edge_count,
        "spatial_partition_cut_edge_ratio": (
            float(cut_edge_count / shared_edge_count) if shared_edge_count else 0.0
        ),
    }


def _normalize_chunk_uvs(uvs: np.ndarray) -> np.ndarray:
    normalized = np.asarray(uvs, dtype=np.float32).copy()
    uv_min = normalized.min(axis=0)
    uv_max = normalized.max(axis=0)
    span = np.maximum(uv_max - uv_min, 1e-6)
    normalized = (normalized - uv_min) / span
    return np.clip(normalized, 0.0, 1.0)


def _make_observer(observer: _Observer | None) -> Callable[[str, Callable[[], _T]], _T]:
    if observer is None:
        return lambda _name, fn: fn()
    return lambda name, fn: observer(name, fn)


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"vertices must have shape (N, 3) with N > 0, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"faces must have shape (F, 3) with F > 0, got {faces.shape}")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite values")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("faces contain indices outside the vertex array")
