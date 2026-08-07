"""Experimental MLX/Metal batched QEM simplification.

The execution model follows CuMesh's parallel simplifier: compute all edge
costs, propagate a deterministic local minimum to incident faces, collapse
only conflict-free local minima, then rebuild topology. CUDA code is not used;
the dense stages are native MLX custom Metal kernels on Apple GPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlx.core as mx
import numpy as np

from ._native import apply_mlx_qem_collapse_batch as _apply_mlx_qem_collapse_batch
from ._native import build_mlx_qem_topology as _build_mlx_qem_topology


_THREADGROUP_SIZE = 256
_MAX_INT32 = np.iinfo(np.int32).max
_TOPOLOGY_POLICIES = {"manifold-preserving", "cumesh-reference"}


@dataclass(frozen=True)
class _Topology:
    faces: np.ndarray
    edges: np.ndarray
    edge_face_counts: np.ndarray
    boundary_vertices: np.ndarray
    vertex_face_offsets: np.ndarray
    vertex_faces: np.ndarray
    vertex_edge_offsets: np.ndarray
    vertex_edges: np.ndarray
    stats: dict[str, Any]


def simplify_mesh_mlx_parallel_qem(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_faces: int,
    threshold: float = 1.0e-8,
    lambda_edge_length: float = 1.0e-2,
    lambda_skinny: float = 1.0e-3,
    max_rounds: int = 32,
    topology_policy: str = "manifold-preserving",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Simplify a manifold mesh with conflict-free batched collapses on MLX GPU.

    This is an experimental reference-mechanism candidate. It deliberately
    stays separate from the production/default export path until real cached
    meshes prove its topology and visual quality.
    """

    work_vertices, work_faces = _validate_mesh(vertices, faces)
    requested_target = int(target_faces)
    if requested_target <= 0:
        raise ValueError("target_faces must be positive")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if not np.isfinite(lambda_edge_length) or lambda_edge_length < 0:
        raise ValueError("lambda_edge_length must be non-negative and finite")
    if not np.isfinite(lambda_skinny) or lambda_skinny < 0:
        raise ValueError("lambda_skinny must be non-negative and finite")
    if max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    normalized_topology_policy = str(topology_policy).strip().lower()
    if normalized_topology_policy not in _TOPOLOGY_POLICIES:
        raise ValueError(
            "topology_policy must be 'manifold-preserving' or "
            "'cumesh-reference'"
        )

    source_vertices = int(work_vertices.shape[0])
    source_faces = int(work_faces.shape[0])
    if source_faces <= requested_target:
        return work_vertices, work_faces.astype(np.int64), _stats(
            source_vertices=source_vertices,
            source_faces=source_faces,
            final_vertices=source_vertices,
            final_faces=source_faces,
            target_faces=requested_target,
            rounds=0,
            collapses=0,
            faces_removed=0,
            topology_seconds=0.0,
            gpu_seconds=0.0,
            apply_seconds=0.0,
            threshold=float(threshold),
            target_reached=True,
            topology_policy=normalized_topology_policy,
            round_limit=int(max_rounds),
        )

    stream = mx.new_stream(mx.gpu)
    vertex_qem_kernel, topology_guard_kernel, edge_cost_kernel, face_min_kernel, edge_select_kernel = _kernels()
    topology_seconds = 0.0
    gpu_seconds = 0.0
    apply_seconds = 0.0
    collapse_count = 0
    faces_removed = 0
    current_threshold = float(threshold)
    rounds = 0
    topology_backend = "not-run"
    topology_cpu_workers = 0
    apply_backend = "not-run"
    apply_cpu_workers = 0
    mlx_round_cache_peak = 0
    mlx_round_active_peak = 0
    mlx_round_cache_clears = 0

    for round_index in range(int(max_rounds)):
        if work_faces.shape[0] <= requested_target:
            break
        rounds = round_index + 1
        topology_started = time.perf_counter()
        topology = _build_topology(work_vertices.shape[0], work_faces)
        topology_seconds += time.perf_counter() - topology_started
        topology_backend = str(topology.stats["backend"])
        topology_cpu_workers = max(topology_cpu_workers, int(topology.stats["cpu_workers"]))

        gpu_started = time.perf_counter()
        with mx.stream(stream):
            vertices_mx = mx.array(work_vertices, dtype=mx.float32)
            faces_mx = mx.array(topology.faces, dtype=mx.int32)
            edges_mx = mx.array(topology.edges, dtype=mx.int32)
            edge_face_counts_mx = mx.array(topology.edge_face_counts, dtype=mx.int32)
            boundary_mx = mx.array(topology.boundary_vertices, dtype=mx.uint8)
            vertex_face_offsets_mx = mx.array(topology.vertex_face_offsets, dtype=mx.int32)
            vertex_faces_mx = mx.array(topology.vertex_faces, dtype=mx.int32)
            vertex_edge_offsets_mx = mx.array(topology.vertex_edge_offsets, dtype=mx.int32)
            vertex_edges_mx = mx.array(topology.vertex_edges, dtype=mx.int32)

            vertex_qems = vertex_qem_kernel(
                inputs=[
                    vertices_mx,
                    faces_mx,
                    vertex_face_offsets_mx,
                    vertex_faces_mx,
                    int(work_vertices.shape[0]),
                ],
                grid=(int(work_vertices.shape[0]), 1, 1),
                threadgroup=(_THREADGROUP_SIZE, 1, 1),
                output_shapes=[(int(work_vertices.shape[0]), 10)],
                output_dtypes=[mx.float32],
            )[0]
            topology_valid = topology_guard_kernel(
                inputs=[
                    faces_mx,
                    edges_mx,
                    edge_face_counts_mx,
                    boundary_mx,
                    vertex_face_offsets_mx,
                    vertex_faces_mx,
                    vertex_edge_offsets_mx,
                    vertex_edges_mx,
                    int(topology.edges.shape[0]),
                    int(normalized_topology_policy == "manifold-preserving"),
                ],
                grid=(int(topology.edges.shape[0]), 1, 1),
                threadgroup=(_THREADGROUP_SIZE, 1, 1),
                output_shapes=[(int(topology.edges.shape[0]),)],
                output_dtypes=[mx.uint8],
            )[0]
            edge_costs = edge_cost_kernel(
                inputs=[
                    vertices_mx,
                    faces_mx,
                    edges_mx,
                    boundary_mx,
                    topology_valid,
                    vertex_qems,
                    vertex_face_offsets_mx,
                    vertex_faces_mx,
                    int(topology.edges.shape[0]),
                    float(lambda_edge_length),
                    float(lambda_skinny),
                ],
                grid=(int(topology.edges.shape[0]), 1, 1),
                threadgroup=(_THREADGROUP_SIZE, 1, 1),
                output_shapes=[(int(topology.edges.shape[0]),)],
                output_dtypes=[mx.float32],
            )[0]
            face_best_edges = face_min_kernel(
                inputs=[
                    faces_mx,
                    vertex_edge_offsets_mx,
                    vertex_edges_mx,
                    edge_costs,
                    int(topology.faces.shape[0]),
                ],
                grid=(int(topology.faces.shape[0]), 1, 1),
                threadgroup=(_THREADGROUP_SIZE, 1, 1),
                output_shapes=[(int(topology.faces.shape[0]),)],
                output_dtypes=[mx.int32],
            )[0]
            selected = edge_select_kernel(
                inputs=[
                    topology.edges.shape[0],
                    edges_mx,
                    edge_costs,
                    vertex_face_offsets_mx,
                    vertex_faces_mx,
                    face_best_edges,
                    current_threshold,
                ],
                grid=(int(topology.edges.shape[0]), 1, 1),
                threadgroup=(_THREADGROUP_SIZE, 1, 1),
                output_shapes=[(int(topology.edges.shape[0]),)],
                output_dtypes=[mx.uint8],
            )[0]

            # Topology changes require a host decision once per round. All MLX
            # work above remains lazy and is synchronized together here.
            mx.eval(selected, edge_costs)
        selected_mask = np.array(selected, dtype=np.uint8, copy=True).astype(bool, copy=False)
        costs_host = np.array(edge_costs, dtype=np.float32, copy=True)
        gpu_seconds += time.perf_counter() - gpu_started

        # Mesh sizes change every round, so cached buffers from the previous
        # shape are not reusable enough to justify retaining several GiB. Copy
        # the two host decisions, release the lazy graph, and clear the MLX
        # allocator cache before rebuilding topology for the next round.
        mlx_round_active_peak = max(mlx_round_active_peak, int(mx.get_active_memory()))
        del (
            vertices_mx,
            faces_mx,
            edges_mx,
            edge_face_counts_mx,
            boundary_mx,
            vertex_face_offsets_mx,
            vertex_faces_mx,
            vertex_edge_offsets_mx,
            vertex_edges_mx,
            vertex_qems,
            topology_valid,
            edge_costs,
            face_best_edges,
            selected,
        )
        mlx_round_cache_peak = max(mlx_round_cache_peak, int(mx.get_cache_memory()))
        mx.clear_cache()
        mlx_round_cache_clears += 1

        selected_ids = np.flatnonzero(selected_mask)
        if selected_ids.size == 0:
            current_threshold *= 10.0
            continue

        selected_ids = _cap_collapse_batch(
            selected_ids,
            costs_host,
            topology.edge_face_counts,
            face_budget=int(work_faces.shape[0]) - requested_target,
        )
        if selected_ids.size == 0:
            current_threshold *= 10.0
            continue

        apply_started = time.perf_counter()
        previous_face_count = int(work_faces.shape[0])
        work_vertices, work_faces, apply_stats = _apply_collapse_batch(
            work_vertices,
            topology.faces,
            topology.edges[selected_ids],
            topology.boundary_vertices,
        )
        apply_backend = str(apply_stats["backend"])
        apply_cpu_workers = max(apply_cpu_workers, int(apply_stats["cpu_workers"]))
        removed = previous_face_count - int(work_faces.shape[0])
        apply_seconds += time.perf_counter() - apply_started
        collapse_count += int(selected_ids.size)
        faces_removed += removed
        if removed <= 0:
            current_threshold *= 10.0
            continue
        if removed / max(1, previous_face_count) < 1.0e-2:
            current_threshold *= 10.0

    final_faces = int(work_faces.shape[0])
    stats = _stats(
        source_vertices=source_vertices,
        source_faces=source_faces,
        final_vertices=int(work_vertices.shape[0]),
        final_faces=final_faces,
        target_faces=requested_target,
        rounds=rounds,
        collapses=collapse_count,
        faces_removed=faces_removed,
        topology_seconds=topology_seconds,
        gpu_seconds=gpu_seconds,
        apply_seconds=apply_seconds,
        threshold=current_threshold,
        target_reached=final_faces <= requested_target,
        topology_backend=topology_backend,
        topology_cpu_workers=topology_cpu_workers,
        apply_backend=apply_backend,
        apply_cpu_workers=apply_cpu_workers,
        topology_policy=normalized_topology_policy,
        round_limit=int(max_rounds),
    )
    stats["mlx_round_active_bytes_peak"] = mlx_round_active_peak
    stats["mlx_round_cache_bytes_peak"] = mlx_round_cache_peak
    stats["mlx_round_cache_clears"] = mlx_round_cache_clears
    stats["mlx_active_bytes_before_cache_clear"] = int(mx.get_active_memory())
    stats["mlx_cache_bytes_before_clear"] = int(mx.get_cache_memory())
    mx.clear_cache()
    stats["mlx_active_bytes_after_cache_clear"] = int(mx.get_active_memory())
    stats["mlx_cache_bytes_after_clear"] = int(mx.get_cache_memory())
    stats["mlx_cache_cleared_after_qem"] = True
    return work_vertices, work_faces.astype(np.int64, copy=False), stats


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    work_vertices = np.ascontiguousarray(np.asarray(vertices), dtype=np.float32)
    work_faces_64 = np.ascontiguousarray(np.asarray(faces), dtype=np.int64)
    if work_vertices.ndim != 2 or work_vertices.shape[1] != 3 or work_vertices.shape[0] == 0:
        raise ValueError("vertices must have shape (V, 3) with V > 0")
    if work_faces_64.ndim != 2 or work_faces_64.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.isfinite(work_vertices).all():
        raise ValueError("vertices must contain only finite values")
    if work_vertices.shape[0] > _MAX_INT32:
        raise ValueError("MLX parallel QEM requires fewer than 2^31 vertices")
    if work_faces_64.size > 0:
        if int(work_faces_64.min()) < 0 or int(work_faces_64.max()) >= work_vertices.shape[0]:
            raise ValueError("faces contain indices outside the vertex array")
    return work_vertices, work_faces_64.astype(np.int32)


def _build_topology(vertex_count: int, faces: np.ndarray) -> _Topology:
    native = _build_mlx_qem_topology(
        int(vertex_count),
        np.ascontiguousarray(faces, dtype=np.int32),
    )
    return _Topology(
        faces=np.asarray(native["faces"]),
        edges=np.asarray(native["edges"]),
        edge_face_counts=np.asarray(native["edge_face_counts"]),
        boundary_vertices=np.asarray(native["boundary_vertices"]),
        vertex_face_offsets=np.asarray(native["vertex_face_offsets"]),
        vertex_faces=np.asarray(native["vertex_faces"]),
        vertex_edge_offsets=np.asarray(native["vertex_edge_offsets"]),
        vertex_edges=np.asarray(native["vertex_edges"]),
        stats=dict(native["stats"]),
    )


def _build_topology_numpy(vertex_count: int, faces: np.ndarray) -> _Topology:
    """Retain the original implementation as a parity oracle for native CSR."""

    face_count = int(faces.shape[0])
    if face_count * 3 > _MAX_INT32:
        raise ValueError("MLX parallel QEM adjacency exceeds int32 indexing capacity")

    left = faces.reshape(-1)
    right = faces[:, [1, 2, 0]].reshape(-1)
    low = np.minimum(left, right).astype(np.uint64)
    high = np.maximum(left, right).astype(np.uint64)
    packed = (low << np.uint64(32)) | high
    packed.sort(kind="stable")
    unique_start = np.empty(packed.shape[0], dtype=bool)
    if packed.size > 0:
        unique_start[0] = True
        unique_start[1:] = packed[1:] != packed[:-1]
        starts = np.flatnonzero(unique_start)
        unique_keys = packed[starts]
        run_ends = np.concatenate((starts[1:], np.array([packed.size], dtype=np.int64)))
        edge_face_counts = (run_ends - starts).astype(np.int32)
    else:
        unique_keys = np.empty((0,), dtype=np.uint64)
        edge_face_counts = np.empty((0,), dtype=np.int32)
    edges = np.empty((unique_keys.shape[0], 2), dtype=np.int32)
    edges[:, 0] = (unique_keys >> np.uint64(32)).astype(np.int32)
    edges[:, 1] = (unique_keys & np.uint64(0xFFFFFFFF)).astype(np.int32)

    boundary_vertices = np.zeros(vertex_count, dtype=np.uint8)
    boundary_edges = edges[edge_face_counts != 2]
    if boundary_edges.size > 0:
        boundary_vertices[boundary_edges.reshape(-1)] = 1

    vertex_face_offsets, vertex_faces = _build_csr(
        vertex_count,
        faces.reshape(-1),
        np.repeat(np.arange(face_count, dtype=np.int32), 3),
    )
    vertex_edge_offsets, vertex_edges = _build_csr(
        vertex_count,
        edges.reshape(-1),
        np.repeat(np.arange(edges.shape[0], dtype=np.int32), 2),
    )
    return _Topology(
        faces=np.ascontiguousarray(faces, dtype=np.int32),
        edges=np.ascontiguousarray(edges, dtype=np.int32),
        edge_face_counts=edge_face_counts,
        boundary_vertices=boundary_vertices,
        vertex_face_offsets=vertex_face_offsets,
        vertex_faces=vertex_faces,
        vertex_edge_offsets=vertex_edge_offsets,
        vertex_edges=vertex_edges,
        stats={
            "backend": "numpy-stable-sort-csr",
            "framework": "numpy",
            "execution_device": "cpu",
            "cpu_workers": 1,
        },
    )


def _build_csr(vertex_count: int, vertex_ids: np.ndarray, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(vertex_ids, kind="stable")
    sorted_vertices = vertex_ids[order]
    sorted_items = np.ascontiguousarray(item_ids[order], dtype=np.int32)
    counts = np.bincount(sorted_vertices, minlength=vertex_count)
    offsets_64 = np.empty(vertex_count + 1, dtype=np.int64)
    offsets_64[0] = 0
    np.cumsum(counts, out=offsets_64[1:])
    if offsets_64[-1] > _MAX_INT32:
        raise ValueError("MLX parallel QEM CSR exceeds int32 indexing capacity")
    return offsets_64.astype(np.int32), sorted_items


def _cap_collapse_batch(
    selected_ids: np.ndarray,
    costs: np.ndarray,
    edge_face_counts: np.ndarray,
    *,
    face_budget: int,
) -> np.ndarray:
    if selected_ids.size == 0 or face_budget <= 0:
        return np.empty((0,), dtype=np.int64)
    order = np.lexsort((selected_ids, costs[selected_ids]))
    ordered = selected_ids[order]
    removals = edge_face_counts[ordered].astype(np.int64)
    cumulative = np.cumsum(removals)
    within = int(np.searchsorted(cumulative, face_budget, side="right"))
    if within == 0:
        return ordered[:1]
    return ordered[:within]


def _apply_collapse_batch(
    vertices: np.ndarray,
    faces: np.ndarray,
    selected_edges: np.ndarray,
    boundary_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    result = _apply_mlx_qem_collapse_batch(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int32),
        np.ascontiguousarray(selected_edges, dtype=np.int32),
        np.ascontiguousarray(boundary_vertices, dtype=np.uint8),
    )
    return (
        np.asarray(result["vertices"]),
        np.asarray(result["faces"]),
        dict(result["stats"]),
    )


def _apply_collapse_batch_numpy(
    vertices: np.ndarray,
    faces: np.ndarray,
    selected_edges: np.ndarray,
    boundary_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain the NumPy implementation as a parity oracle."""

    keep = selected_edges[:, 0]
    drop = selected_edges[:, 1]
    keep_boundary = boundary_vertices[keep] != 0
    drop_boundary = boundary_vertices[drop] != 0
    midpoint = 0.5 * (vertices[keep] + vertices[drop])
    targets = midpoint
    targets = np.where((keep_boundary & ~drop_boundary)[:, None], vertices[keep], targets)
    targets = np.where((~keep_boundary & drop_boundary)[:, None], vertices[drop], targets)

    moved_vertices = vertices.copy()
    moved_vertices[keep] = targets
    vertex_map = np.arange(vertices.shape[0], dtype=np.int32)
    vertex_map[drop] = keep
    remapped_faces = vertex_map[faces]
    keep_faces = (
        (remapped_faces[:, 0] != remapped_faces[:, 1])
        & (remapped_faces[:, 1] != remapped_faces[:, 2])
        & (remapped_faces[:, 0] != remapped_faces[:, 2])
    )
    remapped_faces = remapped_faces[keep_faces]

    used = np.zeros(vertices.shape[0], dtype=bool)
    used[remapped_faces.reshape(-1)] = True
    compact_map = np.cumsum(used, dtype=np.int64) - 1
    compact_vertices = np.ascontiguousarray(moved_vertices[used], dtype=np.float32)
    compact_faces = np.ascontiguousarray(compact_map[remapped_faces], dtype=np.int32)
    return compact_vertices, compact_faces


def _stats(
    *,
    source_vertices: int,
    source_faces: int,
    final_vertices: int,
    final_faces: int,
    target_faces: int,
    rounds: int,
    collapses: int,
    faces_removed: int,
    topology_seconds: float,
    gpu_seconds: float,
    apply_seconds: float,
    threshold: float,
    target_reached: bool,
    topology_backend: str = "not-run",
    topology_cpu_workers: int = 0,
    apply_backend: str = "not-run",
    apply_cpu_workers: int = 0,
    topology_policy: str = "manifold-preserving",
    round_limit: int = 0,
) -> dict[str, Any]:
    device = mx.device_info()
    preserve_manifold = topology_policy == "manifold-preserving"
    return {
        "backend": "mlx-parallel-qem",
        "requested_backend": "mlx-parallel-qem",
        "algorithm": "cumesh-local-minimum-batched-edge-collapse",
        "framework": "mlx",
        "execution_device": "apple-gpu",
        "gpu_submission_confirmed": rounds > 0,
        "metal_device": str(device.get("device_name", "unknown")),
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "target_faces": target_faces,
        "final_vertices": final_vertices,
        "final_faces": final_faces,
        "target_reached": bool(target_reached),
        "simplified": final_faces < source_faces,
        "qem_simplification_backend": "mlx-metal-local-minimum-batched-edge-collapse",
        "qem_equivalence_status": (
            "manifold-preserving-cumesh-derived-quality-unproven"
            if preserve_manifold
            else "cumesh-reference-mechanism"
        ),
        "qem_topology_policy": topology_policy,
        "qem_topology_guards": (
            [
                "interior-boundary-lock",
                "two-manifold-edge",
                "link-condition",
                "canonical-face-duplicate",
                "normal-alignment",
                "minimum-area-aspect",
            ]
            if preserve_manifold
            else []
        ),
        "qem_rounds": rounds,
        "qem_round_limit": round_limit,
        "qem_round_limit_exhausted": bool(round_limit > 0 and rounds >= round_limit and not target_reached),
        "qem_collapses_applied": collapses,
        "qem_faces_removed": faces_removed,
        "qem_final_threshold": threshold,
        "topology_cpu_seconds": topology_seconds,
        "topology_backend": topology_backend,
        "topology_cpu_workers": topology_cpu_workers,
        "mlx_gpu_seconds": gpu_seconds,
        "topology_apply_cpu_seconds": apply_seconds,
        "topology_apply_backend": apply_backend,
        "topology_apply_cpu_workers": apply_cpu_workers,
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        "quality_tier": "experimental_quality_unproven",
        "production_ready": False,
        "production_blockers": ["mlx_parallel_qem_quality_unproven"],
    }


@lru_cache(maxsize=1)
def _kernels() -> tuple[Any, Any, Any, Any, Any]:
    vertex_qem_kernel = mx.fast.metal_kernel(
        name="mlx_spatial_vertex_qem",
        input_names=["vertices", "faces", "vertex_face_offsets", "vertex_faces", "vertex_count"],
        output_names=["qems"],
        source=_VERTEX_QEM_SOURCE,
        ensure_row_contiguous=True,
    )
    topology_guard_kernel = mx.fast.metal_kernel(
        name="mlx_spatial_edge_topology_guard",
        input_names=[
            "faces",
            "edges",
            "edge_face_counts",
            "boundary_vertices",
            "vertex_face_offsets",
            "vertex_faces",
            "vertex_edge_offsets",
            "vertex_edges",
            "edge_count",
            "preserve_manifold",
        ],
        output_names=["topology_valid"],
        header=_EDGE_TOPOLOGY_HEADER,
        source=_EDGE_TOPOLOGY_SOURCE,
        ensure_row_contiguous=True,
    )
    edge_cost_kernel = mx.fast.metal_kernel(
        name="mlx_spatial_edge_qem_cost",
        input_names=[
            "vertices",
            "faces",
            "edges",
            "boundary_vertices",
            "topology_valid",
            "qems",
            "vertex_face_offsets",
            "vertex_faces",
            "edge_count",
            "lambda_edge_length",
            "lambda_skinny",
        ],
        output_names=["costs"],
        header=_EDGE_COST_HEADER,
        source=_EDGE_COST_SOURCE,
        ensure_row_contiguous=True,
    )
    face_min_kernel = mx.fast.metal_kernel(
        name="mlx_spatial_face_local_min_edge",
        input_names=["faces", "vertex_edge_offsets", "vertex_edges", "costs", "face_count"],
        output_names=["face_best_edges"],
        source=_FACE_MIN_SOURCE,
        ensure_row_contiguous=True,
    )
    edge_select_kernel = mx.fast.metal_kernel(
        name="mlx_spatial_select_conflict_free_edges",
        input_names=[
            "edge_count",
            "edges",
            "costs",
            "vertex_face_offsets",
            "vertex_faces",
            "face_best_edges",
            "threshold",
        ],
        output_names=["selected"],
        source=_EDGE_SELECT_SOURCE,
        ensure_row_contiguous=True,
    )
    return vertex_qem_kernel, topology_guard_kernel, edge_cost_kernel, face_min_kernel, edge_select_kernel


_VERTEX_QEM_SOURCE = r"""
uint vertex_id = thread_position_in_grid.x;
if (vertex_id >= uint(vertex_count)) return;
float q[10] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
for (int item = vertex_face_offsets[vertex_id]; item < vertex_face_offsets[vertex_id + 1]; ++item) {
    int face_id = vertex_faces[item];
    int i0 = faces[face_id * 3];
    int i1 = faces[face_id * 3 + 1];
    int i2 = faces[face_id * 3 + 2];
    float3 a = float3(vertices[i0 * 3], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    float3 b = float3(vertices[i1 * 3], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    float3 c = float3(vertices[i2 * 3], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);
    float3 normal = metal::cross(b - a, c - a);
    float length = metal::length(normal);
    if (!metal::isfinite(length) || length <= 1.0e-18f) continue;
    normal /= length;
    float d = -metal::dot(normal, a);
    float p[4] = {normal.x, normal.y, normal.z, d};
    q[0] += p[0] * p[0]; q[1] += p[0] * p[1]; q[2] += p[0] * p[2]; q[3] += p[0] * p[3];
    q[4] += p[1] * p[1]; q[5] += p[1] * p[2]; q[6] += p[1] * p[3];
    q[7] += p[2] * p[2]; q[8] += p[2] * p[3]; q[9] += p[3] * p[3];
}
for (uint index = 0; index < 10; ++index) qems[vertex_id * 10 + index] = q[index];
"""


_EDGE_TOPOLOGY_HEADER = r"""
inline int mlx_spatial_other_endpoint(device const int* edges, int edge_id, int vertex_id) {
    int a = edges[edge_id * 2];
    int b = edges[edge_id * 2 + 1];
    return a == vertex_id ? b : a;
}

inline bool mlx_spatial_face_contains(device const int* faces, int face_id, int vertex_id) {
    return faces[face_id * 3] == vertex_id ||
        faces[face_id * 3 + 1] == vertex_id ||
        faces[face_id * 3 + 2] == vertex_id;
}

inline int2 mlx_spatial_other_face_vertices(
    device const int* faces,
    int face_id,
    int excluded_vertex) {
    int2 other = int2(-1, -1);
    int write = 0;
    for (uint corner = 0; corner < 3; ++corner) {
        int vertex_id = faces[face_id * 3 + corner];
        if (vertex_id == excluded_vertex) continue;
        if (write == 0) other.x = vertex_id;
        else other.y = vertex_id;
        write += 1;
    }
    return other;
}
"""


_EDGE_TOPOLOGY_SOURCE = r"""
uint edge_id = thread_position_in_grid.x;
if (edge_id >= uint(edge_count)) return;
if (preserve_manifold == 0) {
    topology_valid[edge_id] = 1;
    return;
}
int e0 = edges[edge_id * 2];
int e1 = edges[edge_id * 2 + 1];

// Match the topology-preserving native QEM contract: only collapse an
// interior two-manifold edge whose endpoints are not on a boundary.
if (edge_face_counts[edge_id] != 2 ||
    boundary_vertices[e0] != 0 || boundary_vertices[e1] != 0) {
    topology_valid[edge_id] = 0;
    return;
}

int opposite0 = -1;
int opposite1 = -1;
int incident_count = 0;
for (int item = vertex_face_offsets[e0]; item < vertex_face_offsets[e0 + 1]; ++item) {
    int face_id = vertex_faces[item];
    if (!mlx_spatial_face_contains(faces, face_id, e1)) continue;
    int2 other = mlx_spatial_other_face_vertices(faces, face_id, e0);
    int opposite = other.x == e1 ? other.y : other.x;
    if (incident_count == 0) opposite0 = opposite;
    else if (incident_count == 1) opposite1 = opposite;
    incident_count += 1;
}
if (incident_count != 2 || opposite0 < 0 || opposite1 < 0 || opposite0 == opposite1) {
    topology_valid[edge_id] = 0;
    return;
}

// Link condition: the endpoints must share exactly the two opposite vertices
// of the incident triangles. Extra common neighbours create a pinch.
int common_count = 0;
bool found_opposite0 = false;
bool found_opposite1 = false;
for (int item0 = vertex_edge_offsets[e0]; item0 < vertex_edge_offsets[e0 + 1]; ++item0) {
    int neighbour = mlx_spatial_other_endpoint(edges, vertex_edges[item0], e0);
    if (neighbour == e1) continue;
    bool shared = false;
    for (int item1 = vertex_edge_offsets[e1]; item1 < vertex_edge_offsets[e1 + 1]; ++item1) {
        if (mlx_spatial_other_endpoint(edges, vertex_edges[item1], e1) == neighbour) {
            shared = true;
            break;
        }
    }
    if (!shared) continue;
    common_count += 1;
    found_opposite0 = found_opposite0 || neighbour == opposite0;
    found_opposite1 = found_opposite1 || neighbour == opposite1;
}
if (common_count != 2 || !found_opposite0 || !found_opposite1) {
    topology_valid[edge_id] = 0;
    return;
}

// Retargeting e1 -> e0 must not create duplicate canonical faces.
for (int item1 = vertex_face_offsets[e1]; item1 < vertex_face_offsets[e1 + 1]; ++item1) {
    int face1 = vertex_faces[item1];
    if (mlx_spatial_face_contains(faces, face1, e0)) continue;
    int2 pair1 = mlx_spatial_other_face_vertices(faces, face1, e1);
    for (int item0 = vertex_face_offsets[e0]; item0 < vertex_face_offsets[e0 + 1]; ++item0) {
        int face0 = vertex_faces[item0];
        if (mlx_spatial_face_contains(faces, face0, e1)) continue;
        int2 pair0 = mlx_spatial_other_face_vertices(faces, face0, e0);
        if ((pair0.x == pair1.x && pair0.y == pair1.y) ||
            (pair0.x == pair1.y && pair0.y == pair1.x)) {
            topology_valid[edge_id] = 0;
            return;
        }
    }
}
topology_valid[edge_id] = 1;
"""


_EDGE_COST_HEADER = r"""
inline bool mlx_spatial_process_incident(
    int face_id,
    int keep_vertex,
    int other_vertex,
    device const float* vertices,
    device const int* faces,
    float3 new_position,
    thread float& skinny_cost,
    thread int& triangle_count) {
    int3 ids = int3(faces[face_id * 3], faces[face_id * 3 + 1], faces[face_id * 3 + 2]);
    if (ids.x == other_vertex || ids.y == other_vertex || ids.z == other_vertex) return true;
    float3 a = float3(vertices[ids.x * 3], vertices[ids.x * 3 + 1], vertices[ids.x * 3 + 2]);
    float3 b = float3(vertices[ids.y * 3], vertices[ids.y * 3 + 1], vertices[ids.y * 3 + 2]);
    float3 c = float3(vertices[ids.z * 3], vertices[ids.z * 3 + 1], vertices[ids.z * 3 + 2]);
    float3 na = ids.x == keep_vertex ? new_position : a;
    float3 nb = ids.y == keep_vertex ? new_position : b;
    float3 nc = ids.z == keep_vertex ? new_position : c;
    float3 old_normal = metal::cross(b - a, c - a);
    float3 new_normal = metal::cross(nb - na, nc - na);
    float old_area2 = metal::length(old_normal);
    float new_area2 = metal::length(new_normal);
    if (!metal::isfinite(old_area2) || !metal::isfinite(new_area2) ||
        old_area2 <= 1.0e-12f || new_area2 <= 1.0e-12f) return false;
    float alignment = metal::dot(old_normal, new_normal) / (old_area2 * new_area2);
    if (!metal::isfinite(alignment) || alignment <= 0.2f) return false;
    float new_area = 0.5f * new_area2;
    float denominator = metal::length_squared(nc - nb) + metal::length_squared(nb - na) + metal::length_squared(nc - na);
    denominator = metal::max(denominator, 1.0e-12f);
    float shape = 4.0f * metal::sqrt(3.0f) * new_area / denominator;
    if (!metal::isfinite(shape) || shape < 1.0e-4f) return false;
    skinny_cost += 1.0f - metal::clamp(shape, 0.0f, 1.0f);
    triangle_count += 1;
    return true;
}

inline float mlx_spatial_eval_qem(device const float* qems, int a, int b, float3 p) {
    float q[10];
    for (uint i = 0; i < 10; ++i) q[i] = qems[a * 10 + i] + qems[b * 10 + i];
    return q[0] * p.x * p.x + 2.0f * q[1] * p.x * p.y + 2.0f * q[2] * p.x * p.z
        + 2.0f * q[3] * p.x + q[4] * p.y * p.y + 2.0f * q[5] * p.y * p.z
        + 2.0f * q[6] * p.y + q[7] * p.z * p.z + 2.0f * q[8] * p.z + q[9];
}
"""


_EDGE_COST_SOURCE = r"""
uint edge_id = thread_position_in_grid.x;
if (edge_id >= uint(edge_count)) return;
int e0 = edges[edge_id * 2];
int e1 = edges[edge_id * 2 + 1];
if (topology_valid[edge_id] == 0) {
    costs[edge_id] = INFINITY;
    return;
}
float3 v0 = float3(vertices[e0 * 3], vertices[e0 * 3 + 1], vertices[e0 * 3 + 2]);
float3 v1 = float3(vertices[e1 * 3], vertices[e1 * 3 + 1], vertices[e1 * 3 + 2]);
bool b0 = boundary_vertices[e0] != 0;
bool b1 = boundary_vertices[e1] != 0;
float weight0 = b0 && !b1 ? 1.0f : (!b0 && b1 ? 0.0f : 0.5f);
float3 target = v0 * weight0 + v1 * (1.0f - weight0);
float length_squared = metal::length_squared(v1 - v0);
float cost = mlx_spatial_eval_qem(qems, e0, e1, target) + lambda_edge_length * length_squared;
float skinny_cost = 0.0f;
int triangle_count = 0;
for (int item = vertex_face_offsets[e0]; item < vertex_face_offsets[e0 + 1]; ++item) {
    if (!mlx_spatial_process_incident(vertex_faces[item], e0, e1, vertices, faces, target, skinny_cost, triangle_count)) {
        costs[edge_id] = INFINITY;
        return;
    }
}
for (int item = vertex_face_offsets[e1]; item < vertex_face_offsets[e1 + 1]; ++item) {
    if (!mlx_spatial_process_incident(vertex_faces[item], e1, e0, vertices, faces, target, skinny_cost, triangle_count)) {
        costs[edge_id] = INFINITY;
        return;
    }
}
if (triangle_count > 0) skinny_cost /= float(triangle_count);
cost += lambda_skinny * skinny_cost * length_squared;
costs[edge_id] = metal::isfinite(cost) ? cost : INFINITY;
"""


_FACE_MIN_SOURCE = r"""
uint face_id = thread_position_in_grid.x;
if (face_id >= uint(face_count)) return;
int3 ids = int3(faces[face_id * 3], faces[face_id * 3 + 1], faces[face_id * 3 + 2]);
float best_cost = INFINITY;
int best_edge = 2147483647;
for (uint corner = 0; corner < 3; ++corner) {
    int vertex_id = corner == 0 ? ids.x : (corner == 1 ? ids.y : ids.z);
    for (int item = vertex_edge_offsets[vertex_id]; item < vertex_edge_offsets[vertex_id + 1]; ++item) {
        int edge_id = vertex_edges[item];
        float cost = costs[edge_id];
        if (cost < best_cost || (cost == best_cost && edge_id < best_edge)) {
            best_cost = cost;
            best_edge = edge_id;
        }
    }
}
face_best_edges[face_id] = best_edge;
"""


_EDGE_SELECT_SOURCE = r"""
uint edge_id = thread_position_in_grid.x;
if (edge_id >= uint(edge_count)) return;
float cost = costs[edge_id];
if (!metal::isfinite(cost) || cost > threshold) {
    selected[edge_id] = 0;
    return;
}
int e0 = edges[edge_id * 2];
int e1 = edges[edge_id * 2 + 1];
for (int item = vertex_face_offsets[e0]; item < vertex_face_offsets[e0 + 1]; ++item) {
    if (face_best_edges[vertex_faces[item]] != int(edge_id)) {
        selected[edge_id] = 0;
        return;
    }
}
for (int item = vertex_face_offsets[e1]; item < vertex_face_offsets[e1 + 1]; ++item) {
    if (face_best_edges[vertex_faces[item]] != int(edge_id)) {
        selected[edge_id] = 0;
        return;
    }
}
selected[edge_id] = 1;
"""


__all__ = ["simplify_mesh_mlx_parallel_qem"]
