"""Sparse coordinate helpers for MLX spatial model code.

These helpers use row-major ordering: the last coordinate dimension changes
fastest, matching NumPy/PyTorch-style flattening for C-contiguous arrays.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce
from operator import mul

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class FlexibleDualGridMesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class MeshHoleFillStats:
    boundary_edges_before: int
    clean_boundary_loops: int
    filled_loops: int
    skipped_large_loops: int
    skipped_complex_components: int
    vertices_added: int
    faces_added: int


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("shape must contain positive dimensions")
    return dims


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    for axis in range(len(shape)):
        strides.append(reduce(mul, shape[axis + 1 :], 1))
    return tuple(strides)


def dense_coordinates(shape: Sequence[int]) -> mx.array:
    """Return integer coordinates for every cell in an N-D dense grid.

    Args:
        shape: Positive grid dimensions.

    Returns:
        An `int32` MLX array with shape `(*shape, ndim)`. Coordinates are in
        row-major order, so `dense_coordinates((2, 3))[0, 2] == [0, 2]`.
    """
    dims = _shape_tuple(shape)
    size = reduce(mul, dims, 1)
    return unflatten_indices(mx.arange(size), dims).reshape((*dims, len(dims)))


def flatten_coordinates(coordinates: mx.array, shape: Sequence[int]) -> mx.array:
    """Flatten N-D coordinates into row-major linear indices.

    Args:
        coordinates: MLX array with shape `(..., ndim)`.
        shape: Positive grid dimensions matching `ndim`.

    Returns:
        An `int32` MLX array with shape `coordinates.shape[:-1]` containing
        row-major linear indices.
    """
    dims = _shape_tuple(shape)
    if len(coordinates.shape) == 0 or coordinates.shape[-1] != len(dims):
        raise ValueError("coordinates must have shape (..., ndim)")

    coords = coordinates.astype(mx.int32)
    strides = mx.array(_row_major_strides(dims), dtype=mx.int32)
    return mx.sum(coords * strides, axis=-1)


def unflatten_indices(indices: mx.array, shape: Sequence[int]) -> mx.array:
    """Unflatten row-major linear indices into N-D coordinates.

    Args:
        indices: MLX array of row-major linear indices with any shape.
        shape: Positive grid dimensions.

    Returns:
        An `int32` MLX array with shape `(*indices.shape, ndim)`.
    """
    dims = _shape_tuple(shape)
    values = indices.astype(mx.int32)
    coords = []
    for stride, dim in zip(_row_major_strides(dims), dims, strict=True):
        coords.append((values // stride) % dim)
    return mx.stack(coords, axis=-1).astype(mx.int32)


def in_bounds_mask(coordinates: mx.array, shape: Sequence[int]) -> mx.array:
    """Return a mask for coordinates that lie inside an N-D grid.

    Args:
        coordinates: MLX array with shape `(..., ndim)`.
        shape: Positive grid dimensions matching `ndim`.

    Returns:
        A boolean MLX array with shape `coordinates.shape[:-1]`.
    """
    dims = _shape_tuple(shape)
    if len(coordinates.shape) == 0 or coordinates.shape[-1] != len(dims):
        raise ValueError("coordinates must have shape (..., ndim)")

    coords = coordinates.astype(mx.int32)
    upper = mx.array(dims, dtype=mx.int32)
    return mx.all((coords >= 0) & (coords < upper), axis=-1)


def flexi_dual_grid_fields_to_mesh(
    coordinates: mx.array,
    fields: mx.array,
    *,
    grid_size: int,
    aabb: Sequence[Sequence[float]] = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
) -> FlexibleDualGridMesh:
    """Convert TRELLIS.2 7-channel FlexiDualGrid fields into a triangle mesh."""

    coords = np.array(coordinates, dtype=np.int32)
    values = np.array(fields, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"FlexiDualGrid coordinates must have shape (num_tokens, 4), got {coords.shape}")
    if values.ndim != 2 or values.shape != (coords.shape[0], 7):
        raise ValueError(f"FlexiDualGrid fields must have shape ({coords.shape[0]}, 7), got {values.shape}")
    if coords.shape[0] == 0:
        raise ValueError("FlexiDualGrid fields must contain at least one voxel")
    if np.any(coords[:, 0] != 0):
        raise ValueError("FlexiDualGrid mesh extraction currently supports batch index 0 only")

    dual_vertices = 2.0 * _sigmoid(values[:, 0:3]) - 0.5
    intersected = values[:, 3:6] > 0
    split_weight = _softplus(values[:, 6:7])
    return flexible_dual_grid_to_mesh_np(
        coords[:, 1:],
        dual_vertices,
        intersected,
        split_weight,
        aabb=aabb,
        grid_size=grid_size,
    )


def flexible_dual_grid_to_mesh_np(
    coords: np.ndarray,
    dual_vertices: np.ndarray,
    intersected_flag: np.ndarray,
    split_weight: np.ndarray | None,
    *,
    aabb: Sequence[Sequence[float]],
    grid_size: int | Sequence[int],
) -> FlexibleDualGridMesh:
    """NumPy inference-mode port of `o_voxel.convert.flexible_dual_grid_to_mesh`."""

    coords = np.asarray(coords, dtype=np.int32)
    dual_vertices = np.asarray(dual_vertices, dtype=np.float32)
    intersected_flag = np.asarray(intersected_flag, dtype=np.bool_)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (num_tokens, 3), got {coords.shape}")
    if dual_vertices.shape != (coords.shape[0], 3):
        raise ValueError(f"dual_vertices must have shape ({coords.shape[0]}, 3), got {dual_vertices.shape}")
    if intersected_flag.shape != (coords.shape[0], 3):
        raise ValueError(f"intersected_flag must have shape ({coords.shape[0]}, 3), got {intersected_flag.shape}")
    if split_weight is not None:
        split_weight = np.asarray(split_weight, dtype=np.float32)
        if split_weight.shape != (coords.shape[0], 1):
            raise ValueError(f"split_weight must have shape ({coords.shape[0]}, 1), got {split_weight.shape}")

    bounds = np.asarray(aabb, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise ValueError(f"aabb must have shape (2, 3), got {bounds.shape}")
    if isinstance(grid_size, int):
        grid = np.array([grid_size, grid_size, grid_size], dtype=np.int32)
    else:
        grid = np.asarray(tuple(int(dim) for dim in grid_size), dtype=np.int32)
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError(f"grid_size must contain three positive dimensions, got {grid}")

    voxel_size = (bounds[1] - bounds[0]) / grid.astype(np.float32)
    vertices = (coords.astype(np.float32) + dual_vertices) * voxel_size.reshape(1, 3) + bounds[0].reshape(1, 3)
    coord_to_index = {tuple(int(value) for value in row): index for index, row in enumerate(coords)}

    quad_indices: list[list[int]] = []
    for index, coord in enumerate(coords):
        for axis in range(3):
            if not intersected_flag[index, axis]:
                continue
            quad = []
            for offset in _EDGE_NEIGHBOR_VOXEL_OFFSET[axis]:
                neighbor = tuple(int(value) for value in coord + offset)
                neighbor_index = coord_to_index.get(neighbor)
                if neighbor_index is None:
                    quad = []
                    break
                quad.append(neighbor_index)
            if quad:
                quad_indices.append(quad)

    if not quad_indices:
        return FlexibleDualGridMesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int64),
        )

    quads = np.asarray(quad_indices, dtype=np.int64)
    if split_weight is None:
        split_1 = quads[:, _QUAD_SPLIT_1]
        split_2 = quads[:, _QUAD_SPLIT_2]
        align_1 = _split_alignment(vertices, split_1)
        align_2 = _split_alignment(vertices, split_2)
        faces = np.where((align_1 > align_2)[:, None], split_1, split_2).reshape(-1, 3)
    else:
        weights = split_weight[quads, 0]
        split_02 = weights[:, 0] * weights[:, 2]
        split_13 = weights[:, 1] * weights[:, 3]
        faces = np.where(
            (split_02 > split_13)[:, None],
            quads[:, _QUAD_SPLIT_1],
            quads[:, _QUAD_SPLIT_2],
        ).reshape(-1, 3)

    return FlexibleDualGridMesh(vertices=vertices.astype(np.float32, copy=False), faces=faces.astype(np.int64, copy=False))


def fill_flexible_dual_grid_mesh_holes(
    mesh: FlexibleDualGridMesh,
    *,
    max_hole_perimeter: float = 3e-2,
    max_hole_edges: int = 64,
) -> tuple[FlexibleDualGridMesh, MeshHoleFillStats]:
    """Fill clean small boundary loops in a FlexiDualGrid mesh.

    Upstream TRELLIS.2 calls `Mesh.fill_holes(max_hole_perimeter=3e-2)` after
    latent decode. This CPU path intentionally handles only simple manifold
    boundary loops; complex or large boundaries are left untouched.
    """

    if max_hole_perimeter <= 0:
        raise ValueError(f"max_hole_perimeter must be positive, got {max_hole_perimeter}")
    if max_hole_edges < 3:
        raise ValueError(f"max_hole_edges must be at least 3, got {max_hole_edges}")

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

    boundary_edges, skipped_complex_components = _find_boundary_edges_and_complex_components(faces)
    loops = _order_clean_boundary_loops(boundary_edges)
    directed_boundary = {(int(start), int(end)) for start, end in boundary_edges}
    new_vertices: list[np.ndarray] = []
    new_faces: list[tuple[int, int, int]] = []
    skipped_large = 0

    for loop in loops:
        if len(loop) > max_hole_edges:
            skipped_large += 1
            continue
        loop_vertices = vertices[np.asarray(loop, dtype=np.int64)]
        perimeter = float(np.linalg.norm(np.roll(loop_vertices, -1, axis=0) - loop_vertices, axis=1).sum())
        if perimeter > max_hole_perimeter:
            skipped_large += 1
            continue

        center_index = vertices.shape[0] + len(new_vertices)
        new_vertices.append(loop_vertices.mean(axis=0).astype(np.float32, copy=False))
        forward_votes = sum((loop[i], loop[(i + 1) % len(loop)]) in directed_boundary for i in range(len(loop)))
        reverse_boundary = forward_votes >= len(loop) - forward_votes
        for i, start in enumerate(loop):
            end = loop[(i + 1) % len(loop)]
            if reverse_boundary:
                new_faces.append((center_index, int(end), int(start)))
            else:
                new_faces.append((center_index, int(start), int(end)))

    if new_vertices:
        filled_mesh = FlexibleDualGridMesh(
            vertices=np.concatenate([vertices, np.asarray(new_vertices, dtype=np.float32)], axis=0),
            faces=np.concatenate([faces, np.asarray(new_faces, dtype=np.int64)], axis=0),
        )
    else:
        filled_mesh = FlexibleDualGridMesh(vertices=vertices, faces=faces)

    stats = MeshHoleFillStats(
        boundary_edges_before=int(boundary_edges.shape[0]),
        clean_boundary_loops=len(loops),
        filled_loops=len(new_vertices),
        skipped_large_loops=skipped_large,
        skipped_complex_components=skipped_complex_components,
        vertices_added=len(new_vertices),
        faces_added=len(new_faces),
    )
    return filled_mesh, stats


def mesh_to_obj_payload(mesh: FlexibleDualGridMesh) -> bytes:
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError("FlexiDualGrid mesh must contain vertices and faces")
    lines = ["# mlx-spatial TRELLIS.2 FlexiDualGrid shape mesh"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces)
    lines.append("")
    return "\n".join(lines).encode("utf-8")


_EDGE_NEIGHBOR_VOXEL_OFFSET = np.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=np.int32,
)
_QUAD_SPLIT_1 = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
_QUAD_SPLIT_2 = np.array([0, 1, 3, 3, 1, 2], dtype=np.int64)


def _split_alignment(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals_0 = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    normals_1 = np.cross(
        vertices[triangles[:, 2]] - vertices[triangles[:, 1]],
        vertices[triangles[:, 3]] - vertices[triangles[:, 1]],
    )
    return np.abs(np.sum(normals_0 * normals_1, axis=1))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _find_boundary_edges_and_complex_components(faces: np.ndarray) -> tuple[np.ndarray, int]:
    directed = np.empty((faces.shape[0] * 3, 2), dtype=np.int64)
    directed[0::3] = faces[:, [0, 1]]
    directed[1::3] = faces[:, [1, 2]]
    directed[2::3] = faces[:, [2, 0]]
    keys = np.sort(directed, axis=1)
    key_view = np.ascontiguousarray(keys).view([("a", keys.dtype), ("b", keys.dtype)]).reshape(-1)
    _, first_indices, counts = np.unique(key_view, return_index=True, return_counts=True)
    boundary_edges = directed[first_indices[counts == 1]]
    if boundary_edges.shape[0] == 0:
        return boundary_edges, 0

    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))

    visited_edges: set[tuple[int, int]] = set()
    complex_components = 0
    for start, end in boundary_edges:
        edge = (int(min(start, end)), int(max(start, end)))
        if edge in visited_edges:
            continue
        stack = [int(start)]
        component_vertices: set[int] = set()
        component_edges: set[tuple[int, int]] = set()
        while stack:
            vertex = stack.pop()
            if vertex in component_vertices:
                continue
            component_vertices.add(vertex)
            for neighbor in adjacency[vertex]:
                component_edge = (min(vertex, neighbor), max(vertex, neighbor))
                component_edges.add(component_edge)
                if component_edge not in visited_edges:
                    visited_edges.add(component_edge)
                    stack.append(neighbor)
        if (
            any(len(adjacency[vertex]) != 2 for vertex in component_vertices)
            or len(component_edges) != len(component_vertices)
        ):
            complex_components += 1
    return boundary_edges, complex_components


def _order_clean_boundary_loops(boundary_edges: np.ndarray) -> list[list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for start, end in boundary_edges:
        adjacency[int(start)].append(int(end))
        adjacency[int(end)].append(int(start))

    loops: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()
    for raw_start, raw_end in boundary_edges:
        first_edge = (int(min(raw_start, raw_end)), int(max(raw_start, raw_end)))
        if first_edge in visited_edges:
            continue

        start = int(raw_start)
        previous = -1
        current = start
        loop: list[int] = []
        component_edges: set[tuple[int, int]] = set()
        clean = True
        while True:
            if len(adjacency[current]) != 2 or current in loop:
                clean = current == start and len(loop) >= 3
                break
            loop.append(current)
            candidates = [neighbor for neighbor in adjacency[current] if neighbor != previous]
            if not candidates:
                clean = False
                break
            nxt = candidates[0]
            edge = (min(current, nxt), max(current, nxt))
            component_edges.add(edge)
            previous, current = current, nxt
            if current == start:
                clean = len(loop) >= 3
                break

        for edge in component_edges:
            visited_edges.add(edge)
        if clean and all(len(adjacency[vertex]) == 2 for vertex in loop):
            loops.append(loop)
    return loops


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)
