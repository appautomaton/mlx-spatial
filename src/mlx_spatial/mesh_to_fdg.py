"""Convert triangle mesh to FlexiDualGrid representation.

Pure Python/NumPy replacement for o_voxel.convert.mesh_to_flexible_dual_grid().
Produces voxel indices, dual vertices, and intersected flags that are compatible
with flexible_dual_grid_to_mesh_np() for round-trip mesh extraction.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EDGE_OFFSETS = np.array(
    [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ],
    dtype=np.int32,
)

_DEFAULT_AABB = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=np.float32)


def mesh_to_flexible_dual_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid_size: int | Sequence[int],
    *,
    aabb: Sequence[Sequence[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert triangle mesh to FlexiDualGrid representation.

    Args:
        vertices: (num_vertices, 3) float32 array of vertex positions.
        faces: (num_faces, 3) int32/int64 array of face vertex indices.
        grid_size: int or (3,) sequence of grid dimensions.
        aabb: optional (2, 3) axis-aligned bounding box. Defaults to
            [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]].

    Returns:
        coords: (num_voxels, 3) int32 array of voxel grid coordinates (z,y,x).
        dual_vertices: (num_voxels, 3) float32 array of dual vertex offsets
            relative to voxel integer coordinate, range roughly [-0.5, 1.5].
        intersected: (num_voxels, 3) bool array of axis-intersected flags.

    Raises:
        ValueError: if inputs have invalid shapes or the mesh is empty.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (num_vertices, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (num_faces, 3), got {faces.shape}")
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("mesh must contain vertices and faces")
    if np.any(faces < 0) or np.any(faces >= vertices.shape[0]):
        raise ValueError("faces contain vertex indices outside vertex array")

    bounds = _DEFAULT_AABB if aabb is None else np.asarray(aabb, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise ValueError(f"aabb must have shape (2, 3), got {bounds.shape}")

    if isinstance(grid_size, int):
        grid = np.array([grid_size, grid_size, grid_size], dtype=np.int32)
    else:
        grid = np.asarray(tuple(int(dim) for dim in grid_size), dtype=np.int32)
    if grid.shape != (3,) or np.any(grid <= 0):
        raise ValueError(f"grid_size must contain three positive dimensions, got {grid}")

    voxel_size = (bounds[1] - bounds[0]) / grid.astype(np.float32)

    occupied, dual_map = _voxelize_and_compute_duals(vertices, faces, grid, bounds, voxel_size)
    if not occupied:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.bool_),
        )

    coords_list: list[tuple[int, int, int]] = []
    dual_list: list[tuple[float, float, float]] = []
    for coord, (div_z, count) in sorted(occupied.items()):
        if count > 0:
            z, y, x = coord
            accum = dual_map[coord]
            center = (accum[0] / count, accum[1] / count, accum[2] / count)
            dw = (center[0] - bounds[0, 0]) / voxel_size[0] - z
            dh = (center[1] - bounds[0, 1]) / voxel_size[1] - y
            dd = (center[2] - bounds[0, 2]) / voxel_size[2] - x
            coords_list.append((z, y, x))
            dual_list.append((dw, dh, dd))

    if not coords_list:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.bool_),
        )

    coords = np.array(coords_list, dtype=np.int32)
    dual_vertices = np.array(dual_list, dtype=np.float32)
    coord_to_idx = {tuple(int(v) for v in row): i for i, row in enumerate(coords)}

    num_voxels = coords.shape[0]
    intersected = np.zeros((num_voxels, 3), dtype=np.bool_)
    for idx in range(num_voxels):
        cz, cy, cx = int(coords[idx, 0]), int(coords[idx, 1]), int(coords[idx, 2])
        for axis in range(3):
            all_present = True
            for off in _EDGE_OFFSETS[axis]:
                neighbor = (cz + int(off[0]), cy + int(off[1]), cx + int(off[2]))
                if neighbor not in coord_to_idx:
                    all_present = False
                    break
            if all_present:
                intersected[idx, axis] = True

    dual_vertices = np.clip(dual_vertices, -0.5, 1.5)
    return coords, dual_vertices.astype(np.float32), intersected


def _voxelize_and_compute_duals(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: np.ndarray,
    bounds: np.ndarray,
    voxel_size: np.ndarray,
) -> tuple[dict[tuple[int, int, int], tuple[float, int]], dict[tuple[int, int, int], tuple[float, float, float]]]:
    """Voxelize triangle mesh into grid and accumulate dual vertex data.

    Input vertices are in world (x, y, z) order.
    Grid coordinates use (z, y, x) order (FDG convention).
    bounds is (2, 3) where row 0 = (z_min, y_min, x_min), row 1 = (z_max, y_max, x_max).
    """
    occupied: dict[tuple[int, int, int], tuple[float, int]] = {}
    accum: dict[tuple[int, int, int], tuple[float, float, float]] = {}

    tri_verts = vertices[faces]

    for t in range(tri_verts.shape[0]):
        v0, v1, v2 = tri_verts[t, 0], tri_verts[t, 1], tri_verts[t, 2]

        tri_wx_min = float(min(v0[0], v1[0], v2[0]))
        tri_wy_min = float(min(v0[1], v1[1], v2[1]))
        tri_wz_min = float(min(v0[2], v1[2], v2[2]))
        tri_wx_max = float(max(v0[0], v1[0], v2[0]))
        tri_wy_max = float(max(v0[1], v1[1], v2[1]))
        tri_wz_max = float(max(v0[2], v1[2], v2[2]))

        bz_min = float(bounds[0, 0])
        by_min = float(bounds[0, 1])
        bx_min = float(bounds[0, 2])
        vs_z = float(voxel_size[0])
        vs_y = float(voxel_size[1])
        vs_x = float(voxel_size[2])

        eps = np.finfo(np.float32).eps
        gz_min = max(int(np.floor((tri_wz_min - bz_min - eps) / vs_z)), 0)
        gz_max = min(int(np.floor((tri_wz_max - bz_min + eps) / vs_z)) + 1, int(grid[0]))
        gy_min = max(int(np.floor((tri_wy_min - by_min - eps) / vs_y)), 0)
        gy_max = min(int(np.floor((tri_wy_max - by_min + eps) / vs_y)) + 1, int(grid[1]))
        gx_min = max(int(np.floor((tri_wx_min - bx_min - eps) / vs_x)), 0)
        gx_max = min(int(np.floor((tri_wx_max - bx_min + eps) / vs_x)) + 1, int(grid[2]))

        for gz in range(gz_min, gz_max):
            for gy in range(gy_min, gy_max):
                for gx in range(gx_min, gx_max):
                    box_origin_world = np.array(
                        [
                            bx_min + gx * vs_x,
                            by_min + gy * vs_y,
                            bz_min + gz * vs_z,
                        ],
                        dtype=np.float32,
                    )
                    box_size_world = np.array([vs_x, vs_y, vs_z], dtype=np.float32)
                    if not _triangle_box_overlap(v0, v1, v2, box_origin_world, box_size_world):
                        continue
                    cx, cy, cz = _voxel_intersection_centroid(
                        v0, v1, v2,
                        box_origin_world[0], box_origin_world[0] + box_size_world[0],
                        box_origin_world[1], box_origin_world[1] + box_size_world[1],
                        box_origin_world[2], box_origin_world[2] + box_size_world[2],
                    )
                    key = (gz, gy, gx)
                    if key not in occupied:
                        occupied[key] = (cz, 1)
                        accum[key] = (cz, cy, cx)
                    else:
                        prev_div, prev_cnt = occupied[key]
                        occupied[key] = (prev_div + cz, prev_cnt + 1)
                        a = accum[key]
                        accum[key] = (a[0] + cz, a[1] + cy, a[2] + cx)

    return occupied, accum


def _triangle_box_overlap(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    box_center: np.ndarray,
    half_size: np.ndarray,
) -> bool:
    """SAT-based triangle-AABB overlap test (Akenine-Möller).

    The box is centered at box_center with half-extents half_size.
    """
    hs = half_size * 0.5
    c = box_center + hs

    v0c = v0 - c
    v1c = v1 - c
    v2c = v2 - c

    f0 = v1c - v0c
    f1 = v2c - v1c
    f2 = v0c - v2c

    axes = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ]

    for a_idx in range(3):
        for e_idx in range(3):
            a = axes[a_idx]
            e = [f0, f1, f2][e_idx]
            axis = np.cross(a, e)
            p0 = np.dot(v0c, axis)
            p1 = np.dot(v1c, axis)
            p2 = np.dot(v2c, axis)
            r = (
                hs[0] * abs(axis[0])
                + hs[1] * abs(axis[1])
                + hs[2] * abs(axis[2])
            )
            p_min = min(p0, p1, p2)
            p_max = max(p0, p1, p2)
            if p_min > r or p_max < -r:
                return False

    for a_idx in range(3):
        a = axes[a_idx]
        p0 = np.dot(v0c, a)
        p1 = np.dot(v1c, a)
        p2 = np.dot(v2c, a)
        r = hs[a_idx]
        p_min = min(p0, p1, p2)
        p_max = max(p0, p1, p2)
        if p_min > r or p_max < -r:
            return False

    normal = np.cross(f0, f1)
    p0 = np.dot(v0c, normal)
    r = (
        hs[0] * abs(normal[0])
        + hs[1] * abs(normal[1])
        + hs[2] * abs(normal[2])
    )
    if -p0 > r or p0 > r:
        return False

    return True


def _voxel_intersection_centroid(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    bx_min: float,
    bx_max: float,
    by_min: float,
    by_max: float,
    bz_min: float,
    bz_max: float,
) -> tuple[float, float, float]:
    """Compute centroid of triangle-voxel intersection region.

    Collects triangle vertices inside the voxel and triangle-edge
    intersections with voxel faces, then returns their average.

    Returns (cx, cy, cz) in world (x, y, z) coordinates.
    """
    pts: list[tuple[float, float, float]] = []

    for v in (v0, v1, v2):
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        if bx_min <= vx <= bx_max and by_min <= vy <= by_max and bz_min <= vz <= bz_max:
            pts.append((vx, vy, vz))

    edges = [(v0, v1), (v1, v2), (v2, v0)]
    for a, b in edges:
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx_f, by_f, bz_f = float(b[0]), float(b[1]), float(b[2])
        for face_coord, face_axis in (
            (bx_min, 0),
            (bx_max, 0),
            (by_min, 1),
            (by_max, 1),
            (bz_min, 2),
            (bz_max, 2),
        ):
            if face_axis == 0:
                da = ax - face_coord
                db = bx_f - face_coord
            elif face_axis == 1:
                da = ay - face_coord
                db = by_f - face_coord
            else:
                da = az - face_coord
                db = bz_f - face_coord

            if abs(da - db) < 1e-12:
                continue
            t = da / (da - db)
            if t < 0.0 or t > 1.0:
                continue

            ix = ax + t * (bx_f - ax)
            iy = ay + t * (by_f - ay)
            iz = az + t * (bz_f - az)

            inside = True
            if face_axis != 0 and not (bx_min - 1e-9 <= ix <= bx_max + 1e-9):
                inside = False
            if face_axis != 1 and not (by_min - 1e-9 <= iy <= by_max + 1e-9):
                inside = False
            if face_axis != 2 and not (bz_min - 1e-9 <= iz <= bz_max + 1e-9):
                inside = False
            if inside:
                pts.append((ix, iy, iz))

    if not pts:
        cx = (float(v0[0]) + float(v1[0]) + float(v2[0])) / 3.0
        cy = (float(v0[1]) + float(v1[1]) + float(v2[1])) / 3.0
        cz = (float(v0[2]) + float(v1[2]) + float(v2[2])) / 3.0
        return (cx, cy, cz)

    n = float(len(pts))
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n
    cz = sum(p[2] for p in pts) / n
    return (cx, cy, cz)
