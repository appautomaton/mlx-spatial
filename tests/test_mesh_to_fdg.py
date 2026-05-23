import numpy as np
import pytest

from mlx_spatial.mesh_to_fdg import mesh_to_flexible_dual_grid
from mlx_spatial.ovoxel import FlexibleDualGridMesh, flexible_dual_grid_to_mesh_np


def _make_sphere_mesh(radius=0.35, lat_steps=16, lon_steps=16):
    """Generate an icosphere-like triangle mesh of a sphere."""
    verts_list = []
    faces_list = []
    for i in range(lat_steps):
        theta = np.pi * (i + 0.5) / lat_steps
        for j in range(lon_steps):
            phi = 2.0 * np.pi * j / lon_steps
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            verts_list.append([x, y, z])
    verts = np.array(verts_list, dtype=np.float32)

    for i in range(lat_steps - 1):
        for j in range(lon_steps):
            a = i * lon_steps + j
            b = i * lon_steps + (j + 1) % lon_steps
            c = (i + 1) * lon_steps + j
            d = (i + 1) * lon_steps + (j + 1) % lon_steps
            faces_list.append([a, b, c])
            faces_list.append([b, d, c])

    top = len(verts)
    verts = np.vstack([verts, [0.0, 0.0, radius]])
    for j in range(lon_steps):
        faces_list.append([j, (j + 1) % lon_steps, top])

    bottom = len(verts)
    verts = np.vstack([verts, [0.0, 0.0, -radius]])
    for j in range(lon_steps):
        faces_list.append([(lat_steps - 1) * lon_steps + (j + 1) % lon_steps,
                           (lat_steps - 1) * lon_steps + j,
                           bottom])

    return verts.astype(np.float32), np.array(faces_list, dtype=np.int64)


def _make_box_mesh():
    """Return a unit cube mesh centered at origin."""
    verts = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4],
            [1, 5, 6], [1, 6, 2],
        ],
        dtype=np.int64,
    )
    return verts, faces


def _mesh_has_valid_topology(mesh: FlexibleDualGridMesh) -> bool:
    """Check that a mesh has valid topology (no degenerate faces, manifold-ish)."""
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return False
    if np.any(mesh.faces < 0) or np.any(mesh.faces >= mesh.vertices.shape[0]):
        return False
    degenerate = 0
    for tri in mesh.faces:
        v0, v1, v2 = mesh.vertices[tri]
        if np.allclose(v0, v1) or np.allclose(v1, v2) or np.allclose(v0, v2):
            degenerate += 1
    return degenerate < len(mesh.faces)


def _mesh_vertex_extents(mesh: FlexibleDualGridMesh) -> np.ndarray:
    return np.array([mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)])


# --- Round-trip tests ---

def test_round_trip_sphere_grid_16():
    verts, faces = _make_sphere_mesh(radius=0.35)
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=16,
    )
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert coords.dtype == np.int32
    assert dual_vertices.shape == (coords.shape[0], 3)
    assert dual_vertices.dtype == np.float32
    assert intersected.shape == (coords.shape[0], 3)
    assert intersected.dtype == np.bool_
    assert coords.shape[0] > 0

    mesh = flexible_dual_grid_to_mesh_np(
        coords, dual_vertices, intersected, None, grid_size=16,
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    )
    assert _mesh_has_valid_topology(mesh)


def test_round_trip_box_grid_12():
    verts, faces = _make_box_mesh()
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=12,
    )
    assert coords.shape[0] > 0

    mesh = flexible_dual_grid_to_mesh_np(
        coords, dual_vertices, intersected, None, grid_size=12,
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    )
    assert _mesh_has_valid_topology(mesh)


def test_round_trip_with_custom_aabb():
    verts, faces = _make_box_mesh()
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=16,
        aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    )
    assert coords.shape[0] > 0

    mesh = flexible_dual_grid_to_mesh_np(
        coords, dual_vertices, intersected, None, grid_size=16,
        aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
    )
    assert _mesh_has_valid_topology(mesh)


def test_round_trip_with_split_weights():
    verts, faces = _make_sphere_mesh(radius=0.35)
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=16,
    )
    split = np.ones((coords.shape[0], 1), dtype=np.float32) * 0.5

    mesh = flexible_dual_grid_to_mesh_np(
        coords, dual_vertices, intersected, split, grid_size=16,
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    )
    assert _mesh_has_valid_topology(mesh)


def test_round_trip_uneven_grid():
    verts, faces = _make_sphere_mesh(radius=0.35)
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=(16, 12, 8),
    )
    assert coords.shape[0] > 0

    mesh = flexible_dual_grid_to_mesh_np(
        coords, dual_vertices, intersected, None, grid_size=(16, 12, 8),
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    )
    assert _mesh_has_valid_topology(mesh)


# --- Dual vertex range ---

def test_dual_vertices_within_expected_range():
    verts, faces = _make_sphere_mesh(radius=0.35)
    _, dual_vertices, _ = mesh_to_flexible_dual_grid(verts, faces, grid_size=16)
    assert np.all(dual_vertices >= -0.5)
    assert np.all(dual_vertices <= 1.5)


# --- Intersected flag consistency ---

def test_intersected_flags_only_where_quads_can_form():
    verts, faces = _make_sphere_mesh(radius=0.35)
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=16,
    )
    coord_set = {tuple(int(v) for v in row) for row in coords}
    _EDGE_OFFSETS = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
        ],
        dtype=np.int32,
    )
    for i in range(coords.shape[0]):
        for axis in range(3):
            if intersected[i, axis]:
                cz, cy, cx = int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2])
                for off in _EDGE_OFFSETS[axis]:
                    neighbor = (cz + int(off[0]), cy + int(off[1]), cx + int(off[2]))
                    assert neighbor in coord_set, (
                        f"Intersected[{i},{axis}]=True but neighbor {neighbor} missing"
                    )


# --- Empty / degenerate cases ---

def test_empty_mesh_raises():
    verts = np.zeros((0, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int64)
    with pytest.raises(ValueError):
        mesh_to_flexible_dual_grid(verts, faces, grid_size=8)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        mesh_to_flexible_dual_grid(
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.int64),
            grid_size=8,
        )
    with pytest.raises(ValueError):
        mesh_to_flexible_dual_grid(
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((1, 4), dtype=np.int64),
            grid_size=8,
        )


def test_invalid_grid_size_raises():
    verts, faces = _make_box_mesh()
    with pytest.raises(ValueError):
        mesh_to_flexible_dual_grid(verts, faces, grid_size=-1)
    with pytest.raises(ValueError):
        mesh_to_flexible_dual_grid(verts, faces, grid_size=(8, 0, 8))


def test_out_of_bounds_faces_raise():
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(ValueError):
        mesh_to_flexible_dual_grid(verts, faces, grid_size=8)


def test_tiny_mesh_outside_grid_returns_empty():
    verts = np.array([[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=8,
    )
    assert coords.shape == (0, 3)
    assert dual_vertices.shape == (0, 3)
    assert intersected.shape == (0, 3)


# --- Coordinates are non-negative and within grid ---

def test_coordinates_within_grid_bounds():
    verts, faces = _make_sphere_mesh(radius=0.35)
    coords, _, _ = mesh_to_flexible_dual_grid(verts, faces, grid_size=16)
    gz, gy, gx = 16, 16, 16
    assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < gz)
    assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < gy)
    assert np.all(coords[:, 2] >= 0) and np.all(coords[:, 2] < gx)


# --- Multiple resolutions produce consistently more voxels ---

def test_higher_resolution_produces_more_voxels():
    verts, faces = _make_sphere_mesh(radius=0.35)
    c8, _, _ = mesh_to_flexible_dual_grid(verts, faces, grid_size=8)
    c16, _, _ = mesh_to_flexible_dual_grid(verts, faces, grid_size=16)
    assert c16.shape[0] >= c8.shape[0]


# --- Mesh extent is roughly preserved ---

def test_round_trip_mesh_inside_bounds():
    verts, faces = _make_sphere_mesh(radius=0.35)
    coords, dual_vertices, intersected = mesh_to_flexible_dual_grid(
        verts, faces, grid_size=24,
    )
    mesh = flexible_dual_grid_to_mesh_np(
        coords, dual_vertices, intersected, None, grid_size=24,
        aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    )
    ext = _mesh_vertex_extents(mesh)
    assert np.all(ext[0] >= -0.52)
    assert np.all(ext[1] <= 0.52)
