import mlx.core as mx
import numpy as np

from mlx_spatial.ovoxel import (
    FlexibleDualGridMesh,
    dense_coordinates,
    fill_flexible_dual_grid_mesh_holes,
    fill_narrow_band_mesh_holes,
    flexi_dual_grid_fields_to_mesh,
    flatten_coordinates,
    in_bounds_mask,
    mesh_to_obj_payload,
    remesh_narrow_band_dual_contouring,
    unflatten_indices,
)


def test_dense_coordinates_uses_row_major_order_and_int32_dtype():
    coords = dense_coordinates((2, 3))

    assert coords.shape == (2, 3, 2)
    assert coords.dtype == mx.int32
    assert coords.tolist() == [
        [[0, 0], [0, 1], [0, 2]],
        [[1, 0], [1, 1], [1, 2]],
    ]


def test_flatten_and_unflatten_coordinates_round_trip():
    coords = dense_coordinates((2, 3, 4))
    flat = flatten_coordinates(coords, (2, 3, 4))
    round_tripped = unflatten_indices(flat, (2, 3, 4))

    assert flat.shape == (2, 3, 4)
    assert flat.dtype == mx.int32
    assert flat[1, 2, 3].item() == 23
    assert round_tripped.tolist() == coords.tolist()


def test_in_bounds_mask_for_generic_sparse_grid_coordinates():
    sparse_coords = mx.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [2, 0, 0],
            [1, -1, 0],
        ],
        dtype=mx.int32,
    )

    mask = in_bounds_mask(sparse_coords, (2, 3, 4))

    assert mask.shape == (4,)
    assert mask.tolist() == [True, True, False, False]


def test_invalid_shapes_are_rejected():
    for shape in ((), (0,), (2, -1)):
        try:
            dense_coordinates(shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape {shape} should be rejected")


def test_coordinate_rank_mismatches_are_rejected():
    coords = mx.array([[0, 1, 2]], dtype=mx.int32)


    for fn in (flatten_coordinates, in_bounds_mask):
        try:
            fn(coords, (2, 3))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} should reject rank mismatch")


def test_flexi_dual_grid_mesh_extracts_nonempty_tiny_fixture():
    coords = []
    fields = []
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                coords.append([0, z, y, x])
                fields.append([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    mesh = flexi_dual_grid_fields_to_mesh(
        mx.array(coords, dtype=mx.int32),
        mx.array(fields, dtype=mx.float32),
        grid_size=2,
    )
    payload = mesh_to_obj_payload(mesh)

    assert mesh.vertices.shape == (8, 3)
    assert mesh.faces.shape[0] > 0
    assert b"\nv " in payload
    assert b"\nf " in payload


def test_fill_flexible_dual_grid_mesh_holes_fills_small_clean_loop():
    mesh = FlexibleDualGridMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.005, 0.0, 0.0],
                [0.0, 0.005, 0.0],
                [0.0, 0.0, 0.005],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    )

    filled, stats = fill_flexible_dual_grid_mesh_holes(mesh)

    assert _boundary_edge_count(mesh.faces) == 3
    assert _boundary_edge_count(filled.faces) == 0
    assert stats.boundary_edges_before == 3
    assert stats.clean_boundary_loops == 1
    assert stats.filled_loops == 1
    assert stats.vertices_added == 1
    assert stats.faces_added == 3


def test_fill_flexible_dual_grid_mesh_holes_leaves_large_loop_open():
    mesh = FlexibleDualGridMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    )

    filled, stats = fill_flexible_dual_grid_mesh_holes(mesh)

    np.testing.assert_array_equal(filled.faces, mesh.faces)
    assert stats.clean_boundary_loops == 1
    assert stats.filled_loops == 0
    assert stats.skipped_large_loops == 1


def _boundary_edge_count(faces: np.ndarray) -> int:
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in faces:
        for start, end in ((a, b), (b, c), (c, a)):
            edge = (int(min(start, end)), int(max(start, end)))
            counts[edge] = counts.get(edge, 0) + 1
    return sum(count == 1 for count in counts.values())


def test_remesh_narrow_band_dual_contouring_builds_watertight_tiny_cube():
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int32,
    )
    dual_vertices = np.zeros((8, 3), dtype=np.float32)
    intersected = np.ones((8, 3), dtype=bool)

    mesh = remesh_narrow_band_dual_contouring(coords, dual_vertices, intersected, grid_size=2)

    assert mesh.vertices.shape == (8, 3)
    assert mesh.faces.shape[0] > 0
    assert mesh.faces.shape[1] == 3
    assert _boundary_edge_count(mesh.faces) == 0


def test_remesh_narrow_band_dual_contouring_produces_consistent_uv_and_grid_scaling():
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.int32,
    )
    dual_vertices = np.zeros((4, 3), dtype=np.float32)
    intersected = np.ones((4, 3), dtype=bool)

    mesh = remesh_narrow_band_dual_contouring(
        coords,
        dual_vertices,
        intersected,
        aabb=((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        grid_size=2,
    )

    assert mesh.vertices.shape[0] == 4
    assert mesh.vertices.shape[1] == 3
    assert np.all(mesh.vertices >= -1.0)
    assert np.all(mesh.vertices <= 1.0)
    assert mesh.faces.shape[0] > 0


def test_remesh_narrow_band_dual_contouring_returns_empty_mesh_for_empty_input():
    mesh = remesh_narrow_band_dual_contouring(
        np.zeros((0, 3), dtype=np.int32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=bool),
        grid_size=4,
    )

    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 3)


def test_remesh_narrow_band_dual_contouring_rejects_shape_mismatch():
    coords = np.zeros((4, 3), dtype=np.int32)
    bad_dual = np.zeros((3, 3), dtype=np.float32)
    flags = np.ones((4, 3), dtype=bool)

    try:
        remesh_narrow_band_dual_contouring(coords, bad_dual, flags, grid_size=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for shape mismatch")


def test_fill_narrow_band_mesh_holes_fills_boundary_loop():
    mesh = FlexibleDualGridMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.01],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    )

    filled, stats = fill_narrow_band_mesh_holes(mesh)

    assert stats.filled_loops == 1
    assert stats.faces_added == 3
    assert stats.vertices_added == 1
    assert _boundary_edge_count(filled.faces) == 0


def test_fill_narrow_band_mesh_holes_skips_large_area_holes():
    mesh = FlexibleDualGridMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.01],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64),
    )

    filled, stats = fill_narrow_band_mesh_holes(mesh, max_hole_area=0.001)

    assert stats.filled_loops == 0
    assert stats.skipped_large_loops == 1
