import mlx.core as mx

from mlx_spatial.ovoxel import (
    dense_coordinates,
    flexi_dual_grid_fields_to_mesh,
    flatten_coordinates,
    in_bounds_mask,
    mesh_to_obj_payload,
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
