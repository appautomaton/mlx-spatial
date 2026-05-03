import mlx.core as mx

from mlx_spatial.topology import (
    adjacency_pairs_26,
    grid_cells,
    grid_edges,
    neighbor_offsets_26,
)


def test_neighbor_offsets_26_uses_lexicographic_order_and_int32_dtype():
    offsets = neighbor_offsets_26()

    assert offsets.shape == (26, 3)
    assert offsets.dtype == mx.int32
    assert offsets[0].tolist() == [-1, -1, -1]
    assert offsets[11].tolist() == [0, -1, 1]
    assert offsets[12].tolist() == [0, 0, -1]
    assert offsets[13].tolist() == [0, 0, 1]
    assert offsets[-1].tolist() == [1, 1, 1]
    assert [0, 0, 0] not in offsets.tolist()


def test_adjacency_pairs_26_are_ordered_by_source_then_offset():
    coordinates = mx.array(
        [
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        dtype=mx.int32,
    )

    pairs = adjacency_pairs_26(coordinates, (3, 3, 3))

    assert pairs.dtype == mx.int32
    assert pairs.tolist() == [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 0],
        [1, 2],
        [1, 3],
        [2, 0],
        [2, 1],
        [2, 3],
        [3, 0],
        [3, 1],
        [3, 2],
    ]


def test_adjacency_rejects_invalid_inputs():
    valid = mx.array([[0, 0, 0]], dtype=mx.int32)

    for coordinates in (
        mx.array([0, 0, 0], dtype=mx.int32),
        mx.array([[0, 0]], dtype=mx.int32),
        mx.array([[0, 0, 0], [0, 0, 0]], dtype=mx.int32),
        mx.array([[3, 0, 0]], dtype=mx.int32),
    ):
        try:
            adjacency_pairs_26(coordinates, (3, 3, 3))
        except ValueError:
            pass
        else:
            raise AssertionError(f"coordinates {coordinates.tolist()} should be rejected")

    for shape in ((3, 3), (3, 0, 3)):
        try:
            adjacency_pairs_26(valid, shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape {shape} should be rejected")


def test_grid_edges_uses_axis_order_then_row_major_starts():
    edges = grid_edges((2, 2, 2))

    assert edges.dtype == mx.int32
    assert edges.tolist() == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]


def test_grid_cells_uses_row_major_cell_and_lexicographic_corner_order():
    cells = grid_cells((2, 2, 2))

    assert cells.dtype == mx.int32
    assert cells.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_grid_edges_and_cells_reject_invalid_shapes():
    for fn in (grid_edges, grid_cells):
        for shape in ((2, 2), (2, -1, 2)):
            try:
                fn(shape)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} should reject shape {shape}")
