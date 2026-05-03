import mlx.core as mx

from mlx_spatial.sparse_conv import kernel_offsets, sparse_conv_map, sparse_conv_map_vectorized


def test_kernel_offsets_uses_lexicographic_order_and_includes_center():
    offsets = kernel_offsets((3, 3, 3))

    assert offsets.shape == (27, 3)
    assert offsets.dtype == mx.int32
    assert offsets[0].tolist() == [-1, -1, -1]
    assert offsets[13].tolist() == [0, 0, 0]
    assert offsets[-1].tolist() == [1, 1, 1]


def test_kernel_offsets_supports_non_cubic_odd_kernels():
    offsets = kernel_offsets((1, 3, 5))

    assert offsets.shape == (15, 3)
    assert offsets[0].tolist() == [0, -1, -2]
    assert offsets[-1].tolist() == [0, 1, 2]


def test_kernel_offsets_rejects_invalid_kernel_sizes():
    for kernel_size in ((3, 3), (0, 3, 3), (2, 3, 3)):
        try:
            kernel_offsets(kernel_size)
        except ValueError:
            pass
        else:
            raise AssertionError(f"kernel_size {kernel_size} should be rejected")


def test_sparse_conv_map_uses_target_plus_offset_sign_and_exact_row_order():
    coordinates = mx.array(
        [
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        dtype=mx.int32,
    )

    rows = sparse_conv_map(coordinates, (3, 3, 3))

    assert rows.dtype == mx.int32
    assert rows.tolist() == [
        [0, 0, 13],
        [0, 1, 14],
        [0, 2, 17],
        [0, 3, 26],
        [1, 0, 12],
        [1, 1, 13],
        [1, 2, 16],
        [1, 3, 25],
        [2, 0, 9],
        [2, 1, 10],
        [2, 2, 13],
        [2, 3, 22],
        [3, 0, 0],
        [3, 1, 1],
        [3, 2, 4],
        [3, 3, 13],
    ]


def test_sparse_conv_map_omits_missing_neighbors_for_generic_sparse_grid():
    coordinates = mx.array([[0, 0, 0], [0, 0, 1]], dtype=mx.int32)

    rows = sparse_conv_map(coordinates, (2, 2, 2))

    assert rows.tolist() == [
        [0, 0, 13],
        [0, 1, 14],
        [1, 0, 12],
        [1, 1, 13],
    ]


def test_sparse_conv_map_vectorized_matches_reference_order():
    coordinates = mx.array(
        [
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ],
        dtype=mx.int32,
    )

    expected = sparse_conv_map(coordinates, (3, 3, 3))
    actual = sparse_conv_map_vectorized(coordinates, (3, 3, 3), row_chunk_size=2)

    assert actual.tolist() == expected.tolist()


def test_sparse_conv_map_rejects_invalid_inputs():
    valid = mx.array([[0, 0, 0]], dtype=mx.int32)

    for coordinates in (
        mx.array([0, 0, 0], dtype=mx.int32),
        mx.array([[0, 0]], dtype=mx.int32),
        mx.array([[0, 0, 0], [0, 0, 0]], dtype=mx.int32),
        mx.array([[3, 0, 0]], dtype=mx.int32),
    ):
        try:
            sparse_conv_map(coordinates, (3, 3, 3))
        except ValueError:
            pass
        else:
            raise AssertionError(f"coordinates {coordinates.tolist()} should be rejected")

    for shape in ((3, 3), (3, 0, 3)):
        try:
            sparse_conv_map(valid, shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape {shape} should be rejected")

    try:
        sparse_conv_map(valid, (3, 3, 3), kernel_size=(2, 3, 3))
    except ValueError:
        pass
    else:
        raise AssertionError("even kernel size should be rejected")
