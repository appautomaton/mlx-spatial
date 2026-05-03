import mlx.core as mx

from mlx_spatial.sparse_conv import weighted_sparse_conv, weighted_sparse_conv_chunked


def test_weighted_sparse_conv_computes_exact_reference_output():
    source_features = mx.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=mx.float32,
    )
    map_rows = mx.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 2, 1],
        ],
        dtype=mx.int32,
    )
    kernel_weights = mx.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ],
        dtype=mx.float32,
    )

    output = weighted_sparse_conv(source_features, map_rows, kernel_weights, target_count=3)

    assert output.shape == (3, 2)
    assert output.dtype == mx.float32
    assert output.tolist() == [
        [24.0, 36.0],
        [15.0, 22.0],
        [0.0, 0.0],
    ]


def test_weighted_sparse_conv_accumulates_duplicate_targets_deterministically():
    source_features = mx.array([[1.0], [2.0], [3.0]], dtype=mx.float32)
    map_rows = mx.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 2, 0],
        ],
        dtype=mx.int32,
    )
    kernel_weights = mx.array([[[10.0]], [[100.0]]], dtype=mx.float32)

    output = weighted_sparse_conv(source_features, map_rows, kernel_weights, target_count=2)

    assert output.tolist() == [[240.0], [0.0]]


def test_weighted_sparse_conv_chunked_matches_reference_output():
    source_features = mx.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=mx.float32,
    )
    map_rows = mx.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 2, 1],
        ],
        dtype=mx.int32,
    )
    kernel_weights = mx.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ],
        dtype=mx.float32,
    )

    expected = weighted_sparse_conv(source_features, map_rows, kernel_weights, target_count=3)
    actual = weighted_sparse_conv_chunked(source_features, map_rows, kernel_weights, target_count=3, row_chunk_size=1)

    assert actual.tolist() == expected.tolist()


def test_weighted_sparse_conv_supports_empty_maps():
    source_features = mx.array([[1.0, 2.0]], dtype=mx.float32)
    map_rows = mx.array([], dtype=mx.int32).reshape((0, 3))
    kernel_weights = mx.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=mx.float32)

    output = weighted_sparse_conv(source_features, map_rows, kernel_weights, target_count=2)

    assert output.shape == (2, 3)
    assert output.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_weighted_sparse_conv_rejects_invalid_shapes_and_channels():
    source_features = mx.array([[1.0, 2.0]], dtype=mx.float32)
    map_rows = mx.array([[0, 0, 0]], dtype=mx.int32)
    kernel_weights = mx.array([[[1.0], [2.0]]], dtype=mx.float32)

    invalid_cases = (
        (mx.array([1.0, 2.0], dtype=mx.float32), map_rows, kernel_weights, 1),
        (source_features, mx.array([0, 0, 0], dtype=mx.int32), kernel_weights, 1),
        (source_features, mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32), kernel_weights, 1),
        (source_features, map_rows, mx.array([[1.0], [2.0]], dtype=mx.float32), 1),
        (source_features, map_rows, mx.array([[[1.0]]], dtype=mx.float32), 1),
        (source_features, map_rows, kernel_weights, -1),
    )
    for features, rows, weights, target_count in invalid_cases:
        try:
            weighted_sparse_conv(features, rows, weights, target_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid weighted sparse convolution input should be rejected")


def test_weighted_sparse_conv_rejects_out_of_bounds_indices():
    source_features = mx.array([[1.0]], dtype=mx.float32)
    kernel_weights = mx.array([[[1.0]]], dtype=mx.float32)

    invalid_maps = (
        mx.array([[-1, 0, 0]], dtype=mx.int32),
        mx.array([[1, 0, 0]], dtype=mx.int32),
        mx.array([[0, -1, 0]], dtype=mx.int32),
        mx.array([[0, 1, 0]], dtype=mx.int32),
        mx.array([[0, 0, -1]], dtype=mx.int32),
        mx.array([[0, 0, 1]], dtype=mx.int32),
    )
    for map_rows in invalid_maps:
        try:
            weighted_sparse_conv(source_features, map_rows, kernel_weights, target_count=1)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds map row index should be rejected")
