import mlx.core as mx

from mlx_spatial.sparse_conv import gather_sparse_features, scatter_sparse_features


def test_gather_sparse_features_uses_source_index_in_map_row_order():
    source_features = mx.array(
        [
            [10.0, 1.0],
            [20.0, 2.0],
            [30.0, 3.0],
        ],
        dtype=mx.float32,
    )
    map_rows = mx.array(
        [
            [0, 2, 4],
            [1, 0, 5],
            [1, 2, 6],
        ],
        dtype=mx.int32,
    )

    gathered = gather_sparse_features(source_features, map_rows)

    assert gathered.shape == (3, 2)
    assert gathered.dtype == mx.float32
    assert gathered.tolist() == [[30.0, 3.0], [10.0, 1.0], [30.0, 3.0]]


def test_scatter_sparse_features_accumulates_duplicate_targets():
    row_features = mx.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=mx.float32,
    )
    map_rows = mx.array(
        [
            [0, 1, 0],
            [2, 0, 1],
            [0, 2, 2],
            [2, 1, 3],
        ],
        dtype=mx.int32,
    )

    scattered = scatter_sparse_features(row_features, map_rows, target_count=4)

    assert scattered.shape == (4, 2)
    assert scattered.dtype == mx.float32
    assert scattered.tolist() == [
        [4.0, 40.0],
        [0.0, 0.0],
        [6.0, 60.0],
        [0.0, 0.0],
    ]


def test_empty_maps_preserve_feature_channels():
    source_features = mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32)
    row_features = mx.array([], dtype=mx.float32).reshape((0, 3))
    map_rows = mx.array([], dtype=mx.int32).reshape((0, 3))

    gathered = gather_sparse_features(source_features, map_rows)
    scattered = scatter_sparse_features(row_features, map_rows, target_count=2)

    assert gathered.shape == (0, 3)
    assert gathered.tolist() == []
    assert scattered.shape == (2, 3)
    assert scattered.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_sparse_feature_helpers_reject_invalid_map_rows():
    features = mx.array([[1.0, 2.0]], dtype=mx.float32)
    row_features = mx.array([[1.0, 2.0]], dtype=mx.float32)

    invalid_maps = (
        mx.array([0, 0, 0], dtype=mx.int32),
        mx.array([[0, 0]], dtype=mx.int32),
        mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32),
    )
    for map_rows in invalid_maps:
        try:
            gather_sparse_features(features, map_rows)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid gather map_rows should be rejected")

        try:
            scatter_sparse_features(row_features, map_rows, target_count=1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid scatter map_rows should be rejected")


def test_sparse_feature_helpers_reject_invalid_feature_shapes():
    map_rows = mx.array([[0, 0, 0]], dtype=mx.int32)

    try:
        gather_sparse_features(mx.array([1.0, 2.0], dtype=mx.float32), map_rows)
    except ValueError:
        pass
    else:
        raise AssertionError("rank-1 source_features should be rejected")

    try:
        scatter_sparse_features(mx.array([1.0, 2.0], dtype=mx.float32), map_rows, target_count=1)
    except ValueError:
        pass
    else:
        raise AssertionError("rank-1 row_features should be rejected")

    try:
        scatter_sparse_features(mx.array([[1.0], [2.0]], dtype=mx.float32), map_rows, target_count=1)
    except ValueError:
        pass
    else:
        raise AssertionError("row count mismatch should be rejected")


def test_sparse_feature_helpers_reject_out_of_bounds_indices():
    source_features = mx.array([[1.0]], dtype=mx.float32)
    row_features = mx.array([[1.0]], dtype=mx.float32)

    for map_rows in (
        mx.array([[0, -1, 0]], dtype=mx.int32),
        mx.array([[0, 1, 0]], dtype=mx.int32),
    ):
        try:
            gather_sparse_features(source_features, map_rows)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds source_index should be rejected")

    for map_rows, target_count in (
        (mx.array([[-1, 0, 0]], dtype=mx.int32), 1),
        (mx.array([[1, 0, 0]], dtype=mx.int32), 1),
    ):
        try:
            scatter_sparse_features(row_features, map_rows, target_count=target_count)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-bounds target_index should be rejected")

    try:
        scatter_sparse_features(row_features, mx.array([[0, 0, 0]], dtype=mx.int32), target_count=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative target_count should be rejected")
