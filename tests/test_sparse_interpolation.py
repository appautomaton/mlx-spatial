import mlx.core as mx
import numpy as np

import mlx_spatial
from mlx_spatial.sparse_conv import sparse_nearest_interpolate, sparse_trilinear_interpolate


def test_sparse_nearest_interpolate_uses_rounded_sparse_grid_lookup():
    coords = mx.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
        dtype=mx.int32,
    )
    feats = mx.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=mx.float32)
    queries = mx.array(
        [
            [0.1, 0.1, 0.2],
            [0.0, 0.0, 0.7],
            [1.2, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=mx.float32,
    )

    values, valid = sparse_nearest_interpolate(queries, coords, feats, shape=(2, 2, 2), return_valid_mask=True)

    assert values.tolist() == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [0.0, 0.0]]
    assert valid.tolist() == [True, True, True, False]


def test_sparse_trilinear_interpolate_matches_dense_reference():
    rows = []
    feats = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                rows.append([z, y, x])
                feats.append([float(z + y * 2 + x * 4)])
    coords = mx.array(rows, dtype=mx.int32)
    features = mx.array(feats, dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(queries, coords, features, shape=(2, 2, 2), return_valid_mask=True)

    np.testing.assert_allclose(np.array(values)[:, 0], np.array([0.0, 3.5], dtype=np.float32))
    assert valid.tolist() == [True, True]


def test_sparse_trilinear_interpolate_reports_missing_required_corners():
    coords = mx.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=mx.int32,
    )
    features = mx.ones((4, 1), dtype=mx.float32)
    queries = mx.array([[0.5, 0.5, 0.5]], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(queries, coords, features, shape=(2, 2, 2), return_valid_mask=True)

    assert values.tolist() == [[0.5]]
    assert valid.tolist() == [True]
    try:
        sparse_trilinear_interpolate(queries, coords, features, shape=(2, 2, 2), require_all_corners=True)
    except ValueError as error:
        assert "requires all non-zero interpolation corners" in str(error)
    else:
        raise AssertionError("missing trilinear corners should be rejected when required")


def test_sparse_interpolation_supports_batched_coordinates():
    coords = mx.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
        ],
        dtype=mx.int32,
    )
    features = mx.array([[1.0], [10.0], [20.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.7]], dtype=mx.float32)

    values = sparse_nearest_interpolate(queries, coords, features, shape=(1, 1, 2))

    assert values.tolist() == [[1.0], [20.0]]


def test_sparse_interpolation_rejects_invalid_inputs():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[1.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)

    for bad_coords in (
        mx.array([0, 0, 0], dtype=mx.int32),
        mx.array([[0, 0]], dtype=mx.int32),
        mx.array([[0, 0, 0], [0, 0, 0]], dtype=mx.int32),
    ):
        try:
            sparse_nearest_interpolate(queries, bad_coords, features)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid source coordinates should be rejected")

    try:
        sparse_nearest_interpolate(mx.array([[0.0, 0.0]], dtype=mx.float32), coords, features)
    except ValueError:
        pass
    else:
        raise AssertionError("query coordinate rank mismatch should be rejected")

    try:
        sparse_nearest_interpolate(queries, coords, mx.array([1.0], dtype=mx.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("rank-1 source features should be rejected")

    try:
        sparse_trilinear_interpolate(queries, coords, features, shape=(2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid interpolation shape should be rejected")


def test_sparse_interpolation_exports_from_package_root():
    assert mlx_spatial.sparse_nearest_interpolate is sparse_nearest_interpolate
    assert mlx_spatial.sparse_trilinear_interpolate is sparse_trilinear_interpolate
