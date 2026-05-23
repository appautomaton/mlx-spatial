import mlx.core as mx
import numpy as np

import mlx_spatial
from mlx_spatial.spatial_interp import sparse_nearest_interpolate, sparse_trilinear_interpolate


# ---------------------------------------------------------------------------
# Dense-grid reference helpers
# ---------------------------------------------------------------------------

def _dense_nearest(values_3d: np.ndarray, query: np.ndarray) -> float:
    """Reference nearest-neighbour on a dense 3D grid."""
    ix = int(np.floor(query[0] + 0.5))
    iy = int(np.floor(query[1] + 0.5))
    iz = int(np.floor(query[2] + 0.5))
    shape = values_3d.shape
    if 0 <= ix < shape[0] and 0 <= iy < shape[1] and 0 <= iz < shape[2]:
        return values_3d[ix, iy, iz]
    return 0.0


def _dense_trilinear(values_3d: np.ndarray, query: np.ndarray) -> float:
    """Reference trilinear interpolation on a dense 3D grid."""
    x, y, z = float(query[0]), float(query[1]), float(query[2])
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    z0 = int(np.floor(z))
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1
    shape = values_3d.shape

    def _get(i: int, j: int, k: int) -> float:
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
            return float(values_3d[i, j, k])
        return 0.0

    xd = x - x0
    yd = y - y0
    zd = z - z0

    c000 = _get(x0, y0, z0) * (1 - xd) * (1 - yd) * (1 - zd)
    c001 = _get(x0, y0, z1) * (1 - xd) * (1 - yd) * zd
    c010 = _get(x0, y1, z0) * (1 - xd) * yd * (1 - zd)
    c011 = _get(x0, y1, z1) * (1 - xd) * yd * zd
    c100 = _get(x1, y0, z0) * xd * (1 - yd) * (1 - zd)
    c101 = _get(x1, y0, z1) * xd * (1 - yd) * zd
    c110 = _get(x1, y1, z0) * xd * yd * (1 - zd)
    c111 = _get(x1, y1, z1) * xd * yd * zd
    return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111


def _dense_grid_from_sparse(coords: np.ndarray, features: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Build a dense 3D grid from sparse voxel data for reference comparison."""
    grid = np.zeros((*shape, features.shape[1]), dtype=np.float32)
    for idx in range(coords.shape[0]):
        c = tuple(coords[idx])
        grid[c] = features[idx]
    return grid


def _sparse_has_coord(coords: np.ndarray, target: np.ndarray) -> bool:
    return any(np.array_equal(c, target) for c in coords)


# ---------------------------------------------------------------------------
# nearest_interpolate tests
# ---------------------------------------------------------------------------

def test_nearest_rounded_coordinates():
    """Nearest interpolate rounds coordinates toward nearest integer grid point."""
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int64)
    features = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    queries = mx.array([
        [0.1, 0.1, 0.0],
        [0.6, 0.0, 0.0],
        [0.2, 0.8, 0.0],
    ], dtype=mx.float32)

    values = sparse_nearest_interpolate(
        queries, mx.array(coords), mx.array(features), shape=(2, 2, 1),
    )

    np.testing.assert_allclose(np.array(values), np.array([[1.0], [2.0], [3.0]], dtype=np.float32))


def test_nearest_missing_voxel_returns_zero():
    """Queries to non-existent sparse voxels return zero."""
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[5.0, 5.0]], dtype=mx.float32)
    queries = mx.array([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=mx.float32)

    values, valid = sparse_nearest_interpolate(
        queries, coords, features, shape=(2, 2, 2), return_valid_mask=True,
    )

    assert values.tolist() == [[5.0, 5.0], [0.0, 0.0]]
    assert valid.tolist() == [True, False]


def test_nearest_boundary_clamping():
    """Nearest interpolate respects shape bounds and marks out-of-bounds as invalid."""
    coords = mx.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
    ], dtype=mx.int32)
    features = mx.array([[10.0], [20.0], [30.0]], dtype=mx.float32)

    queries = mx.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.9, 1.9, 1.9],
        [2.0, 2.0, 2.0],
            [2.6, 2.6, 2.6],
    ], dtype=mx.float32)

    # With shape (3, 3, 3) the coordinate (2, 2, 2) is valid
    values, valid = sparse_nearest_interpolate(
        queries, coords, features, shape=(3, 3, 3), return_valid_mask=True,
    )

    assert values[0].item() == 10.0 and valid[0].item() is True
    assert values[1].item() == 10.0 and valid[1].item() is True
    assert values[2].item() == 30.0 and valid[2].item() is True
    assert values[3].item() == 30.0 and valid[3].item() is True
    assert values[4].item() == 0.0 and valid[4].item() is False


def test_nearest_single_channel_and_multi_channel():
    coords = mx.array([[0, 0, 0], [1, 1, 1]], dtype=mx.int32)
    single = mx.array([[3.0], [7.0]], dtype=mx.float32)
    multi = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)

    queries = mx.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=mx.float32)

    v1 = sparse_nearest_interpolate(queries, coords, single, shape=(2, 2, 2))
    v2 = sparse_nearest_interpolate(queries, coords, multi, shape=(2, 2, 2))

    assert v1.tolist() == [[3.0], [7.0]]
    assert v2.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_nearest_batched_coordinates():
    coords = mx.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=mx.int32)
    features = mx.array([[1.0], [2.0], [10.0]], dtype=mx.float32)

    queries = mx.array([
        [0, 0.0, 0.0, 0.0],
        [1, 0.0, 0.0, 0.0],
        [0, 0.0, 0.0, 0.5],
    ], dtype=mx.float32)

    values = sparse_nearest_interpolate(queries, coords, features, shape=(1, 1, 2))
    assert values.tolist() == [[1.0], [10.0], [2.0]]


# ---------------------------------------------------------------------------
# trilinear_interpolate tests
# ---------------------------------------------------------------------------

def test_trilinear_dense_full_grid_parity():
    """Trilinear interpolation matches dense-grid reference on a full cube."""
    shape = (2, 2, 2)
    coords_list = []
    feats_list = []
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                coords_list.append([z, y, x])
                feats_list.append([float(z * 4 + y * 2 + x)])
    coords = mx.array(coords_list, dtype=mx.int32)
    features = mx.array(feats_list, dtype=mx.float32)

    queries = mx.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [0.0, 0.5, 1.0],
        [0.25, 0.75, 0.25],
    ], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(
        queries, coords, features, shape=shape, return_valid_mask=True,
    )

    dense = _dense_grid_from_sparse(
        np.array(coords_list, dtype=np.int64),
        np.array(feats_list, dtype=np.float32),
        shape,
    )
    expected = np.array([[_dense_trilinear(dense[..., 0], q)] for q in np.array(queries)], dtype=np.float32)

    np.testing.assert_allclose(np.array(values), expected, atol=1e-6)
    assert valid.tolist() == [True, True, True, True, True]


def test_trilinear_missing_corners_zero_weighted():
    """Missing corners are skipped but contribute zero to result."""
    coords = mx.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=mx.int32)
    features = mx.ones((7, 1), dtype=mx.float32)

    queries = mx.array([
        [0.5, 0.5, 0.5],
    ], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(
        queries, coords, features, shape=(2, 2, 2), return_valid_mask=True,
    )

    assert valid.tolist() == [True]
    np.testing.assert_allclose(np.array(values), np.array([[0.875]], dtype=np.float32), atol=1e-6)


def test_trilinear_require_all_corners_enforcement():
    """require_all_corners=True raises when sparse data is incomplete."""
    coords = mx.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=mx.int32)
    features = mx.ones((4, 1), dtype=mx.float32)
    queries = mx.array([[0.5, 0.5, 0.5]], dtype=mx.float32)

    try:
        sparse_trilinear_interpolate(queries, coords, features, shape=(2, 2, 2), require_all_corners=True)
    except ValueError as err:
        assert "requires all non-zero interpolation corners" in str(err)
    else:
        raise AssertionError("missing corners should raise ValueError")


def test_trilinear_single_channel_and_multi_channel():
    coords = mx.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
    ], dtype=mx.int32)
    single = mx.ones((8, 1), dtype=mx.float32)
    multi = mx.array([
        [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8],
        [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.0, 0.0],
    ], dtype=mx.float32)

    queries = mx.array([[0.5, 0.5, 0.5]], dtype=mx.float32)

    v1 = sparse_trilinear_interpolate(queries, coords, single, shape=(2, 2, 2))
    v2 = sparse_trilinear_interpolate(queries, coords, multi, shape=(2, 2, 2))

    np.testing.assert_allclose(np.array(v1), np.array([[1.0]], dtype=np.float32), atol=1e-6)

    dense = _dense_grid_from_sparse(np.array(coords, dtype=np.int64), np.array(multi).astype(np.float32), (2, 2, 2))
    expected = np.array([
        [_dense_trilinear(dense[..., 0], [0.5, 0.5, 0.5]),
         _dense_trilinear(dense[..., 1], [0.5, 0.5, 0.5])],
    ], dtype=np.float32)
    np.testing.assert_allclose(np.array(v2), expected, atol=1e-6)


def test_trilinear_batched_coordinates():
    coords = mx.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
    ], dtype=mx.int32)
    features = mx.ones((16, 1), dtype=mx.float32)

    queries = mx.array([
        [0, 0.5, 0.5, 0.5],
        [1, 0.5, 0.5, 0.5],
    ], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(
        queries, coords, features, shape=(2, 2, 2), return_valid_mask=True,
    )

    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(np.array(values), np.array([[1.0], [1.0]], dtype=np.float32), atol=1e-6)


def test_trilinear_boundary_clamping():
    """Out-of-bounds corners are treated as missing; valid_mask reflects this."""
    coords = mx.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ], dtype=mx.int32)
    features = mx.ones((4, 1), dtype=mx.float32)

    for q in ([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 1.5, 0.5]):
        values, valid = sparse_trilinear_interpolate(
            mx.array([q], dtype=mx.float32), coords, features, shape=(2, 2, 2),
            return_valid_mask=True,
        )
        assert valid.tolist() == [True]


def test_trilinear_all_missing_corners():
    """A query whose all 8 corners are missing returns zero with invalid mask."""
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[42.0]], dtype=mx.float32)

    queries = mx.array([[1.5, 1.5, 1.5]], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(
        queries, coords, features, shape=(2, 2, 2), return_valid_mask=True,
    )

    assert values.tolist() == [[0.0]]
    assert valid.tolist() == [False]


def test_trilinear_non_uniform_shape():
    """Trilinear handles non-cubic grid shapes correctly."""
    coords_list = []
    feats_list = []
    shape = (3, 2, 4)
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                coords_list.append([z, y, x])
    coords = mx.array(coords_list, dtype=mx.int32)
    num_coords = len(coords_list)
    features = mx.array([[float(i)] for i in range(num_coords)], dtype=mx.float32)

    queries = mx.array([
        [1.5, 0.5, 1.5],
        [2.0, 1.0, 3.0],
        [0.0, 0.0, 0.0],
    ], dtype=mx.float32)

    values, valid = sparse_trilinear_interpolate(
        queries, coords, features, shape=shape, return_valid_mask=True,
    )

    dense = _dense_grid_from_sparse(np.array(coords_list, dtype=np.int64), np.array(features).astype(np.float32), shape)
    for i in range(3):
        q_np = np.array(queries)[i]
        if valid[i].item():
            expected = np.array([_dense_trilinear(dense[..., 0], q_np)], dtype=np.float32)
            np.testing.assert_allclose(np.array(values)[i], expected, atol=1e-6)

    assert valid.tolist() == [True, True, True]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

def test_rejects_invalid_source_coordinate_shape():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[1.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)

    for bad in (
        mx.array([0, 0, 0], dtype=mx.int32),
        mx.array([[0, 0]], dtype=mx.int32),
    ):
        try:
            sparse_nearest_interpolate(queries, bad, features)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject coords shape {bad.shape}")


def test_rejects_query_rank_mismatch():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[1.0]], dtype=mx.float32)
    try:
        sparse_nearest_interpolate(mx.array([[0.0, 0.0]], dtype=mx.float32), coords, features)
    except ValueError:
        pass
    else:
        raise AssertionError("rank mismatch should be rejected")


def test_rejects_feature_rank_mismatch():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)
    try:
        sparse_nearest_interpolate(queries, coords, mx.array([1.0], dtype=mx.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("rank-1 features should be rejected")


def test_trilinear_rejects_non_3d_spatial():
    coords = mx.array([[0, 0]], dtype=mx.int32)
    features = mx.array([[1.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0]], dtype=mx.float32)
    try:
        sparse_trilinear_interpolate(queries, coords, features)
    except ValueError:
        pass
    else:
        raise AssertionError("trilinear should require 3 spatial dims")


def test_rejects_invalid_shape():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[1.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)

    for bad_shape in ((2, 2), (2, 2, 0)):
        try:
            sparse_trilinear_interpolate(queries, coords, features, shape=bad_shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should reject shape {bad_shape}")


def test_rejects_out_of_bounds_source_coordinates():
    coords = mx.array([[0, 0, 0], [5, 5, 5]], dtype=mx.int32)
    features = mx.array([[1.0], [2.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)
    try:
        sparse_nearest_interpolate(queries, coords, features, shape=(2, 2, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-bounds source coordinates should be rejected")


def test_rejects_duplicate_source_coordinates():
    coords = mx.array([[0, 0, 0], [0, 0, 0]], dtype=mx.int32)
    features = mx.array([[1.0], [2.0]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)
    try:
        sparse_nearest_interpolate(queries, coords, features)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate coordinates should be rejected")


def test_rejects_empty_features():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[]], dtype=mx.float32)
    queries = mx.array([[0.0, 0.0, 0.0]], dtype=mx.float32)
    try:
        sparse_nearest_interpolate(queries, coords, features)
    except ValueError:
        pass
    else:
        raise AssertionError("empty feature channels should be rejected")


def test_rejects_non_finite_values():
    coords = mx.array([[0, 0, 0]], dtype=mx.int32)
    features = mx.array([[1.0]], dtype=mx.float32)
    for bad_queries in (
        mx.array([[np.inf, 0.0, 0.0]], dtype=mx.float32),
        mx.array([[np.nan, 0.0, 0.0]], dtype=mx.float32),
    ):
        try:
            sparse_nearest_interpolate(bad_queries, coords, features)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite queries should be rejected")


# ---------------------------------------------------------------------------
# Export / module tests
# ---------------------------------------------------------------------------

def test_functions_exported_from_package_root():
    assert mlx_spatial.sparse_nearest_interpolate is sparse_nearest_interpolate
    assert mlx_spatial.sparse_trilinear_interpolate is sparse_trilinear_interpolate


def test_functions_exported_from_sparse_conv():
    from mlx_spatial.sparse_conv import sparse_nearest_interpolate as _sc_near
    from mlx_spatial.sparse_conv import sparse_trilinear_interpolate as _sc_tril
    assert _sc_near is sparse_nearest_interpolate
    assert _sc_tril is sparse_trilinear_interpolate


# ---------------------------------------------------------------------------
# Large-scale parity / stress tests
# ---------------------------------------------------------------------------

def test_trilinear_large_grid_parity_random_queries():
    """Parity check: trilinear on a 4x4x4 dense grid with random queries."""
    shape = (4, 4, 4)
    coords_list = []
    feats_list = []
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                coords_list.append([z, y, x])
                feats_list.append([float(z * 16 + y * 4 + x)])
    coords = mx.array(coords_list, dtype=mx.int32)
    features = mx.array(feats_list, dtype=mx.float32)

    rng = np.random.default_rng(42)
    q = rng.uniform(0, 3.999, size=(50, 3)).astype(np.float32)
    queries = mx.array(q)

    values, valid = sparse_trilinear_interpolate(
        queries, coords, features, shape=shape, return_valid_mask=True,
    )

    dense = _dense_grid_from_sparse(
        np.array(coords_list, dtype=np.int64),
        np.array(feats_list, dtype=np.float32),
        shape,
    )
    for i in range(q.shape[0]):
        expected = _dense_trilinear(dense[..., 0], q[i])
        np.testing.assert_allclose(float(np.array(values)[i, 0]), expected, atol=1e-5, rtol=1e-5)
    assert all(valid)


def test_nearest_large_grid_parity_random_queries():
    """Parity check: nearest on a 4x4x4 dense grid with random queries."""
    shape = (4, 4, 4)
    coords_list = []
    feats_list = []
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                coords_list.append([z, y, x])
                feats_list.append([float(z * 16 + y * 4 + x)])
    coords = mx.array(coords_list, dtype=mx.int32)
    features = mx.array(feats_list, dtype=mx.float32)

    rng = np.random.default_rng(42)
    q = rng.uniform(-0.5, 4.5, size=(50, 3)).astype(np.float32)
    queries = mx.array(q)

    values, valid = sparse_nearest_interpolate(
        queries, coords, features, shape=shape, return_valid_mask=True,
    )

    dense = _dense_grid_from_sparse(
        np.array(coords_list, dtype=np.int64),
        np.array(feats_list, dtype=np.float32),
        shape,
    )
    for i in range(q.shape[0]):
        expected = _dense_nearest(dense[..., 0], q[i])
        if valid[i].item():
            np.testing.assert_allclose(float(np.array(values)[i, 0]), expected, atol=1e-6)
        else:
            assert float(np.array(values)[i, 0]) == 0.0


def test_no_shape_parameter_allows_any_valid_query():
    coords = mx.array([
        [100, 200, 300],
        [101, 201, 301],
    ], dtype=mx.int32)
    features = mx.array([[1.0], [2.0]], dtype=mx.float32)

    queries = mx.array([
        [100.0, 200.0, 300.0],
        [100.4, 200.6, 300.2],
    ], dtype=mx.float32)

    values, valid = sparse_nearest_interpolate(queries, coords, features, return_valid_mask=True)
    assert values.tolist() == [[1.0], [0.0]]
    assert valid.tolist() == [True, False]
