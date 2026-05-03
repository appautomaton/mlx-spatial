import mlx.core as mx

import mlx_spatial


def test_package_exports_regular_grid():
    assert mlx_spatial.regular_grid is not None


def test_regular_grid_returns_mlx_array_with_expected_shape_and_values():
    grid = mlx_spatial.regular_grid((2, 3))

    assert isinstance(grid, mx.array)
    assert grid.shape == (2, 3)
    assert grid.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_regular_grid_rejects_empty_or_non_positive_shape():
    for shape in ((), (0,), (2, -1)):
        try:
            mlx_spatial.regular_grid(shape)
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape {shape} should be rejected")
