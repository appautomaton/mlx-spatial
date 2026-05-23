import numpy as np
import mlx.core as mx

from mlx_spatial.grid import (
    create_uv_grid,
    fourier_embeddings,
    fourier_embeddings_np,
    regular_grid,
)


def test_regular_grid_returns_dense_integer_grid():
    grid = regular_grid((2, 3))

    assert grid.shape == (2, 3)
    assert grid.dtype == mx.int32
    assert grid.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_create_uv_grid_normalized_generates_xy_centered_pattern():
    grid = create_uv_grid(3, 2)

    assert grid.shape == (2, 3, 2)
    assert grid.dtype == mx.float32
    result = np.array(grid)
    np.testing.assert_allclose(result[0, 0], [-1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result[0, -1], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result[-1, 0], [-1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(result[-1, -1], [1.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(result[0, 1], [0.0, 1.0], atol=1e-6)


def test_create_uv_grid_unnormalized_returns_pixel_coordinates():
    grid = create_uv_grid(3, 2, normalize=False)

    result = np.array(grid)
    np.testing.assert_allclose(result[0, 0], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[-1, -1], [2.0, 1.0], atol=1e-6)


def test_create_uv_grid_rejects_invalid_size():
    try:
        create_uv_grid(0, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for zero width")


def test_fourier_embeddings_2d_generates_sin_cos_pairs():
    positions = mx.array([[0.0, 0.0], [1.0, 0.5]], dtype=mx.float32)

    emb = fourier_embeddings(positions, embed_dim=8, omega_0=1.0)

    assert emb.shape == (2, 8)
    assert emb.dtype == mx.float32
    result = np.array(emb)
    assert np.all(np.isfinite(result))
    assert np.abs(result).sum() > 0


def test_fourier_embeddings_3d_positions():
    positions = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

    emb = fourier_embeddings(positions, embed_dim=12, omega_0=10.0)

    result = np.array(emb)
    assert result.shape == (2, 12)
    assert np.all(np.isfinite(result))


def test_fourier_embeddings_batch_shape_preserved():
    positions = np.ones((3, 4, 2), dtype=np.float32)

    emb = fourier_embeddings(positions, embed_dim=8, omega_0=5.0)

    result = np.array(emb)
    assert result.shape == (3, 4, 8)


def test_fourier_embeddings_rejects_bad_embed_dim():
    positions = np.zeros((1, 3), dtype=np.float32)

    try:
        fourier_embeddings(positions, embed_dim=7)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-divisible embed_dim")


def test_fourier_embeddings_np_matches_mlx_output():
    positions = np.array([[0.0, 0.0], [0.5, -0.5], [0.3, 0.7]], dtype=np.float32)

    mlx_emb = np.array(fourier_embeddings(positions, embed_dim=4, omega_0=2.0))
    np_emb = fourier_embeddings_np(positions, embed_dim=4, omega_0=2.0)

    np.testing.assert_allclose(mlx_emb, np_emb, rtol=1e-6, atol=1e-6)


def test_fourier_embeddings_np_round_trips_with_known_formula():
    positions = np.array([[0.25]], dtype=np.float32)

    emb = fourier_embeddings_np(positions, embed_dim=4, omega_0=1.0)

    assert emb.shape == (1, 4)
    freqs = np.arange(2, dtype=np.float32)
    freqs = freqs / 1.0
    freqs = np.pi * (1.0 ** freqs)
    expected_sin = np.sin(0.25 * freqs)
    expected_cos = np.cos(0.25 * freqs)
    expected = np.concatenate([expected_sin, expected_cos]).astype(np.float32)
    np.testing.assert_allclose(emb[0], expected, atol=1e-6)


def test_fourier_embeddings_np_rejects_scalar_input():
    try:
        fourier_embeddings_np(np.float32(1.0), embed_dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for scalar positions")
