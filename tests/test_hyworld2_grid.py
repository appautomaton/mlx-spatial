import mlx.core as mx
import numpy as np

from mlx_spatial.hyworld2_grid import (
    create_hyworld2_uv_grid,
    hyworld2_patch_rope_positions,
    hyworld2_position_grid_to_embed,
)
from mlx_spatial.hyworld2_worldmirror import build_worldmirror_rope_positions


def test_grid_create_uv_grid_matches_vendor_reference_corners():
    grid = create_hyworld2_uv_grid(width=4, height=2, aspect_ratio=2.0)
    values = np.array(grid)
    diag = np.sqrt(5.0)
    span_x = 2.0 / diag
    span_y = 1.0 / diag

    assert values.shape == (2, 4, 2)
    np.testing.assert_allclose(values[0, 0], [-span_x * 3.0 / 4.0, -span_y * 1.0 / 2.0], atol=1e-6)
    np.testing.assert_allclose(values[-1, -1], [span_x * 3.0 / 4.0, span_y * 1.0 / 2.0], atol=1e-6)


def test_grid_position_embed_matches_vendor_sinusoidal_formula():
    grid = mx.array(
        [
            [[0.0, 0.0], [1.0, 0.5]],
            [[-1.0, -0.5], [0.25, -0.25]],
        ],
        dtype=mx.float32,
    )

    embed = hyworld2_position_grid_to_embed(grid, embed_dim=8, omega_0=100.0)

    omega = np.array([1.0, 0.1], dtype=np.float32)
    flat = np.array(grid).reshape(-1, 2)
    out_x = flat[:, 0:1] * omega[None, :]
    out_y = flat[:, 1:2] * omega[None, :]
    expected = np.concatenate([np.sin(out_x), np.cos(out_x), np.sin(out_y), np.cos(out_y)], axis=1)
    np.testing.assert_allclose(np.array(embed).reshape(-1, 8), expected, atol=1e-6)


def test_grid_patch_rope_positions_match_worldmirror_builder():
    expected = build_worldmirror_rope_positions(
        frames=2,
        patch_grid=(2, 3),
        patch_start_idx=2,
        normalized=False,
    )

    actual = hyworld2_patch_rope_positions(frames=2, patch_grid=(2, 3), patch_start_idx=2)

    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-6)
