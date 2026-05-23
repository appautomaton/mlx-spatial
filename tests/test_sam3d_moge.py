import numpy as np
from PIL import Image

from mlx_spatial.sam3d_moge import (
    _moge_num_tokens,
    _moge_resized_size,
    _remap_moge_points_exp,
    _resize_hwc_float,
    normalized_view_plane_uv_numpy,
    recover_focal_shift_numpy,
)


def test_moge_num_tokens_matches_resolution_level_9_contract():
    assert _moge_num_tokens(resolution_level=9) == 2551


def test_moge_resized_size_preserves_patch_area_token_budget():
    height, width = 4480, 6720

    resized_height, resized_width = _moge_resized_size(height, width, num_tokens=2551)

    assert resized_height // 14 == 41
    assert resized_width // 14 == 61


def test_normalized_view_plane_uv_matches_moge_geometry_contract():
    uv = normalized_view_plane_uv_numpy(width=3, height=2, aspect_ratio=1.5)

    assert uv.shape == (2, 3, 2)
    assert np.allclose(uv[0, 0], [-0.5547002, -0.2773501], atol=1e-6)
    assert np.allclose(uv[-1, -1], [0.5547002, 0.2773501], atol=1e-6)


def test_remap_moge_points_exp_scales_xy_by_positive_z():
    raw = np.array([[[2.0, -3.0, 0.0], [1.0, 2.0, np.log(2.0)]]], dtype=np.float32)

    remapped = _remap_moge_points_exp(raw)

    assert np.allclose(remapped[0, 0], [2.0, -3.0, 1.0])
    assert np.allclose(remapped[0, 1], [2.0, 4.0, 2.0])


def test_resize_hwc_float_preserves_float_precision_without_uint8_quantization():
    image = np.array(
        [
            [[0.001, 0.111, 0.222], [0.333, 0.444, 0.555]],
            [[0.666, 0.777, 0.888], [0.999, 0.123, 0.234]],
        ],
        dtype=np.float32,
    )

    resized = _resize_hwc_float(image, (2, 2), Image.Resampling.BICUBIC)

    np.testing.assert_allclose(resized, image, rtol=0, atol=1e-7)


def test_recover_focal_shift_numpy_recovers_simple_projective_fixture():
    height, width = 16, 16
    focal = 1.25
    shift = 0.35
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    depth = 1.5 + rows + cols
    uv = normalized_view_plane_uv_numpy(width=width, height=height)
    points = np.zeros((height, width, 3), dtype=np.float32)
    points[..., :2] = uv * (depth[..., None] + shift) / focal
    points[..., 2] = depth
    mask = np.ones((height, width), dtype=bool)

    recovered_focal, recovered_shift = recover_focal_shift_numpy(points, mask)

    assert np.isclose(recovered_focal, focal, atol=1e-2)
    assert np.isclose(recovered_shift, shift, atol=1e-2)
