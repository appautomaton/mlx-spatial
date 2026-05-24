import numpy as np
import pytest
import mlx.core as mx

from mlx_spatial.gs_rasterize import (
    GaussianRasterizeResult,
    GaussianSplatRenderer,
    rasterize_gaussians,
    rasterize_gaussians_cpu_reference,
)


def _camera(width: int = 9, height: int = 9) -> dict[str, np.ndarray]:
    return {
        "view_matrix": np.eye(4, dtype=np.float32),
        "intrinsics": np.array(
            [
                [8.0, 0.0, width / 2.0],
                [0.0, 8.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }


def _base_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    quats = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    scales = np.array([[0.18, 0.18, 0.18]], dtype=np.float32)
    opacities = np.array([0.8], dtype=np.float32)
    colors = np.array([[1.0, 0.25, 0.0]], dtype=np.float32)
    return means, quats, scales, opacities, colors


def test_rasterize_gaussians_shape_dtype_and_metadata():
    means, quats, scales, opacities, colors = _base_inputs()
    result = rasterize_gaussians(
        means,
        quats,
        scales,
        opacities,
        colors,
        _camera(),
        (9, 9),
        use_metal=False,
    )

    assert isinstance(result, GaussianRasterizeResult)
    assert result.rgba.shape == (9, 9, 4)
    assert result.depth.shape == (9, 9)
    assert result.rgba.dtype == mx.float32
    assert result.depth.dtype == mx.float32
    assert result.pixel_count > 0
    assert result.metadata["backend"] == "cpu"
    assert result.metadata["gaussian_count"] == 1
    assert result.metadata["visible_gaussian_count"] == 1
    assert result.metadata["pixel_count"] == result.pixel_count
    assert result.metadata["footprint_model"] == "anisotropic projected 3D covariance"
    assert result.metadata["quaternion_convention"] == "XYZW scalar-last"
    assert result.metadata["quaternion_rotation_applied"] is True


def test_projection_intrinsics_centered_gaussian_lands_in_center_pixels():
    means, quats, scales, opacities, colors = _base_inputs()
    result = rasterize_gaussians_cpu_reference(
        means,
        quats,
        scales,
        opacities,
        colors,
        _camera(),
        (9, 9),
    )
    alpha = np.asarray(result.rgba)[..., 3]
    peak_y, peak_x = np.unravel_index(np.argmax(alpha), alpha.shape)

    assert (peak_y, peak_x) in {(4, 4), (4, 5), (5, 4), (5, 5)}
    assert alpha[4:6, 4:6].max() == pytest.approx(alpha.max())
    assert np.asarray(result.depth)[peak_y, peak_x] == pytest.approx(2.0, abs=1e-5)


def test_rgba_contract_returns_straight_rgb_not_premultiplied():
    means, quats, scales, opacities, colors = _base_inputs()
    result = rasterize_gaussians_cpu_reference(
        means,
        quats,
        scales,
        opacities,
        colors,
        _camera(),
        (9, 9),
    )
    center = np.asarray(result.rgba)[4, 4]

    assert 0.0 < center[3] < 1.0
    np.testing.assert_allclose(center[:3], colors[0], rtol=0.0, atol=1e-6)


def test_alpha_compositing_depth_order_is_input_order_independent():
    means = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    quats = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    scales = np.array([[0.16, 0.16, 0.16], [0.16, 0.16, 0.16]], dtype=np.float32)
    opacities = np.array([0.7, 0.7], dtype=np.float32)
    colors = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    first = rasterize_gaussians_cpu_reference(means, quats, scales, opacities, colors, _camera(), (9, 9))
    second = rasterize_gaussians_cpu_reference(
        means[::-1],
        quats[::-1],
        scales[::-1],
        opacities[::-1],
        colors[::-1],
        _camera(),
        (9, 9),
    )

    np.testing.assert_allclose(np.asarray(first.rgba), np.asarray(second.rgba), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(first.depth), np.asarray(second.depth), rtol=0.0, atol=1e-6)
    center = np.asarray(first.rgba)[4, 4]
    assert center[0] > center[2]
    assert np.asarray(first.depth)[4, 4] < 2.5


def test_degree_zero_sh_features_convert_to_rgb():
    means, quats, scales, opacities, _ = _base_inputs()
    sh_dc = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    result = rasterize_gaussians_cpu_reference(
        means,
        quats,
        scales,
        opacities,
        sh_dc,
        _camera(),
        (9, 9),
    )
    center = np.asarray(result.rgba)[4, 4]
    assert center[0] == pytest.approx(center[1])
    assert center[1] == pytest.approx(center[2])
    assert center[0] > 0.0


def test_higher_degree_sh_evaluates_via_eval_sh():
    means, quats, scales, opacities, _ = _base_inputs()
    sh_coeffs = np.zeros((1, 4, 3), dtype=np.float32)
    sh_coeffs[:, 0, :] = 0.5

    result = rasterize_gaussians_cpu_reference(
        means,
        quats,
        scales,
        opacities,
        sh_coeffs,
        _camera(),
        (9, 9),
        sh_degree=1,
    )
    center = np.asarray(result.rgba)[4, 4]
    assert center[0] == pytest.approx(center[1])
    assert center[1] == pytest.approx(center[2])
    assert center[0] > 0.0


def test_anisotropic_scale_and_quaternion_rotate_screen_footprint():
    means = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    quats_identity = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    angle = np.pi / 2.0
    quats_rotated = np.array([[0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]], dtype=np.float32)
    scales = np.array([[0.32, 0.06, 0.06]], dtype=np.float32)
    opacities = np.array([0.9], dtype=np.float32)
    colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    camera = _camera(width=17, height=17)

    horizontal = rasterize_gaussians_cpu_reference(
        means, quats_identity, scales, opacities, colors, camera, (17, 17), min_alpha=0.0
    )
    vertical = rasterize_gaussians_cpu_reference(
        means, quats_rotated, scales, opacities, colors, camera, (17, 17), min_alpha=0.0
    )

    h_var_x, h_var_y = _alpha_variance(np.asarray(horizontal.rgba)[..., 3])
    v_var_x, v_var_y = _alpha_variance(np.asarray(vertical.rgba)[..., 3])

    assert h_var_x > h_var_y * 4.0
    assert v_var_y > v_var_x * 4.0
    assert np.asarray(horizontal.rgba)[8, 11, 3] > np.asarray(horizontal.rgba)[11, 8, 3]
    assert np.asarray(vertical.rgba)[11, 8, 3] > np.asarray(vertical.rgba)[8, 11, 3]


@pytest.mark.heavy
def test_metal_matches_cpu_reference_for_tiny_image():
    means = np.array(
        [
            [-0.08, 0.0, 2.0],
            [0.10, 0.0, 2.4],
            [0.0, 0.08, 2.2],
        ],
        dtype=np.float32,
    )
    quats = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (3, 1))
    scales = np.full((3, 3), 0.14, dtype=np.float32)
    opacities = np.array([0.55, 0.45, 0.35], dtype=np.float32)
    colors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    cpu = rasterize_gaussians_cpu_reference(means, quats, scales, opacities, colors, _camera(), (8, 8))
    renderer = GaussianSplatRenderer(use_metal=True, allow_cpu_fallback=False)
    try:
        metal = rasterize_gaussians(
            means,
            quats,
            scales,
            opacities,
            colors,
            _camera(),
            (8, 8),
            renderer=renderer,
        )
    except Exception as exc:
        pytest.skip(f"MLX custom Metal kernel unavailable in this environment: {exc}")

    assert metal.metadata["backend"] == "metal"
    np.testing.assert_allclose(np.asarray(metal.rgba), np.asarray(cpu.rgba), rtol=0.01, atol=0.01)
    np.testing.assert_allclose(np.asarray(metal.depth), np.asarray(cpu.depth), rtol=0.01, atol=0.01)


@pytest.mark.heavy
def test_metal_matches_cpu_reference_for_anisotropic_rotated_gaussian():
    means = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)
    angle = np.pi / 4.0
    quats = np.array([[0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]], dtype=np.float32)
    scales = np.array([[0.30, 0.08, 0.05]], dtype=np.float32)
    opacities = np.array([0.75], dtype=np.float32)
    colors = np.array([[0.2, 0.7, 1.0]], dtype=np.float32)
    camera = _camera(width=12, height=12)

    cpu = rasterize_gaussians_cpu_reference(means, quats, scales, opacities, colors, camera, (12, 12))
    renderer = GaussianSplatRenderer(use_metal=True, allow_cpu_fallback=False)
    try:
        metal = rasterize_gaussians(
            means,
            quats,
            scales,
            opacities,
            colors,
            camera,
            (12, 12),
            renderer=renderer,
        )
    except Exception as exc:
        pytest.skip(f"MLX custom Metal kernel unavailable in this environment: {exc}")

    assert metal.metadata["backend"] == "metal"
    np.testing.assert_allclose(np.asarray(metal.rgba), np.asarray(cpu.rgba), rtol=0.01, atol=0.01)
    np.testing.assert_allclose(np.asarray(metal.depth), np.asarray(cpu.depth), rtol=0.01, atol=0.01)


def _alpha_variance(alpha: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0 : alpha.shape[0], 0 : alpha.shape[1]]
    total = float(alpha.sum())
    mean_x = float((alpha * xx).sum() / total)
    mean_y = float((alpha * yy).sum() / total)
    var_x = float((alpha * (xx - mean_x) ** 2).sum() / total)
    var_y = float((alpha * (yy - mean_y) ** 2).sum() / total)
    return var_x, var_y
