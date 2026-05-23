import numpy as np
from PIL import Image

from mlx_spatial.sam3d_preprocess import (
    _normalize_pointmap_object_centric_ssi,
    _resize_float_image,
    _resize_pointmap,
    load_sam3d_mask,
    preprocess_sam3d_image_mask,
    preprocess_sam3d_official_tensors,
)


def test_preprocess_sam3d_image_mask_embeds_mask_as_alpha(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8), mode="RGB").save(image_path)
    Image.fromarray(np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8), mode="L").save(mask_path)

    result = preprocess_sam3d_image_mask(image_path, mask_path)

    assert result.rgba.shape == (2, 3, 4)
    assert result.size == (3, 2)
    assert result.foreground_pixels == 3
    assert result.rgba[..., :3].mean() == 127
    assert result.rgba[..., 3].tolist() == [[0, 255, 0], [255, 255, 0]]


def test_load_sam3d_mask_uses_last_channel_for_image_like_masks(tmp_path):
    mask_path = tmp_path / "mask.png"
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = np.array([[0, 1], [255, 0]], dtype=np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(mask_path)

    mask = load_sam3d_mask(mask_path)

    assert mask.tolist() == [[False, True], [True, False]]


def test_preprocess_sam3d_image_mask_rejects_size_mismatch(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(image_path)
    Image.fromarray(np.zeros((3, 2), dtype=np.uint8), mode="L").save(mask_path)

    try:
        preprocess_sam3d_image_mask(image_path, mask_path)
    except ValueError as error:
        assert "does not match image size" in str(error)
    else:
        raise AssertionError("expected mask size mismatch to fail")


def test_preprocess_sam3d_official_tensors_crops_pads_and_resizes_pointmap():
    rgba = np.zeros((4, 6, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[1:3, 2:5, 3] = 255
    yy, xx = np.meshgrid(np.arange(4, dtype=np.float32), np.arange(6, dtype=np.float32), indexing="ij")
    pointmap = np.stack((xx, yy, xx + yy), axis=-1)

    result = preprocess_sam3d_official_tensors(rgba, pointmap=pointmap, output_size=8, crop_box_size_factor=1.0)

    assert result.image.shape == (3, 8, 8)
    assert result.mask.shape == (1, 8, 8)
    assert result.rgb_image.shape == (3, 8, 8)
    assert result.rgb_image_mask.shape == (1, 8, 8)
    assert result.pointmap is not None
    assert result.pointmap.shape == (3, 8, 8)
    assert result.rgb_pointmap is not None
    assert result.rgb_pointmap.shape == (3, 8, 8)
    assert result.pointmap_scale is not None
    assert result.pointmap_shift is not None
    assert result.output_size == 8
    assert result.crop_box == (2, 0, 4, 2)
    assert result.pointmap_scale.shape == (3,)
    assert result.pointmap_shift.shape == (3,)
    assert float(result.mask.sum()) > 0.0
    assert np.isfinite(result.pointmap[:, result.mask[0] > 0]).all()


def test_preprocess_sam3d_full_pointmap_padding_matches_official_zero_fill():
    rgba = np.zeros((2, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[:, 1:3, 3] = 255
    pointmap = np.ones((2, 4, 3), dtype=np.float32)

    result = preprocess_sam3d_official_tensors(rgba, pointmap=pointmap, output_size=4, crop_box_size_factor=1.0)

    assert result.rgb_pointmap is not None
    assert np.isfinite(result.rgb_pointmap).all()
    assert np.any(np.isclose(result.rgb_pointmap, 0.0))


def test_preprocess_sam3d_cropped_pointmap_out_of_image_fill_matches_official_zero():
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 128
    rgba[:3, :3, 3] = 255

    yy, xx = np.mgrid[:4, :4]
    pointmap = np.stack(
        [xx.astype(np.float32), yy.astype(np.float32), (xx + yy).astype(np.float32)],
        axis=-1,
    )

    result = preprocess_sam3d_official_tensors(
        rgba,
        pointmap=pointmap,
        output_size=6,
        crop_box_size_factor=3.0,
    )

    assert result.crop_box == (-2, -2, 4, 4)
    assert result.pointmap is not None
    assert np.isfinite(result.pointmap[:, 0, 0]).all()
    assert np.allclose(result.pointmap[:, 0, 0], 0.0)


def test_object_centric_scene_scale_ignores_partially_nan_points_like_torch():
    pointmap = np.array(
        [
            [[0.0, 0.0, 1.0], [100.0, np.nan, np.nan]],
            [[2.0, 0.0, 1.0], [4.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    mask = np.array([[True, False], [True, True]])

    _normalized, scale, shift = _normalize_pointmap_object_centric_ssi(pointmap, mask)

    np.testing.assert_allclose(shift, np.array([2.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(scale, np.array([2.0, 2.0, 2.0], dtype=np.float32))


def test_resize_pointmap_nearest_uses_torch_floor_mapping():
    pointmap = np.arange(6, dtype=np.float32).reshape(1, 6, 1)

    resized = _resize_pointmap(pointmap, 4, order=0)

    assert resized[0, :, 0].tolist() == [0.0, 1.0, 3.0, 4.0]


def test_resize_float_image_nearest_uses_torch_floor_mapping_for_masks():
    mask = np.arange(6, dtype=np.float32).reshape(1, 6, 1)

    resized = _resize_float_image(mask, 4, resample=Image.Resampling.NEAREST)

    assert resized[0, :, 0].tolist() == [0.0, 1.0, 3.0, 4.0]


def test_resize_float_image_preserves_float_precision_without_uint8_quantization():
    image = np.array(
        [
            [[0.001, 0.111, 0.222], [0.333, 0.444, 0.555]],
            [[0.666, 0.777, 0.888], [0.999, 0.123, 0.234]],
        ],
        dtype=np.float32,
    )

    resized = _resize_float_image(image, 2, resample=Image.Resampling.BILINEAR)

    np.testing.assert_allclose(resized, image, rtol=0, atol=1e-7)


def test_preprocess_sam3d_official_tensors_rejects_empty_mask():
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)

    try:
        preprocess_sam3d_official_tensors(rgba)
    except ValueError as error:
        assert "non-empty mask" in str(error)
    else:
        raise AssertionError("expected empty SAM3D mask to fail")
