from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

import mlx_spatial
from mlx_spatial.mapanything_preprocess import (
    MAPANYTHING_DINOV2_IMAGE_MEAN,
    MAPANYTHING_DINOV2_IMAGE_STD,
    crop_resize_mapanything_image,
    discover_mapanything_images,
    find_mapanything_closest_aspect_ratio,
    mapanything_target_size,
    normalize_mapanything_image,
    preprocess_mapanything_images,
)


def _write_rgb(path, size=(640, 480), color=(20, 40, 80)):
    Image.new("RGB", size, color).save(path)


def test_find_closest_aspect_ratio_uses_vendored_518_mapping():
    assert find_mapanything_closest_aspect_ratio(4 / 3, 518) == (518, 392)
    assert find_mapanything_closest_aspect_ratio(1.0, 518) == (518, 518)
    assert find_mapanything_closest_aspect_ratio(0.75, 518) == (392, 518)


def test_mapanything_target_size_supports_vendored_resize_modes():
    assert mapanything_target_size(4 / 3, resize_mode="fixed_mapping") == (518, 392)
    assert mapanything_target_size(4 / 3, resize_mode="square", size=518) == (518, 518)
    assert mapanything_target_size(4 / 3, resize_mode="longest_side", size=518) == (518, 392)
    assert mapanything_target_size(4 / 3, resize_mode="fixed_size", size=(521, 401)) == (
        518,
        392,
    )


def test_discover_mapanything_images_sorts_supported_images_and_applies_stride_before_filter(
    tmp_path,
):
    _write_rgb(tmp_path / "b.png")
    (tmp_path / "c.txt").write_text("ignored", encoding="utf-8")
    _write_rgb(tmp_path / "a.jpg")

    images = discover_mapanything_images(tmp_path, stride=2)

    assert [path.name for path in images] == ["a.jpg"]


def test_discover_mapanything_images_rejects_empty_or_invalid_inputs(tmp_path):
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid images found"):
        discover_mapanything_images(tmp_path)
    with pytest.raises(FileNotFoundError):
        discover_mapanything_images(tmp_path / "missing")
    with pytest.raises(ValueError, match="stride"):
        discover_mapanything_images(tmp_path, stride=0)


def test_crop_resize_mapanything_image_matches_expected_4_3_bucket():
    image = Image.new("RGB", (640, 480), (1, 2, 3))

    processed = crop_resize_mapanything_image(image, (518, 392))

    assert processed.size == (518, 392)


def test_normalize_mapanything_image_uses_dinov2_constants():
    image = Image.new("RGB", (14, 14), (255, 255, 255))

    tensor = normalize_mapanything_image(image)

    assert tensor.shape == (1, 3, 14, 14)
    assert tensor.dtype == mx.float32
    values = np.array(tensor)
    expected = (1.0 - np.array(MAPANYTHING_DINOV2_IMAGE_MEAN)) / np.array(
        MAPANYTHING_DINOV2_IMAGE_STD
    )
    np.testing.assert_allclose(values[0, :, 0, 0], expected, rtol=1e-6, atol=1e-6)


def test_preprocess_mapanything_images_returns_mlx_views_and_metadata(tmp_path):
    _write_rgb(tmp_path / "b.png", size=(640, 480), color=(10, 20, 30))
    _write_rgb(tmp_path / "a.jpg", size=(640, 480), color=(40, 50, 60))

    preprocessed = preprocess_mapanything_images(tmp_path)

    assert preprocessed.frame_count == 2
    assert [path.name for path in preprocessed.image_paths] == ["a.jpg", "b.png"]
    assert preprocessed.target_size == (518, 392)
    assert preprocessed.average_aspect_ratio == 640 / 480
    assert preprocessed.patch_size == 14
    for index, view in enumerate(preprocessed.views):
        assert view.idx == index
        assert view.instance == str(index)
        assert view.data_norm_type == "dinov2"
        assert view.original_size == (640, 480)
        assert view.processed_size == (392, 518)
        assert view.true_shape == (392, 518)
        assert view.img.shape == (1, 3, 392, 518)
        assert view.img.dtype == mx.float32


def test_preprocess_mapanything_desk_images_use_official_4_3_bucket_when_present():
    desk = Path("inputs/map-anything/desk")
    if not desk.is_dir():
        pytest.skip(f"Desk inputs not present: {desk}")

    preprocessed = preprocess_mapanything_images(desk)

    assert preprocessed.frame_count == 2
    assert preprocessed.target_size == (518, 392)
    assert preprocessed.average_aspect_ratio == 4 / 3
    assert all(view.original_size == (4032, 3024) for view in preprocessed.views)
    assert all(view.processed_size == (392, 518) for view in preprocessed.views)
    assert all(view.img.shape == (1, 3, 392, 518) for view in preprocessed.views)


def test_preprocess_mapanything_helpers_are_public_exports():
    assert mlx_spatial.preprocess_mapanything_images is preprocess_mapanything_images
    assert mlx_spatial.MAPANYTHING_DINOV2_IMAGE_MEAN == MAPANYTHING_DINOV2_IMAGE_MEAN
