import numpy as np
import mlx.core as mx
from PIL import Image

from mlx_spatial.hyworld2_preprocess import (
    HYWORLD2_DEFAULT_MEMORY_PROFILE,
    HYWORLD2_OFFICIAL_TARGET_SIZE,
    adaptive_hyworld2_target_size,
    discover_hyworld2_images,
    memory_profile_config,
    preprocess_hyworld2_images,
    resize_crop_hyworld2_image,
)


def _write_rgb(path, size=(640, 480), color=(20, 40, 80)):
    Image.new("RGB", size, color).save(path)


def test_memory_profiles_resolve_target_and_frame_limits():
    assert HYWORLD2_DEFAULT_MEMORY_PROFILE == "large"
    balanced = memory_profile_config("balanced")
    large = memory_profile_config("large")

    assert balanced.target_size == 518
    assert balanced.max_frames == 8
    assert large.target_size == HYWORLD2_OFFICIAL_TARGET_SIZE
    assert large.max_frames == 32


def test_discover_hyworld2_images_sorts_supported_images_and_limits_frames(tmp_path):
    _write_rgb(tmp_path / "b.png")
    _write_rgb(tmp_path / "a.webp")
    (tmp_path / "ignored.txt").write_text("x", encoding="utf-8")

    images = discover_hyworld2_images(tmp_path, max_frames=1)

    assert [path.name for path in images] == ["a.webp"]


def test_adaptive_target_size_uses_first_image_longest_edge_and_patch_multiple(tmp_path):
    image_path = tmp_path / "image.png"
    _write_rgb(image_path, size=(621, 480))

    assert adaptive_hyworld2_target_size((image_path,), 518) == 518
    assert adaptive_hyworld2_target_size((image_path,), 1000) == 616


def test_resize_crop_hyworld2_image_returns_patch_multiple_size():
    image = Image.new("RGB", (640, 480), (1, 2, 3))

    resized = resize_crop_hyworld2_image(image, 518)

    assert resized.size == (518, 392)
    assert resized.width % 14 == 0
    assert resized.height % 14 == 0


def test_preprocess_hyworld2_images_returns_mlx_batch_and_trace_metadata(tmp_path):
    _write_rgb(tmp_path / "a.png", size=(640, 480), color=(10, 20, 30))
    _write_rgb(tmp_path / "b.png", size=(640, 480), color=(40, 50, 60))

    preprocessed = preprocess_hyworld2_images(tmp_path, memory_profile="balanced")

    assert preprocessed.tensor.shape == (1, 2, 3, 392, 518)
    assert preprocessed.tensor.dtype == mx.float32
    assert preprocessed.processed_size == (392, 518)
    assert preprocessed.patch_grid == (28, 37)
    assert preprocessed.token_count == 2 * 28 * 37
    assert preprocessed.original_sizes == ((640, 480), (640, 480))
    values = np.array(preprocessed.tensor)
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_preprocess_hyworld2_images_composites_rgba_over_white(tmp_path):
    rgba = Image.new("RGBA", (28, 28), (0, 0, 0, 0))
    rgba.save(tmp_path / "transparent.png")

    preprocessed = preprocess_hyworld2_images(tmp_path, memory_profile="balanced")

    assert float(np.array(preprocessed.tensor).mean()) == 1.0


def test_preprocess_hyworld2_images_rejects_mixed_processed_shapes(tmp_path):
    _write_rgb(tmp_path / "a.png", size=(640, 480))
    _write_rgb(tmp_path / "b.png", size=(480, 640))

    try:
        preprocess_hyworld2_images(tmp_path, memory_profile="balanced")
    except ValueError as error:
        assert "share processed H,W" in str(error)
    else:
        raise AssertionError("mixed processed HY-World sizes should be rejected")
