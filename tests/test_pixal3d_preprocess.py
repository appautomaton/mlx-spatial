import numpy as np
from PIL import Image

from mlx_spatial.pixal3d_preprocess import preprocess_pixal3d_image


def test_pixal3d_preprocess_uses_rgba_alpha_and_black_composite(tmp_path):
    image_path = tmp_path / "rgba.png"
    rgba = np.zeros((24, 32, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[6:18, 8:24, :3] = (200, 100, 50)
    rgba[6:18, 8:24, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(image_path)

    result = preprocess_pixal3d_image(image_path)

    assert result.ready
    assert result.image is not None
    assert result.image.image.mode == "RGB"
    assert result.image.had_input_alpha is True
    assert result.image.generated_alpha is False
    assert result.image.preprocess_variant == "rgba-alpha-black"
    assert result.image.metadata()["foreground_crop_scale"] == 1.1
    assert result.image.metadata()["bg_color"] == (0, 0, 0)
    assert result.image.output_size[0] == result.image.output_size[1]
    output = np.array(result.image.image)
    assert output.max() == 200
    assert not np.any(output == 255)


def test_pixal3d_preprocess_blocks_rgb_without_background_remover(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (16, 16), (128, 64, 32)).save(image_path)

    result = preprocess_pixal3d_image(image_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "input-preprocessing"
    assert result.blocker.operation == "run Pixal3D RGB background removal"
    assert "RMBG" in result.blocker.reason


def test_pixal3d_preprocess_accepts_background_remover_rgba_output(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (16, 12), (128, 64, 32)).save(image_path)

    def fake_remover(image):
        rgba = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.array(image, dtype=np.uint8)
        rgba[2:10, 3:13, 3] = 255
        return Image.fromarray(rgba, mode="RGBA")

    result = preprocess_pixal3d_image(image_path, background_remover=fake_remover)

    assert result.ready
    assert result.image is not None
    assert result.image.generated_alpha is True
    assert result.image.preprocess_variant == "rmbg-black"
    assert result.image.metadata()["background_removed"] is True


def test_pixal3d_preprocess_blocks_empty_alpha_foreground(tmp_path):
    image_path = tmp_path / "empty-alpha.png"
    Image.new("RGBA", (16, 16), (128, 64, 32, 0)).save(image_path)

    result = preprocess_pixal3d_image(image_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.stage == "input-preprocessing"
    assert result.blocker.operation == "crop Pixal3D alpha foreground"
