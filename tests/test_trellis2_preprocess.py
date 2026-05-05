from PIL import Image
from safetensors.mlx import save_file
import mlx.core as mx

import mlx_spatial
import mlx_spatial.trellis2_preprocess as preprocess_module
from mlx_spatial.trellis2_preprocess import (
    Trellis2PreprocessBlocker,
    Trellis2PreprocessResult,
    Trellis2PreprocessedImage,
    preprocess_trellis2_image,
)


def _write_rgba_image(path, *, size=(8, 8), alpha_box=(2, 2, 6, 6), alpha_value=255):
    image = Image.new("RGBA", size, (200, 100, 50, 0))
    pixels = image.load()
    for y in range(alpha_box[1], alpha_box[3]):
        for x in range(alpha_box[0], alpha_box[2]):
            pixels[x, y] = (200, 100, 50, alpha_value)
    image.save(path)


def _write_rmbg_root(root, *, deform_conv=False):
    root.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "bb.weight": mx.array([1.0], dtype=mx.float32),
            "decoder.weight": mx.array([2.0], dtype=mx.float32),
            "squeeze_module.weight": mx.array([3.0], dtype=mx.float32),
        },
        root / "model.safetensors",
    )
    (root / "config.json").write_text("{}")
    (root / "BiRefNet_config.py").write_text("config = {}\n")
    architecture = "from torchvision.ops import deform_conv2d\n" if deform_conv else "class BiRefNet: pass\n"
    (root / "birefnet.py").write_text(architecture)


def test_preprocess_rgba_image_uses_existing_alpha_and_composites_rgb(tmp_path):
    image_path = tmp_path / "alpha.png"
    _write_rgba_image(image_path)

    result = preprocess_trellis2_image(image_path)

    assert result.ready
    assert result.blocker is None
    assert result.image is not None
    assert result.image.input_path == image_path
    assert result.image.input_mode == "RGBA"
    assert result.image.input_size == (8, 8)
    assert result.image.had_input_alpha
    assert not result.image.generated_alpha
    assert result.image.output_mode == "RGB"
    assert result.image.output_size == (2, 2)
    assert result.image.image.getpixel((1, 1)) == (200, 100, 50)


def test_preprocess_resizes_large_images_to_max_side_before_crop(tmp_path):
    image_path = tmp_path / "large.png"
    _write_rgba_image(image_path, size=(2048, 1024), alpha_box=(512, 256, 1536, 768))

    result = preprocess_trellis2_image(image_path)

    assert result.ready
    assert result.image is not None
    assert max(result.image.output_size) <= 1024
    assert result.image.input_size == (2048, 1024)


def test_preprocess_rgb_image_blocks_without_background_remover(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)

    result = preprocess_trellis2_image(image_path)

    assert not result.ready
    assert result.image is None
    assert result.blocker is not None
    assert result.blocker.stage == "image-preprocessing-background"
    assert result.blocker.operation == "MLX RMBG background removal"
    assert "RGB or fully opaque images require" in result.blocker.reason


def test_preprocess_rgb_image_reports_missing_rmbg_assets_when_configured(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)

    result = preprocess_trellis2_image(image_path, rmbg_root=tmp_path / "missing-rmbg")

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.operation == "local RMBG asset validation"
    assert "model.safetensors" in result.blocker.reason


def test_preprocess_rgb_image_propagates_rmbg_port_blocker(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
    rmbg_root = tmp_path / "rmbg"
    _write_rmbg_root(rmbg_root, deform_conv=True)

    result = preprocess_trellis2_image(image_path, rmbg_root=rmbg_root)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.operation == "MLX BiRefNet checkpoint key mapping"
    assert "required forward keys" in result.blocker.reason


def test_preprocess_rgb_image_reports_incomplete_rmbg_forward_keys(tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)
    rmbg_root = tmp_path / "rmbg"
    _write_rmbg_root(rmbg_root)

    result = preprocess_trellis2_image(image_path, rmbg_root=rmbg_root)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.operation == "MLX BiRefNet checkpoint key mapping"


def test_preprocess_rgb_image_uses_mlx_rmbg_forward(monkeypatch, tmp_path):
    image_path = tmp_path / "rgb.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_path)

    def fake_assess(_path, _root):
        return None

    def fake_remove_background(image, *, root):
        output = image.convert("RGBA")
        alpha = Image.new("L", image.size, 0)
        pixels = alpha.load()
        for y in range(2, 6):
            for x in range(2, 6):
                pixels[x, y] = 255
        output.putalpha(alpha)
        return output

    monkeypatch.setattr(preprocess_module, "_assess_rmbg", fake_assess)
    monkeypatch.setattr(preprocess_module, "remove_background_rmbg2_mlx", fake_remove_background)

    result = preprocess_module.preprocess_trellis2_image(image_path, rmbg_root=tmp_path / "rmbg")

    assert result.ready
    assert result.image is not None
    assert result.image.generated_alpha
    assert not result.image.had_input_alpha
    assert result.image.output_mode == "RGB"


def test_preprocess_blocks_when_alpha_has_no_foreground(tmp_path):
    image_path = tmp_path / "empty.png"
    Image.new("RGBA", (8, 8), (10, 20, 30, 0)).save(image_path)

    result = preprocess_trellis2_image(image_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.operation == "alpha foreground crop"


def test_preprocess_reports_invalid_image_deterministically(tmp_path):
    image_path = tmp_path / "not-image.png"
    image_path.write_text("not an image")

    result = preprocess_trellis2_image(image_path)

    assert not result.ready
    assert result.blocker is not None
    assert result.blocker.operation == "image decode"
    assert "could not be decoded" in result.blocker.reason


def test_preprocess_exports_are_public():
    assert mlx_spatial.preprocess_trellis2_image is preprocess_trellis2_image
    assert mlx_spatial.Trellis2PreprocessBlocker is Trellis2PreprocessBlocker
    assert mlx_spatial.Trellis2PreprocessResult is Trellis2PreprocessResult
    assert mlx_spatial.Trellis2PreprocessedImage is Trellis2PreprocessedImage
