"""Pixal3D image preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError


PIXAL3D_PREPROCESS_MAX_SIDE = 1024
PIXAL3D_ALPHA_FOREGROUND_THRESHOLD = 0.8 * 255
PIXAL3D_FOREGROUND_CROP_SCALE = 1.1
PIXAL3D_DEFAULT_BACKGROUND_COLOR = (0, 0, 0)

Pixal3DBackgroundRemover = Callable[[Image.Image], Image.Image]


@dataclass(frozen=True)
class Pixal3DPreprocessBlocker:
    """Structured blocker for input image preprocessing failures."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class Pixal3DPreprocessedImage:
    """Foreground-isolated Pixal3D image and trace metadata."""

    input_path: Path
    image: Image.Image
    input_mode: str
    input_size: tuple[int, int]
    resized_size: tuple[int, int]
    had_input_alpha: bool
    generated_alpha: bool
    preprocess_variant: str
    bg_color: tuple[int, int, int]
    crop_box: tuple[float, float, float, float]
    foreground_bbox: tuple[int, int, int, int]

    @property
    def output_mode(self) -> str:
        return self.image.mode

    @property
    def output_size(self) -> tuple[int, int]:
        return self.image.size

    def metadata(self) -> dict[str, object]:
        return {
            "raw_path": str(self.input_path),
            "input_mode": self.input_mode,
            "input_size": self.input_size,
            "resized_size": self.resized_size,
            "output_mode": self.output_mode,
            "output_size": self.output_size,
            "had_input_alpha": self.had_input_alpha,
            "generated_alpha": self.generated_alpha,
            "background_removed": self.generated_alpha,
            "preprocess_variant": self.preprocess_variant,
            "bg_color": self.bg_color,
            "foreground_alpha_threshold": PIXAL3D_ALPHA_FOREGROUND_THRESHOLD,
            "foreground_crop_scale": PIXAL3D_FOREGROUND_CROP_SCALE,
            "foreground_bbox": self.foreground_bbox,
            "crop_box": self.crop_box,
            "reference": "vendors/Pixal3D pixal3d_image_to_3d.Pixal3DImageTo3DPipeline.preprocess_image",
        }


@dataclass(frozen=True)
class Pixal3DPreprocessResult:
    """Preprocess result with either an image or a blocker."""

    input_path: Path
    image: Pixal3DPreprocessedImage | None = None
    blocker: Pixal3DPreprocessBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.image is not None and self.blocker is None


def preprocess_pixal3d_image(
    image_path: str | Path,
    *,
    background_remover: Pixal3DBackgroundRemover | None = None,
    bg_color: tuple[int, int, int] = PIXAL3D_DEFAULT_BACKGROUND_COLOR,
) -> Pixal3DPreprocessResult:
    """Decode and preprocess an image with the Pixal3D vendor image contract."""

    path = Path(image_path)
    try:
        with Image.open(path) as opened:
            image = opened.copy()
    except FileNotFoundError:
        return _blocked(path, "decode Pixal3D input image", f"image file not found: {path}")
    except (UnidentifiedImageError, OSError) as error:
        return _blocked(path, "decode Pixal3D input image", f"image file could not be decoded: {error}")

    input_mode = image.mode
    input_size = image.size
    had_input_alpha = _has_useful_alpha(image)
    resized = _resize_max_side(image, max_side=PIXAL3D_PREPROCESS_MAX_SIDE)

    generated_alpha = False
    if had_input_alpha:
        alpha_image = resized
        variant = "rgba-alpha-black"
    elif background_remover is not None:
        alpha_image = background_remover(resized.convert("RGB"))
        generated_alpha = True
        variant = "rmbg-black"
    else:
        return _blocked(
            path,
            "run Pixal3D RGB background removal",
            "RGB or fully opaque images require a Pixal3D-compatible RMBG/background remover",
            {
                "input_mode": input_mode,
                "input_size": input_size,
                "reference": "vendor Pixal3D uses rembg_model for RGB/fully opaque inputs",
            },
        )

    if alpha_image.mode != "RGBA":
        return _blocked(
            path,
            "validate Pixal3D foreground alpha matte",
            f"background remover must return RGBA output, got {alpha_image.mode}",
            {"preprocess_variant": variant},
        )

    composed = _crop_and_composite_alpha(alpha_image, bg_color=bg_color)
    if composed is None:
        return _blocked(
            path,
            "crop Pixal3D alpha foreground",
            "alpha channel contains no foreground pixels above Pixal3D threshold",
            {
                "foreground_alpha_threshold": PIXAL3D_ALPHA_FOREGROUND_THRESHOLD,
                "preprocess_variant": variant,
            },
        )
    image_rgb, foreground_bbox, crop_box = composed

    return Pixal3DPreprocessResult(
        input_path=path,
        image=Pixal3DPreprocessedImage(
            input_path=path,
            image=image_rgb,
            input_mode=input_mode,
            input_size=input_size,
            resized_size=resized.size,
            had_input_alpha=had_input_alpha,
            generated_alpha=generated_alpha,
            preprocess_variant=variant,
            bg_color=tuple(int(channel) for channel in bg_color),
            crop_box=crop_box,
            foreground_bbox=foreground_bbox,
        ),
    )


def _has_useful_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.array(image)[:, :, 3]
    return bool(not np.all(alpha == 255))


def _resize_max_side(image: Image.Image, *, max_side: int) -> Image.Image:
    largest_side = max(image.size)
    scale = min(1.0, max_side / largest_side)
    if scale >= 1.0:
        return image
    next_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(next_size, Image.Resampling.LANCZOS)


def _crop_and_composite_alpha(
    image: Image.Image,
    *,
    bg_color: tuple[int, int, int],
) -> tuple[Image.Image, tuple[int, int, int, int], tuple[float, float, float, float]] | None:
    output_np = np.array(image)
    alpha = output_np[:, :, 3]
    foreground = np.argwhere(alpha > PIXAL3D_ALPHA_FOREGROUND_THRESHOLD)
    if foreground.size == 0:
        return None

    bbox = (
        int(np.min(foreground[:, 1])),
        int(np.min(foreground[:, 0])),
        int(np.max(foreground[:, 1])),
        int(np.max(foreground[:, 0])),
    )
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = max(1, int(size * PIXAL3D_FOREGROUND_CROP_SCALE))
    crop_box = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    cropped = image.crop(crop_box)
    cropped_np = np.array(cropped).astype(np.float32) / 255.0
    rgb = cropped_np[:, :, :3]
    alpha_channel = cropped_np[:, :, 3:4]
    bg = np.array(bg_color, dtype=np.float32) / 255.0
    composed = rgb * alpha_channel + bg * (1.0 - alpha_channel)
    image_rgb = Image.fromarray((np.clip(composed, 0, 1) * 255).astype(np.uint8), mode="RGB")
    return image_rgb, bbox, crop_box


def _blocked(
    path: Path,
    operation: str,
    reason: str,
    metadata: dict[str, object] | None = None,
) -> Pixal3DPreprocessResult:
    return Pixal3DPreprocessResult(
        input_path=path,
        blocker=Pixal3DPreprocessBlocker(
            stage="input-preprocessing",
            operation=operation,
            reason=reason,
            metadata=metadata or {},
        ),
    )
