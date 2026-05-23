"""TRELLIS.2 image preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

from .trellis2_rmbg import Rmbg2PortBlocker, assess_rmbg2_mlx_port


class Rmbg2MlxRuntimeError(RuntimeError):
    """Lazy proxy error for RMBG forward failures."""


def remove_background_rmbg2_mlx(image: Image.Image, *, root: str | Path) -> Image.Image:
    from .trellis2_rmbg_forward import Rmbg2MlxRuntimeError as ForwardRuntimeError
    from .trellis2_rmbg_forward import remove_background_rmbg2_mlx as remove_background

    try:
        return remove_background(image, root=root)
    except ForwardRuntimeError as error:
        raise Rmbg2MlxRuntimeError(str(error)) from error


@dataclass(frozen=True)
class Trellis2PreprocessBlocker:
    stage: str
    operation: str
    reference: str
    reason: str
    next_slice: str


@dataclass(frozen=True)
class Trellis2PreprocessedImage:
    input_path: Path
    image: Image.Image
    input_mode: str
    input_size: tuple[int, int]
    had_input_alpha: bool
    generated_alpha: bool

    @property
    def output_mode(self) -> str:
        return self.image.mode

    @property
    def output_size(self) -> tuple[int, int]:
        return self.image.size


@dataclass(frozen=True)
class Trellis2PreprocessResult:
    input_path: Path
    image: Trellis2PreprocessedImage | None = None
    blocker: Trellis2PreprocessBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.image is not None and self.blocker is None


BackgroundRemover = Callable[[Image.Image], Image.Image]


def preprocess_trellis2_image(
    image_path: str | Path,
    *,
    background_remover: BackgroundRemover | None = None,
    rmbg_root: str | Path | None = None,
) -> Trellis2PreprocessResult:
    """Decode and preprocess an image for the TRELLIS.2 image-to-3D stage."""

    path = Path(image_path)
    try:
        with Image.open(path) as opened:
            image = opened.copy()
    except FileNotFoundError:
        return _blocked(path, "image decode", f"image file not found: {path}")
    except (UnidentifiedImageError, OSError) as error:
        return _blocked(path, "image decode", f"image file could not be decoded: {error}")

    input_mode = image.mode
    input_size = image.size
    has_alpha = _has_useful_alpha(image)
    image = _resize_max_side(image, max_side=1024)

    generated_alpha = False
    if has_alpha:
        output = image
    elif background_remover is not None:
        output = background_remover(image.convert("RGB"))
        generated_alpha = True
    elif rmbg_root is not None:
        assessment = _assess_rmbg(path, rmbg_root)
        if assessment is not None:
            return assessment
        try:
            output = remove_background_rmbg2_mlx(image.convert("RGB"), root=rmbg_root)
        except Rmbg2MlxRuntimeError as error:
            return _blocked(
                path,
                "MLX BiRefNet forward pass",
                str(error),
                next_slice="fix the local MLX BiRefNet forward path for RMBG RGB preprocessing",
            )
        generated_alpha = True
    else:
        return _blocked(
            path,
            "MLX RMBG background removal",
            "RGB or fully opaque images require local RMBG/BiRefNet background removal",
            next_slice="validate local RMBG assets and wire MLX BiRefNet for RGB preprocessing",
        )

    if output.mode != "RGBA":
        return _blocked(
            path,
            "background alpha matte",
            f"background remover must return RGBA output, got {output.mode}",
            next_slice="return an RGBA alpha matte from the RMBG preprocessing path",
        )

    composed = _crop_and_composite_alpha(output)
    if composed is None:
        return _blocked(
            path,
            "alpha foreground crop",
            "alpha channel contains no foreground pixels above threshold",
            next_slice="provide an image with foreground alpha or improve background removal output",
        )

    return Trellis2PreprocessResult(
        input_path=path,
        image=Trellis2PreprocessedImage(
            input_path=path,
            image=composed,
            input_mode=input_mode,
            input_size=input_size,
            had_input_alpha=has_alpha,
            generated_alpha=generated_alpha,
        ),
    )


def _assess_rmbg(path: Path, rmbg_root: str | Path) -> Trellis2PreprocessResult | None:
    try:
        assessment = assess_rmbg2_mlx_port(rmbg_root)
    except FileNotFoundError as error:
        return _blocked(
            path,
            "local RMBG asset validation",
            str(error),
            next_slice="place compatible RMBG-2.0 assets under weights/rmbg2",
        )
    if assessment.blocker is None:
        return None
    return _blocked_from_rmbg(path, assessment.blocker)


def _blocked_from_rmbg(path: Path, blocker: Rmbg2PortBlocker) -> Trellis2PreprocessResult:
    return Trellis2PreprocessResult(
        input_path=path,
        blocker=Trellis2PreprocessBlocker(
            stage=blocker.stage,
            operation=blocker.operation,
            reference=blocker.reference,
            reason=blocker.reason,
            next_slice=blocker.next_slice,
        ),
    )


def _has_useful_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.array(image)[:, :, 3]
    return bool(np.any(alpha != 255))


def _resize_max_side(image: Image.Image, *, max_side: int) -> Image.Image:
    largest_side = max(image.size)
    scale = min(1.0, max_side / largest_side)
    if scale >= 1.0:
        return image
    next_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(next_size, Image.Resampling.LANCZOS)


def _crop_and_composite_alpha(image: Image.Image) -> Image.Image | None:
    output_np = np.array(image)
    alpha = output_np[:, :, 3]
    foreground = np.argwhere(alpha > 0.8 * 255)
    if foreground.size == 0:
        return None

    bbox = (
        np.min(foreground[:, 1]),
        np.min(foreground[:, 0]),
        np.max(foreground[:, 1]),
        np.max(foreground[:, 0]),
    )
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = max(1, int(size * 1))
    crop_box = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )
    cropped = image.crop(crop_box)
    cropped_np = np.array(cropped).astype(np.float32) / 255
    rgb = cropped_np[:, :, :3] * cropped_np[:, :, 3:4]
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _blocked(
    path: Path,
    operation: str,
    reason: str,
    *,
    next_slice: str = "implement TRELLIS.2 image preprocessing for this input",
) -> Trellis2PreprocessResult:
    return Trellis2PreprocessResult(
        input_path=path,
        blocker=Trellis2PreprocessBlocker(
            stage="image-preprocessing-background",
            operation=operation,
            reference="vendors/TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py:127-162",
            reason=reason,
            next_slice=next_slice,
        ),
    )
