"""MapAnything image discovery and preprocessing without Torch runtime deps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose


MAPANYTHING_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAPANYTHING_PATCH_SIZE = 14
MAPANYTHING_DEFAULT_RESOLUTION_SET = 518
MAPANYTHING_DEFAULT_NORM_TYPE = "dinov2"
MAPANYTHING_DINOV2_IMAGE_MEAN = (0.485, 0.456, 0.406)
MAPANYTHING_DINOV2_IMAGE_STD = (0.229, 0.224, 0.225)

MAPANYTHING_RESOLUTION_MAPPINGS = {
    518: {
        1.000: (518, 518),
        1.321: (518, 392),
        1.542: (518, 336),
        1.762: (518, 294),
        2.056: (518, 252),
        3.083: (518, 168),
        0.757: (392, 518),
        0.649: (336, 518),
        0.567: (294, 518),
        0.486: (252, 518),
    },
    512: {
        1.000: (512, 512),
        1.333: (512, 384),
        1.524: (512, 336),
        1.778: (512, 288),
        2.000: (512, 256),
        3.200: (512, 160),
        0.750: (384, 512),
        0.656: (336, 512),
        0.562: (288, 512),
        0.500: (256, 512),
    },
    504: {
        1.000: (504, 504),
        1.333: (504, 378),
        1.565: (504, 322),
        1.800: (504, 280),
        2.118: (504, 238),
        3.273: (504, 154),
        0.750: (378, 504),
        0.639: (322, 504),
        0.556: (280, 504),
        0.472: (238, 504),
    },
}


@dataclass(frozen=True)
class MapAnythingPreprocessedView:
    """One MapAnything image-only view in the format needed by the MLX port."""

    image_path: Path
    img: mx.array
    true_shape: tuple[int, int]
    idx: int
    instance: str
    data_norm_type: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    target_size: tuple[int, int]


@dataclass(frozen=True)
class MapAnythingPreprocessedInput:
    """Preprocessed MapAnything image batch plus trace metadata."""

    views: tuple[MapAnythingPreprocessedView, ...]
    target_size: tuple[int, int]
    average_aspect_ratio: float
    resize_mode: str
    patch_size: int
    resolution_set: int

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return tuple(view.image_path for view in self.views)

    @property
    def frame_count(self) -> int:
        return len(self.views)


def discover_mapanything_images(
    folder_or_list: str | Path | Sequence[str | Path],
    *,
    stride: int = 1,
) -> tuple[Path, ...]:
    """Discover supported MapAnything image inputs in vendored deterministic order."""

    if stride <= 0:
        raise ValueError("stride must be positive")

    root: Path | None
    entries: list[Path]
    if isinstance(folder_or_list, str | Path):
        path = Path(folder_or_list)
        if path.is_dir():
            root = path
            entries = [Path(name) for name in sorted(child.name for child in path.iterdir())]
        elif path.is_file():
            root = None
            entries = [path]
        else:
            raise FileNotFoundError(f"input path not found: {path}")
    else:
        root = None
        entries = [Path(item) for item in folder_or_list]

    images: list[Path] = []
    for index, entry in enumerate(entries):
        if index % stride != 0:
            continue
        candidate = root / entry if root is not None else entry
        if candidate.suffix.lower() in MAPANYTHING_IMAGE_EXTENSIONS and candidate.is_file():
            images.append(candidate)

    if not images:
        raise ValueError("No valid images found")
    return tuple(images)


def find_mapanything_closest_aspect_ratio(
    aspect_ratio: float,
    resolution_set: int = MAPANYTHING_DEFAULT_RESOLUTION_SET,
) -> tuple[int, int]:
    """Return vendored fixed-mapping target size as (width, height)."""

    try:
        mapping = MAPANYTHING_RESOLUTION_MAPPINGS[resolution_set]
    except KeyError as error:
        raise ValueError(f"unsupported MapAnything resolution set: {resolution_set}") from error
    closest_key = min(mapping, key=lambda key: abs(key - aspect_ratio))
    return mapping[closest_key]


def preprocess_mapanything_images(
    folder_or_list: str | Path | Sequence[str | Path],
    *,
    resize_mode: str = "fixed_mapping",
    size: int | tuple[int, int] | None = None,
    norm_type: str = MAPANYTHING_DEFAULT_NORM_TYPE,
    patch_size: int = MAPANYTHING_PATCH_SIZE,
    resolution_set: int = MAPANYTHING_DEFAULT_RESOLUTION_SET,
    stride: int = 1,
) -> MapAnythingPreprocessedInput:
    """Prepare image-only MapAnything inputs as MLX tensors shaped [1, 3, H, W]."""

    _validate_resize_args(resize_mode, size)
    if norm_type != "dinov2":
        raise ValueError("only dinov2 MapAnything normalization is supported")
    image_paths = discover_mapanything_images(folder_or_list, stride=stride)

    loaded: list[tuple[Path, Image.Image, int, int]] = []
    aspect_ratios: list[float] = []
    for image_path in image_paths:
        with Image.open(image_path) as raw:
            image = exif_transpose(raw).convert("RGB")
            width, height = image.size
            loaded.append((image_path, image.copy(), width, height))
            aspect_ratios.append(width / height)

    if not loaded:
        raise ValueError("No valid images found")
    average_aspect_ratio = float(sum(aspect_ratios) / len(aspect_ratios))
    target_size = mapanything_target_size(
        average_aspect_ratio,
        resize_mode=resize_mode,
        size=size,
        patch_size=patch_size,
        resolution_set=resolution_set,
    )

    views: list[MapAnythingPreprocessedView] = []
    for image_path, image, original_width, original_height in loaded:
        processed = crop_resize_mapanything_image(image, target_size)
        tensor = normalize_mapanything_image(processed, norm_type=norm_type)
        views.append(
            MapAnythingPreprocessedView(
                image_path=image_path,
                img=tensor,
                true_shape=(processed.height, processed.width),
                idx=len(views),
                instance=str(len(views)),
                data_norm_type=norm_type,
                original_size=(original_width, original_height),
                processed_size=(processed.height, processed.width),
                target_size=target_size,
            )
        )

    return MapAnythingPreprocessedInput(
        views=tuple(views),
        target_size=target_size,
        average_aspect_ratio=average_aspect_ratio,
        resize_mode=resize_mode,
        patch_size=patch_size,
        resolution_set=resolution_set,
    )


def mapanything_target_size(
    average_aspect_ratio: float,
    *,
    resize_mode: str = "fixed_mapping",
    size: int | tuple[int, int] | None = None,
    patch_size: int = MAPANYTHING_PATCH_SIZE,
    resolution_set: int = MAPANYTHING_DEFAULT_RESOLUTION_SET,
) -> tuple[int, int]:
    """Resolve the shared target size used for all MapAnything input images."""

    _validate_resize_args(resize_mode, size)
    if resize_mode == "fixed_mapping":
        return find_mapanything_closest_aspect_ratio(average_aspect_ratio, resolution_set)
    if resize_mode == "square":
        assert isinstance(size, int)
        target = round(size // patch_size) * patch_size
        return (target, target)
    if resize_mode == "longest_side":
        assert isinstance(size, int)
        if average_aspect_ratio >= 1:
            return (size, round((size // patch_size) / average_aspect_ratio) * patch_size)
        return (round((size // patch_size) * average_aspect_ratio) * patch_size, size)
    assert isinstance(size, tuple)
    return ((size[0] // patch_size) * patch_size, (size[1] // patch_size) * patch_size)


def crop_resize_mapanything_image(image: Image.Image, resolution: tuple[int, int]) -> Image.Image:
    """Match vendored `crop_resize_if_necessary` for image-only inputs."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.array(resolution)
    scale_final = max(output_resolution / image.size) + 1e-8
    rescaled_resolution = np.floor(input_resolution * scale_final).astype(int)
    resample = Image.Resampling.LANCZOS if scale_final < 1 else Image.Resampling.BICUBIC
    resized = image.resize(tuple(int(value) for value in rescaled_resolution), resample=resample)
    width, height = resized.size
    target_width, target_height = resolution
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


def normalize_mapanything_image(
    image: Image.Image,
    *,
    norm_type: str = MAPANYTHING_DEFAULT_NORM_TYPE,
) -> mx.array:
    """Normalize an RGB PIL image like TorchVision ToTensor + DINOv2 Normalize."""

    if norm_type != "dinov2":
        raise ValueError("only dinov2 MapAnything normalization is supported")
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np.array(MAPANYTHING_DINOV2_IMAGE_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(MAPANYTHING_DINOV2_IMAGE_STD, dtype=np.float32)[:, None, None]
    normalized = (array - mean) / std
    return mx.array(normalized[None, ...], dtype=mx.float32)


def _validate_resize_args(resize_mode: str, size: int | tuple[int, int] | None) -> None:
    valid_resize_modes = {"fixed_mapping", "longest_side", "square", "fixed_size"}
    if resize_mode not in valid_resize_modes:
        raise ValueError(f"resize_mode must be one of {sorted(valid_resize_modes)}, got {resize_mode!r}")
    if resize_mode in {"longest_side", "square"} and not isinstance(size, int):
        raise ValueError(f"size must be an int for resize_mode={resize_mode!r}")
    if resize_mode == "fixed_size":
        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or not all(isinstance(value, int) for value in size)
        ):
            raise ValueError("size must be a tuple[int, int] for resize_mode='fixed_size'")
