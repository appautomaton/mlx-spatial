"""HY-World-2.0 WorldMirror input discovery and image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image


HYWORLD2_IMAGE_EXTENSIONS = (".jpeg", ".jpg", ".png", ".webp")
HYWORLD2_PATCH_SIZE = 14
HYWORLD2_OFFICIAL_TARGET_SIZE = 952
HYWORLD2_DEFAULT_MEMORY_PROFILE = "large"


@dataclass(frozen=True)
class HyWorld2MemoryProfile:
    """Resolved preprocessing limits for a named memory profile."""

    name: str
    target_size: int
    max_frames: int
    activation_guard_bytes: int


@dataclass(frozen=True)
class HyWorld2PreprocessedInput:
    """MLX-ready WorldMirror image batch and trace metadata."""

    image_paths: tuple[Path, ...]
    tensor: mx.array
    original_sizes: tuple[tuple[int, int], ...]
    processed_size: tuple[int, int]
    target_size: int
    patch_grid: tuple[int, int]
    token_count: int

    @property
    def frame_count(self) -> int:
        return len(self.image_paths)


HYWORLD2_MEMORY_PROFILE_CONFIGS = {
    "safe": HyWorld2MemoryProfile(
        name="safe",
        target_size=392,
        max_frames=2,
        activation_guard_bytes=1_000_000_000,
    ),
    "balanced": HyWorld2MemoryProfile(
        name="balanced",
        target_size=518,
        max_frames=8,
        activation_guard_bytes=4_000_000_000,
    ),
    "large": HyWorld2MemoryProfile(
        name="large",
        target_size=952,
        max_frames=32,
        activation_guard_bytes=12_000_000_000,
    ),
}


def memory_profile_config(name: str) -> HyWorld2MemoryProfile:
    """Return the named HY-World preprocessing/memory profile."""

    try:
        return HYWORLD2_MEMORY_PROFILE_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"unknown HY-World memory profile: {name!r}") from error


def discover_hyworld2_images(input_path: str | Path, *, max_frames: int) -> tuple[Path, ...]:
    """Discover supported HY-World image inputs in deterministic order."""

    path = Path(input_path)
    if path.is_dir():
        images = tuple(
            sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in HYWORLD2_IMAGE_EXTENSIONS
            )
        )
    elif path.is_file() and path.suffix.lower() in HYWORLD2_IMAGE_EXTENSIONS:
        images = (path,)
    elif path.is_file():
        raise ValueError(f"unsupported HY-World input file type: {path.suffix or '<none>'}")
    else:
        raise FileNotFoundError(f"input path not found: {path}")

    if not images:
        raise ValueError(f"no supported images found under: {path}")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    return images[:max_frames]


def preprocess_hyworld2_images(
    input_path: str | Path,
    *,
    memory_profile: str = HYWORLD2_DEFAULT_MEMORY_PROFILE,
    patch_size: int = HYWORLD2_PATCH_SIZE,
) -> HyWorld2PreprocessedInput:
    """Prepare HY-World image inputs as an MLX tensor shaped [1, S, 3, H, W]."""

    profile = memory_profile_config(memory_profile)
    image_paths = discover_hyworld2_images(input_path, max_frames=profile.max_frames)
    target_size = adaptive_hyworld2_target_size(image_paths, profile.target_size, patch_size=patch_size)

    arrays: list[np.ndarray] = []
    original_sizes: list[tuple[int, int]] = []
    processed_size: tuple[int, int] | None = None

    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb = _rgb_over_white(image)
            original_sizes.append(rgb.size)
            processed = resize_crop_hyworld2_image(rgb, target_size, patch_size=patch_size)
            if processed_size is None:
                processed_size = (processed.height, processed.width)
            elif processed_size != (processed.height, processed.width):
                raise ValueError(
                    "HY-World preprocessing requires all frames to share processed H,W; "
                    f"got {processed_size} and {(processed.height, processed.width)}"
                )
            arrays.append(np.asarray(processed, dtype=np.float32).transpose(2, 0, 1) / 255.0)

    if processed_size is None:
        raise ValueError("HY-World preprocessing found no images")

    batch = np.stack(arrays, axis=0)[None, ...]
    tensor = mx.array(batch, dtype=mx.float32)
    patch_grid = (processed_size[0] // patch_size, processed_size[1] // patch_size)
    token_count = len(image_paths) * patch_grid[0] * patch_grid[1]

    return HyWorld2PreprocessedInput(
        image_paths=image_paths,
        tensor=tensor,
        original_sizes=tuple(original_sizes),
        processed_size=processed_size,
        target_size=target_size,
        patch_grid=patch_grid,
        token_count=token_count,
    )


def adaptive_hyworld2_target_size(
    image_paths: tuple[Path, ...],
    max_target_size: int,
    *,
    patch_size: int = HYWORLD2_PATCH_SIZE,
) -> int:
    """Match official adaptive target sizing from the first image longest edge."""

    if not image_paths:
        raise ValueError("adaptive target sizing requires at least one image")
    with Image.open(image_paths[0]) as image:
        rgb = _rgb_over_white(image)
        longest_edge = max(rgb.size)
    target = min(longest_edge, max_target_size)
    target = (target // patch_size) * patch_size
    return max(target, 2 * patch_size)


def resize_crop_hyworld2_image(
    image: Image.Image,
    target_size: int,
    *,
    patch_size: int = HYWORLD2_PATCH_SIZE,
) -> Image.Image:
    """Resize longest edge to target and center-crop to patch-size multiples."""

    if target_size < 2 * patch_size:
        raise ValueError("target_size must be at least two patches")
    width, height = image.size
    scale = target_size / max(width, height)
    new_width = max(patch_size, int(round(width * scale / patch_size)) * patch_size)
    new_height = max(patch_size, int(round(height * scale / patch_size)) * patch_size)
    resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    crop_width = min(new_width, (new_width // patch_size) * patch_size)
    crop_height = min(new_height, (new_height // patch_size) * patch_size)
    left = max((new_width - crop_width) // 2, 0)
    top = max((new_height - crop_height) // 2, 0)
    return resized.crop((left, top, left + crop_width, top + crop_height))


def _rgb_over_white(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        return background.convert("RGB")
    return image.convert("RGB")
