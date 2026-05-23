"""SAM 3D Objects image and mask preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sam3dPreprocessedInput:
    """Image/mask input in the RGBA convention used by official SAM3D inference."""

    rgba: np.ndarray
    image_path: Path
    mask_path: Path
    foreground_pixels: int
    size: tuple[int, int]


@dataclass(frozen=True)
class Sam3dOfficialPreprocessOutput:
    """Official SAM3D tensor-like preprocessing output using NumPy arrays."""

    image: np.ndarray
    mask: np.ndarray
    rgb_image: np.ndarray
    rgb_image_mask: np.ndarray
    pointmap: np.ndarray | None
    rgb_pointmap: np.ndarray | None
    pointmap_scale: np.ndarray | None
    pointmap_shift: np.ndarray | None
    crop_box: tuple[int, int, int, int]
    output_size: int


def preprocess_sam3d_image_mask(image_path: str | Path, mask_path: str | Path) -> Sam3dPreprocessedInput:
    """Load RGB image plus binary mask and embed the mask in the alpha channel."""

    image_file = Path(image_path)
    mask_file = Path(mask_path)
    if not image_file.is_file():
        raise FileNotFoundError(f"SAM3D image not found: {image_file}")
    if not mask_file.is_file():
        raise FileNotFoundError(f"SAM3D mask not found: {mask_file}")

    image = np.asarray(Image.open(image_file).convert("RGB"), dtype=np.uint8)
    mask = load_sam3d_mask(mask_file)
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"SAM3D mask size {mask.shape[::-1]} does not match image size {image.shape[1::-1]}"
        )
    alpha = (mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate((image, alpha), axis=-1)
    return Sam3dPreprocessedInput(
        rgba=rgba,
        image_path=image_file,
        mask_path=mask_file,
        foreground_pixels=int(mask.sum()),
        size=(int(image.shape[1]), int(image.shape[0])),
    )


def load_sam3d_mask(path: str | Path) -> np.ndarray:
    """Load a SAM3D mask as a boolean array using the official nonzero convention."""

    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., -1]
    if mask.ndim != 2:
        raise ValueError(f"SAM3D mask must be 2D or image-like with an alpha channel: {path}")
    return mask > 0


def preprocess_sam3d_official_tensors(
    rgba: np.ndarray,
    *,
    pointmap: np.ndarray | None = None,
    output_size: int = 518,
    crop_box_size_factor: float = 1.2,
) -> Sam3dOfficialPreprocessOutput:
    """Run the active official SAM3D image/mask/pointmap transform contract.

    The converted SAM3D Objects pipeline uses resize-to-same-size,
    ObjectCentricSSI pointmap normalization, crop-around-mask, pad-to-square,
    and 518px resize for the sparse-structure inputs.
    """

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"SAM3D RGBA input must have shape (H, W, 4), got {rgba.shape}")
    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    image = rgba[..., :3].astype(np.float32) / 255.0
    mask = rgba[..., 3] > 0
    if not np.any(mask):
        raise ValueError("SAM3D preprocessing requires a non-empty mask")
    crop_box = _mask_crop_box(mask, box_size_factor=crop_box_size_factor)
    cropped_image = _crop_and_pad_array(image, crop_box, fill_value=0.0)
    cropped_mask = _crop_and_pad_array(mask.astype(np.float32), crop_box, fill_value=0.0)
    image_square = _pad_to_square(cropped_image, fill_value=0.0)
    mask_square = _pad_to_square(cropped_mask, fill_value=0.0)
    image_resized = _resize_float_image(image_square, output_size, resample=Image.Resampling.BILINEAR)
    mask_resized = _resize_float_image(mask_square, output_size, resample=Image.Resampling.NEAREST)[..., 0]

    full_image_square = _pad_to_square(image, fill_value=0.0)
    full_mask_square = _pad_to_square(mask.astype(np.float32), fill_value=0.0)
    full_image = _resize_float_image(full_image_square, output_size, resample=Image.Resampling.BILINEAR)
    full_mask = _resize_float_image(full_mask_square, output_size, resample=Image.Resampling.NEAREST)[..., 0]

    processed_pointmap = None
    full_pointmap = None
    scale = None
    shift = None
    if pointmap is not None:
        if pointmap.shape[:2] != rgba.shape[:2] or pointmap.ndim != 3 or pointmap.shape[2] != 3:
            raise ValueError(
                f"pointmap must have shape {(rgba.shape[0], rgba.shape[1], 3)}, got {pointmap.shape}"
            )
        normalized_pointmap, scale, shift = _normalize_pointmap_object_centric_ssi(
            pointmap.astype(np.float32, copy=False),
            mask,
        )
        cropped_pointmap = _crop_and_pad_array(normalized_pointmap, crop_box, fill_value=0.0)
        pointmap_square = _pad_to_square(cropped_pointmap, fill_value=float("nan"))
        processed_pointmap = _resize_pointmap(pointmap_square, output_size, order=0)
        full_pointmap = _resize_pointmap(
            _pad_to_square(normalized_pointmap, fill_value=0.0),
            output_size,
            order=0,
        )

    return Sam3dOfficialPreprocessOutput(
        image=_chw(image_resized),
        mask=mask_resized[None, ...].astype(np.float32),
        rgb_image=_chw(full_image),
        rgb_image_mask=full_mask[None, ...].astype(np.float32),
        pointmap=_chw(processed_pointmap) if processed_pointmap is not None else None,
        rgb_pointmap=_chw(full_pointmap) if full_pointmap is not None else None,
        pointmap_scale=scale,
        pointmap_shift=shift,
        crop_box=crop_box,
        output_size=output_size,
    )


def _mask_crop_box(mask: np.ndarray, *, box_size_factor: float) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(mask)
    min_y, max_y = int(rows.min()), int(rows.max())
    min_x, max_x = int(cols.min()), int(cols.max())
    center_y = (min_y + max_y) / 2.0
    center_x = (min_x + max_x) / 2.0
    side = int(max(max_y - min_y, max_x - min_x, 2) * float(box_size_factor))
    half = side // 2
    return (
        int(center_x - half),
        int(center_y - half),
        int(center_x + half),
        int(center_y + half),
    )


def _crop_and_pad_array(array: np.ndarray, box: tuple[int, int, int, int], *, fill_value: float) -> np.ndarray:
    x0, y0, x1, y1 = box
    out_h = y1 - y0
    out_w = x1 - x0
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"invalid SAM3D crop box: {box}")
    if array.ndim == 2:
        output = np.full((out_h, out_w), fill_value, dtype=array.dtype)
    else:
        output = np.full((out_h, out_w, array.shape[2]), fill_value, dtype=array.dtype)
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, array.shape[1])
    src_y1 = min(y1, array.shape[0])
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return output
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    output[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0), ...] = array[
        src_y0:src_y1,
        src_x0:src_x1,
        ...,
    ]
    return output


def _normalize_pointmap_object_centric_ssi(
    pointmap: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pointmap_flat = np.transpose(pointmap, (2, 0, 1)).reshape(3, -1)
    mask_flat = mask.reshape(-1) > 0
    mask_points = pointmap_flat[:, mask_flat]
    if not np.isfinite(mask_points).any():
        scale = np.ones((3,), dtype=np.float32)
        shift = np.zeros((3,), dtype=np.float32)
        return pointmap.astype(np.float32, copy=False), scale, shift

    shift = np.nanmedian(mask_points, axis=1).astype(np.float32)
    points_centered = pointmap_flat - shift[:, None]
    max_dims = np.max(np.abs(points_centered), axis=0)
    scale_value = float(np.nanmedian(max_dims))
    if not np.isfinite(scale_value) or scale_value <= 0:
        scale_value = 1.0
    scale = np.full((3,), scale_value, dtype=np.float32)
    normalized = (pointmap - shift[None, None, :]) / scale[None, None, :]
    return normalized.astype(np.float32, copy=False), scale, shift


def _pad_to_square(array: np.ndarray, *, fill_value: float) -> np.ndarray:
    height, width = array.shape[:2]
    side = max(height, width)
    if array.ndim == 2:
        out = np.full((side, side), fill_value, dtype=array.dtype)
    else:
        out = np.full((side, side, array.shape[2]), fill_value, dtype=array.dtype)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    out[y0 : y0 + height, x0 : x0 + width, ...] = array
    return out


def _resize_float_image(array: np.ndarray, size: int, *, resample: Image.Resampling) -> np.ndarray:
    if resample == Image.Resampling.NEAREST:
        values = array[..., None] if array.ndim == 2 else array
        return _resize_nearest_hwc(values.astype(np.float32, copy=False), size)
    if array.ndim == 2:
        return _resize_float_channels(array[..., None], size, resample)
    channels = []
    for channel in range(array.shape[2]):
        channel_array = np.asarray(array[..., channel], dtype=np.float32)
        if np.nanmin(channel_array) >= 0.0 and np.nanmax(channel_array) <= 1.0:
            channels.append(_resize_float_channel(channel_array, size, resample))
        else:
            try:
                from scipy import ndimage
            except ModuleNotFoundError as error:
                raise ValueError("scipy is required for float pointmap resizing") from error
            zoom = (size / channel_array.shape[0], size / channel_array.shape[1])
            order = 0 if resample == Image.Resampling.NEAREST else 1
            channels.append(ndimage.zoom(channel_array, zoom, order=order))
    return np.stack(channels, axis=-1).astype(np.float32)


def _resize_float_channels(array: np.ndarray, size: int, resample: Image.Resampling) -> np.ndarray:
    return np.stack(
        [
            _resize_float_channel(np.asarray(array[..., channel], dtype=np.float32), size, resample)
            for channel in range(array.shape[2])
        ],
        axis=-1,
    ).astype(np.float32)


def _resize_float_channel(channel_array: np.ndarray, size: int, resample: Image.Resampling) -> np.ndarray:
    if resample == Image.Resampling.NEAREST:
        return _resize_nearest_hwc(channel_array[..., None].astype(np.float32, copy=False), size)[..., 0]
    pil = Image.fromarray(np.clip(channel_array, 0.0, 1.0).astype(np.float32, copy=False), mode="F")
    return np.asarray(pil.resize((size, size), resample=resample), dtype=np.float32)


def _resize_pointmap(array: np.ndarray, size: int, *, order: int) -> np.ndarray:
    if order == 0:
        return _resize_nearest_hwc(array, size)
    try:
        from scipy import ndimage
    except ModuleNotFoundError as error:
        raise ValueError("scipy is required for float pointmap resizing") from error
    zoom = (size / array.shape[0], size / array.shape[1], 1.0)
    return ndimage.zoom(array.astype(np.float32, copy=False), zoom, order=order).astype(np.float32)


def _resize_nearest_hwc(array: np.ndarray, size: int) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    if values.ndim != 3:
        raise ValueError(f"nearest resize expects HWC input, got {values.shape}")
    height, width = values.shape[:2]
    rows = np.floor(np.arange(size, dtype=np.float32) * (height / float(size))).astype(np.int64)
    cols = np.floor(np.arange(size, dtype=np.float32) * (width / float(size))).astype(np.int64)
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    return values[rows[:, None], cols[None, :], :].astype(np.float32, copy=False)


def _chw(array: np.ndarray | None) -> np.ndarray | None:
    if array is None:
        return None
    return np.transpose(array.astype(np.float32, copy=False), (2, 0, 1))
