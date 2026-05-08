"""Shared mesh and texture export helpers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def texture_png_payload(image: np.ndarray) -> bytes:
    """Encode a texture image as a PNG payload for embedded GLB images."""

    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError(f"texture image must be 2D or 3D, got {array.shape}")
    if array.size == 0:
        raise ValueError("texture image must not be empty")
    if array.dtype != np.uint8:
        raise ValueError(f"texture image must use uint8 pixels, got {array.dtype}")
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def pad_glb_buffer(buffer: bytearray, *, pad_byte: int) -> None:
    """Pad a GLB binary buffer to a 4-byte boundary."""

    padding = (4 - len(buffer) % 4) % 4
    if padding:
        buffer.extend(bytes([pad_byte]) * padding)


def rasterize_uv_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize UV triangles to their corresponding 3D surface positions."""

    positions = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    mask = np.zeros((texture_size, texture_size), dtype=bool)
    uv_scale = float(texture_size - 1)

    for face in faces:
        uv0, uv1, uv2 = uvs[face] * uv_scale
        p0, p1, p2 = vertices[face]
        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), texture_size - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), texture_size - 1)
        if max_x < min_x or max_y < min_y:
            continue

        x_values = np.arange(min_x, max_x + 1, dtype=np.float32)
        y_values = np.arange(min_y, max_y + 1, dtype=np.float32)
        px_grid, py_grid = np.meshgrid(x_values, y_values)
        weights = _barycentric_weights(px_grid, py_grid, uv0, uv1, uv2)
        if weights is None:
            continue
        w0, w1, w2 = weights
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue
        face_positions = w0[..., None] * p0 + w1[..., None] * p1 + w2[..., None] * p2
        yy, xx = np.where(inside)
        positions[y_values.astype(np.int64)[yy], x_values.astype(np.int64)[xx]] = face_positions[yy, xx]
        mask[y_values.astype(np.int64)[yy], x_values.astype(np.int64)[xx]] = True

    return positions, mask


def fill_texture_holes(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill uncovered texels by nearest covered texel propagation."""

    if values.ndim != 3:
        raise ValueError(f"texture values must have shape (height, width, channels), got {values.shape}")
    if mask.shape != values.shape[:2]:
        raise ValueError(f"texture mask must have shape {values.shape[:2]}, got {mask.shape}")
    if not np.any(mask):
        raise ValueError("texture hole filling requires at least one covered texel")

    nearest_y, nearest_x = nearest_covered_texels(mask)
    return values.astype(np.float32, copy=False)[nearest_y, nearest_x], np.ones_like(mask, dtype=bool)


def fill_texture_holes_ndimage(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill uncovered texels with local dilation first, then nearest propagation."""

    if values.ndim != 3:
        raise ValueError(f"texture values must have shape (height, width, channels), got {values.shape}")
    if mask.shape != values.shape[:2]:
        raise ValueError(f"texture mask must have shape {values.shape[:2]}, got {mask.shape}")
    if not np.any(mask):
        raise ValueError("texture hole filling requires at least one covered texel")

    from scipy.ndimage import binary_dilation, uniform_filter

    filled = values.astype(np.float32, copy=True)
    current_mask = mask.astype(bool, copy=True)
    for _ in range(iterations):
        if np.all(current_mask):
            return filled, current_mask
        dilated = binary_dilation(current_mask, iterations=1)
        update = dilated & ~current_mask
        if not np.any(update):
            break
        denom = uniform_filter(current_mask.astype(np.float32), size=3)
        for channel_index in range(filled.shape[2]):
            numer = uniform_filter(filled[:, :, channel_index] * current_mask.astype(np.float32), size=3)
            blurred = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
            filled[:, :, channel_index][update] = blurred[update]
        current_mask = current_mask | update

    if np.all(current_mask):
        return filled, current_mask
    return fill_texture_holes(filled, current_mask)


def nearest_covered_texels(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest covered texel coordinates for every texel in a mask."""

    covered = mask.astype(bool, copy=False)
    if covered.ndim != 2:
        raise ValueError(f"texture mask must be 2D, got {covered.shape}")
    if not np.any(covered):
        raise ValueError("nearest texel fill requires at least one covered texel")

    height, width = covered.shape
    yy, xx = np.indices((height, width), dtype=np.int32)
    nearest_y = np.where(covered, yy, -1).astype(np.int32)
    nearest_x = np.where(covered, xx, -1).astype(np.int32)
    step = 1 << max(height, width).bit_length()
    while step > 1:
        step //= 2
        current_valid = nearest_y >= 0
        current_distance = np.where(
            current_valid,
            (nearest_y - yy).astype(np.int64) ** 2 + (nearest_x - xx).astype(np.int64) ** 2,
            np.iinfo(np.int64).max,
        )
        padded_y = np.pad(nearest_y, ((step, step), (step, step)), mode="constant", constant_values=-1)
        padded_x = np.pad(nearest_x, ((step, step), (step, step)), mode="constant", constant_values=-1)
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                candidate_y = padded_y[step + dy : step + dy + height, step + dx : step + dx + width]
                candidate_x = padded_x[step + dy : step + dy + height, step + dx : step + dx + width]
                candidate_valid = candidate_y >= 0
                if not np.any(candidate_valid):
                    continue
                candidate_distance = np.where(
                    candidate_valid,
                    (candidate_y - yy).astype(np.int64) ** 2 + (candidate_x - xx).astype(np.int64) ** 2,
                    np.iinfo(np.int64).max,
                )
                better = candidate_valid & (
                    (candidate_distance < current_distance)
                    | (
                        (candidate_distance == current_distance)
                        & ((candidate_y < nearest_y) | ((candidate_y == nearest_y) & (candidate_x < nearest_x)))
                    )
                )
                nearest_y[better] = candidate_y[better]
                nearest_x[better] = candidate_x[better]
                current_distance[better] = candidate_distance[better]
    return nearest_y, nearest_x


def _barycentric_weights(
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    d00 = uv1[0] - uv0[0]
    d01 = uv2[0] - uv0[0]
    d10 = uv1[1] - uv0[1]
    d11 = uv2[1] - uv0[1]
    denom = d00 * d11 - d01 * d10
    if abs(float(denom)) < 1e-10:
        return None
    dx = px_grid - uv0[0]
    dy = py_grid - uv0[1]
    w1 = (dx * d11 - d01 * dy) / denom
    w2 = (d00 * dy - dx * d10) / denom
    w0 = 1.0 - w1 - w2
    return w0, w1, w2
