"""Torch-free MLX runtime pieces for Valeo NAF feature upsampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mlx.core as mx
import numpy as np
from PIL import Image
from safetensors.mlx import load_file

from .model_assets import NAF_ASSETS, ModelAssetValidation, validate_model_assets


NAF_REPO_ID = "valeoai/NAF"
NAF_RELEASE_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
NAF_DEFAULT_ROOT = NAF_ASSETS.root_hint
NAF_WEIGHTS_FILENAME = "naf_release.safetensors"


@dataclass(frozen=True)
class NafRuntimeConfig:
    """NAF runtime dimensions with upstream defaults."""

    dim: int = 256
    heads_attn: int = 4
    heads_rope: int = 4
    kernel_size: int = 9
    img_layers: int = 2
    group_count: int = 8
    eps: float = 1e-5


@dataclass(frozen=True)
class NafProjectedFeatures:
    """Coordinate-sampled high-resolution NAF features."""

    features: mx.array
    target_size: tuple[int, int]
    point_count: int
    chunk_size: int
    kernel_size: int
    source: str


def naf_conversion_command(root: str | Path = NAF_DEFAULT_ROOT) -> tuple[str, ...]:
    """Return a dev setup command for converting the upstream NAF checkpoint."""

    return (
        "uv",
        "run",
        "--group",
        "torch-ref",
        "python",
        "scripts/pixal3d/convert_naf.py",
        "--output",
        str(Path(root) / NAF_WEIGHTS_FILENAME),
    )


def validate_naf_assets(root: str | Path = NAF_DEFAULT_ROOT) -> ModelAssetValidation:
    """Validate a local converted NAF asset root without importing Torch."""

    return validate_model_assets(root, NAF_ASSETS)


def naf_weights_path(root_or_path: str | Path = NAF_DEFAULT_ROOT) -> Path:
    """Resolve a NAF root or explicit safetensors path."""

    path = Path(root_or_path)
    if path.suffix == ".safetensors":
        return path
    return path / NAF_WEIGHTS_FILENAME


def load_naf_tensors(
    root_or_path: str | Path = NAF_DEFAULT_ROOT,
    *,
    config: NafRuntimeConfig = NafRuntimeConfig(),
) -> dict[str, mx.array]:
    """Load converted NAF safetensors and validate required tensor shapes."""

    path = naf_weights_path(root_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"converted NAF weights not found: {path}")
    tensors = dict(load_file(str(path)))
    validate_naf_tensors(tensors, config=config)
    return tensors


def prepare_naf_image_tensor(image: Image.Image, *, image_size: int) -> mx.array:
    """Convert a PIL image to the unnormalized BCHW tensor expected by NAF."""

    resized = image.convert("RGB").resize((int(image_size), int(image_size)), Image.Resampling.LANCZOS)
    array = np.array(resized).astype(np.float32) / 255.0
    return mx.array(np.transpose(array, (2, 0, 1))[None, :, :, :])


def validate_naf_tensors(
    tensors: Mapping[str, mx.array],
    *,
    config: NafRuntimeConfig = NafRuntimeConfig(),
) -> None:
    """Validate converted NAF tensor names and shapes."""

    expected = naf_required_tensor_shapes(config)
    missing = [name for name in expected if name not in tensors]
    if missing:
        raise ValueError(f"NAF weights missing required tensor(s): {missing[:5]}")
    mismatches = []
    for name, shape in expected.items():
        actual = tuple(int(dim) for dim in tensors[name].shape)
        if actual != shape:
            mismatches.append(f"{name}: expected {shape}, got {actual}")
    if mismatches:
        raise ValueError(f"NAF tensor shape mismatch: {mismatches[:5]}")


def naf_required_tensor_shapes(config: NafRuntimeConfig = NafRuntimeConfig()) -> dict[str, tuple[int, ...]]:
    """Return expected converted NAF tensor shapes for a runtime config."""

    dim = int(config.dim)
    hidden = dim // 2
    if dim <= 0 or dim % 2:
        raise ValueError("NAF dim must be a positive even integer")
    if dim % int(config.heads_rope) != 0:
        raise ValueError("NAF dim must be divisible by heads_rope")
    rope_width = dim // int(config.heads_rope) // 4
    if rope_width <= 0:
        raise ValueError("NAF dim / heads_rope must be at least 4")

    shapes: dict[str, tuple[int, ...]] = {
        "image_encoder.encoder.0.weight": (hidden, 3, 1, 1),
        "image_encoder.encoder.0.bias": (hidden,),
        "image_encoder.sem_encoder.0.weight": (hidden, 3, 3, 3),
        "image_encoder.sem_encoder.0.bias": (hidden,),
        "image_encoder.rope.periods": (rope_width,),
    }
    for branch in ("encoder", "sem_encoder"):
        kernel = 1 if branch == "encoder" else 3
        for layer in range(1, int(config.img_layers) + 1):
            prefix = f"image_encoder.{branch}.{layer}"
            shapes.update(
                {
                    f"{prefix}.norm1.weight": (hidden,),
                    f"{prefix}.norm1.bias": (hidden,),
                    f"{prefix}.conv1.weight": (hidden, hidden, kernel, kernel),
                    f"{prefix}.conv1.bias": (hidden,),
                    f"{prefix}.norm2.weight": (hidden,),
                    f"{prefix}.norm2.bias": (hidden,),
                    f"{prefix}.conv2.weight": (hidden, hidden, kernel, kernel),
                    f"{prefix}.conv2.bias": (hidden,),
                }
            )
    return shapes


def run_naf_image_encoder(
    image_bchw: mx.array,
    tensors: Mapping[str, mx.array],
    *,
    output_size: int | tuple[int, int],
    config: NafRuntimeConfig = NafRuntimeConfig(),
) -> mx.array:
    """Run the NAF image encoder and RoPE branch on an RGB image in ``[0, 1]``."""

    validate_naf_tensors(tensors, config=config)
    if image_bchw.ndim != 4 or int(image_bchw.shape[1]) != 3:
        raise ValueError(f"NAF image must have shape [B,3,H,W], got {image_bchw.shape}")
    target = _normalize_size(output_size)
    image = image_bchw.astype(mx.float32)
    if int(image.shape[-2]) > 4 * target[0] or int(image.shape[-1]) > 4 * target[1]:
        raise ValueError("NAF image downscale-before-encode path is not implemented for this target size")

    encoded = _run_naf_encoder_branch(image, tensors, "encoder", config=config)
    semantic = _run_naf_encoder_branch(image, tensors, "sem_encoder", config=config)
    features = mx.concatenate((encoded, semantic), axis=1)
    features = _adaptive_avg_pool2d_nchw(features, target)
    return _apply_naf_rope(features, tensors["image_encoder.rope.periods"], config=config)


def project_naf_features_at_points(
    image_bchw: mx.array,
    lr_features_bchw: mx.array,
    image_points: mx.array,
    *,
    image_resolution: int,
    output_size: int | tuple[int, int],
    tensors: Mapping[str, mx.array],
    chunk_size: int = 8192,
    config: NafRuntimeConfig = NafRuntimeConfig(),
) -> NafProjectedFeatures:
    """Return NAF-upsampled features sampled at Pixal3D image coordinates."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if lr_features_bchw.ndim != 4:
        raise ValueError(f"lr_features_bchw must have shape [B,C,H,W], got {lr_features_bchw.shape}")
    if image_points.ndim != 3 or int(image_points.shape[-1]) != 2:
        raise ValueError(f"image_points must have shape [B,P,2], got {image_points.shape}")
    batch, channels, lr_h, lr_w = (int(dim) for dim in lr_features_bchw.shape)
    if int(image_bchw.shape[0]) != batch or int(image_points.shape[0]) != batch:
        raise ValueError("NAF image, features, and points must have matching batch dimensions")
    if batch != 1:
        raise ValueError("NAF coordinate projection currently supports batch size 1")
    if channels % int(config.heads_attn) != 0:
        raise ValueError(f"feature channels must be divisible by heads_attn={config.heads_attn}, got {channels}")
    if int(config.dim) % int(config.heads_attn) != 0:
        raise ValueError("NAF dim must be divisible by heads_attn")
    if int(config.kernel_size) <= 0 or int(config.kernel_size) % 2 == 0:
        raise ValueError("NAF kernel_size must be a positive odd integer")

    target = _normalize_size(output_size)
    if target[0] % lr_h != 0 or target[1] % lr_w != 0:
        raise ValueError(f"NAF target size {target} must be divisible by LR feature grid {(lr_h, lr_w)}")
    query_map = run_naf_image_encoder(image_bchw, tensors, output_size=target, config=config)
    key_map = _adaptive_avg_pool2d_nchw(query_map, (lr_h, lr_w))
    points = image_points.astype(mx.float32)
    sampled_chunks: list[mx.array] = []
    point_count = int(points.shape[1])
    for start in range(0, point_count, int(chunk_size)):
        stop = min(start + int(chunk_size), point_count)
        sampled_chunks.append(
            _sample_naf_output_chunk(
                query_map,
                key_map,
                lr_features_bchw.astype(mx.float32),
                points[:, start:stop, :],
                image_resolution=image_resolution,
                target_size=target,
                config=config,
            )
        )
    features = mx.concatenate(sampled_chunks, axis=1) if sampled_chunks else mx.zeros((batch, 0, channels), dtype=lr_features_bchw.dtype)
    return NafProjectedFeatures(
        features=features.astype(lr_features_bchw.dtype),
        target_size=target,
        point_count=point_count,
        chunk_size=int(chunk_size),
        kernel_size=int(config.kernel_size),
        source="mlx-spatial NAF",
    )


def _run_naf_encoder_branch(
    image: mx.array,
    tensors: Mapping[str, mx.array],
    branch: str,
    *,
    config: NafRuntimeConfig,
) -> mx.array:
    kernel = 1 if branch == "encoder" else 3
    prefix = f"image_encoder.{branch}"
    hidden = _conv2d_nchw_reflect(
        image,
        tensors[f"{prefix}.0.weight"],
        tensors[f"{prefix}.0.bias"],
        padding=kernel // 2,
    )
    for layer in range(1, int(config.img_layers) + 1):
        block = f"{prefix}.{layer}"
        hidden = _group_norm_nchw(hidden, tensors[f"{block}.norm1.weight"], tensors[f"{block}.norm1.bias"], groups=config.group_count)
        hidden = _silu(hidden)
        hidden = _conv2d_nchw_reflect(hidden, tensors[f"{block}.conv1.weight"], tensors[f"{block}.conv1.bias"], padding=kernel // 2)
        hidden = _group_norm_nchw(hidden, tensors[f"{block}.norm2.weight"], tensors[f"{block}.norm2.bias"], groups=config.group_count)
        hidden = _silu(hidden)
        hidden = _conv2d_nchw_reflect(hidden, tensors[f"{block}.conv2.weight"], tensors[f"{block}.conv2.bias"], padding=kernel // 2)
    return hidden


def _sample_naf_output_chunk(
    query_map: mx.array,
    key_map: mx.array,
    value_map: mx.array,
    image_points: mx.array,
    *,
    image_resolution: int,
    target_size: tuple[int, int],
    config: NafRuntimeConfig,
) -> mx.array:
    target_h, target_w = target_size
    source_x = (image_points[..., 0] + 0.5) * target_w / int(image_resolution) - 0.5
    source_y = (image_points[..., 1] + 0.5) * target_h / int(image_resolution) - 0.5
    source_x = mx.clip(source_x, 0.0, float(target_w - 1))
    source_y = mx.clip(source_y, 0.0, float(target_h - 1))
    x0 = mx.floor(source_x).astype(mx.int32)
    y0 = mx.floor(source_y).astype(mx.int32)
    x1 = mx.clip(x0 + 1, 0, target_w - 1)
    y1 = mx.clip(y0 + 1, 0, target_h - 1)
    wx = source_x - x0.astype(source_x.dtype)
    wy = source_y - y0.astype(source_y.dtype)

    v00 = _naf_output_at_integer_pixels(query_map, key_map, value_map, y0[0], x0[0], config=config)
    v01 = _naf_output_at_integer_pixels(query_map, key_map, value_map, y0[0], x1[0], config=config)
    v10 = _naf_output_at_integer_pixels(query_map, key_map, value_map, y1[0], x0[0], config=config)
    v11 = _naf_output_at_integer_pixels(query_map, key_map, value_map, y1[0], x1[0], config=config)
    wx = wx[0, :, None]
    wy = wy[0, :, None]
    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    return (top * (1.0 - wy) + bottom * wy)[None, :, :]


def _naf_output_at_integer_pixels(
    query_map: mx.array,
    key_map: mx.array,
    value_map: mx.array,
    y: mx.array,
    x: mx.array,
    *,
    config: NafRuntimeConfig,
) -> mx.array:
    heads = int(config.heads_attn)
    kernel = int(config.kernel_size)
    radius = kernel // 2
    _, query_channels, target_h, target_w = (int(dim) for dim in query_map.shape)
    _, value_channels, lr_h, lr_w = (int(dim) for dim in value_map.shape)
    query_head_dim = query_channels // heads
    value_head_dim = value_channels // heads
    dilation_h = target_h // lr_h
    dilation_w = target_w // lr_w

    query_bhwc = mx.transpose(query_map, (0, 2, 3, 1))
    query = query_bhwc[0, y, x, :].reshape(-1, heads, query_head_dim)
    key_bhwc = mx.transpose(key_map, (0, 2, 3, 1))
    value_bhwc = mx.transpose(value_map, (0, 2, 3, 1))

    key_neighbors: list[mx.array] = []
    value_neighbors: list[mx.array] = []
    for dy in range(-radius, radius + 1):
        yy = mx.clip(y + dy * dilation_h, 0, target_h - 1) // dilation_h
        for dx in range(-radius, radius + 1):
            xx = mx.clip(x + dx * dilation_w, 0, target_w - 1) // dilation_w
            key_neighbors.append(key_bhwc[0, yy, xx, :].reshape(-1, heads, query_head_dim))
            value_neighbors.append(value_bhwc[0, yy, xx, :].reshape(-1, heads, value_head_dim))

    keys = mx.stack(key_neighbors, axis=1)
    values = mx.stack(value_neighbors, axis=1)
    scores = mx.sum(query[:, None, :, :] * keys, axis=-1) * (query_head_dim**-0.5)
    weights = mx.softmax(scores, axis=1)
    output = mx.sum(weights[..., None] * values, axis=1)
    return output.reshape(-1, value_channels)


def _apply_naf_rope(x: mx.array, periods: mx.array, *, config: NafRuntimeConfig) -> mx.array:
    batch, channels, height, width = (int(dim) for dim in x.shape)
    heads = int(config.heads_rope)
    if channels % heads:
        raise ValueError(f"NAF RoPE channels must be divisible by heads_rope={heads}, got {channels}")
    head_dim = channels // heads
    if head_dim % 4:
        raise ValueError("NAF RoPE head dimension must be divisible by 4")
    values = mx.reshape(x, (batch, heads, head_dim, height, width))
    values = mx.transpose(values, (0, 1, 3, 4, 2)).reshape(batch, heads, height * width, head_dim)

    coords_h = (mx.arange(height, dtype=mx.float32) + 0.5) / height
    coords_w = (mx.arange(width, dtype=mx.float32) + 0.5) / width
    grid_h, grid_w = mx.meshgrid(coords_h, coords_w, indexing="ij")
    coords = mx.stack((grid_h.reshape(-1), grid_w.reshape(-1)), axis=-1) * 2.0 - 1.0
    angles = 2.0 * math.pi * coords[:, :, None] / periods.astype(mx.float32)[None, None, :]
    angles = angles.reshape(height * width, -1)
    angles = mx.tile(angles, (1, 2))
    cos = mx.cos(angles)[None, None, :, :]
    sin = mx.sin(angles)[None, None, :, :]
    first, second = mx.split(values, 2, axis=-1)
    rotated = mx.concatenate((-second, first), axis=-1)
    values = values * cos + rotated * sin
    values = values.reshape(batch, heads, height, width, head_dim)
    values = mx.transpose(values, (0, 1, 4, 2, 3)).reshape(batch, channels, height, width)
    return values.astype(x.dtype)


def _adaptive_avg_pool2d_nchw(x: mx.array, output_size: tuple[int, int]) -> mx.array:
    batch, channels, height, width = (int(dim) for dim in x.shape)
    out_h, out_w = output_size
    if (height, width) == (out_h, out_w):
        return x
    if height % out_h != 0 or width % out_w != 0:
        raise ValueError(f"adaptive average pool requires divisible sizes, got {(height, width)} -> {output_size}")
    kh = height // out_h
    kw = width // out_w
    return mx.mean(x.reshape(batch, channels, out_h, kh, out_w, kw), axis=(3, 5))


def _conv2d_nchw_reflect(x: mx.array, weight: mx.array, bias: mx.array | None, *, padding: int) -> mx.array:
    if padding:
        x = _reflect_pad_nchw(x, padding)
    out = mx.conv2d(
        mx.transpose(x, (0, 2, 3, 1)),
        mx.transpose(weight.astype(x.dtype), (0, 2, 3, 1)),
        stride=1,
        padding=0,
    )
    if bias is not None:
        out = out + bias.astype(out.dtype)[None, None, None, :]
    return mx.transpose(out, (0, 3, 1, 2))


def _reflect_pad_nchw(x: mx.array, padding: int) -> mx.array:
    _, _, height, width = (int(dim) for dim in x.shape)
    if height <= 1 or width <= 1:
        raise ValueError("reflect padding requires spatial dimensions greater than 1")
    h_idx = _reflect_indices(height, padding)
    w_idx = _reflect_indices(width, padding)
    return x[:, :, h_idx, :][:, :, :, w_idx]


def _reflect_indices(length: int, padding: int) -> mx.array:
    raw = mx.arange(-padding, length + padding, dtype=mx.int32)
    period = 2 * length - 2
    wrapped = raw % period
    return mx.where(wrapped < length, wrapped, period - wrapped).astype(mx.int32)


def _group_norm_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array,
    *,
    groups: int,
    eps: float = 1e-5,
) -> mx.array:
    batch, channels, height, width = (int(dim) for dim in x.shape)
    group_count = min(int(groups), channels)
    if channels % group_count:
        group_count = 1
    values = x.astype(mx.float32).reshape(batch, group_count, channels // group_count, height, width)
    mean = mx.mean(values, axis=(2, 3, 4), keepdims=True)
    centered = values - mean
    var = mx.mean(centered * centered, axis=(2, 3, 4), keepdims=True)
    values = (centered * mx.rsqrt(var + eps)).reshape(batch, channels, height, width)
    values = values * weight.astype(values.dtype)[None, :, None, None] + bias.astype(values.dtype)[None, :, None, None]
    return values.astype(x.dtype)


def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int):
        return (int(size), int(size))
    if len(size) != 2:
        raise ValueError("size must be an int or (height, width)")
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("size dimensions must be positive")
    return height, width


def _silu(values: mx.array) -> mx.array:
    return values * mx.sigmoid(values)
