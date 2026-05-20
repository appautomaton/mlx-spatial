"""DINO Vision Transformer backbone for HY-World 2.0.

Gap ID: HW-09 (PatchEmbed + DinoVisionTransformer). Matches
``vendors/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/layers/vision_transformer.py``
and the DINO patch-embed path in the VGT forward pass.

The DinoVisionTransformer takes raw images, applies patch convolution,
adds positional embeddings and register tokens, then runs a depth-long
sequence of DINO ViT blocks. The primary output is ``x_norm_patchtokens``,
which the VGT uses as its patch token input.

This module uses the tensor-dict pattern: all weights are looked up by key
in a ``tensors: dict[str, mx.array]`` rather than stored as nn.Module
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .hyworld2_layers import layer_norm
from .hyworld2_transformer import run_dino_block


_RESNET_MEAN = (0.485, 0.456, 0.406)
_RESNET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DinoVisionTransformer:
    """Callable MLX facade for HY-World's DINO ViT patch backbone."""

    config: object
    tensors: dict[str, mx.array]
    patch_grid: tuple[int, int] | None = None

    def extract_patch_tokens(
        self,
        image_tensor: mx.array,
    ) -> tuple[mx.array | None, object | None]:
        patch_grid = self.patch_grid
        if patch_grid is None:
            height = int(image_tensor.shape[-2])
            width = int(image_tensor.shape[-1])
            patch_size = _int_or(self.config.patch_size)
            patch_grid = (height // patch_size, width // patch_size)
        return run_dino_vit(
            image_tensor,
            self.config,
            self.tensors,
            patch_grid=patch_grid,
        )

    def forward_features(
        self,
        image_tensor: mx.array,
    ) -> tuple[dict[str, mx.array] | None, object | None]:
        patch_tokens, blocker = self.extract_patch_tokens(image_tensor)
        if blocker is not None or patch_tokens is None:
            return None, blocker
        return {"x_norm_patchtokens": patch_tokens}, None

    def __call__(
        self,
        image_tensor: mx.array,
    ) -> tuple[mx.array | None, object | None]:
        return self.extract_patch_tokens(image_tensor)


def run_dino_vit(
    image_tensor: mx.array,
    config: object,
    tensors: dict[str, mx.array],
    *,
    patch_grid: tuple[int, int],
) -> tuple[mx.array | None, object | None]:
    """Run the DINO ViT backbone to extract patch tokens from images.

    Implements: patch conv → cls_token + pos_embed + register_tokens →
    depth × DinoBlock → final LayerNorm → extract patch tokens.

    Args:
        image_tensor: ``(B, S, 3, H, W)`` input images (float, [0,1] range).
        config: VisualGeometryTransformerConfig with ``patch_size``,
            ``embed_dim``, ``depth``, ``num_register_tokens``, etc.
        tensors: Weight dictionary with ``patch_embed.*`` keys.
        patch_grid: ``(grid_h, grid_w)`` number of patches.

    Returns:
        ``(patch_tokens, blocker)`` where patch_tokens is
        ``(B, S, P, D)`` on success and blocker is ``None``.
        On failure, patch_tokens is ``None`` and blocker describes the error.
    """
    required = (
        "patch_embed.patch_embed.proj.weight",
        "patch_embed.patch_embed.proj.bias",
        "patch_embed.cls_token",
        "patch_embed.pos_embed",
        "patch_embed.register_tokens",
        "patch_embed.norm.weight",
        "patch_embed.norm.bias",
    )
    missing = tuple(name for name in required if name not in tensors)
    if missing:
        from .hyworld2_transformer import _make_blocker
        return None, _make_blocker(
            "model-construction",
            "HY-World DINO patch embedding tensor lookup",
            f"missing tensor for DINO patch embedding: {missing[0]}",
            {"missing": missing},
        )

    batch, frames, channels, height, width = (
        int(d) for d in image_tensor.shape
    )
    embed_dim = _int_or(config.embed_dim)
    depth = _int_or(config.depth)
    num_register_tokens = _int_or(config.num_register_tokens)

    reshaped = mx.reshape(image_tensor, (batch * frames, channels, height, width))
    mean = mx.array(_RESNET_MEAN, dtype=reshaped.dtype)[None, :, None, None]
    std = mx.array(_RESNET_STD, dtype=reshaped.dtype)[None, :, None, None]
    normalized = (reshaped - mean) / std

    proj_weight = tensors["patch_embed.patch_embed.proj.weight"]
    proj_bias = tensors["patch_embed.patch_embed.proj.bias"]
    patch_map = _conv2d_nchw(
        normalized,
        proj_weight,
        proj_bias,
        stride=_int_or(config.patch_size),
        padding=0,
    )
    patch_tokens = mx.reshape(
        mx.transpose(patch_map, (0, 2, 3, 1)),
        (batch * frames, patch_grid[0] * patch_grid[1], embed_dim),
    )

    cls_token = mx.broadcast_to(
        tensors["patch_embed.cls_token"].astype(patch_tokens.dtype),
        (batch * frames, 1, embed_dim),
    )
    dino_tokens = mx.concatenate((cls_token, patch_tokens), axis=1)

    pos_embed, pos_blocker = _interpolate_dino_pos_embed(
        tensors["patch_embed.pos_embed"], patch_grid
    )
    if pos_blocker is not None or pos_embed is None:
        from .hyworld2_transformer import _make_blocker
        return None, pos_blocker if pos_blocker is not None else _make_blocker(
            "model-construction",
            "HY-World DINO positional embedding",
            "unknown error",
            {},
        )
    dino_tokens = dino_tokens + pos_embed.astype(dino_tokens.dtype)

    register_tokens = mx.broadcast_to(
        tensors["patch_embed.register_tokens"].astype(dino_tokens.dtype),
        (batch * frames, num_register_tokens, embed_dim),
    )
    dino_tokens = mx.concatenate(
        (dino_tokens[:, :1, :], register_tokens, dino_tokens[:, 1:, :]), axis=1
    )

    for block_index in range(depth):
        dino_tokens, block_blocker = run_dino_block(
            dino_tokens, config, tensors, block_index=block_index
        )
        if block_blocker is not None:
            return None, block_blocker

    dino_tokens = layer_norm(
        dino_tokens,
        tensors["patch_embed.norm.weight"],
        tensors["patch_embed.norm.bias"],
        eps=1e-6,
    )

    extracted = dino_tokens[:, 1 + num_register_tokens :, :]
    patch_tokens_out = mx.reshape(
        extracted,
        (batch, frames, patch_grid[0] * patch_grid[1], embed_dim),
    )
    return patch_tokens_out, None


def interpolate_dino_pos_embed(
    pos_embed: mx.array,
    patch_grid: tuple[int, int],
) -> tuple[mx.array | None, object | None]:
    """Interpolate DINO positional embeddings for the given patch grid.

    Public wrapper for testing.
    """
    return _interpolate_dino_pos_embed(pos_embed, patch_grid)


def _interpolate_dino_pos_embed(
    pos_embed: mx.array,
    patch_grid: tuple[int, int],
) -> tuple[mx.array | None, object | None]:
    """Bicubic interpolation of DINO positional embeddings."""
    from .hyworld2_transformer import _make_blocker

    patch_count = patch_grid[0] * patch_grid[1]
    expected = patch_count + 1
    if int(pos_embed.shape[1]) == expected:
        return pos_embed, None

    stored_patch_count = int(pos_embed.shape[1]) - 1
    stored_side = int(stored_patch_count**0.5)
    if stored_side * stored_side != stored_patch_count:
        return None, _make_blocker(
            "model-construction",
            "HY-World DINO positional embedding interpolation",
            "checkpoint DINO positional embedding does not contain a square patch grid",
            {
                "checkpoint_pos_embed_shape": tuple(pos_embed.shape),
                "checkpoint_patch_grid": stored_patch_count,
                "requested_patch_grid": patch_grid,
            },
        )

    try:
        from scipy import ndimage
    except ModuleNotFoundError:
        return None, _make_blocker(
            "model-construction",
            "HY-World DINO positional embedding interpolation",
            "scipy is required for exact bicubic DINO positional embedding interpolation",
            {
                "checkpoint_pos_embed_shape": tuple(pos_embed.shape),
                "checkpoint_patch_grid": (stored_side, stored_side),
                "requested_patch_grid": patch_grid,
            },
        )

    pos_np = np.array(pos_embed, dtype=np.float32)
    cls_pos, patch_pos = pos_np[:, :1, :], pos_np[:, 1:, :]
    embed_dim = int(pos_embed.shape[-1])
    side_in = stored_side
    reshaped = patch_pos.reshape(1, side_in, side_in, embed_dim)
    target_shape = (1, patch_grid[0], patch_grid[1], embed_dim)
    zoom_h = patch_grid[0] / side_in
    zoom_w = patch_grid[1] / side_in
    interpolated = _zoom_ndarray(reshaped, (1, zoom_h, zoom_w, 1), order=3)
    if interpolated.shape != target_shape:
        interpolated = _fit_resized_pos_embed(interpolated, patch_grid, embed_dim)
    result = np.concatenate([cls_pos, interpolated.reshape(1, -1, embed_dim)], axis=1)
    return mx.array(result, dtype=pos_embed.dtype), None


def _zoom_ndarray(arr: np.ndarray, zoom_factors: tuple, order: int = 3) -> np.ndarray:
    """Apply scipy zoom with fallback."""
    from scipy.ndimage import zoom
    return zoom(arr, zoom_factors, order=order)


def _fit_resized_pos_embed(
    values: np.ndarray,
    patch_grid: tuple[int, int],
    embed_dim: int,
) -> np.ndarray:
    """Pad/crop resized positional embeddings to the exact target shape."""
    target_h, target_w = patch_grid
    h, w = values.shape[1], values.shape[2]
    output = values
    if h < target_h or w < target_w:
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        output = np.pad(output, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return output[:1, :target_h, :target_w, :embed_dim]


def _conv2d_nchw(
    values: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    """Conv2d operating on NCHW tensors with OIHW weights."""
    nhwc = mx.transpose(values, (0, 2, 3, 1))
    hwio = mx.transpose(weight, (0, 2, 3, 1))
    result = mx.conv2d(nhwc, hwio, stride=stride, padding=padding)
    if bias is not None:
        result = result + bias
    return mx.transpose(result, (0, 3, 1, 2))


def _int_or(value) -> int:
    if isinstance(value, int):
        return value
    return int(value)
