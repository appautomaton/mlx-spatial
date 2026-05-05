"""MLX BiRefNet/RMBG-2.0 inference for TRELLIS.2 preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .checkpoint import load_checkpoint_tensors
from .model_assets import RMBG2_ASSETS
from .trellis2_rmbg import _checkpoint_path


RMBG2_IMAGE_SIZE = 1024
RMBG2_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RMBG2_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Rmbg2MlxRuntimeError(RuntimeError):
    """Raised when the local RMBG-2.0 MLX forward path cannot run."""


@dataclass(frozen=True)
class Rmbg2MlxForwardResult:
    """RMBG-2.0 forward output for one image."""

    rgba: Image.Image
    mask: Image.Image


def remove_background_rmbg2_mlx(
    image: Image.Image,
    *,
    root: str | Path = RMBG2_ASSETS.root_hint,
) -> Image.Image:
    """Run local RMBG-2.0 BiRefNet weights in MLX and return an RGBA image."""

    result = run_rmbg2_mlx(image, root=root)
    return result.rgba


def run_rmbg2_mlx(
    image: Image.Image,
    *,
    root: str | Path = RMBG2_ASSETS.root_hint,
) -> Rmbg2MlxForwardResult:
    """Run local RMBG-2.0 BiRefNet weights in MLX and return the mask and RGBA image."""

    rgb = image.convert("RGB")
    model_input = _prepare_rmbg2_input(rgb)
    tensors = _load_rmbg2_forward_tensors(Path(root))
    try:
        logits = _BiRefNetMlx(tensors)(model_input)
    except KeyError as error:
        raise Rmbg2MlxRuntimeError(f"RMBG-2.0 checkpoint is missing tensor {error}") from error
    except ValueError as error:
        raise Rmbg2MlxRuntimeError(str(error)) from error

    mask = mx.sigmoid(logits[0, 0])
    mx.eval(mask)
    mask_np = np.array(mask, dtype=np.float32)
    mask_np = np.clip(mask_np, 0.0, 1.0)
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.resize(rgb.size)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(mask_img)
    return Rmbg2MlxForwardResult(rgba=rgba, mask=mask_img)


@lru_cache(maxsize=2)
def _load_rmbg2_forward_tensors(root: Path) -> dict[str, mx.array]:
    checkpoint_path = _checkpoint_path(root)
    return load_checkpoint_tensors(
        checkpoint_path,
        prefixes=("bb.", "decoder.", "squeeze_module."),
    )


def _prepare_rmbg2_input(image: Image.Image) -> mx.array:
    resized = image.resize((RMBG2_IMAGE_SIZE, RMBG2_IMAGE_SIZE), Image.Resampling.BILINEAR)
    values = np.asarray(resized, dtype=np.float32) / 255.0
    values = (values - RMBG2_MEAN) / RMBG2_STD
    nchw = np.transpose(values, (2, 0, 1))[None, ...]
    return mx.array(nchw, dtype=mx.float32)


class _BiRefNetMlx:
    def __init__(self, tensors: dict[str, mx.array]):
        self.tensors = tensors

    def __call__(self, x: mx.array) -> mx.array:
        x1, x2, x3, x4 = self._forward_encoder(x)
        x4 = _basic_dec_block(x4, "squeeze_module.0", self.tensors)
        return self._forward_decoder((x, x1, x2, x3, x4))

    def _forward_encoder(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        x1, x2, x3, x4 = _swin_v1_l_forward(x, "bb", self.tensors)
        half = _resize_nchw(x, (int(x.shape[2]) // 2, int(x.shape[3]) // 2))
        x1_half, x2_half, x3_half, x4_half = _swin_v1_l_forward(half, "bb", self.tensors)
        x1 = mx.concatenate((x1, _resize_nchw(x1_half, (int(x1.shape[2]), int(x1.shape[3])))), axis=1)
        x2 = mx.concatenate((x2, _resize_nchw(x2_half, (int(x2.shape[2]), int(x2.shape[3])))), axis=1)
        x3 = mx.concatenate((x3, _resize_nchw(x3_half, (int(x3.shape[2]), int(x3.shape[3])))), axis=1)
        x4 = mx.concatenate((x4, _resize_nchw(x4_half, (int(x4.shape[2]), int(x4.shape[3])))), axis=1)
        x4 = mx.concatenate(
            (
                _resize_nchw(x1, (int(x4.shape[2]), int(x4.shape[3]))),
                _resize_nchw(x2, (int(x4.shape[2]), int(x4.shape[3]))),
                _resize_nchw(x3, (int(x4.shape[2]), int(x4.shape[3]))),
                x4,
            ),
            axis=1,
        )
        return x1, x2, x3, x4

    def _forward_decoder(self, features: tuple[mx.array, mx.array, mx.array, mx.array, mx.array]) -> mx.array:
        x, x1, x2, x3, x4 = features

        x4 = mx.concatenate((x4, _simple_convs(_patches_batch(x, x4), "decoder.ipt_blk5", self.tensors)), axis=1)
        p4 = _basic_dec_block(x4, "decoder.decoder_block4", self.tensors)
        p4 = p4 * mx.sigmoid(_conv2d_nchw(_gdt_convs(p4, "decoder.gdt_convs_4", self.tensors), self.tensors["decoder.gdt_convs_attn_4.0.weight"], self.tensors["decoder.gdt_convs_attn_4.0.bias"], padding=0))
        p3_input = _resize_nchw(p4, (int(x3.shape[2]), int(x3.shape[3]))) + _conv2d_nchw(x3, self.tensors["decoder.lateral_block4.conv.weight"], self.tensors["decoder.lateral_block4.conv.bias"], padding=0)

        p3_input = mx.concatenate((p3_input, _simple_convs(_patches_batch(x, p3_input), "decoder.ipt_blk4", self.tensors)), axis=1)
        p3 = _basic_dec_block(p3_input, "decoder.decoder_block3", self.tensors)
        p3 = p3 * mx.sigmoid(_conv2d_nchw(_gdt_convs(p3, "decoder.gdt_convs_3", self.tensors), self.tensors["decoder.gdt_convs_attn_3.0.weight"], self.tensors["decoder.gdt_convs_attn_3.0.bias"], padding=0))
        p2_input = _resize_nchw(p3, (int(x2.shape[2]), int(x2.shape[3]))) + _conv2d_nchw(x2, self.tensors["decoder.lateral_block3.conv.weight"], self.tensors["decoder.lateral_block3.conv.bias"], padding=0)

        p2_input = mx.concatenate((p2_input, _simple_convs(_patches_batch(x, p2_input), "decoder.ipt_blk3", self.tensors)), axis=1)
        p2 = _basic_dec_block(p2_input, "decoder.decoder_block2", self.tensors)
        p2 = p2 * mx.sigmoid(_conv2d_nchw(_gdt_convs(p2, "decoder.gdt_convs_2", self.tensors), self.tensors["decoder.gdt_convs_attn_2.0.weight"], self.tensors["decoder.gdt_convs_attn_2.0.bias"], padding=0))
        p1_input = _resize_nchw(p2, (int(x1.shape[2]), int(x1.shape[3]))) + _conv2d_nchw(x1, self.tensors["decoder.lateral_block2.conv.weight"], self.tensors["decoder.lateral_block2.conv.bias"], padding=0)

        p1_input = mx.concatenate((p1_input, _simple_convs(_patches_batch(x, p1_input), "decoder.ipt_blk2", self.tensors)), axis=1)
        p1 = _basic_dec_block(p1_input, "decoder.decoder_block1", self.tensors)
        p1 = _resize_nchw(p1, (int(x.shape[2]), int(x.shape[3])))
        p1 = mx.concatenate((p1, _simple_convs(_patches_batch(x, p1), "decoder.ipt_blk1", self.tensors)), axis=1)
        return _conv2d_nchw(p1, self.tensors["decoder.conv_out1.0.weight"], self.tensors["decoder.conv_out1.0.bias"], padding=0)


def _swin_v1_l_forward(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> tuple[mx.array, ...]:
    embed_dim = 192
    depths = (2, 2, 18, 2)
    heads = (6, 12, 24, 48)
    window_size = 12
    dims = tuple(embed_dim * (2**index) for index in range(4))

    x = _swin_patch_embed(x, prefix, tensors)
    wh = int(x.shape[2])
    ww = int(x.shape[3])
    x = mx.reshape(mx.transpose(x, (0, 2, 3, 1)), (int(x.shape[0]), wh * ww, embed_dim))

    outputs: list[mx.array] = []
    for layer_index, depth in enumerate(depths):
        layer_prefix = f"{prefix}.layers.{layer_index}"
        x_out, out_h, out_w, x, wh, ww = _swin_basic_layer(
            x,
            wh,
            ww,
            layer_prefix,
            tensors,
            dim=dims[layer_index],
            depth=depth,
            num_heads=heads[layer_index],
            window_size=window_size,
            downsample=layer_index < len(depths) - 1,
        )
        normed = _layer_norm(x_out, tensors[f"{prefix}.norm{layer_index}.weight"], tensors[f"{prefix}.norm{layer_index}.bias"])
        out = mx.reshape(normed, (int(normed.shape[0]), out_h, out_w, dims[layer_index]))
        outputs.append(mx.transpose(out, (0, 3, 1, 2)))

    return tuple(outputs)


def _swin_patch_embed(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    patch_size = 4
    pad_h = (patch_size - int(x.shape[2]) % patch_size) % patch_size
    pad_w = (patch_size - int(x.shape[3]) % patch_size) % patch_size
    if pad_h or pad_w:
        x = mx.pad(x, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))
    x = _conv2d_nchw(
        x,
        tensors[f"{prefix}.patch_embed.proj.weight"],
        tensors[f"{prefix}.patch_embed.proj.bias"],
        stride=patch_size,
        padding=0,
    )
    bsz, channels, height, width = (int(dim) for dim in x.shape)
    tokens = mx.reshape(mx.transpose(x, (0, 2, 3, 1)), (bsz, height * width, channels))
    tokens = _layer_norm(tokens, tensors[f"{prefix}.patch_embed.norm.weight"], tensors[f"{prefix}.patch_embed.norm.bias"])
    return mx.transpose(mx.reshape(tokens, (bsz, height, width, channels)), (0, 3, 1, 2))


def _swin_basic_layer(
    x: mx.array,
    height: int,
    width: int,
    prefix: str,
    tensors: dict[str, mx.array],
    *,
    dim: int,
    depth: int,
    num_heads: int,
    window_size: int,
    downsample: bool,
) -> tuple[mx.array, int, int, mx.array, int, int]:
    attn_mask = _swin_attention_mask(height, width, window_size, x.dtype)
    for block_index in range(depth):
        shift_size = 0 if block_index % 2 == 0 else window_size // 2
        x = _swin_block(
            x,
            height,
            width,
            f"{prefix}.blocks.{block_index}",
            tensors,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            attn_mask=attn_mask,
        )
    x_out = x
    if not downsample:
        return x_out, height, width, x, height, width
    x_down = _patch_merging(x, height, width, f"{prefix}.downsample", tensors)
    return x_out, height, width, x_down, (height + 1) // 2, (width + 1) // 2


def _swin_block(
    x: mx.array,
    height: int,
    width: int,
    prefix: str,
    tensors: dict[str, mx.array],
    *,
    dim: int,
    num_heads: int,
    window_size: int,
    shift_size: int,
    attn_mask: mx.array,
) -> mx.array:
    shortcut = x
    x = _layer_norm(x, tensors[f"{prefix}.norm1.weight"], tensors[f"{prefix}.norm1.bias"])
    x = mx.reshape(x, (int(x.shape[0]), height, width, dim))
    pad_b = (window_size - height % window_size) % window_size
    pad_r = (window_size - width % window_size) % window_size
    if pad_b or pad_r:
        x = mx.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
    hp = int(x.shape[1])
    wp = int(x.shape[2])
    if shift_size:
        shifted = mx.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))
        mask = attn_mask
    else:
        shifted = x
        mask = None
    windows = _window_partition(shifted, window_size)
    windows = mx.reshape(windows, (-1, window_size * window_size, dim))
    attended = _window_attention(windows, f"{prefix}.attn", tensors, num_heads=num_heads, window_size=window_size, mask=mask)
    attended = mx.reshape(attended, (-1, window_size, window_size, dim))
    shifted = _window_reverse(attended, window_size, hp, wp)
    if shift_size:
        x = mx.roll(shifted, shift=(shift_size, shift_size), axis=(1, 2))
    else:
        x = shifted
    if pad_b or pad_r:
        x = x[:, :height, :width, :]
    x = mx.reshape(x, (int(x.shape[0]), height * width, dim))
    x = shortcut + x
    return x + _swin_mlp(_layer_norm(x, tensors[f"{prefix}.norm2.weight"], tensors[f"{prefix}.norm2.bias"]), f"{prefix}.mlp", tensors)


def _window_attention(
    x: mx.array,
    prefix: str,
    tensors: dict[str, mx.array],
    *,
    num_heads: int,
    window_size: int,
    mask: mx.array | None,
) -> mx.array:
    batch_windows, tokens, channels = (int(dim) for dim in x.shape)
    head_dim = channels // num_heads
    qkv = _linear(x, tensors[f"{prefix}.qkv.weight"], tensors[f"{prefix}.qkv.bias"])
    qkv = mx.reshape(qkv, (batch_windows, tokens, 3, num_heads, head_dim))
    qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
    q = qkv[0] * (head_dim**-0.5)
    k = qkv[1]
    v = qkv[2]
    attn = q @ mx.transpose(k, (0, 1, 3, 2))

    table = tensors[f"{prefix}.relative_position_bias_table"]
    index = tensors[f"{prefix}.relative_position_index"].astype(mx.int32)
    bias = mx.take(table, mx.reshape(index, (-1,)), axis=0)
    bias = mx.reshape(bias, (window_size * window_size, window_size * window_size, num_heads))
    bias = mx.transpose(bias, (2, 0, 1))
    attn = attn + bias[None, :, :, :]
    if mask is not None:
        num_windows = int(mask.shape[0])
        attn = mx.reshape(attn, (batch_windows // num_windows, num_windows, num_heads, tokens, tokens))
        attn = attn + mask[None, :, None, :, :]
        attn = mx.reshape(attn, (-1, num_heads, tokens, tokens))
    attn = mx.softmax(attn, axis=-1)
    out = attn @ v
    out = mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (batch_windows, tokens, channels))
    return _linear(out, tensors[f"{prefix}.proj.weight"], tensors[f"{prefix}.proj.bias"])


def _swin_mlp(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    x = _linear(x, tensors[f"{prefix}.fc1.weight"], tensors[f"{prefix}.fc1.bias"])
    x = _gelu(x)
    return _linear(x, tensors[f"{prefix}.fc2.weight"], tensors[f"{prefix}.fc2.bias"])


def _patch_merging(x: mx.array, height: int, width: int, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    batch, _, channels = (int(dim) for dim in x.shape)
    x = mx.reshape(x, (batch, height, width, channels))
    if height % 2 or width % 2:
        x = mx.pad(x, ((0, 0), (0, height % 2), (0, width % 2), (0, 0)))
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = mx.concatenate((x0, x1, x2, x3), axis=-1)
    x = mx.reshape(x, (batch, -1, 4 * channels))
    x = _layer_norm(x, tensors[f"{prefix}.norm.weight"], tensors[f"{prefix}.norm.bias"])
    return _linear(x, tensors[f"{prefix}.reduction.weight"], None)


@lru_cache(maxsize=32)
def _swin_attention_mask(height: int, width: int, window_size: int, dtype: mx.Dtype) -> mx.array:
    shift_size = window_size // 2
    hp = int(np.ceil(height / window_size)) * window_size
    wp = int(np.ceil(width / window_size)) * window_size
    img_mask = np.zeros((1, hp, wp, 1), dtype=np.float32)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    count = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = count
            count += 1
    windows = _window_partition_np(img_mask, window_size).reshape(-1, window_size * window_size)
    mask = windows[:, None, :] - windows[:, :, None]
    mask = np.where(mask != 0, -100.0, 0.0).astype(np.float32)
    return mx.array(mask, dtype=dtype)


def _window_partition(x: mx.array, window_size: int) -> mx.array:
    batch, height, width, channels = (int(dim) for dim in x.shape)
    x = mx.reshape(x, (batch, height // window_size, window_size, width // window_size, window_size, channels))
    x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
    return mx.reshape(x, (-1, window_size, window_size, channels))


def _window_partition_np(x: np.ndarray, window_size: int) -> np.ndarray:
    batch, height, width, channels = x.shape
    x = x.reshape(batch, height // window_size, window_size, width // window_size, window_size, channels)
    x = np.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(-1, window_size, window_size, channels)


def _window_reverse(windows: mx.array, window_size: int, height: int, width: int) -> mx.array:
    batch = int(windows.shape[0]) // (height * width // window_size // window_size)
    x = mx.reshape(windows, (batch, height // window_size, width // window_size, window_size, window_size, -1))
    x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
    return mx.reshape(x, (batch, height, width, -1))


def _basic_dec_block(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    x = _conv2d_nchw(x, tensors[f"{prefix}.conv_in.weight"], tensors[f"{prefix}.conv_in.bias"])
    x = _batch_norm_nchw(x, prefix=f"{prefix}.bn_in", tensors=tensors)
    x = _relu(x)
    x = _aspp_deformable(x, f"{prefix}.dec_att", tensors)
    x = _conv2d_nchw(x, tensors[f"{prefix}.conv_out.weight"], tensors[f"{prefix}.conv_out.bias"])
    return _batch_norm_nchw(x, prefix=f"{prefix}.bn_out", tensors=tensors)


def _aspp_deformable(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    branch_1 = _aspp_deformable_module(x, f"{prefix}.aspp1", tensors, padding=0)
    branches = [
        _aspp_deformable_module(x, f"{prefix}.aspp_deforms.0", tensors, padding=0),
        _aspp_deformable_module(x, f"{prefix}.aspp_deforms.1", tensors, padding=1),
        _aspp_deformable_module(x, f"{prefix}.aspp_deforms.2", tensors, padding=3),
    ]
    pooled = mx.mean(x, axis=(2, 3), keepdims=True)
    pooled = _conv2d_nchw(pooled, tensors[f"{prefix}.global_avg_pool.1.weight"], tensors.get(f"{prefix}.global_avg_pool.1.bias"), padding=0)
    pooled = _batch_norm_nchw(pooled, prefix=f"{prefix}.global_avg_pool.2", tensors=tensors)
    pooled = _relu(pooled)
    pooled = _resize_nchw(pooled, (int(branch_1.shape[2]), int(branch_1.shape[3])))
    x = mx.concatenate((branch_1, *branches, pooled), axis=1)
    x = _conv2d_nchw(x, tensors[f"{prefix}.conv1.weight"], tensors.get(f"{prefix}.conv1.bias"), padding=0)
    x = _batch_norm_nchw(x, prefix=f"{prefix}.bn1", tensors=tensors)
    return _relu(x)


def _aspp_deformable_module(x: mx.array, prefix: str, tensors: dict[str, mx.array], *, padding: int) -> mx.array:
    x = _deformable_conv2d_forward(x, f"{prefix}.atrous_conv", tensors, padding=padding)
    x = _batch_norm_nchw(x, prefix=f"{prefix}.bn", tensors=tensors)
    return _relu(x)


def _deformable_conv2d_forward(x: mx.array, prefix: str, tensors: dict[str, mx.array], *, padding: int) -> mx.array:
    offset = _conv2d_nchw(x, tensors[f"{prefix}.offset_conv.weight"], tensors[f"{prefix}.offset_conv.bias"], padding=padding)
    modulator = 2.0 * mx.sigmoid(_conv2d_nchw(x, tensors[f"{prefix}.modulator_conv.weight"], tensors[f"{prefix}.modulator_conv.bias"], padding=padding))
    return _deform_conv2d_nchw(
        x,
        tensors[f"{prefix}.regular_conv.weight"],
        tensors.get(f"{prefix}.regular_conv.bias"),
        offset,
        modulator,
        padding=padding,
    )


def _deform_conv2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    offset: mx.array,
    mask: mx.array,
    *,
    padding: int,
) -> mx.array:
    batch, channels, in_h, in_w = (int(dim) for dim in x.shape)
    out_channels, weight_channels, kernel_h, kernel_w = (int(dim) for dim in weight.shape)
    if channels != weight_channels:
        raise ValueError(f"deform conv channel mismatch: input {channels}, weight {weight_channels}")
    out_h = int(offset.shape[2])
    out_w = int(offset.shape[3])
    if int(offset.shape[1]) != 2 * kernel_h * kernel_w:
        raise ValueError(f"deform conv offset channel mismatch for {tuple(weight.shape)} and {tuple(offset.shape)}")
    if int(mask.shape[1]) != kernel_h * kernel_w:
        raise ValueError(f"deform conv mask channel mismatch for {tuple(weight.shape)} and {tuple(mask.shape)}")

    padded = mx.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding))) if padding else x
    base_y = mx.arange(out_h, dtype=mx.float32)[None, :, None]
    base_x = mx.arange(out_w, dtype=mx.float32)[None, None, :]
    output = mx.zeros((batch * out_h * out_w, out_channels), dtype=x.dtype)
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            kernel_index = ky * kernel_w + kx
            yy = base_y + float(ky) + offset[:, 2 * kernel_index, :, :]
            xx = base_x + float(kx) + offset[:, 2 * kernel_index + 1, :, :]
            sampled = _bilinear_sample_nchw(padded, yy, xx)
            sampled = sampled * mask[:, kernel_index : kernel_index + 1, :, :]
            sampled = mx.reshape(mx.transpose(sampled, (0, 2, 3, 1)), (batch * out_h * out_w, channels))
            kernel_weight = weight[:, :, ky, kx]
            output = output + sampled @ mx.transpose(kernel_weight)
    if bias is not None:
        output = output + bias
    output = mx.reshape(output, (batch, out_h, out_w, out_channels))
    return mx.transpose(output, (0, 3, 1, 2))


def _bilinear_sample_nchw(x: mx.array, y: mx.array, x_coord: mx.array) -> mx.array:
    batch, channels, height, width = (int(dim) for dim in x.shape)
    y = mx.clip(y, 0.0, float(height - 1))
    x_coord = mx.clip(x_coord, 0.0, float(width - 1))
    y0 = mx.floor(y).astype(mx.int32)
    x0 = mx.floor(x_coord).astype(mx.int32)
    y1 = mx.minimum(y0 + 1, height - 1)
    x1 = mx.minimum(x0 + 1, width - 1)

    y0f = y0.astype(mx.float32)
    x0f = x0.astype(mx.float32)
    wy = y - y0f
    wx = x_coord - x0f

    v00 = _gather_spatial_nchw(x, y0, x0)
    v01 = _gather_spatial_nchw(x, y0, x1)
    v10 = _gather_spatial_nchw(x, y1, x0)
    v11 = _gather_spatial_nchw(x, y1, x1)
    wa = (1.0 - wy) * (1.0 - wx)
    wb = (1.0 - wy) * wx
    wc = wy * (1.0 - wx)
    wd = wy * wx
    return v00 * wa[:, None, :, :] + v01 * wb[:, None, :, :] + v10 * wc[:, None, :, :] + v11 * wd[:, None, :, :]


def _gather_spatial_nchw(x: mx.array, y: mx.array, x_coord: mx.array) -> mx.array:
    batch, channels, height, width = (int(dim) for dim in x.shape)
    flat = mx.reshape(x, (batch, channels, height * width))
    indices = mx.reshape(y * width + x_coord, (batch, 1, -1))
    indices = mx.broadcast_to(indices, (batch, channels, int(indices.shape[-1])))
    gathered = mx.take_along_axis(flat, indices, axis=2)
    return mx.reshape(gathered, (batch, channels, int(y.shape[1]), int(y.shape[2])))


def _gdt_convs(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    x = _conv2d_nchw(x, tensors[f"{prefix}.0.weight"], tensors[f"{prefix}.0.bias"])
    x = _batch_norm_nchw(x, prefix=f"{prefix}.1", tensors=tensors)
    return _relu(x)


def _simple_convs(x: mx.array, prefix: str, tensors: dict[str, mx.array]) -> mx.array:
    x = _conv2d_nchw(x, tensors[f"{prefix}.conv1.weight"], tensors[f"{prefix}.conv1.bias"])
    return _conv2d_nchw(x, tensors[f"{prefix}.conv_out.weight"], tensors[f"{prefix}.conv_out.bias"])


def _patches_batch(x: mx.array, target: mx.array) -> mx.array:
    batch, channels, height, width = (int(dim) for dim in x.shape)
    patch_h = int(target.shape[2])
    patch_w = int(target.shape[3])
    if height % patch_h or width % patch_w:
        raise ValueError(f"RMBG decoder input patches require divisible shapes, got image {(height, width)} target {(patch_h, patch_w)}")
    h_blocks = height // patch_h
    w_blocks = width // patch_w
    patches = mx.reshape(x, (batch, channels, h_blocks, patch_h, w_blocks, patch_w))
    patches = mx.transpose(patches, (0, 4, 2, 1, 3, 5))
    return mx.reshape(patches, (batch, w_blocks * h_blocks * channels, patch_h, patch_w))


def _conv2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight.astype(x.dtype), (0, 2, 3, 1))
    output = mx.conv2d(x_nhwc, weight_ohwi, stride=stride, padding=padding, dilation=dilation)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _batch_norm_nchw(x: mx.array, *, prefix: str, tensors: dict[str, mx.array], eps: float = 1e-5) -> mx.array:
    weight = tensors[f"{prefix}.weight"].astype(x.dtype)[None, :, None, None]
    bias = tensors[f"{prefix}.bias"].astype(x.dtype)[None, :, None, None]
    running_mean = tensors[f"{prefix}.running_mean"].astype(x.dtype)[None, :, None, None]
    running_var = tensors[f"{prefix}.running_var"].astype(x.dtype)[None, :, None, None]
    return ((x - running_mean) * mx.rsqrt(running_var + eps)) * weight + bias


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-5) -> mx.array:
    return mx.fast.layer_norm(x.astype(mx.float32), weight.astype(mx.float32), bias.astype(mx.float32), eps).astype(x.dtype)


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = x @ mx.transpose(weight.astype(x.dtype))
    if bias is not None:
        output = output + bias.astype(output.dtype)
    return output


def _resize_nchw(x: mx.array, size: tuple[int, int]) -> mx.array:
    height, width = (int(dim) for dim in size)
    if int(x.shape[2]) == height and int(x.shape[3]) == width:
        return x
    scale = (height / int(x.shape[2]), width / int(x.shape[3]))
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    output = nn.Upsample(scale, mode="linear", align_corners=True)(x_nhwc)
    if int(output.shape[1]) != height or int(output.shape[2]) != width:
        output = output[:, :height, :width, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / np.sqrt(2.0)))
