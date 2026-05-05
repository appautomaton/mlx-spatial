"""SAM 3D Objects condition embedder primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .checkpoint import load_checkpoint_tensors


SAM3D_CONDITION_PREFIX = "_base_models.condition_embedder."
SAM3D_DINO_IMAGE_MEAN = (0.485, 0.456, 0.406)
SAM3D_DINO_IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sam3dDinoConfig:
    input_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    num_heads: int = 16
    num_blocks: int = 24
    register_count: int = 4
    prenorm_features: bool = False
    normalize_images: bool = True
    attention_chunk_size: int = 192
    max_attention_bytes: int = 1_600_000_000


@dataclass(frozen=True)
class Sam3dPointPatchConfig:
    input_size: int = 256
    patch_size: int = 8
    embed_dim: int = 512
    num_heads: int = 16
    mlp_ratio: float = 2.0
    remap_output: str = "linear"


@dataclass(frozen=True)
class Sam3dConditionStackOutput:
    tokens: mx.array
    metadata: dict[str, object]


def load_sam3d_condition_tensors(path: str | Path) -> dict[str, mx.array]:
    """Load active SAM3D condition embedder tensors with the checkpoint prefix removed."""

    tensors = load_checkpoint_tensors(path, prefixes=(SAM3D_CONDITION_PREFIX,))
    return {
        key[len(SAM3D_CONDITION_PREFIX) :]: value
        for key, value in tensors.items()
        if key.startswith(SAM3D_CONDITION_PREFIX)
    }


def run_sam3d_dino_vitl14_reg(
    image: np.ndarray | mx.array,
    tensors: dict[str, mx.array],
    *,
    module_index: int,
    config: Sam3dDinoConfig = Sam3dDinoConfig(),
) -> mx.array:
    """Run the active SAM3D DINOv2 ViT-L/14-reg embedder in MLX."""

    prefix = f"module_list.{module_index}.backbone."
    x = _image_to_bchw(image).astype(mx.float32)
    if int(x.shape[1]) == 1:
        x = mx.broadcast_to(x, (int(x.shape[0]), 3, int(x.shape[2]), int(x.shape[3])))
    if int(x.shape[1]) != 3:
        raise ValueError(f"SAM3D DINO input must have 1 or 3 channels, got {tuple(x.shape)}")
    if int(x.shape[2]) != config.input_size or int(x.shape[3]) != config.input_size:
        x = _resize_linear_bchw(x, (config.input_size, config.input_size))
    if config.normalize_images:
        mean = mx.array(SAM3D_DINO_IMAGE_MEAN, dtype=x.dtype)[None, :, None, None]
        std = mx.array(SAM3D_DINO_IMAGE_STD, dtype=x.dtype)[None, :, None, None]
        x = (x - mean) / std

    patch_map = _conv2d_nchw(
        x,
        tensors[f"{prefix}patch_embed.proj.weight"],
        tensors[f"{prefix}patch_embed.proj.bias"],
        stride=config.patch_size,
        padding=0,
    )
    patch_h, patch_w = int(patch_map.shape[2]), int(patch_map.shape[3])
    patch_tokens = mx.reshape(
        mx.transpose(patch_map, (0, 2, 3, 1)),
        (int(patch_map.shape[0]), patch_h * patch_w, config.embed_dim),
    )
    cls = mx.broadcast_to(
        tensors[f"{prefix}cls_token"].astype(patch_tokens.dtype),
        (int(patch_tokens.shape[0]), 1, config.embed_dim),
    )
    tokens = mx.concatenate((cls, patch_tokens), axis=1)
    tokens = tokens + _interpolate_dino_pos_embed(
        tensors[f"{prefix}pos_embed"],
        (patch_h, patch_w),
        config=config,
    ).astype(tokens.dtype)
    registers = mx.broadcast_to(
        tensors[f"{prefix}register_tokens"].astype(tokens.dtype),
        (int(tokens.shape[0]), config.register_count, config.embed_dim),
    )
    hidden = mx.concatenate((tokens[:, :1, :], registers, tokens[:, 1:, :]), axis=1)

    for block_index in range(config.num_blocks):
        hidden = _run_dino_block(
            hidden,
            tensors,
            prefix=f"{prefix}blocks.{block_index}",
            config=config,
        )
        mx.eval(hidden)

    if config.prenorm_features:
        return _layer_norm(hidden, mx.ones((config.embed_dim,), dtype=hidden.dtype), mx.zeros((config.embed_dim,), dtype=hidden.dtype), eps=1e-5)

    normalized = _layer_norm(
        hidden,
        tensors[f"{prefix}norm.weight"],
        tensors[f"{prefix}norm.bias"],
        eps=1e-6,
    )
    return mx.concatenate(
        (
            normalized[:, :1, :],
            normalized[:, 1 + config.register_count :, :],
        ),
        axis=1,
    )


def run_sam3d_point_patch_embed(
    pointmap: np.ndarray | mx.array,
    tensors: dict[str, mx.array],
    config: Sam3dPointPatchConfig = Sam3dPointPatchConfig(),
    *,
    prefix: str = "",
) -> mx.array:
    """Port official `PointPatchEmbed` for SAM3D pointmap conditioning."""

    xyz = _pointmap_to_bchw(pointmap)
    xyz = _resize_nearest_bchw(xyz, config.input_size)
    valid_mask = mx.all(mx.isfinite(xyz), axis=1)
    xyz = mx.transpose(xyz, (0, 2, 3, 1))
    xyz_safe = mx.where(valid_mask[..., None], xyz, mx.zeros_like(xyz))
    xyz_remapped = _remap_points(xyz_safe, config.remap_output)
    x = _linear(xyz_remapped, tensors[f"{prefix}point_proj.weight"], tensors[f"{prefix}point_proj.bias"])
    invalid = tensors[f"{prefix}invalid_xyz_token"].astype(x.dtype)
    x = mx.where(valid_mask[..., None], x, invalid[None, None, None, :])
    batch, height, width, embed_dim = tuple(int(value) for value in x.shape)
    if height % config.patch_size or width % config.patch_size:
        raise ValueError("SAM3D pointmap input_size must be divisible by patch_size")
    windows_h = height // config.patch_size
    windows_w = width // config.patch_size
    x = mx.reshape(
        x,
        (batch, windows_h, config.patch_size, windows_w, config.patch_size, embed_dim),
    )
    x = mx.transpose(x, (0, 1, 3, 2, 4, 5))
    x = mx.reshape(x, (batch * windows_h * windows_w, config.patch_size * config.patch_size, embed_dim))
    cls = mx.broadcast_to(
        tensors[f"{prefix}cls_token"].astype(x.dtype),
        (int(x.shape[0]), 1, embed_dim),
    )
    tokens = mx.concatenate((cls, x), axis=1)
    tokens = tokens + tensors[f"{prefix}pos_embed_window"].astype(tokens.dtype)
    tokens = _point_patch_block(tokens, tensors, f"{prefix}blocks.0", config)
    cls_tokens = mx.reshape(tokens[:, 0, :], (batch, windows_h * windows_w, embed_dim))
    pos_patch = _point_patch_pos_embed(tensors[f"{prefix}pos_embed"], windows_h, windows_w)
    return cls_tokens + pos_patch.astype(cls_tokens.dtype)


def run_sam3d_projection_net(tokens: mx.array, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    """Run EmbedderFuser projection net: LayerNorm + Llama FeedForward."""

    hidden = _layer_norm(tokens, tensors[f"{prefix}0.weight"], tensors[f"{prefix}0.bias"], eps=1e-5)
    gate = _silu(_linear(hidden, tensors[f"{prefix}1.w1.weight"], None))
    up = _linear(hidden, tensors[f"{prefix}1.w3.weight"], None)
    return _linear(gate * up, tensors[f"{prefix}1.w2.weight"], None)


def fuse_sam3d_condition_tokens(
    projected_tokens: tuple[mx.array, ...],
    idx_emb: mx.array,
    *,
    positional_indices: tuple[int, ...],
) -> mx.array:
    """Concatenate projected condition tokens with learned position embeddings."""

    if len(projected_tokens) != len(positional_indices):
        raise ValueError("projected token count must match positional index count")
    fused = []
    for tokens, index in zip(projected_tokens, positional_indices, strict=True):
        fused.append(tokens + idx_emb[index : index + 1, None].astype(tokens.dtype))
    return mx.concatenate(fused, axis=1)


def run_sam3d_ss_condition_stack(
    official,
    tensors: dict[str, mx.array],
    *,
    dino_config: Sam3dDinoConfig = Sam3dDinoConfig(),
    point_config: Sam3dPointPatchConfig = Sam3dPointPatchConfig(),
) -> Sam3dConditionStackOutput:
    """Run the active sparse-structure condition fuser for official preprocessed tensors."""

    if official.pointmap is None or official.rgb_pointmap is None:
        raise ValueError("SAM3D SS condition stack requires cropped and full pointmaps")
    chunks: list[mx.array] = []
    token_metadata: list[dict[str, object]] = []
    idx_emb = tensors["idx_emb"]

    for array, pos_idx, label in (
        (official.image, 0, "image/cropped"),
        (official.rgb_image, 1, "rgb_image/full"),
    ):
        tokens = run_sam3d_projection_net(
            run_sam3d_dino_vitl14_reg(array, tensors, module_index=0, config=dino_config),
            tensors,
            prefix="projection_nets.0.",
        )
        chunks.append(tokens + idx_emb[pos_idx : pos_idx + 1, None].astype(tokens.dtype))
        token_metadata.append(_token_shape_metadata(label, tokens))

    for array, pos_idx, label in (
        (official.mask, 0, "mask/cropped"),
        (official.rgb_image_mask, 1, "rgb_image_mask/full"),
    ):
        tokens = run_sam3d_projection_net(
            run_sam3d_dino_vitl14_reg(array, tensors, module_index=1, config=dino_config),
            tensors,
            prefix="projection_nets.1.",
        )
        chunks.append(tokens + idx_emb[pos_idx : pos_idx + 1, None].astype(tokens.dtype))
        token_metadata.append(_token_shape_metadata(label, tokens))

    for array, pos_idx, label in (
        (official.pointmap, 0, "pointmap/cropped"),
        (official.rgb_pointmap, 1, "rgb_pointmap/full"),
    ):
        tokens = run_sam3d_projection_net(
            run_sam3d_point_patch_embed(array, tensors, point_config, prefix="module_list.2."),
            tensors,
            prefix="projection_nets.2.",
        )
        chunks.append(tokens + idx_emb[pos_idx : pos_idx + 1, None].astype(tokens.dtype))
        token_metadata.append(_token_shape_metadata(label, tokens))

    fused = mx.concatenate(chunks, axis=1)
    mx.eval(fused)
    return Sam3dConditionStackOutput(
        tokens=fused,
        metadata={
            "kind": "ss",
            "token_shape": tuple(int(value) for value in fused.shape),
            "chunks": tuple(token_metadata),
            "dino_prenorm_features": bool(dino_config.prenorm_features),
            "pointmap_windows": int((point_config.input_size // point_config.patch_size) ** 2),
        },
    )


def run_sam3d_slat_condition_stack(
    official,
    tensors: dict[str, mx.array],
    *,
    dino_config: Sam3dDinoConfig = Sam3dDinoConfig(prenorm_features=True),
) -> Sam3dConditionStackOutput:
    """Run the active structured-latent condition fuser for official preprocessed tensors."""

    chunks: list[mx.array] = []
    token_metadata: list[dict[str, object]] = []
    idx_emb = tensors["idx_emb"]
    for module_index, projection_index, arrays in (
        (
            0,
            0,
            ((official.image, 0, "image/cropped"), (official.rgb_image, 1, "rgb_image/full")),
        ),
        (
            1,
            1,
            ((official.mask, 0, "mask/cropped"), (official.rgb_image_mask, 1, "rgb_image_mask/full")),
        ),
    ):
        for array, pos_idx, label in arrays:
            tokens = run_sam3d_projection_net(
                run_sam3d_dino_vitl14_reg(array, tensors, module_index=module_index, config=dino_config),
                tensors,
                prefix=f"projection_nets.{projection_index}.",
            )
            chunks.append(tokens + idx_emb[pos_idx : pos_idx + 1, None].astype(tokens.dtype))
            token_metadata.append(_token_shape_metadata(label, tokens))

    fused = mx.concatenate(chunks, axis=1)
    mx.eval(fused)
    return Sam3dConditionStackOutput(
        tokens=fused,
        metadata={
            "kind": "slat",
            "token_shape": tuple(int(value) for value in fused.shape),
            "chunks": tuple(token_metadata),
            "dino_prenorm_features": bool(dino_config.prenorm_features),
        },
    )


def _token_shape_metadata(label: str, tokens: mx.array) -> dict[str, object]:
    return {"name": label, "shape": tuple(int(value) for value in tokens.shape)}


def _run_dino_block(
    hidden: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dDinoConfig,
) -> mx.array:
    residual = hidden
    normalized = _layer_norm(hidden, tensors[f"{prefix}.norm1.weight"], tensors[f"{prefix}.norm1.bias"], eps=1e-6)
    attended = _self_attention(
        normalized,
        tensors,
        prefix=f"{prefix}.attn",
        num_heads=config.num_heads,
        attention_chunk_size=config.attention_chunk_size,
        max_attention_bytes=config.max_attention_bytes,
    )
    hidden = residual + attended * tensors[f"{prefix}.ls1.gamma"].astype(attended.dtype)
    residual = hidden
    normalized = _layer_norm(hidden, tensors[f"{prefix}.norm2.weight"], tensors[f"{prefix}.norm2.bias"], eps=1e-6)
    mlp = _linear(
        nn.gelu(_linear(normalized, tensors[f"{prefix}.mlp.fc1.weight"], tensors[f"{prefix}.mlp.fc1.bias"])),
        tensors[f"{prefix}.mlp.fc2.weight"],
        tensors[f"{prefix}.mlp.fc2.bias"],
    )
    return residual + mlp * tensors[f"{prefix}.ls2.gamma"].astype(mlp.dtype)


def _point_patch_block(tokens: mx.array, tensors: dict[str, mx.array], prefix: str, config: Sam3dPointPatchConfig) -> mx.array:
    residual = tokens
    hidden = _layer_norm(tokens, tensors[f"{prefix}.norm1.weight"], tensors[f"{prefix}.norm1.bias"], eps=1e-6)
    hidden = _self_attention(hidden, tensors, prefix=f"{prefix}.attn", num_heads=config.num_heads)
    tokens = residual + hidden
    residual = tokens
    hidden = _layer_norm(tokens, tensors[f"{prefix}.norm2.weight"], tensors[f"{prefix}.norm2.bias"], eps=1e-6)
    hidden = _linear(nn.gelu(_linear(hidden, tensors[f"{prefix}.mlp.fc1.weight"], tensors[f"{prefix}.mlp.fc1.bias"])), tensors[f"{prefix}.mlp.fc2.weight"], tensors[f"{prefix}.mlp.fc2.bias"])
    return residual + hidden


def _self_attention(
    tokens: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    num_heads: int,
    attention_chunk_size: int | None = None,
    max_attention_bytes: int | None = None,
) -> mx.array:
    batch, token_count, dim = tuple(int(value) for value in tokens.shape)
    head_dim = dim // num_heads
    qkv = _linear(tokens, tensors[f"{prefix}.qkv.weight"], tensors[f"{prefix}.qkv.bias"])
    query, key, value = (
        mx.transpose(mx.reshape(part, (batch, token_count, num_heads, head_dim)), (0, 2, 1, 3))
        for part in mx.split(qkv, 3, axis=-1)
    )
    scale = head_dim**-0.5
    if attention_chunk_size is None:
        weights = mx.softmax((query @ mx.transpose(key, (0, 1, 3, 2))) * scale, axis=-1)
        attended = weights @ value
    else:
        if max_attention_bytes is not None:
            estimated = batch * num_heads * min(token_count, attention_chunk_size) * token_count * 4
            if estimated > max_attention_bytes:
                raise ValueError(
                    "SAM3D DINO exact attention chunk exceeds activation guard "
                    f"({estimated} > {max_attention_bytes} bytes)"
                )
        chunks = []
        key_t = mx.transpose(key, (0, 1, 3, 2))
        for start in range(0, token_count, attention_chunk_size):
            stop = min(start + attention_chunk_size, token_count)
            weights = mx.softmax((query[:, :, start:stop, :] @ key_t) * scale, axis=-1)
            chunks.append(weights @ value)
        attended = mx.concatenate(chunks, axis=2)
    merged = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)), (batch, token_count, dim))
    return _linear(merged, tensors[f"{prefix}.proj.weight"], tensors[f"{prefix}.proj.bias"])


def _interpolate_dino_pos_embed(
    pos_embed: mx.array,
    patch_grid: tuple[int, int],
    *,
    config: Sam3dDinoConfig,
) -> mx.array:
    patch_height, patch_width = patch_grid
    expected = patch_height * patch_width + 1
    if int(pos_embed.shape[1]) == expected:
        return pos_embed
    from scipy import ndimage

    pos_np = np.array(pos_embed, dtype=np.float32)
    stored_count = pos_np.shape[1] - 1
    stored_side = int(stored_count**0.5)
    if stored_side * stored_side != stored_count:
        raise ValueError(f"SAM3D DINO positional embedding patch count is not square: {pos_np.shape}")
    cls = pos_np[:, :1, :]
    patch = pos_np[:, 1:, :].reshape((stored_side, stored_side, config.embed_dim))
    resized = ndimage.zoom(
        patch,
        (patch_height / stored_side, patch_width / stored_side, 1.0),
        order=3,
        mode="nearest",
        prefilter=True,
    )
    resized = _fit_resized_pos_embed(resized, patch_grid)
    patch_tokens = resized.reshape((1, patch_height * patch_width, config.embed_dim))
    return mx.array(np.concatenate((cls, patch_tokens), axis=1), dtype=pos_embed.dtype)


def _fit_resized_pos_embed(values: np.ndarray, patch_grid: tuple[int, int]) -> np.ndarray:
    height, width = patch_grid
    output = values
    if output.shape[0] < height or output.shape[1] < width:
        output = np.pad(output, ((0, max(height - output.shape[0], 0)), (0, max(width - output.shape[1], 0)), (0, 0)), mode="edge")
    return output[:height, :width, :]


def _point_patch_pos_embed(pos_embed: mx.array, height: int, width: int) -> mx.array:
    if int(pos_embed.shape[2]) == height and int(pos_embed.shape[3]) == width:
        resized = pos_embed
    else:
        resized = _resize_linear_bchw(pos_embed, (height, width))
    return mx.reshape(mx.transpose(resized, (0, 2, 3, 1)), (1, height * width, int(pos_embed.shape[1])))


def _pointmap_to_bchw(pointmap: np.ndarray | mx.array) -> mx.array:
    values = mx.array(pointmap, dtype=mx.float32) if not isinstance(pointmap, mx.array) else pointmap.astype(mx.float32)
    if values.ndim == 3:
        if int(values.shape[0]) == 3:
            values = values[None, ...]
        elif int(values.shape[-1]) == 3:
            values = mx.transpose(values, (2, 0, 1))[None, ...]
        else:
            raise ValueError(f"SAM3D pointmap must have 3 channels, got {tuple(values.shape)}")
    if values.ndim != 4 or int(values.shape[1]) != 3:
        raise ValueError(f"SAM3D pointmap must have shape [B,3,H,W], got {tuple(values.shape)}")
    return values


def _image_to_bchw(image: np.ndarray | mx.array) -> mx.array:
    values = mx.array(image, dtype=mx.float32) if not isinstance(image, mx.array) else image.astype(mx.float32)
    if values.ndim == 2:
        values = values[None, None, ...]
    elif values.ndim == 3:
        if int(values.shape[0]) in {1, 3}:
            values = values[None, ...]
        elif int(values.shape[-1]) in {1, 3}:
            values = mx.transpose(values, (2, 0, 1))[None, ...]
        else:
            raise ValueError(f"SAM3D image tensor must have 1 or 3 channels, got {tuple(values.shape)}")
    if values.ndim != 4 or int(values.shape[1]) not in {1, 3}:
        raise ValueError(f"SAM3D image tensor must have shape [B,C,H,W], got {tuple(values.shape)}")
    return values


def _resize_nearest_bchw(values: mx.array, size: int) -> mx.array:
    height, width = int(values.shape[2]), int(values.shape[3])
    if height == size and width == size:
        return values
    rows = np.clip(((np.arange(size) + 0.5) * height / size).astype(np.int64), 0, height - 1)
    cols = np.clip(((np.arange(size) + 0.5) * width / size).astype(np.int64), 0, width - 1)
    arr = np.array(values, dtype=np.float32)
    return mx.array(arr[:, :, rows[:, None], cols[None, :]], dtype=values.dtype)


def _resize_linear_bchw(values: mx.array, size: tuple[int, int] | int) -> mx.array:
    if isinstance(size, int):
        size = (size, size)
    height, width = size
    scale = (height / int(values.shape[2]), width / int(values.shape[3]))
    resized = nn.Upsample(scale, mode="linear", align_corners=False)(mx.transpose(values, (0, 2, 3, 1)))
    return mx.transpose(resized[:, :height, :width, :], (0, 3, 1, 2))


def _remap_points(points: mx.array, remap_output: str) -> mx.array:
    if remap_output == "linear":
        return points
    if remap_output == "sinh":
        return mx.arcsinh(points)
    if remap_output == "exp":
        xy = points[..., :2] / (1.0 + points[..., 2:3])
        z = mx.log1p(points[..., 2:3])
        return mx.concatenate((xy, z), axis=-1)
    if remap_output == "exp_disparity":
        z = mx.maximum(points[..., 2:3], 1e-8)
        return mx.concatenate((points[..., :2] / z, mx.log(z)), axis=-1)
    if remap_output == "sinh_exp":
        return mx.concatenate((mx.arcsinh(points[..., :2]), mx.log(mx.maximum(points[..., 2:3], 1e-8))), axis=-1)
    raise ValueError(f"unsupported SAM3D point remap: {remap_output}")


def _linear(values: mx.array, weight: mx.array, bias: mx.array | None) -> mx.array:
    output = values @ mx.transpose(weight.astype(values.dtype))
    if bias is not None:
        output = output + bias.astype(output.dtype)
    return output


def _layer_norm(values: mx.array, weight: mx.array, bias: mx.array, *, eps: float) -> mx.array:
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = mx.mean(centered * centered, axis=-1, keepdims=True)
    return centered * mx.rsqrt(variance + eps) * weight.astype(values.dtype) + bias.astype(values.dtype)


def _conv2d_nchw(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight.astype(x.dtype), (0, 2, 3, 1))
    out = mx.conv2d(x_nhwc, weight_ohwi, stride=stride, padding=padding)
    if bias is not None:
        out = out + bias.astype(out.dtype)[None, None, None, :]
    return mx.transpose(out, (0, 3, 1, 2))


def _silu(values: mx.array) -> mx.array:
    return values * mx.sigmoid(values)
