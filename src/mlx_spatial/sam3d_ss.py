"""SAM 3D Objects sparse-structure decoder utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml

from .checkpoint import load_checkpoint_tensors


@dataclass(frozen=True)
class Sam3dSSDecoderConfig:
    out_channels: int = 1
    latent_channels: int = 8
    num_res_blocks: int = 2
    channels: tuple[int, ...] = (512, 128, 32)
    num_res_blocks_middle: int = 2
    norm_type: str = "layer"
    reshape_input_to_cube: bool = False
    use_fp16: bool = False


@dataclass(frozen=True)
class Sam3dSSDecoderOutput:
    occupancy: np.ndarray
    coords_original: np.ndarray
    coords: np.ndarray
    downsample_factor: int
    metadata: dict[str, object]


def read_sam3d_ss_decoder_config(path: str | Path) -> Sam3dSSDecoderConfig:
    """Read the active SAM3D sparse-structure decoder YAML."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SAM3D SS decoder config must be a mapping")
    target = raw.get("_target_")
    if target not in {
        "sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae.SparseStructureDecoderTdfyWrapper",
        None,
    }:
        raise ValueError(f"unsupported SAM3D SS decoder target: {target}")
    return Sam3dSSDecoderConfig(
        out_channels=int(raw.get("out_channels", 1)),
        latent_channels=int(raw.get("latent_channels", 8)),
        num_res_blocks=int(raw.get("num_res_blocks", 2)),
        channels=tuple(int(value) for value in raw.get("channels", (512, 128, 32))),
        num_res_blocks_middle=int(raw.get("num_res_blocks_middle", 2)),
        norm_type=str(raw.get("norm_type", "layer")),
        reshape_input_to_cube=bool(raw.get("reshape_input_to_cube", False)),
        use_fp16=bool(raw.get("use_fp16", False)),
    )


def load_sam3d_ss_decoder_tensors(path: str | Path) -> dict[str, mx.array]:
    return load_checkpoint_tensors(path, prefixes=("input_layer.", "middle_block.", "blocks.", "out_layer."))


def run_sam3d_ss_decoder(
    latent: mx.array | np.ndarray,
    tensors: dict[str, mx.array],
    config: Sam3dSSDecoderConfig,
    *,
    prune_neighbor_axes_dist: int = 1,
    max_coords: int = 42_000,
) -> Sam3dSSDecoderOutput:
    """Run the dense SAM3D sparse-structure decoder and extract final coords."""

    x = mx.array(latent, dtype=mx.float32) if not isinstance(latent, mx.array) else latent.astype(mx.float32)
    x = _ss_latent_to_ncdhw(x, config)
    h = _conv3d_ncdhw(x, tensors["input_layer.weight"], tensors["input_layer.bias"], padding=1)
    for index in range(config.num_res_blocks_middle):
        h = _res_block3d(h, tensors, f"middle_block.{index}", config.norm_type)
    block_index = 0
    for level, channels in enumerate(config.channels):
        for _ in range(config.num_res_blocks):
            h = _res_block3d(h, tensors, f"blocks.{block_index}", config.norm_type)
            block_index += 1
        if level < len(config.channels) - 1:
            h = _upsample_block3d(h, tensors, f"blocks.{block_index}")
            block_index += 1
    h = _channel_layer_norm_ncdhw(h, tensors["out_layer.0.weight"], tensors["out_layer.0.bias"])
    h = _silu(h)
    h = _conv3d_ncdhw(h, tensors["out_layer.2.weight"], tensors["out_layer.2.bias"], padding=1)
    mx.eval(h)
    occupancy = np.array(h, dtype=np.float32)
    coords_original = extract_sam3d_ss_coords(occupancy)
    coords = coords_original
    if prune_neighbor_axes_dist > 0 and coords.shape[0]:
        coords = prune_sam3d_sparse_structure(coords, max_neighbor_axes_dist=prune_neighbor_axes_dist)
    coords, downsample_factor = downsample_sam3d_sparse_structure(coords, max_coords=max_coords)
    return Sam3dSSDecoderOutput(
        occupancy=occupancy,
        coords_original=coords_original,
        coords=coords,
        downsample_factor=downsample_factor,
        metadata={
            "occupancy_shape": tuple(int(value) for value in occupancy.shape),
            "coords_original_count": int(coords_original.shape[0]),
            "coords_count": int(coords.shape[0]),
            "downsample_factor": int(downsample_factor),
            "prune_neighbor_axes_dist": int(prune_neighbor_axes_dist),
        },
    )


def extract_sam3d_ss_coords(occupancy: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    """Match upstream `torch.argwhere(ss > 0)[:, [0, 2, 3, 4]]`."""

    occ = np.asarray(occupancy)
    if occ.ndim != 5:
        raise ValueError(f"SAM3D SS occupancy must have shape [B,C,D,H,W], got {occ.shape}")
    active = np.argwhere(occ > threshold)
    if active.size == 0:
        return np.zeros((0, 4), dtype=np.int32)
    return active[:, [0, 2, 3, 4]].astype(np.int32, copy=False)


def prune_sam3d_sparse_structure(coords_batch: np.ndarray, *, max_neighbor_axes_dist: int = 1) -> np.ndarray:
    """Remove interior occupied voxels using the official neighbor-count rule."""

    coords_batch = _as_coord_batch(coords_batch)
    if coords_batch.shape[0] == 0:
        return coords_batch
    kept_batches: list[np.ndarray] = []
    full_count = (2 * max_neighbor_axes_dist + 1) ** 3
    for batch_id in np.unique(coords_batch[:, 0]):
        batch_rows = coords_batch[coords_batch[:, 0] == batch_id]
        coords = batch_rows[:, 1:]
        min_xyz = coords.min(axis=0)
        coords0 = coords - min_xyz[None, :]
        grid_shape = coords0.max(axis=0) + 1
        occ = np.zeros(tuple(int(v) for v in grid_shape), dtype=bool)
        occ[coords0[:, 0], coords0[:, 1], coords0[:, 2]] = True
        keep = []
        pad = max_neighbor_axes_dist
        padded = np.pad(occ, pad, mode="constant", constant_values=False)
        for index, coord in enumerate(coords0):
            shifted = coord + pad
            window = padded[
                shifted[0] - pad : shifted[0] + pad + 1,
                shifted[1] - pad : shifted[1] + pad + 1,
                shifted[2] - pad : shifted[2] + pad + 1,
            ]
            if int(window.sum()) < full_count:
                keep.append(index)
        if keep:
            kept_batches.append(batch_rows[np.asarray(keep, dtype=np.int64)])
    if not kept_batches:
        return np.zeros((0, 4), dtype=np.int32)
    return np.concatenate(kept_batches, axis=0).astype(np.int32, copy=False)


def downsample_sam3d_sparse_structure(
    coords_batch: np.ndarray,
    *,
    max_coords: int = 42_000,
    downsample_factor: int = 2,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Downsample SAM3D sparse coords when the active count exceeds the mesh guard."""

    coords_batch = _as_coord_batch(coords_batch)
    if coords_batch.shape[0] <= max_coords:
        return coords_batch, 1
    coords = coords_batch[:, 1:].astype(np.float32)
    batches = coords_batch[:, :1]
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)
    original_size = coords_max - coords_min + 1.0
    target_size = original_size / float(downsample_factor)
    target_min = coords_min + (original_size - target_size) / 2.0
    target_max = target_min + target_size - 1.0
    denom = np.maximum(coords_max - coords_min, 1.0)
    coords_normalized = (coords - coords_min[None, :]) / denom[None, :]
    rescaled = np.round(coords_normalized * (target_size - 1.0)[None, :] + target_min[None, :]).astype(np.int32)
    rescaled = np.minimum(np.maximum(rescaled, target_min.astype(np.int32)), target_max.astype(np.int32))
    unique = np.unique(np.concatenate((batches.astype(np.int32), rescaled), axis=1), axis=0)
    if unique.shape[0] > max_coords:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(unique.shape[0])[:max_coords]
        unique = unique[indices]
    return unique.astype(np.int32, copy=False), downsample_factor


def _as_coord_batch(coords_batch: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_batch, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"SAM3D sparse coords must have shape (N, 4), got {coords.shape}")
    return coords


def _ss_latent_to_ncdhw(latent: mx.array, config: Sam3dSSDecoderConfig) -> mx.array:
    if latent.ndim == 5:
        return latent
    if latent.ndim != 3:
        raise ValueError(f"SAM3D SS latent must have shape [B,T,C] or [B,C,D,H,W], got {tuple(latent.shape)}")
    batch, tokens, channels = tuple(int(value) for value in latent.shape)
    if channels != config.latent_channels:
        raise ValueError(f"SAM3D SS latent channel mismatch: {channels} != {config.latent_channels}")
    side = round(tokens ** (1 / 3))
    if side**3 != tokens:
        raise ValueError(f"SAM3D SS flat latent token count must be a cube, got {tokens}")
    return mx.transpose(mx.reshape(latent, (batch, side, side, side, channels)), (0, 4, 1, 2, 3))


def _res_block3d(x: mx.array, tensors: dict[str, mx.array], prefix: str, norm_type: str) -> mx.array:
    residual = x
    h = _norm3d(x, tensors[f"{prefix}.norm1.weight"], tensors[f"{prefix}.norm1.bias"], norm_type)
    h = _silu(h)
    h = _conv3d_ncdhw(h, tensors[f"{prefix}.conv1.weight"], tensors[f"{prefix}.conv1.bias"], padding=1)
    h = _norm3d(h, tensors[f"{prefix}.norm2.weight"], tensors[f"{prefix}.norm2.bias"], norm_type)
    h = _silu(h)
    h = _conv3d_ncdhw(h, tensors[f"{prefix}.conv2.weight"], tensors[f"{prefix}.conv2.bias"], padding=1)
    skip_weight = tensors.get(f"{prefix}.skip_connection.weight")
    if skip_weight is not None:
        residual = _conv3d_ncdhw(residual, skip_weight, tensors.get(f"{prefix}.skip_connection.bias"), padding=0)
    return h + residual


def _upsample_block3d(x: mx.array, tensors: dict[str, mx.array], prefix: str) -> mx.array:
    h = _conv3d_ncdhw(x, tensors[f"{prefix}.conv.weight"], tensors[f"{prefix}.conv.bias"], padding=1)
    return _pixel_shuffle_3d(h, 2)


def _pixel_shuffle_3d(x: mx.array, scale_factor: int) -> mx.array:
    batch, channels, depth, height, width = tuple(int(value) for value in x.shape)
    out_channels = channels // (scale_factor**3)
    x = mx.reshape(x, (batch, out_channels, scale_factor, scale_factor, scale_factor, depth, height, width))
    x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    return mx.reshape(x, (batch, out_channels, depth * scale_factor, height * scale_factor, width * scale_factor))


def _norm3d(x: mx.array, weight: mx.array, bias: mx.array, norm_type: str) -> mx.array:
    if norm_type == "layer":
        return _channel_layer_norm_ncdhw(x, weight, bias)
    if norm_type == "group":
        return _group_norm_ncdhw(x, weight, bias, groups=32)
    raise ValueError(f"unsupported SAM3D SS decoder norm_type: {norm_type}")


def _channel_layer_norm_ncdhw(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-6) -> mx.array:
    values = mx.transpose(x, (0, 2, 3, 4, 1)).astype(mx.float32)
    mean = mx.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    var = mx.mean(centered * centered, axis=-1, keepdims=True)
    values = centered * mx.rsqrt(var + eps)
    values = values * weight.astype(values.dtype) + bias.astype(values.dtype)
    return mx.transpose(values.astype(x.dtype), (0, 4, 1, 2, 3))


def _group_norm_ncdhw(x: mx.array, weight: mx.array, bias: mx.array, *, groups: int, eps: float = 1e-5) -> mx.array:
    batch, channels, depth, height, width = tuple(int(value) for value in x.shape)
    groups = min(groups, channels)
    values = mx.reshape(x.astype(mx.float32), (batch, groups, channels // groups, depth, height, width))
    mean = mx.mean(values, axis=(2, 3, 4, 5), keepdims=True)
    centered = values - mean
    var = mx.mean(centered * centered, axis=(2, 3, 4, 5), keepdims=True)
    values = mx.reshape(centered * mx.rsqrt(var + eps), (batch, channels, depth, height, width))
    values = values * weight.astype(values.dtype)[None, :, None, None, None] + bias.astype(values.dtype)[None, :, None, None, None]
    return values.astype(x.dtype)


def _conv3d_ncdhw(x: mx.array, weight: mx.array, bias: mx.array | None, *, padding: int) -> mx.array:
    x_ndhwc = mx.transpose(x, (0, 2, 3, 4, 1))
    weight_okkki = mx.transpose(weight.astype(x.dtype), (0, 2, 3, 4, 1))
    out = mx.conv3d(x_ndhwc, weight_okkki, padding=padding)
    if bias is not None:
        out = out + bias.astype(out.dtype)[None, None, None, None, :]
    return mx.transpose(out, (0, 4, 1, 2, 3))


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)
