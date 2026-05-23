"""SAM 3D Objects structured-latent decoders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import yaml

from .checkpoint import load_checkpoint_tensors
from .sam3d_slat import _absolute_position_embedding
from .sam3d_ss_flow import _gelu_tanh, _linear
from .sam3d_transformer import sam3d_layer_norm, sam3d_scaled_dot_product_attention


SAM3D_GAUSSIAN_DECODER_TARGET = (
    "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_gs."
    "SLatGaussianDecoderTdfyWrapper"
)
SAM3D_MESH_DECODER_TARGET = (
    "sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh."
    "SLatMeshDecoderTdfyWrapper"
)


@dataclass(frozen=True)
class Sam3dSLatDecoderConfig:
    resolution: int = 64
    model_channels: int = 768
    latent_channels: int = 8
    num_blocks: int = 12
    num_heads: int = 12
    window_size: int = 8
    target: str | None = None
    representation_config: dict[str, object] | None = None


@dataclass(frozen=True)
class Sam3dMeshDecoderConfig(Sam3dSLatDecoderConfig):
    use_color: bool = False


def read_sam3d_slat_decoder_config(path: str | Path) -> Sam3dSLatDecoderConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SAM3D SLat decoder config must be a mapping")
    target = raw.get("_target_")
    if target not in {SAM3D_GAUSSIAN_DECODER_TARGET, None}:
        raise ValueError(f"unsupported SAM3D SLat gaussian decoder target: {target}")
    return Sam3dSLatDecoderConfig(
        resolution=int(raw.get("resolution", 64)),
        model_channels=int(raw.get("model_channels", 768)),
        latent_channels=int(raw.get("latent_channels", 8)),
        num_blocks=int(raw.get("num_blocks", 12)),
        num_heads=int(raw.get("num_heads", 12)),
        window_size=int(raw.get("window_size", 8)),
        target=str(target) if target is not None else None,
        representation_config=_mapping_or_none(raw.get("representation_config"), "representation_config"),
    )


def read_sam3d_mesh_decoder_config(path: str | Path) -> Sam3dMeshDecoderConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SAM3D SLat mesh decoder config must be a mapping")
    target = raw.get("_target_")
    if target not in {SAM3D_MESH_DECODER_TARGET, None}:
        raise ValueError(f"unsupported SAM3D SLat mesh decoder target: {target}")
    representation_config = _mapping_or_none(raw.get("representation_config"), "representation_config")
    return Sam3dMeshDecoderConfig(
        resolution=int(raw.get("resolution", 64)),
        model_channels=int(raw.get("model_channels", 768)),
        latent_channels=int(raw.get("latent_channels", 8)),
        num_blocks=int(raw.get("num_blocks", 12)),
        num_heads=int(raw.get("num_heads", 12)),
        window_size=int(raw.get("window_size", 8)),
        target=str(target) if target is not None else None,
        representation_config=representation_config,
        use_color=bool((representation_config or {}).get("use_color", False)),
    )


def load_sam3d_slat_decoder_tensors(path: str | Path) -> dict[str, mx.array]:
    return load_checkpoint_tensors(path, prefixes=("input_layer.", "blocks.", "out_layer.", "offset_perturbation"))


def load_sam3d_mesh_decoder_tensors(path: str | Path) -> dict[str, mx.array]:
    return load_checkpoint_tensors(path, prefixes=("input_layer.", "blocks.", "upsample.", "out_layer."))


def run_sam3d_slat_decoder_torso(
    coords: np.ndarray,
    feats: mx.array,
    tensors: dict[str, mx.array],
    config: Sam3dSLatDecoderConfig = Sam3dSLatDecoderConfig(),
) -> mx.array:
    """Run the shared SAM3D SLat decoder transformer before decoder-specific heads."""

    coords_np = _validate_coords(coords)
    if feats.ndim != 2 or int(feats.shape[0]) != coords_np.shape[0]:
        raise ValueError(f"SAM3D decoder feats must have shape (N,C), got {tuple(feats.shape)} for coords {coords_np.shape}")
    hidden = _linear(feats, tensors["input_layer.weight"], tensors["input_layer.bias"])
    hidden = hidden + _absolute_position_embedding(coords_np[:, 1:], config.model_channels).astype(hidden.dtype)
    for index in range(config.num_blocks):
        hidden = _decoder_block(
            coords_np,
            hidden,
            tensors,
            prefix=f"blocks.{index}.",
            config=config,
            shift_window=(config.window_size // 2) * (index % 2),
        )
        mx.eval(hidden)
    return sam3d_layer_norm(hidden, eps=1e-5)


def run_sam3d_slat_decoder_network(
    coords: np.ndarray,
    feats: mx.array,
    tensors: dict[str, mx.array],
    config: Sam3dSLatDecoderConfig = Sam3dSLatDecoderConfig(),
) -> mx.array:
    """Run the shared SAM3D SLat decoder transformer and output layer."""

    hidden = run_sam3d_slat_decoder_torso(coords, feats, tensors, config)
    return _linear(hidden.astype(feats.dtype), tensors["out_layer.weight"], tensors["out_layer.bias"])


def _decoder_block(
    coords: np.ndarray,
    feats: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSLatDecoderConfig,
    shift_window: int,
) -> mx.array:
    hidden = sam3d_layer_norm(feats)
    hidden = _decoder_windowed_self_attention(
        coords,
        hidden,
        tensors,
        prefix=f"{prefix}attn.",
        config=config,
        shift_window=shift_window,
    )
    feats = feats + hidden
    hidden = sam3d_layer_norm(feats)
    hidden = _decoder_feed_forward(hidden, tensors, prefix=f"{prefix}mlp.")
    return feats + hidden


def _decoder_windowed_self_attention(
    coords: np.ndarray,
    feats: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    config: Sam3dSLatDecoderConfig,
    shift_window: int,
) -> mx.array:
    tokens, channels = int(feats.shape[0]), int(feats.shape[1])
    head_dim = channels // config.num_heads
    qkv = _linear(feats, tensors[f"{prefix}to_qkv.weight"], tensors[f"{prefix}to_qkv.bias"])
    qkv = mx.reshape(qkv, (tokens, 3, config.num_heads, head_dim))
    fwd, bwd, seq_lens = _window_partition(coords, window_size=config.window_size, shift_window=shift_window)
    qkv_sorted = qkv[mx.array(fwd, dtype=mx.int32)]
    outputs = []
    start = 0
    for seq_len in seq_lens:
        stop = start + seq_len
        q = qkv_sorted[start:stop, 0, :, :][None, :, :, :]
        k = qkv_sorted[start:stop, 1, :, :][None, :, :, :]
        v = qkv_sorted[start:stop, 2, :, :][None, :, :, :]
        outputs.append(sam3d_scaled_dot_product_attention(q, k, v)[0])
        start = stop
    if not outputs:
        attended = mx.zeros((tokens, config.num_heads, head_dim), dtype=feats.dtype)
    else:
        attended = mx.concatenate(outputs, axis=0)
    attended = attended[mx.array(bwd, dtype=mx.int32)]
    return _linear(mx.reshape(attended, (tokens, channels)), tensors[f"{prefix}to_out.weight"], tensors[f"{prefix}to_out.bias"])


def _window_partition(coords: np.ndarray, *, window_size: int, shift_window: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    shifted = np.asarray(coords, dtype=np.int64).copy()
    shifted[:, 1:] += int(shift_window)
    windows = shifted.copy()
    windows[:, 1:] //= int(window_size)
    max_coords = windows[:, 1:].max(axis=0)
    num_windows = np.ceil((max_coords + 1) / 1).astype(np.int64)
    offsets = np.cumprod(np.array([1, *num_windows[::-1]], dtype=np.int64))[::-1]
    codes = (windows * offsets[None, :]).sum(axis=1)
    fwd = np.argsort(codes, kind="stable")
    bwd = np.empty_like(fwd)
    bwd[fwd] = np.arange(fwd.shape[0], dtype=np.int64)
    sorted_codes = codes[fwd]
    _, counts = np.unique(sorted_codes, return_counts=True)
    return fwd.astype(np.int32, copy=False), bwd.astype(np.int32, copy=False), [int(value) for value in counts.tolist()]


def _decoder_feed_forward(feats: mx.array, tensors: dict[str, mx.array], *, prefix: str) -> mx.array:
    hidden = _linear(feats, tensors[f"{prefix}mlp.0.weight"], tensors[f"{prefix}mlp.0.bias"])
    hidden = _gelu_tanh(hidden)
    return _linear(hidden, tensors[f"{prefix}mlp.2.weight"], tensors[f"{prefix}mlp.2.bias"])


def _mapping_or_none(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"SAM3D decoder {label} must be a mapping")
    return dict(value)


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"SAM3D decoder coords must have shape (N,4), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("SAM3D decoder coords must not be empty")
    return values
