"""SAM 3D Objects MoGe pointmap runtime boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from .checkpoint import inspect_checkpoint, load_checkpoint_tensors
from .sam3d_assets import Sam3dAssetBlocker


SAM3D_MOGE_DEFAULT_ROOT = "weights/moge-vitl-mlx"
SAM3D_MOGE_PATCH_SIZE = 14
SAM3D_MOGE_EMBED_DIM = 1024
SAM3D_MOGE_DEPTH = 24
SAM3D_MOGE_NUM_HEADS = 16
SAM3D_MOGE_INTERMEDIATE_BLOCKS = (20, 21, 22, 23)
SAM3D_MOGE_NUM_TOKENS_RANGE = (1275, 2551)
SAM3D_MOGE_MASK_THRESHOLD = 0.5
SAM3D_MOGE_MEMORY_PROFILES = {
    "safe": {"max_tokens": 1600, "attention_chunk": 128, "max_attention_bytes": 800_000_000},
    "balanced": {"max_tokens": 2800, "attention_chunk": 192, "max_attention_bytes": 1_600_000_000},
    "large": {"max_tokens": 4096, "attention_chunk": 256, "max_attention_bytes": 3_000_000_000},
}
SAM3D_MOGE_REQUIRED_KEYS = (
    "image_mean",
    "image_std",
    "backbone.patch_embed.proj.weight",
    "backbone.patch_embed.proj.bias",
    "backbone.cls_token",
    "backbone.pos_embed",
    "backbone.norm.weight",
    "backbone.norm.bias",
    "head.projects.0.weight",
    "head.upsample_blocks.0.0.0.weight",
    "head.output_block.0.2.weight",
    "head.output_block.1.2.weight",
)


@dataclass(frozen=True)
class Sam3dMogeInspection:
    root: Path
    checkpoint_path: Path
    ready: bool
    tensor_count: int
    sample_keys: tuple[str, ...]
    blocker: Sam3dAssetBlocker | None = None


@dataclass(frozen=True)
class Sam3dMogePointmap:
    pointmap: np.ndarray
    intrinsics: np.ndarray
    mask: np.ndarray
    depth: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class Sam3dMogeResult:
    pointmap: Sam3dMogePointmap | None = None
    inspection: Sam3dMogeInspection | None = None
    blocker: Sam3dAssetBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.pointmap is not None and self.blocker is None


def inspect_sam3d_moge_assets(root: str | Path = SAM3D_MOGE_DEFAULT_ROOT) -> Sam3dMogeInspection:
    """Inspect the converted MoGe safetensors checkpoint without running inference."""

    root_path = Path(root)
    checkpoint = root_path / "model.safetensors"
    if not checkpoint.is_file():
        return Sam3dMogeInspection(
            root=root_path,
            checkpoint_path=checkpoint,
            ready=False,
            tensor_count=0,
            sample_keys=(),
            blocker=Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="validate converted MoGe safetensors checkpoint",
                reason=f"MoGe checkpoint not found: {checkpoint}",
                metadata={"root": str(root_path), "expected": str(checkpoint)},
            ),
        )
    try:
        infos = inspect_checkpoint(checkpoint)
    except (OSError, ValueError) as error:
        return Sam3dMogeInspection(
            root=root_path,
            checkpoint_path=checkpoint,
            ready=False,
            tensor_count=0,
            sample_keys=(),
            blocker=Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="inspect converted MoGe safetensors checkpoint",
                reason=str(error),
                metadata={"root": str(root_path), "checkpoint": str(checkpoint)},
            ),
        )
    names = {info.name for info in infos}
    missing = tuple(key for key in SAM3D_MOGE_REQUIRED_KEYS if key not in names)
    if missing:
        return Sam3dMogeInspection(
            root=root_path,
            checkpoint_path=checkpoint,
            ready=False,
            tensor_count=len(infos),
            sample_keys=tuple(info.name for info in infos[:10]),
            blocker=Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="validate converted MoGe checkpoint key contract",
                reason=f"MoGe checkpoint is missing required tensor: {missing[0]}",
                metadata={"missing": missing, "checkpoint": str(checkpoint)},
            ),
        )
    return Sam3dMogeInspection(
        root=root_path,
        checkpoint_path=checkpoint,
        ready=True,
        tensor_count=len(infos),
        sample_keys=tuple(info.name for info in infos[:10]),
    )


def run_sam3d_moge_pointmap(
    image_rgb: np.ndarray,
    *,
    root: str | Path = SAM3D_MOGE_DEFAULT_ROOT,
    memory_profile: str = "balanced",
) -> Sam3dMogeResult:
    """Run the MLX MoGe v1 pointmap stage used by SAM3D Objects."""

    inspection = inspect_sam3d_moge_assets(root)
    if inspection.blocker is not None:
        return Sam3dMogeResult(inspection=inspection, blocker=inspection.blocker)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        blocker = Sam3dAssetBlocker(
            stage="moge-pointmap",
            operation="validate MoGe RGB input",
            reason=f"MoGe input must have shape (H, W, 3), got {image_rgb.shape}",
            metadata={"shape": tuple(int(value) for value in image_rgb.shape)},
        )
        return Sam3dMogeResult(inspection=inspection, blocker=blocker)
    if memory_profile not in SAM3D_MOGE_MEMORY_PROFILES:
        blocker = Sam3dAssetBlocker(
            stage="moge-pointmap",
            operation="validate MoGe memory profile",
            reason=f"unsupported SAM3D MoGe memory profile: {memory_profile}",
            metadata={"memory_profile": memory_profile, "supported": tuple(SAM3D_MOGE_MEMORY_PROFILES)},
        )
        return Sam3dMogeResult(inspection=inspection, blocker=blocker)

    try:
        tensors = load_checkpoint_tensors(
            inspection.checkpoint_path,
            prefixes=("image_", "backbone.", "head."),
        )
    except (OSError, ValueError, TypeError) as error:
        blocker = Sam3dAssetBlocker(
            stage="moge-pointmap",
            operation="load converted MoGe MLX tensors",
            reason=str(error),
            metadata={"checkpoint": str(inspection.checkpoint_path)},
        )
        return Sam3dMogeResult(inspection=inspection, blocker=blocker)

    shape_blocker = _validate_full_moge_tensors(tensors)
    if shape_blocker is not None:
        return Sam3dMogeResult(inspection=inspection, blocker=shape_blocker)

    try:
        pointmap = _run_moge_forward(image_rgb, tensors, memory_profile=memory_profile)
    except _MogeBlocked as blocked:
        return Sam3dMogeResult(inspection=inspection, blocker=blocked.blocker)
    except (FloatingPointError, OverflowError, ValueError) as error:
        blocker = Sam3dAssetBlocker(
            stage="moge-pointmap",
            operation="run MLX MoGe v1 forward pass",
            reason=str(error),
            metadata={"checkpoint": str(inspection.checkpoint_path), "memory_profile": memory_profile},
        )
        return Sam3dMogeResult(inspection=inspection, blocker=blocker)

    return Sam3dMogeResult(pointmap=pointmap, inspection=inspection)


class _MogeBlocked(Exception):
    def __init__(self, blocker: Sam3dAssetBlocker):
        super().__init__(blocker.reason)
        self.blocker = blocker


def _run_moge_forward(
    image_rgb: np.ndarray,
    tensors: dict[str, mx.array],
    *,
    memory_profile: str,
) -> Sam3dMogePointmap:
    config = SAM3D_MOGE_MEMORY_PROFILES[memory_profile]
    height, width = (int(value) for value in image_rgb.shape[:2])
    aspect_ratio = width / height
    num_tokens = _moge_num_tokens(resolution_level=9)
    resized_height, resized_width = _moge_resized_size(height, width, num_tokens=num_tokens)
    patch_height = max(resized_height // SAM3D_MOGE_PATCH_SIZE, 1)
    patch_width = max(resized_width // SAM3D_MOGE_PATCH_SIZE, 1)
    token_count = patch_height * patch_width + 1
    if token_count > int(config["max_tokens"]):
        raise _MogeBlocked(
            Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="guard MLX MoGe DINO token count",
                reason=(
                    "MLX MoGe exact DINO token count would exceed the configured memory profile "
                    f"({token_count} > {config['max_tokens']})"
                ),
                metadata={
                    "token_count": token_count,
                    "max_tokens": int(config["max_tokens"]),
                    "memory_profile": memory_profile,
                    "resized_size": (resized_height, resized_width),
                    "patch_grid": (patch_height, patch_width),
                },
            )
        )

    image_14 = _prepare_moge_image_tensor(
        image_rgb,
        tensors,
        resized_size=(resized_height, resized_width),
        patch_grid=(patch_height, patch_width),
    )
    features = _run_moge_dinov2(
        image_14,
        tensors,
        patch_grid=(patch_height, patch_width),
        attention_chunk_size=int(config["attention_chunk"]),
        max_attention_bytes=int(config["max_attention_bytes"]),
    )
    points, mask_logits = _run_moge_head(
        features,
        tensors,
        resized_size=(resized_height, resized_width),
        output_size=(height, width),
        aspect_ratio=aspect_ratio,
    )
    points_np = np.array(points, dtype=np.float32)[0]
    mask_np = np.array(mask_logits, dtype=np.float32)[0, 0]
    points_np = _remap_moge_points_exp(np.transpose(points_np, (1, 2, 0)))
    mask_binary = mask_np > SAM3D_MOGE_MASK_THRESHOLD
    finite_mask = np.isfinite(points_np).all(axis=-1)
    mask_binary = mask_binary & finite_mask
    focal, shift = recover_focal_shift_numpy(points_np, mask_binary)
    fx = focal / 2.0 * np.sqrt(1.0 + aspect_ratio**2) / aspect_ratio
    fy = focal / 2.0 * np.sqrt(1.0 + aspect_ratio**2)
    intrinsics = np.array(
        [[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    depth = points_np[..., 2] + np.float32(shift)
    camera_points = points_np + np.array([0.0, 0.0, shift], dtype=np.float32)[None, None, :]
    camera_points = np.where(mask_binary[..., None], camera_points, np.inf).astype(np.float32, copy=False)
    depth = np.where(mask_binary, depth, np.inf).astype(np.float32, copy=False)

    # Official SAM3D rotates MoGe/R3 camera points into PyTorch3D camera space.
    # For this camera convention handoff, X and Y flip while Z remains depth.
    sam3d_points = camera_points.copy()
    sam3d_points[..., 0] *= -1.0
    sam3d_points[..., 1] *= -1.0

    return Sam3dMogePointmap(
        pointmap=sam3d_points,
        intrinsics=intrinsics,
        mask=mask_binary,
        depth=depth,
        metadata={
            "model": "MoGe v1 dinov2_vitl14",
            "num_tokens": int(num_tokens),
            "resized_size": (int(resized_height), int(resized_width)),
            "patch_grid": (int(patch_height), int(patch_width)),
            "token_count": int(token_count),
            "memory_profile": memory_profile,
            "force_projection": False,
            "apply_mask": True,
            "camera_convention": "MoGe/R3 -> PyTorch3D by flipping X and Y",
            "focal": float(focal),
            "shift": float(shift),
        },
    )


def _moge_num_tokens(*, resolution_level: int) -> int:
    min_tokens, max_tokens = SAM3D_MOGE_NUM_TOKENS_RANGE
    return int(min_tokens + (resolution_level / 9.0) * (max_tokens - min_tokens))


def _moge_resized_size(height: int, width: int, *, num_tokens: int) -> tuple[int, int]:
    resize_factor = np.sqrt((num_tokens * SAM3D_MOGE_PATCH_SIZE**2) / float(height * width))
    resized_width = max(int(width * resize_factor), SAM3D_MOGE_PATCH_SIZE)
    resized_height = max(int(height * resize_factor), SAM3D_MOGE_PATCH_SIZE)
    return resized_height, resized_width


def _prepare_moge_image_tensor(
    image_rgb: np.ndarray,
    tensors: dict[str, mx.array],
    *,
    resized_size: tuple[int, int],
    patch_grid: tuple[int, int],
) -> mx.array:
    resized_height, resized_width = resized_size
    patch_height, patch_width = patch_grid
    image = image_rgb.astype(np.float32, copy=False) / 255.0
    resized = _resize_hwc_float(image, (resized_height, resized_width), Image.Resampling.BICUBIC)
    aligned = _resize_hwc_float(
        resized,
        (patch_height * SAM3D_MOGE_PATCH_SIZE, patch_width * SAM3D_MOGE_PATCH_SIZE),
        Image.Resampling.BILINEAR,
    )
    nchw = np.transpose(aligned[None, ...], (0, 3, 1, 2))
    image_mx = mx.array(nchw, dtype=mx.float32)
    mean = tensors["image_mean"].astype(mx.float32)
    std = tensors["image_std"].astype(mx.float32)
    return (image_mx - mean) / std


def _resize_hwc_float(values: np.ndarray, size: tuple[int, int], resample: Image.Resampling) -> np.ndarray:
    height, width = (int(value) for value in size)
    clipped = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(np.round(clipped * 255.0).astype(np.uint8), mode="RGB")
    resized = image.resize((width, height), resample=resample)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _run_moge_dinov2(
    image: mx.array,
    tensors: dict[str, mx.array],
    *,
    patch_grid: tuple[int, int],
    attention_chunk_size: int,
    max_attention_bytes: int,
) -> tuple[tuple[mx.array, mx.array], ...]:
    patch_height, patch_width = patch_grid
    patch_map = _conv2d_nchw(
        image,
        tensors["backbone.patch_embed.proj.weight"],
        tensors["backbone.patch_embed.proj.bias"],
        stride=SAM3D_MOGE_PATCH_SIZE,
    )
    patch_tokens = mx.reshape(
        mx.transpose(patch_map, (0, 2, 3, 1)),
        (1, patch_height * patch_width, SAM3D_MOGE_EMBED_DIM),
    )
    cls = mx.broadcast_to(tensors["backbone.cls_token"].astype(patch_tokens.dtype), (1, 1, SAM3D_MOGE_EMBED_DIM))
    hidden = mx.concatenate((cls, patch_tokens), axis=1)
    hidden = hidden + _interpolate_dino_pos_embed(tensors["backbone.pos_embed"], patch_grid).astype(hidden.dtype)
    mx.eval(hidden)

    features: list[tuple[mx.array, mx.array]] = []
    for block_index in range(SAM3D_MOGE_DEPTH):
        hidden = _run_moge_dino_block(
            hidden,
            tensors,
            block_index=block_index,
            attention_chunk_size=attention_chunk_size,
            max_attention_bytes=max_attention_bytes,
        )
        if block_index in SAM3D_MOGE_INTERMEDIATE_BLOCKS:
            normalized = _layer_norm(
                hidden,
                tensors["backbone.norm.weight"],
                tensors["backbone.norm.bias"],
                eps=1e-6,
            )
            features.append((normalized[:, 1:, :], normalized[:, :1, :]))
            mx.eval(features[-1][0], features[-1][1])
        mx.eval(hidden)
    return tuple(features)


def _run_moge_dino_block(
    hidden: mx.array,
    tensors: dict[str, mx.array],
    *,
    block_index: int,
    attention_chunk_size: int,
    max_attention_bytes: int,
) -> mx.array:
    prefix = f"backbone.blocks.{block_index}"
    residual = hidden
    normalized = _layer_norm(
        hidden,
        tensors[f"{prefix}.norm1.weight"],
        tensors[f"{prefix}.norm1.bias"],
        eps=1e-6,
    )
    attended = _moge_self_attention(
        normalized,
        tensors,
        prefix=prefix,
        attention_chunk_size=attention_chunk_size,
        max_attention_bytes=max_attention_bytes,
    )
    hidden = residual + attended * tensors[f"{prefix}.ls1.gamma"].astype(attended.dtype)

    residual = hidden
    normalized = _layer_norm(
        hidden,
        tensors[f"{prefix}.norm2.weight"],
        tensors[f"{prefix}.norm2.bias"],
        eps=1e-6,
    )
    mlp = _linear(
        nn.gelu(_linear(normalized, tensors[f"{prefix}.mlp.fc1.weight"], tensors[f"{prefix}.mlp.fc1.bias"])),
        tensors[f"{prefix}.mlp.fc2.weight"],
        tensors[f"{prefix}.mlp.fc2.bias"],
    )
    return residual + mlp * tensors[f"{prefix}.ls2.gamma"].astype(mlp.dtype)


def _moge_self_attention(
    hidden: mx.array,
    tensors: dict[str, mx.array],
    *,
    prefix: str,
    attention_chunk_size: int,
    max_attention_bytes: int,
) -> mx.array:
    batch, token_count, _ = tuple(int(dim) for dim in hidden.shape)
    head_dim = SAM3D_MOGE_EMBED_DIM // SAM3D_MOGE_NUM_HEADS
    estimated = batch * SAM3D_MOGE_NUM_HEADS * min(token_count, attention_chunk_size) * token_count * 4
    if estimated > max_attention_bytes:
        raise _MogeBlocked(
            Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="guard MLX MoGe exact DINO attention",
                reason=(
                    "MLX MoGe exact attention chunk would exceed the configured activation guard "
                    f"({estimated} > {max_attention_bytes} bytes)"
                ),
                metadata={
                    "estimated_attention_bytes": int(estimated),
                    "max_attention_bytes": int(max_attention_bytes),
                    "token_count": int(token_count),
                    "attention_chunk_size": int(attention_chunk_size),
                },
            )
        )
    qkv = _linear(hidden, tensors[f"{prefix}.attn.qkv.weight"], tensors[f"{prefix}.attn.qkv.bias"])
    query, key, value = (
        mx.transpose(mx.reshape(part, (batch, token_count, SAM3D_MOGE_NUM_HEADS, head_dim)), (0, 2, 1, 3))
        for part in mx.split(qkv, 3, axis=-1)
    )
    attended_chunks = []
    scale = head_dim**-0.5
    for start in range(0, token_count, attention_chunk_size):
        stop = min(start + attention_chunk_size, token_count)
        logits = (query[:, :, start:stop, :] @ mx.transpose(key, (0, 1, 3, 2))) * scale
        weights = mx.softmax(logits.astype(mx.float32), axis=-1).astype(value.dtype)
        attended_chunks.append(weights @ value)
    attended = mx.concatenate(attended_chunks, axis=2)
    merged = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)), (batch, token_count, SAM3D_MOGE_EMBED_DIM))
    return _linear(merged, tensors[f"{prefix}.attn.proj.weight"], tensors[f"{prefix}.attn.proj.bias"])


def _run_moge_head(
    features: tuple[tuple[mx.array, mx.array], ...],
    tensors: dict[str, mx.array],
    *,
    resized_size: tuple[int, int],
    output_size: tuple[int, int],
    aspect_ratio: float,
) -> tuple[mx.array, mx.array]:
    resized_height, resized_width = resized_size
    patch_height = resized_height // SAM3D_MOGE_PATCH_SIZE
    patch_width = resized_width // SAM3D_MOGE_PATCH_SIZE
    projected: list[mx.array] = []
    for index, (patch_tokens, _cls_token) in enumerate(features):
        feature = mx.reshape(
            mx.transpose(patch_tokens, (0, 2, 1)),
            (1, SAM3D_MOGE_EMBED_DIM, patch_height, patch_width),
        )
        projected.append(
            _conv2d_nchw(
                feature,
                tensors[f"head.projects.{index}.weight"],
                tensors[f"head.projects.{index}.bias"],
                stride=1,
                padding=0,
            )
        )
    x = projected[0]
    for feature in projected[1:]:
        x = x + feature
    mx.eval(x)

    for stage_index in range(3):
        uv = _normalized_view_plane_uv_mx(
            width=int(x.shape[3]),
            height=int(x.shape[2]),
            aspect_ratio=aspect_ratio,
            dtype=x.dtype,
        )
        x = mx.concatenate((x, uv), axis=1)
        x = _conv_transpose2d_nchw(
            x,
            tensors[f"head.upsample_blocks.{stage_index}.0.0.weight"],
            tensors[f"head.upsample_blocks.{stage_index}.0.0.bias"],
            stride=2,
        )
        x = _conv2d_nchw_replicate(
            x,
            tensors[f"head.upsample_blocks.{stage_index}.0.1.weight"],
            tensors[f"head.upsample_blocks.{stage_index}.0.1.bias"],
            padding=1,
        )
        for block_index in (1, 2):
            x = _run_moge_residual_conv_block(x, tensors, f"head.upsample_blocks.{stage_index}.{block_index}")
        mx.eval(x)

    x = _resize_nchw_mlx(x, output_size, mode="linear")
    uv = _normalized_view_plane_uv_mx(
        width=int(x.shape[3]),
        height=int(x.shape[2]),
        aspect_ratio=aspect_ratio,
        dtype=x.dtype,
    )
    x = mx.concatenate((x, uv), axis=1)
    points = _run_moge_output_block(x, tensors, "head.output_block.0")
    mask = _run_moge_output_block(x, tensors, "head.output_block.1")
    mx.eval(points, mask)
    return points.astype(mx.float32), mask.astype(mx.float32)


def _run_moge_residual_conv_block(x: mx.array, tensors: dict[str, mx.array], prefix: str) -> mx.array:
    residual = x
    hidden = _group_norm_nchw(
        x,
        tensors[f"{prefix}.layers.0.weight"],
        tensors[f"{prefix}.layers.0.bias"],
        groups=1,
    )
    hidden = mx.maximum(hidden, 0)
    hidden = _conv2d_nchw_replicate(
        hidden,
        tensors[f"{prefix}.layers.2.weight"],
        tensors[f"{prefix}.layers.2.bias"],
        padding=1,
    )
    groups = max(int(tensors[f"{prefix}.layers.3.weight"].shape[0]) // 32, 1)
    hidden = _group_norm_nchw(
        hidden,
        tensors[f"{prefix}.layers.3.weight"],
        tensors[f"{prefix}.layers.3.bias"],
        groups=groups,
    )
    hidden = mx.maximum(hidden, 0)
    hidden = _conv2d_nchw_replicate(
        hidden,
        tensors[f"{prefix}.layers.5.weight"],
        tensors[f"{prefix}.layers.5.bias"],
        padding=1,
    )
    return hidden + residual


def _run_moge_output_block(x: mx.array, tensors: dict[str, mx.array], prefix: str) -> mx.array:
    x = _conv2d_nchw_replicate(x, tensors[f"{prefix}.0.weight"], tensors[f"{prefix}.0.bias"], padding=1)
    x = mx.maximum(x, 0)
    return _conv2d_nchw_replicate(x, tensors[f"{prefix}.2.weight"], tensors[f"{prefix}.2.bias"], padding=0)


def _interpolate_dino_pos_embed(pos_embed: mx.array, patch_grid: tuple[int, int]) -> mx.array:
    patch_height, patch_width = patch_grid
    expected = patch_height * patch_width + 1
    if int(pos_embed.shape[1]) == expected:
        return pos_embed
    from scipy import ndimage

    pos_np = np.array(pos_embed, dtype=np.float32)
    stored_count = pos_np.shape[1] - 1
    stored_side = int(stored_count**0.5)
    if stored_side * stored_side != stored_count:
        raise _MogeBlocked(
            Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="interpolate MoGe DINO positional embedding",
                reason="MoGe DINO positional embedding does not contain a square patch grid",
                metadata={"pos_embed_shape": tuple(int(value) for value in pos_embed.shape)},
            )
        )
    cls = pos_np[:, :1, :]
    patch = pos_np[:, 1:, :].reshape((stored_side, stored_side, pos_np.shape[-1]))
    resized = ndimage.zoom(
        patch,
        (patch_height / stored_side, patch_width / stored_side, 1.0),
        order=3,
        mode="nearest",
        prefilter=True,
    )
    resized = _fit_resized_pos_embed(resized, patch_grid)
    patch_tokens = resized.reshape((1, patch_height * patch_width, pos_np.shape[-1]))
    return mx.array(np.concatenate((cls, patch_tokens), axis=1), dtype=pos_embed.dtype)


def _fit_resized_pos_embed(values: np.ndarray, patch_grid: tuple[int, int]) -> np.ndarray:
    height, width = patch_grid
    output = values
    if output.shape[0] < height or output.shape[1] < width:
        output = np.pad(
            output,
            ((0, max(height - output.shape[0], 0)), (0, max(width - output.shape[1], 0)), (0, 0)),
            mode="edge",
        )
    return output[:height, :width, :]


def _normalized_view_plane_uv_mx(
    *,
    width: int,
    height: int,
    aspect_ratio: float,
    dtype: mx.Dtype,
) -> mx.array:
    uv = normalized_view_plane_uv_numpy(width=width, height=height, aspect_ratio=aspect_ratio)
    uv = np.transpose(uv, (2, 0, 1))[None, ...]
    return mx.array(uv, dtype=dtype)


def normalized_view_plane_uv_numpy(
    *,
    width: int,
    height: int,
    aspect_ratio: float | None = None,
    dtype=np.float32,
) -> np.ndarray:
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / np.sqrt(1.0 + aspect_ratio**2)
    span_y = 1.0 / np.sqrt(1.0 + aspect_ratio**2)
    u = np.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
    )
    v = np.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
    )
    grid_u, grid_v = np.meshgrid(u, v, indexing="xy")
    return np.stack((grid_u, grid_v), axis=-1)


def recover_focal_shift_numpy(
    points: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    focal: float | None = None,
    downsample_size: tuple[int, int] = (64, 64),
) -> tuple[float, float]:
    height, width = points.shape[:2]
    uv = normalized_view_plane_uv_numpy(width=width, height=height)
    points_lr = _nearest_resize_hwc(points, downsample_size)
    uv_lr = _nearest_resize_hwc(uv, downsample_size)
    if mask is not None:
        mask_lr = _nearest_resize_mask(mask, downsample_size)
        points_lr = points_lr[mask_lr]
        uv_lr = uv_lr[mask_lr]
    else:
        points_lr = points_lr.reshape(-1, 3)
        uv_lr = uv_lr.reshape(-1, 2)
    finite = np.isfinite(points_lr).all(axis=-1)
    points_lr = points_lr[finite]
    uv_lr = uv_lr[finite]
    if points_lr.shape[0] < 2:
        return 1.0, 0.0
    if focal is None:
        shift, recovered_focal = _solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = _solve_optimal_shift(uv_lr, points_lr, focal)
        recovered_focal = float(focal)
    return float(recovered_focal), float(shift)


def _solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import least_squares

    uv_flat = uv.reshape(-1, 2).astype(np.float64)
    xy = xyz[..., :2].reshape(-1, 2).astype(np.float64)
    z = xyz[..., 2].reshape(-1).astype(np.float64)

    def residual(shift):
        xy_proj = xy / (z + shift[0])[:, None]
        denom = np.square(xy_proj).sum()
        focal = 1.0 if denom <= 1e-12 else float((xy_proj * uv_flat).sum() / denom)
        return (focal * xy_proj - uv_flat).ravel()

    solution = least_squares(residual, x0=np.array([0.0]), ftol=1e-3, method="lm")
    shift = float(solution.x.squeeze())
    xy_proj = xy / (z + shift)[:, None]
    denom = np.square(xy_proj).sum()
    focal = 1.0 if denom <= 1e-12 else float((xy_proj * uv_flat).sum() / denom)
    return shift, focal


def _solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float) -> float:
    from scipy.optimize import least_squares

    uv_flat = uv.reshape(-1, 2).astype(np.float64)
    xy = xyz[..., :2].reshape(-1, 2).astype(np.float64)
    z = xyz[..., 2].reshape(-1).astype(np.float64)

    def residual(shift):
        xy_proj = xy / (z + shift[0])[:, None]
        return (float(focal) * xy_proj - uv_flat).ravel()

    solution = least_squares(residual, x0=np.array([0.0]), ftol=1e-3, method="lm")
    return float(solution.x.squeeze())


def _nearest_resize_hwc(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    height, width = (int(value) for value in size)
    rows = np.clip(((np.arange(height) + 0.5) * values.shape[0] / height).astype(np.int64), 0, values.shape[0] - 1)
    cols = np.clip(((np.arange(width) + 0.5) * values.shape[1] / width).astype(np.int64), 0, values.shape[1] - 1)
    return values[rows[:, None], cols[None, :], :]


def _nearest_resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return _nearest_resize_hwc(mask.astype(np.float32)[..., None], size)[..., 0] > 0.5


def _remap_moge_points_exp(points: np.ndarray) -> np.ndarray:
    z = np.exp(points[..., 2:3])
    return np.concatenate((points[..., :2] * z, z), axis=-1).astype(np.float32, copy=False)


def _validate_full_moge_tensors(tensors: dict[str, mx.array]) -> Sam3dAssetBlocker | None:
    expected = {
        "backbone.patch_embed.proj.weight": (SAM3D_MOGE_EMBED_DIM, 3, SAM3D_MOGE_PATCH_SIZE, SAM3D_MOGE_PATCH_SIZE),
        "backbone.patch_embed.proj.bias": (SAM3D_MOGE_EMBED_DIM,),
        "backbone.cls_token": (1, 1, SAM3D_MOGE_EMBED_DIM),
        "backbone.pos_embed": (1, 1370, SAM3D_MOGE_EMBED_DIM),
        "backbone.norm.weight": (SAM3D_MOGE_EMBED_DIM,),
        "backbone.norm.bias": (SAM3D_MOGE_EMBED_DIM,),
        "head.projects.0.weight": (512, SAM3D_MOGE_EMBED_DIM, 1, 1),
        "head.output_block.0.2.weight": (3, 32, 1, 1),
        "head.output_block.1.2.weight": (1, 32, 1, 1),
    }
    for name, shape in expected.items():
        actual = tuple(int(value) for value in tensors[name].shape)
        if actual != shape:
            return Sam3dAssetBlocker(
                stage="moge-pointmap",
                operation="validate full MoGe v1 tensor shapes",
                reason=f"MoGe tensor {name} has shape {actual}, expected {shape}",
                metadata={"tensor": name, "actual": actual, "expected": shape},
            )
    for block_index in range(SAM3D_MOGE_DEPTH):
        prefix = f"backbone.blocks.{block_index}"
        for suffix in (
            "norm1.weight",
            "norm1.bias",
            "norm2.weight",
            "norm2.bias",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
            "ls1.gamma",
            "ls2.gamma",
        ):
            key = f"{prefix}.{suffix}"
            if key not in tensors:
                return Sam3dAssetBlocker(
                    stage="moge-pointmap",
                    operation="validate full MoGe DINO block tensors",
                    reason=f"MoGe checkpoint is missing required tensor: {key}",
                    metadata={"missing": key, "block_index": block_index},
                )
    return None


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


def _group_norm_nchw(values: mx.array, weight: mx.array, bias: mx.array, *, groups: int, eps: float = 1e-5) -> mx.array:
    batch, channels, height, width = tuple(int(value) for value in values.shape)
    grouped = mx.reshape(values, (batch, groups, channels // groups, height, width))
    mean = mx.mean(grouped, axis=(2, 3, 4), keepdims=True)
    centered = grouped - mean
    variance = mx.mean(centered * centered, axis=(2, 3, 4), keepdims=True)
    normalized = mx.reshape(centered * mx.rsqrt(variance + eps), (batch, channels, height, width))
    return normalized * weight.astype(values.dtype)[None, :, None, None] + bias.astype(values.dtype)[None, :, None, None]


def _conv2d_nchw(
    values: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> mx.array:
    values_nhwc = mx.transpose(values, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight.astype(values.dtype), (0, 2, 3, 1))
    output = mx.conv2d(values_nhwc, weight_ohwi, stride=stride, padding=padding)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _conv2d_nchw_replicate(
    values: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    padding: int,
) -> mx.array:
    if padding:
        values = mx.pad(values, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="edge")
    return _conv2d_nchw(values, weight, bias, stride=1, padding=0)


def _conv_transpose2d_nchw(
    values: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    *,
    stride: int,
) -> mx.array:
    values_nhwc = mx.transpose(values, (0, 2, 3, 1))
    weight_hwio = mx.transpose(weight.astype(values.dtype), (1, 2, 3, 0))
    output = mx.conv_transpose2d(values_nhwc, weight_hwio, stride=stride, padding=0)
    if bias is not None:
        output = output + bias.astype(output.dtype)[None, None, None, :]
    return mx.transpose(output, (0, 3, 1, 2))


def _resize_nchw_mlx(values: mx.array, size: tuple[int, int], *, mode: str) -> mx.array:
    height, width = (int(value) for value in size)
    if int(values.shape[2]) == height and int(values.shape[3]) == width:
        return values
    scale = (height / int(values.shape[2]), width / int(values.shape[3]))
    output = nn.Upsample(scale, mode=mode, align_corners=False)(mx.transpose(values, (0, 2, 3, 1)))
    if int(output.shape[1]) < height or int(output.shape[2]) < width:
        pad_h = max(height - int(output.shape[1]), 0)
        pad_w = max(width - int(output.shape[2]), 0)
        output = mx.pad(output, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    output = output[:, :height, :width, :]
    return mx.transpose(output, (0, 3, 1, 2))
