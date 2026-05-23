"""LiTo inference orchestration.

The default generation path is reserved for checkpoint-backed LiTo output.
The local source-contract pipeline remains available only as an explicit smoke
mode so synthetic PLY files cannot be mistaken for real LiTo results.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors

from .lito_assets import LITO_DEFAULT_ROOT, validate as validate_lito_assets
from .lito_condition import LitoCondition
from .lito_dit import LITO_MEMORY_PROFILES as _DIT_MEMORY_PROFILES
from .lito_dit import LitoDiT
from .lito_render import LitoRenderer
from .lito_real_backend import (
    LITO_INIT_COORD_CAP_PROFILE,
    LitoBackendUnavailable,
    LitoRealBackendConfig,
    create_lito_real_backend,
    resolve_lito_init_coord_cap,
    write_lito_gaussians_ply,
)
from .lito_tokenizer import LitoTokenizer


logger = logging.getLogger(__name__)

LITO_MEMORY_PROFILES = _DIT_MEMORY_PROFILES
LITO_DEFAULT_MEMORY_PROFILE = "balanced"
LitoMemoryProfileName = Literal["safe", "balanced", "large"]

LITO_SOFT_MEMORY_LIMIT_GB = 90.0
LITO_HARD_MEMORY_LIMIT_GB = 100.0

# Source: upstream Apple LiTo demos/fastapi_lito_demo.py:555-560 and :668-670.
LITO_RECOMMENDED_NUM_STEPS = 20
# Source: upstream Apple LiTo demos/fastapi_lito_demo.py:555-560 and :624-647.
LITO_RECOMMENDED_SEED_POLICY = "unseeded_runtime"
# Source: upstream Apple LiTo demos/fastapi_lito_demo.py:555-560 and lito_dit_trainer.py:760-768.
LITO_RECOMMENDED_CFG_SCALE = 3.0
# Source: upstream Apple LiTo demos/fastapi_lito_demo.py:141-148 and :641-645.
LITO_RECOMMENDED_RESOLUTION = (518, 518)
# Source: upstream Apple LiTo src/lito/models/dino.py:43-55.
LITO_RECOMMENDED_IMAGE_MEAN = (0.485, 0.456, 0.406)
# Source: upstream Apple LiTo src/lito/models/dino.py:43-55.
LITO_RECOMMENDED_IMAGE_STD = (0.229, 0.224, 0.225)
# Source: upstream Apple LiTo src/lito/trainers/lito_dit_trainer.py:880-906.
LITO_RECOMMENDED_SAMPLER = "heun"
# Source: upstream Apple LiTo demos/fastapi_lito_demo.py:356-379.
LITO_RECOMMENDED_DECODE_STEPS_FOR_SAMPLE_XYZ = 50
# Source: upstream Apple LiTo demos/fastapi_lito_demo.py:338-345 and :356-379.
LITO_RECOMMENDED_MLX_COMPUTE_DTYPE = "float16"

LITO_STAGE_NAMES = ("preprocess", "condition", "tokenize", "dit", "decode", "render", "export")

LITO_REAL_TENSOR_SENTINELS = {
    "tokenizer/lito_new.safetensors": ("gs_decoder.gs_output_shape_mlp.2.linear.bias",),
    "image_to_3d/lito_dit_rgba.safetensors": ("velocity_estimator_ema.module.final_layer.linear.bias",),
}

_IMAGE_FLOAT_DTYPE = np.float32  # float32 preserves image/preprocess fixture schema before MLX half conversion.
_GAUSSIAN_FLOAT_DTYPE = np.float32  # float32 matches 3DGS export and rasterizer parameter schema.


@dataclass(frozen=True)
class LitoMemoryProfile:
    """Small source-contract execution knobs for pipeline smoke generation."""

    name: LitoMemoryProfileName
    input_points: int
    gaussian_count: int
    render_size: int
    use_metal_render: bool


LITO_MEMORY_PROFILE_CONFIGS: dict[str, LitoMemoryProfile] = {
    "safe": LitoMemoryProfile("safe", input_points=1024, gaussian_count=64, render_size=48, use_metal_render=False),
    "balanced": LitoMemoryProfile(
        "balanced",
        input_points=2048,
        gaussian_count=128,
        render_size=64,
        use_metal_render=False,
    ),
    "large": LitoMemoryProfile("large", input_points=4096, gaussian_count=256, render_size=96, use_metal_render=False),
}


class LitoMemoryLimitExceeded(RuntimeError):
    """Raised when LiTo generation crosses the configured memory ceiling."""


class LitoRealGenerationNotImplemented(RuntimeError):
    """Raised when checkpoint-backed generation is requested before the backend exists."""


@dataclass(frozen=True)
class LitoRealAssetSummary:
    """Header-only proof that a converted LiTo root contains real checkpoint tensors."""

    root: Path
    checkpoint_key_counts: dict[str, int]
    sentinel_shapes: dict[str, tuple[int, ...]]
    sentinel_dtypes: dict[str, str]


@dataclass(frozen=True)
class LitoGenerationResult:
    """Outputs and metrics from one LiTo source-contract generation."""

    gaussians: dict[str, np.ndarray]
    rendered_image: np.ndarray | None
    output_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class _PreprocessedImage:
    straight_rgb: np.ndarray
    alpha: np.ndarray
    rgba: np.ndarray
    resolution: tuple[int, int]


class LitoInferencePipeline:
    """LiTo image-to-3DGS pipeline wrapper."""

    def __init__(
        self,
        weights_root: str | Path = LITO_DEFAULT_ROOT,
        *,
        memory_profile: str = LITO_DEFAULT_MEMORY_PROFILE,
        max_init_coords_per_batch: int | str | None = LITO_INIT_COORD_CAP_PROFILE,
        source_contract_smoke: bool = False,
    ) -> None:
        self.weights_root = Path(weights_root)
        self.memory_profile = memory_profile_config(memory_profile)
        self.max_init_coords_per_batch = normalize_lito_init_coord_cap(max_init_coords_per_batch)
        self.source_contract_smoke = bool(source_contract_smoke)
        self.real_asset_summary = None if self.source_contract_smoke else _require_checkpoint_backed_assets(
            self.weights_root
        )
        if self.source_contract_smoke:
            self.conditioner = LitoCondition.load(self.weights_root)
            self.tokenizer = LitoTokenizer.load(self.weights_root)
            self.dit = LitoDiT(memory_profile=self.memory_profile.name)
            self.renderer = LitoRenderer(
                str(self.weights_root),
                use_metal=self.memory_profile.use_metal_render,
                allow_cpu_fallback=True,
            )

    def generate(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None = None,
        output_format: str = "ply",
        num_steps: int = LITO_RECOMMENDED_NUM_STEPS,
        seed: int | None = None,
        cfg_scale: float = LITO_RECOMMENDED_CFG_SCALE,
        resolution: int | tuple[int, int] = LITO_RECOMMENDED_RESOLUTION,
        render_size: int | None = None,
    ) -> LitoGenerationResult:
        """Run LiTo generation."""

        if not self.source_contract_smoke:
            return self._generate_checkpoint_backed(
                image_path,
                output_path=output_path,
                output_format=output_format,
                num_steps=num_steps,
                seed=seed,
                cfg_scale=cfg_scale,
                resolution=resolution,
                render_size=render_size,
            )

        return self._generate_source_contract_smoke(
            image_path,
            output_path=output_path,
            output_format=output_format,
            num_steps=num_steps,
            seed=seed,
            cfg_scale=cfg_scale,
            resolution=resolution,
            render_size=render_size,
        )

    def _generate_checkpoint_backed(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None,
        output_format: str,
        num_steps: int,
        seed: int | None,
        cfg_scale: float,
        resolution: int | tuple[int, int],
        render_size: int | None,
    ) -> LitoGenerationResult:
        assert self.real_asset_summary is not None
        output_format = _normalize_output_format(output_format, output_path)
        output = Path(output_path) if output_path is not None else None
        metrics: dict[str, dict[str, float]] = {}
        image = Path(image_path)
        _ = render_size
        preprocessed = self._stage(
            "preprocess",
            metrics,
            lambda: _preprocess_image(image, resolution=_coerce_resolution(resolution)),
        )
        backend = create_lito_real_backend(
            LitoRealBackendConfig(
                weights_root=self.weights_root,
                asset_summary=self.real_asset_summary,
                memory_profile=self.memory_profile.name,
                max_init_coords_per_batch=self.max_init_coords_per_batch,
                raw_weights_root=_infer_raw_weights_root(self.weights_root),
                allow_cuda=False,
                mlx_compute_dtype=LITO_RECOMMENDED_MLX_COMPUTE_DTYPE,
            )
        )
        condition = self._stage(
            "condition",
            metrics,
            lambda: backend.condition_rgba(preprocessed.rgba),
        )
        latent = self._stage(
            "dit",
            metrics,
            lambda: backend.sample_dit_latents(
                condition,
                seed=seed,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
                method=LITO_RECOMMENDED_SAMPLER,
            ),
        )
        gaussians = self._stage(
            "decode",
            metrics,
            lambda: backend.decode_sampled_latents_to_gaussians(latent),
        )
        artifacts = self._stage(
            "export",
            metrics,
            lambda: _export_real_outputs(gaussians, output, output_format),
        )
        metadata = {
            "pipeline": "lito-checkpoint-backed",
            "weights_root": str(self.weights_root),
            "memory_profile": self.memory_profile.name,
            "max_init_coords_per_batch": self.max_init_coords_per_batch,
            "num_steps": int(num_steps),
            "cfg_scale": float(cfg_scale),
            "seed": seed,
            "seed_policy": LITO_RECOMMENDED_SEED_POLICY,
            "recommended_sampler": LITO_RECOMMENDED_SAMPLER,
            "recommended_compute_dtype": LITO_RECOMMENDED_MLX_COMPUTE_DTYPE,
            "image_path": str(image),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "checkpoints": dict(self.real_asset_summary.checkpoint_key_counts),
        }
        return LitoGenerationResult(
            gaussians=gaussians,
            rendered_image=None,
            output_path=output,
            metadata=metadata,
            metrics=metrics,
        )

    def _generate_source_contract_smoke(
        self,
        image_path: str | Path,
        *,
        output_path: str | Path | None = None,
        output_format: str = "ply",
        num_steps: int = LITO_RECOMMENDED_NUM_STEPS,
        seed: int | None = None,
        cfg_scale: float = LITO_RECOMMENDED_CFG_SCALE,
        resolution: int | tuple[int, int] = LITO_RECOMMENDED_RESOLUTION,
        render_size: int | None = None,
    ) -> LitoGenerationResult:
        """Run deterministic local LiTo source-contract smoke generation."""

        output_format = _normalize_output_format(output_format, output_path)
        output = Path(output_path) if output_path is not None else None
        metrics: dict[str, dict[str, float]] = {}
        image = Path(image_path)
        active_render_size = int(render_size or self.memory_profile.render_size)

        preprocessed = self._stage(
            "preprocess",
            metrics,
            lambda: _preprocess_image(image, resolution=_coerce_resolution(resolution)),
        )
        cond_tokens = self._stage(
            "condition",
            metrics,
            lambda: self.conditioner(preprocessed.straight_rgb, preprocessed.alpha),
        )
        latent_tokens = self._stage(
            "tokenize",
            metrics,
            lambda: self._tokenize_preprocessed(preprocessed),
        )
        latent = self._stage(
            "dit",
            metrics,
            lambda: self.dit.sample(
                cond_tokens,
                num_steps=num_steps,
                seed=seed if seed is not None else 0,
                initial_latent=latent_tokens,
                memory_profile=self.memory_profile.name,
            ),
        )
        gaussians = self._stage(
            "decode",
            metrics,
            lambda: _decode_latent_to_gaussians(
                latent,
                gaussian_count=self.memory_profile.gaussian_count,
                render_size=active_render_size,
                cfg_scale=cfg_scale,
            ),
        )
        render_result = self._stage(
            "render",
            metrics,
            lambda: self.renderer.render(gaussians),
        )
        artifacts = self._stage(
            "export",
            metrics,
            lambda: _export_outputs(gaussians, output, output_format),
        )

        metadata = {
            "pipeline": "lito-source-contract-smoke",
            "weights_root": str(self.weights_root),
            "memory_profile": self.memory_profile.name,
            "num_steps": int(num_steps),
            "cfg_scale": float(cfg_scale),
            "seed": seed,
            "seed_policy": LITO_RECOMMENDED_SEED_POLICY,
            "recommended_sampler": LITO_RECOMMENDED_SAMPLER,
            "recommended_compute_dtype": LITO_RECOMMENDED_MLX_COMPUTE_DTYPE,
            "image_path": str(image),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "render": dict(render_result.metadata),
        }
        return LitoGenerationResult(
            gaussians=gaussians,
            rendered_image=np.asarray(render_result.image),
            output_path=output,
            metadata=metadata,
            metrics=metrics,
        )

    def _tokenize_preprocessed(self, preprocessed: _PreprocessedImage) -> mx.array:
        inputs = _image_to_point_ray_inputs(preprocessed, point_count=self.memory_profile.input_points)
        return self.tokenizer(
            inputs["xyz_w"],
            inputs["rgb"],
            inputs["ray_origin_direction_w"],
        )

    def _stage(self, name: str, metrics: dict[str, dict[str, float]], call: Callable[[], Any]) -> Any:
        if name not in LITO_STAGE_NAMES:
            raise ValueError(f"unknown LiTo stage: {name}")
        _reset_peak_memory()
        start = time.perf_counter()
        result = call()
        _eval_output(result)
        _synchronize()
        wall_time_s = time.perf_counter() - start
        peak_active_memory_gb = _check_memory(name)
        metrics[name] = {
            "wall_time_s": wall_time_s,
            "peak_active_memory_gb": peak_active_memory_gb,
            "peak_cache_memory_gb": _memory_gb("cache"),
        }
        return result


def memory_profile_config(name: str) -> LitoMemoryProfile:
    """Return a LiTo pipeline memory profile by name."""

    try:
        return LITO_MEMORY_PROFILE_CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"unknown LiTo memory profile: {name!r}") from error


def normalize_lito_init_coord_cap(value: int | str | None) -> int | str | None:
    """Normalize a public LiTo init-coordinate cap override."""

    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token == LITO_INIT_COORD_CAP_PROFILE:
            return LITO_INIT_COORD_CAP_PROFILE
        if token == "none":
            return None
        if token.isdecimal():
            value = int(token)
        else:
            raise ValueError(
                "--max-init-coords-per-batch must be 'profile', 'none', or a positive integer, "
                f"got {value!r}"
            )
    return resolve_lito_init_coord_cap("safe", value)


def _require_checkpoint_backed_assets(root: Path) -> LitoRealAssetSummary:
    validation = validate_lito_assets(root)
    if not validation.ready:
        raise FileNotFoundError(
            "checkpoint-backed LiTo generation requires converted assets under "
            f"{validation.root}; missing: {', '.join(validation.missing)}. "
            "Use `mlx-spatial-lito download-command` and `python -m mlx_spatial.lito_assets convert`, "
            "or pass --source-contract-smoke for synthetic smoke only."
        )

    checkpoint_key_counts: dict[str, int] = {}
    sentinel_shapes: dict[str, tuple[int, ...]] = {}
    sentinel_dtypes: dict[str, str] = {}
    for relative_path, required_keys in LITO_REAL_TENSOR_SENTINELS.items():
        path = root / relative_path
        with safe_open(path, framework="np") as handle:
            keys = set(handle.keys())
            checkpoint_key_counts[relative_path] = len(keys)
            missing = [key for key in required_keys if key not in keys]
            if missing:
                raise ValueError(
                    f"LiTo checkpoint {path} is present but missing required real tensor keys: "
                    f"{', '.join(missing)}"
                )
            for key in required_keys:
                tensor_slice = handle.get_slice(key)
                summary_key = f"{relative_path}:{key}"
                sentinel_shapes[summary_key] = tuple(int(dim) for dim in tensor_slice.get_shape())
                sentinel_dtypes[summary_key] = str(tensor_slice.get_dtype())
    return LitoRealAssetSummary(
        root=root,
        checkpoint_key_counts=checkpoint_key_counts,
        sentinel_shapes=sentinel_shapes,
        sentinel_dtypes=sentinel_dtypes,
    )


def _check_memory(stage: str) -> float:
    active_gb = _memory_gb("active")
    peak_gb = _memory_gb("peak")
    peak_active_gb = max(active_gb, peak_gb)
    if peak_active_gb >= LITO_HARD_MEMORY_LIMIT_GB:
        raise LitoMemoryLimitExceeded(
            f"stage {stage}: peak active memory {peak_active_gb:.1f} GB exceeded "
            f"{LITO_HARD_MEMORY_LIMIT_GB:.1f} GB ceiling"
        )
    if peak_active_gb >= LITO_SOFT_MEMORY_LIMIT_GB:
        logger.warning(
            "stage %s: peak active memory %.1f GB crossed %.1f GB soft threshold",
            stage,
            peak_active_gb,
            LITO_SOFT_MEMORY_LIMIT_GB,
        )
    return peak_active_gb


def _preprocess_image(path: Path, *, resolution: tuple[int, int]) -> _PreprocessedImage:
    if not path.is_file():
        raise FileNotFoundError(f"LiTo input image not found: {path}")
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGBA")
    image = _crop_and_pad_object(image)
    image = image.resize(resolution, Image.Resampling.LANCZOS)
    rgba = np.asarray(image, dtype=_IMAGE_FLOAT_DTYPE) / 255.0
    straight_rgb = rgba[None, None, :, :, :3].astype(_IMAGE_FLOAT_DTYPE, copy=False)
    alpha = rgba[None, None, :, :, 3:4].astype(_IMAGE_FLOAT_DTYPE, copy=False)
    return _PreprocessedImage(straight_rgb=straight_rgb, alpha=alpha, rgba=rgba, resolution=resolution)


def _crop_and_pad_object(
    image: Image.Image,
    *,
    fill_ratio: float = 0.8,
    th_alpha: float = 0.8,
    pad_x_ratio: float = 0.5,
    pad_y_ratio: float = 0.5,
) -> Image.Image:
    alpha = np.asarray(image.getchannel("A"), dtype=_IMAGE_FLOAT_DTYPE) / 255.0
    rows = np.any(alpha > th_alpha, axis=1)
    cols = np.any(alpha > th_alpha, axis=0)
    if not np.any(rows) or not np.any(cols):
        return image.copy()

    h, w = alpha.shape
    y_indices = np.flatnonzero(rows)
    x_indices = np.flatnonzero(cols)
    y_min = int(y_indices[0])
    y_max = int(y_indices[-1])
    x_min = int(x_indices[0])
    x_max = int(x_indices[-1])

    # Mirrors upstream determine_crop_and_pad(..., keep_optical_axis=True).
    center_x = w / 2.0
    center_y = h / 2.0
    bbox_size = max(
        max(abs(x_max - center_x), abs(x_min - center_x)),
        max(abs(y_max - center_y), abs(y_min - center_y)),
    )
    bbox_size *= 2.0
    crop_size = max(1, int(bbox_size / fill_ratio))
    pad_size = crop_size - bbox_size

    crop_x1 = int(center_x - bbox_size / 2.0 - pad_size * pad_x_ratio)
    crop_y1 = int(center_y - bbox_size / 2.0 - pad_size * pad_y_ratio)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    pad_left = max(0, -crop_x1)
    pad_top = max(0, -crop_y1)
    pad_right = max(0, crop_x2 - w)
    pad_bottom = max(0, crop_y2 - h)

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(w, crop_x2)
    crop_y2 = min(h, crop_y2)

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    cropped = rgba[crop_y1:crop_y2, crop_x1:crop_x2]
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    return Image.fromarray(cropped, mode="RGBA")


def _image_to_point_ray_inputs(preprocessed: _PreprocessedImage, *, point_count: int) -> dict[str, mx.array]:
    rgba = preprocessed.rgba
    height, width = rgba.shape[:2]
    flat = rgba.reshape((-1, 4))
    total = flat.shape[0]
    indices = np.linspace(0, total - 1, point_count, dtype=np.int64)
    ys = (indices // width).astype(_IMAGE_FLOAT_DTYPE)
    xs = (indices % width).astype(_IMAGE_FLOAT_DTYPE)
    x_norm = (xs / max(width - 1, 1)) * 2.0 - 1.0
    y_norm = (ys / max(height - 1, 1)) * 2.0 - 1.0
    samples = flat[indices]
    rgb = samples[:, :3].astype(_IMAGE_FLOAT_DTYPE, copy=False)
    alpha = samples[:, 3:4].astype(_IMAGE_FLOAT_DTYPE, copy=False)
    xyz = np.concatenate([x_norm[:, None] * alpha, -y_norm[:, None] * alpha, 1.0 + alpha], axis=1)
    direction = np.stack([x_norm, -y_norm, np.ones_like(x_norm)], axis=1)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(min=1e-6)
    origin = np.zeros_like(direction, dtype=_IMAGE_FLOAT_DTYPE)
    rays = np.concatenate([origin, direction], axis=1)
    return {
        "xyz_w": mx.array(xyz[None, :, :], dtype=mx.float16),
        "rgb": mx.array(rgb[None, :, :], dtype=mx.float16),
        "ray_origin_direction_w": mx.array(rays[None, :, :], dtype=mx.float16),
    }


def _decode_latent_to_gaussians(
    latent: mx.array,
    *,
    gaussian_count: int,
    render_size: int,
    cfg_scale: float,
) -> dict[str, np.ndarray]:
    latent_np = np.asarray(latent, dtype=_GAUSSIAN_FLOAT_DTYPE)
    if latent_np.ndim != 3 or latent_np.shape[-1] < 14:
        raise ValueError(f"latent must have shape (B, N, >=14), got {latent_np.shape}")
    rows = latent_np[0, :gaussian_count]
    xyz = np.tanh(rows[:, 0:3])
    xyz[:, 2] = 2.25 + 0.35 * np.abs(xyz[:, 2])
    scaling = 0.025 + 0.08 * _sigmoid(rows[:, 3:6])
    quaternion = rows[:, 6:10]
    quaternion[:, 3] += 1.0
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True).clip(min=1e-6)
    opacity = _sigmoid(rows[:, 10:11] * float(cfg_scale) / max(LITO_RECOMMENDED_CFG_SCALE, 1e-6))
    color = _sigmoid(rows[:, 11:14])
    lf = (
        np.tanh(rows[:, 14:18])
        if rows.shape[1] >= 18
        else np.zeros((rows.shape[0], 4), dtype=_GAUSSIAN_FLOAT_DTYPE)
    )
    intrinsic = np.array(
        [[[render_size * 0.9, 0.0, (render_size - 1) / 2.0], [0.0, render_size * 0.9, (render_size - 1) / 2.0], [0.0, 0.0, 1.0]]],
        dtype=_GAUSSIAN_FLOAT_DTYPE,
    )
    return {
        "xyz_w": xyz.astype(_GAUSSIAN_FLOAT_DTYPE),
        "scaling": scaling.astype(_GAUSSIAN_FLOAT_DTYPE),
        "quaternion": quaternion.astype(_GAUSSIAN_FLOAT_DTYPE),
        "opacity": opacity.astype(_GAUSSIAN_FLOAT_DTYPE),
        "rgb_sh": color[:, None, :].astype(_GAUSSIAN_FLOAT_DTYPE),
        "lf": lf.astype(_GAUSSIAN_FLOAT_DTYPE),
        "intrinsic": intrinsic,
        "H_c2w": np.eye(4, dtype=_GAUSSIAN_FLOAT_DTYPE)[None, :, :],
        "height_px": np.array([render_size], dtype=np.int64),
        "width_px": np.array([render_size], dtype=np.int64),
        "image_size": np.array([render_size, render_size], dtype=np.int64),
    }


def _export_outputs(gaussians: dict[str, np.ndarray], output: Path | None, output_format: str) -> dict[str, Path]:
    if output is None:
        return {}
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if output_format == "ply":
        _write_gaussians_ply(output, gaussians)
        artifacts["ply"] = output
        sidecar = output.with_suffix(".safetensors")
        _write_gaussians_safetensors(sidecar, gaussians)
        artifacts["safetensors"] = sidecar
    elif output_format == "safetensors":
        _write_gaussians_safetensors(output, gaussians)
        artifacts["safetensors"] = output
    elif output_format == "splat":
        _write_gaussians_splat(output, gaussians)
        artifacts["splat"] = output
        sidecar = output.with_suffix(".safetensors")
        _write_gaussians_safetensors(sidecar, gaussians)
        artifacts["safetensors"] = sidecar
    else:  # pragma: no cover - normalized before dispatch
        raise ValueError(f"unsupported LiTo export format: {output_format}")
    return artifacts


def _export_real_outputs(gaussians: dict[str, np.ndarray], output: Path | None, output_format: str) -> dict[str, Path]:
    if output is None:
        return {}
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    if output_format == "ply":
        write_lito_gaussians_ply(output, gaussians)
        artifacts["ply"] = output
        sidecar = output.with_suffix(".safetensors")
        _write_gaussians_safetensors(sidecar, gaussians)
        artifacts["safetensors"] = sidecar
    elif output_format == "safetensors":
        _write_gaussians_safetensors(output, gaussians)
        artifacts["safetensors"] = output
    elif output_format == "splat":
        raise LitoRealGenerationNotImplemented("checkpoint-backed LiTo splat export is not implemented yet")
    else:  # pragma: no cover - normalized before dispatch
        raise ValueError(f"unsupported LiTo export format: {output_format}")
    return artifacts


def _write_gaussians_ply(path: Path, gaussians: dict[str, np.ndarray]) -> None:
    xyz = np.asarray(gaussians["xyz_w"], dtype=_GAUSSIAN_FLOAT_DTYPE)
    scaling = np.asarray(gaussians["scaling"], dtype=_GAUSSIAN_FLOAT_DTYPE)
    quat = np.asarray(gaussians["quaternion"], dtype=_GAUSSIAN_FLOAT_DTYPE)
    opacity = np.asarray(gaussians["opacity"], dtype=_GAUSSIAN_FLOAT_DTYPE).reshape((-1, 1))
    color = np.asarray(gaussians["rgb_sh"], dtype=_GAUSSIAN_FLOAT_DTYPE)[:, 0, :]
    vertex_count = int(xyz.shape[0])
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "comment mlx-spatial LiTo source-contract smoke 3DGS export",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "property float opacity",
            "property float red",
            "property float green",
            "property float blue",
            "end_header",
        ]
    )
    rows = np.concatenate([xyz, scaling, quat, opacity, color], axis=1)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(header)
        handle.write("\n")
        for row in rows:
            handle.write(" ".join(f"{float(value):.8g}" for value in row))
            handle.write("\n")


def _write_gaussians_safetensors(path: Path, gaussians: dict[str, np.ndarray]) -> None:
    tensors = {
        key: np.asarray(value)
        for key, value in gaussians.items()
        if key in {"xyz_w", "scaling", "quaternion", "opacity", "rgb_sh", "lf", "intrinsic", "H_c2w"}
    }
    save_safetensors(tensors, str(path))


def _write_gaussians_splat(path: Path, gaussians: dict[str, np.ndarray]) -> None:
    xyz = np.asarray(gaussians["xyz_w"], dtype=_GAUSSIAN_FLOAT_DTYPE)
    color = np.asarray(gaussians["rgb_sh"], dtype=_GAUSSIAN_FLOAT_DTYPE)[:, 0, :]
    opacity = np.asarray(gaussians["opacity"], dtype=_GAUSSIAN_FLOAT_DTYPE).reshape((-1, 1))
    payload = np.concatenate([xyz, color, opacity], axis=1).astype("<f4", copy=False)
    with path.open("wb") as handle:
        handle.write(b"LITO_SPLAT_SMOKE\0")
        handle.write(struct.pack("<I", int(payload.shape[0])))
        handle.write(payload.tobytes(order="C"))


def _normalize_output_format(output_format: str, output_path: str | Path | None) -> str:
    fmt = output_format.lower()
    if fmt == "auto":
        if output_path is None:
            return "ply"
        suffix = Path(output_path).suffix.lower().lstrip(".")
        return suffix if suffix in {"ply", "splat", "safetensors"} else "ply"
    if fmt not in {"ply", "splat", "safetensors"}:
        raise ValueError(f"unsupported LiTo output format: {output_format}")
    return fmt


def _coerce_resolution(resolution: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(resolution, int):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        return (resolution, resolution)
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError(f"resolution must be a positive int or pair, got {resolution}")
    return int(resolution[0]), int(resolution[1])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _eval_output(value: Any) -> None:
    arrays: list[mx.array] = []
    _collect_mx_arrays(value, arrays)
    if arrays:
        mx.eval(*arrays)


def _collect_mx_arrays(value: Any, arrays: list[mx.array]) -> None:
    if isinstance(value, mx.array):
        arrays.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_mx_arrays(item, arrays)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_mx_arrays(item, arrays)


def _reset_peak_memory() -> None:
    metal = getattr(mx, "metal", None)
    if metal is None or not hasattr(metal, "reset_peak_memory"):
        return
    try:
        metal.reset_peak_memory()
    except Exception:
        return


def _synchronize() -> None:
    try:
        mx.synchronize()
    except Exception:
        return


def _memory_gb(kind: str) -> float:
    return float(_memory_bytes(kind)) / float(1024**3)


def _memory_bytes(kind: str) -> int:
    metal = getattr(mx, "metal", None)
    if metal is None:
        return 0
    method_name = {
        "active": "get_active_memory",
        "peak": "get_peak_memory",
        "cache": "get_cache_memory",
    }[kind]
    method = getattr(metal, method_name, None)
    if method is None:
        return 0
    try:
        return int(method())
    except Exception:
        return 0


def _infer_raw_weights_root(weights_root: Path) -> Path | None:
    if weights_root.name.endswith("-mlx"):
        candidate = weights_root.with_name(weights_root.name.removesuffix("-mlx") + "-raw")
        if candidate.is_dir():
            return candidate
    candidate = Path("weights/lito-raw")
    return candidate if candidate.is_dir() else None


def metrics_to_json(metrics: dict[str, dict[str, float]]) -> str:
    """Return deterministic JSON for CLI metrics output."""

    return json.dumps(metrics, indent=2, sort_keys=True)


__all__ = [
    "LITO_DEFAULT_MEMORY_PROFILE",
    "LITO_HARD_MEMORY_LIMIT_GB",
    "LITO_MEMORY_PROFILES",
    "LITO_REAL_TENSOR_SENTINELS",
    "LITO_RECOMMENDED_CFG_SCALE",
    "LITO_RECOMMENDED_IMAGE_MEAN",
    "LITO_RECOMMENDED_IMAGE_STD",
    "LITO_RECOMMENDED_MLX_COMPUTE_DTYPE",
    "LITO_RECOMMENDED_NUM_STEPS",
    "LITO_RECOMMENDED_RESOLUTION",
    "LITO_RECOMMENDED_SAMPLER",
    "LITO_RECOMMENDED_SEED_POLICY",
    "LITO_SOFT_MEMORY_LIMIT_GB",
    "LITO_STAGE_NAMES",
    "LitoGenerationResult",
    "LitoInferencePipeline",
    "LitoBackendUnavailable",
    "LitoMemoryLimitExceeded",
    "LitoRealAssetSummary",
    "LitoRealGenerationNotImplemented",
    "memory_profile_config",
    "normalize_lito_init_coord_cap",
]
