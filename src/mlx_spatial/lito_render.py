"""LiTo LF-conditioned Gaussian rendering adapter.

Adapter-only: this module does not modify ``gs_rasterize.py``. LiTo-specific
light-field conditioning is applied as a pre-rasterization layer that maps the
vendor-reference Gaussian schema onto the existing mlx-spatial rasterizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

from mlx_spatial.gs_rasterize import rasterize_gaussians

_RENDER_DTYPE = np.float32  # float32 matches gsplat/gs_rasterize projection and compositing precision.
_MLX_RENDER_DTYPE = mx.float32  # float32 probe keeps MLX eval barriers aligned with rasterizer outputs.


@dataclass(frozen=True)
class LitoRenderResult:
    """Rendered LiTo image and alpha in upstream trainer layout."""

    image: Any
    alpha: Any
    rgba: Any
    depth: Any | None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class LitoGaussianParams:
    """Rasterizer-compatible Gaussian tensors after LF conditioning."""

    xyz_w: np.ndarray
    scaling: np.ndarray
    quaternion: np.ndarray
    opacity: np.ndarray
    rgb_sh: np.ndarray


class LitoRenderer:
    """Adapter for LiTo ``render_gaussians`` onto ``gs_rasterize``.

    ``source_contract_backend="source_contract_local"`` activates the synthetic
    Slice 0B fixture backend. That backend is only for local source-contract
    tests; normal calls use the existing Gaussian rasterizer or its no-Metal
    NumPy fallback.
    """

    def __init__(
        self,
        weights_root: str | None = None,
        *,
        use_metal: bool = True,
        allow_cpu_fallback: bool = True,
        source_contract_backend: str | None = None,
        lf_color_scale: float = 0.10,
        lf_opacity_scale: float = 0.25,
    ) -> None:
        self.weights_root = weights_root
        self.use_metal = bool(use_metal)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.source_contract_backend = source_contract_backend
        self.lf_color_scale = float(lf_color_scale)
        self.lf_opacity_scale = float(lf_opacity_scale)

    @classmethod
    def load(cls, weights_root: str | None = None, **kwargs: Any) -> "LitoRenderer":
        """Construct the stateless render adapter."""

        return cls(weights_root=weights_root, **kwargs)

    def __call__(
        self,
        gaussians: Mapping[str, Any],
        camera: Mapping[str, Any] | None = None,
        lf_condition: Any | None = None,
        *,
        return_result: bool = False,
    ) -> Any:
        result = self.render(gaussians=gaussians, camera=camera, lf_condition=lf_condition)
        return result if return_result else result.image

    def render(
        self,
        gaussians: Mapping[str, Any],
        camera: Mapping[str, Any] | None = None,
        lf_condition: Any | None = None,
    ) -> LitoRenderResult:
        """Render LiTo Gaussian parameters in ``(B, Q, H, W, C)`` layout."""

        _reset_peak_memory()
        t0 = perf_counter()
        peak_before = _safe_peak_memory_gb()
        if self.source_contract_backend == "source_contract_local":
            result = _render_source_contract_fixture(gaussians)
        else:
            result = self._render_adapter(gaussians, camera, lf_condition)
        _eval_render_result(result)
        _sync_mlx()
        wall_time = perf_counter() - t0
        peak_after = _safe_peak_memory_gb()
        metrics = {
            **result.metrics,
            "wall_time_s": wall_time,
            "peak_active_memory_gb": max(peak_before, peak_after),
        }
        return LitoRenderResult(
            image=result.image,
            alpha=result.alpha,
            rgba=result.rgba,
            depth=result.depth,
            metrics=metrics,
            metadata=result.metadata,
        )

    def prepare_gaussians(
        self,
        gaussians: Mapping[str, Any],
        lf_condition: Any | None = None,
    ) -> LitoGaussianParams:
        """Apply the LiTo LF layer and return rasterizer-compatible tensors."""

        xyz = _as_render_np(gaussians["xyz_w"], "xyz_w")
        scaling = _as_render_np(gaussians["scaling"], "scaling")
        quaternion = _normalize_quaternion(_as_render_np(gaussians["quaternion"], "quaternion"))
        opacity = _as_render_np(gaussians["opacity"], "opacity").reshape((-1, 1))
        rgb_sh = _as_render_np(gaussians["rgb_sh"], "rgb_sh")

        xyz = _flatten_gaussian_tensor(xyz, "xyz_w", trailing=3)
        scaling = _flatten_gaussian_tensor(scaling, "scaling", trailing=3)
        quaternion = _flatten_gaussian_tensor(quaternion, "quaternion", trailing=4)
        opacity = _flatten_gaussian_tensor(opacity, "opacity", trailing=1)
        if rgb_sh.ndim == 2:
            rgb_sh = rgb_sh[:, None, :]
        elif rgb_sh.ndim > 3:
            rgb_sh = rgb_sh.reshape((-1, *rgb_sh.shape[-2:]))
        if rgb_sh.ndim != 3 or rgb_sh.shape[-1] != 3:
            raise ValueError(f"rgb_sh must have shape (N, K, 3), got {rgb_sh.shape}")

        lf = lf_condition if lf_condition is not None else gaussians.get("lf")
        if lf is not None:
            rgb_sh, opacity = self._apply_lf_conditioning(rgb_sh, opacity, lf)

        count = xyz.shape[0]
        for name, value, trailing in (
            ("scaling", scaling, 3),
            ("quaternion", quaternion, 4),
            ("opacity", opacity, 1),
            ("rgb_sh", rgb_sh, 3),
        ):
            if value.shape[0] != count:
                raise ValueError(f"{name} first dimension must match xyz_w ({count}), got {value.shape}")
            if value.shape[-1] != trailing:
                raise ValueError(f"{name} trailing dimension must be {trailing}, got {value.shape}")

        return LitoGaussianParams(
            xyz_w=xyz,
            scaling=np.maximum(scaling, 1e-6).astype(_RENDER_DTYPE, copy=False),
            quaternion=quaternion,
            opacity=np.clip(opacity, 0.0, 1.0).astype(_RENDER_DTYPE, copy=False),
            rgb_sh=rgb_sh.astype(_RENDER_DTYPE, copy=False),
        )

    def _apply_lf_conditioning(
        self,
        rgb_sh: np.ndarray,
        opacity: np.ndarray,
        lf_condition: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        lf = _as_render_np(lf_condition, "lf_condition")
        lf = _flatten_gaussian_tensor(lf, "lf_condition", trailing=lf.shape[-1])
        if lf.shape[0] != rgb_sh.shape[0]:
            raise ValueError(
                f"lf_condition first dimension must match rgb_sh, got {lf.shape} and {rgb_sh.shape}"
            )
        if lf.shape[1] < 1:
            raise ValueError("lf_condition must have at least one channel")

        if lf.shape[1] >= 3:
            rgb_gain = 1.0 + self.lf_color_scale * np.tanh(lf[:, :3])
            conditioned_rgb = np.clip(rgb_sh * rgb_gain[:, None, :], 0.0, 1.0)
        else:
            conditioned_rgb = rgb_sh

        alpha_channel = lf[:, 3:4] if lf.shape[1] >= 4 else lf[:, :1]
        opacity_gain = 1.0 + self.lf_opacity_scale * np.tanh(alpha_channel)
        conditioned_opacity = np.clip(opacity * opacity_gain, 0.0, 1.0)
        return conditioned_rgb.astype(_RENDER_DTYPE), conditioned_opacity.astype(_RENDER_DTYPE)

    def _render_adapter(
        self,
        gaussians: Mapping[str, Any],
        camera: Mapping[str, Any] | None,
        lf_condition: Any | None,
    ) -> LitoRenderResult:
        params = self.prepare_gaussians(gaussians, lf_condition=lf_condition)
        camera_params, image_size = _coerce_camera(gaussians, camera)
        if _mlx_runtime_available():
            rendered = rasterize_gaussians(
                params.xyz_w,
                params.quaternion,
                params.scaling,
                params.opacity,
                params.rgb_sh,
                camera_params,
                image_size,
                use_metal=self.use_metal,
                allow_cpu_fallback=self.allow_cpu_fallback,
                sh_degree=_sh_degree_from_coeffs(params.rgb_sh.shape[-2]),
            )
            rgba_hw = rendered.rgba
            depth_hw = rendered.depth
            image = rgba_hw[None, None, :, :, :3]
            alpha = rgba_hw[None, None, :, :, 3:4]
            rgba = rgba_hw[None, None, :, :, :]
            metadata = {
                **rendered.metadata,
                "adapter": "lito_render",
                "source_contract_backend": None,
                "lf_conditioned": lf_condition is not None or "lf" in gaussians,
            }
            return LitoRenderResult(image=image, alpha=alpha, rgba=rgba, depth=depth_hw, metadata=metadata)

        if not self.allow_cpu_fallback:
            raise RuntimeError("MLX Metal device is unavailable and allow_cpu_fallback=False")
        return _rasterize_numpy_fallback(
            params,
            camera_params,
            image_size,
            metadata={
                "adapter": "lito_render",
                "backend": "cpu-no-metal",
                "source_contract_backend": None,
                "lf_conditioned": lf_condition is not None or "lf" in gaussians,
            },
        )


def _render_source_contract_fixture(gaussians: Mapping[str, Any]) -> LitoRenderResult:
    height, width = _image_size_from_gaussians(gaussians)
    opacity = _as_render_np(gaussians["opacity"], "opacity")
    yy, xx = np.meshgrid(
        np.arange(height, dtype=_RENDER_DTYPE),
        np.arange(width, dtype=_RENDER_DTYPE),
        indexing="ij",
    )
    image = np.stack(
        [
            xx / _RENDER_DTYPE(width - 1),
            yy / _RENDER_DTYPE(height - 1),
            np.full((height, width), opacity.mean(dtype=_RENDER_DTYPE), dtype=_RENDER_DTYPE),
        ],
        axis=-1,
    ).astype(_RENDER_DTYPE)
    alpha = np.clip((image[..., :1] * 0.35 + image[..., 1:2] * 0.35 + 0.3), 0.0, 1.0).astype(_RENDER_DTYPE)
    rgba = np.concatenate([image, alpha], axis=-1)
    return LitoRenderResult(
        image=image[None, None, :, :, :],
        alpha=alpha[None, None, :, :, :],
        rgba=rgba[None, None, :, :, :],
        depth=None,
        metadata={
            "adapter": "lito_render",
            "backend": "source_contract_local",
            "source_contract_backend": "source_contract_local",
            "fixture_role": "gaussian_camera_image_contract",
            "lf_conditioned": "lf" in gaussians,
        },
    )


def _rasterize_numpy_fallback(
    params: LitoGaussianParams,
    camera: dict[str, np.ndarray],
    image_size: tuple[int, int],
    *,
    metadata: dict[str, object],
) -> LitoRenderResult:
    height, width = image_size
    view = camera["view_matrix"]
    intr = camera["intrinsics"]
    means_h = np.concatenate([params.xyz_w, np.ones((params.xyz_w.shape[0], 1), dtype=_RENDER_DTYPE)], axis=1)
    camera_xyz = (view @ means_h.T).T[:, :3]
    z = camera_xyz[:, 2]
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    px = fx * (camera_xyz[:, 0] / np.maximum(z, 1e-6)) + cx
    py = fy * (camera_xyz[:, 1] / np.maximum(z, 1e-6)) + cy
    radius = np.maximum(
        np.max(params.scaling, axis=-1) * max(fx, fy) / np.maximum(z, 1e-6) * 3.0,
        0.5,
    )
    visible = (
        (z > 1e-6)
        & (params.opacity[:, 0] > 0.0)
        & (px + radius >= 0.0)
        & (px - radius < width)
        & (py + radius >= 0.0)
        & (py - radius < height)
    )
    order = np.argsort(z[visible], kind="stable")
    indices = np.nonzero(visible)[0][order]

    rgba = np.zeros((height, width, 4), dtype=_RENDER_DTYPE)
    depth_weighted = np.zeros((height, width), dtype=_RENDER_DTYPE)
    transmittance = np.ones((height, width), dtype=_RENDER_DTYPE)
    yy, xx = np.mgrid[0:height, 0:width].astype(_RENDER_DTYPE)
    sample_x = xx + 0.5
    sample_y = yy + 0.5
    colors = _rgb_from_sh(params.rgb_sh)
    for index in indices:
        sigma = max(float(radius[index]) / 3.0, 1e-6)
        dx = sample_x - float(px[index])
        dy = sample_y - float(py[index])
        dist2 = dx * dx + dy * dy
        mask = dist2 <= float(radius[index] * radius[index])
        alpha = np.zeros((height, width), dtype=_RENDER_DTYPE)
        alpha[mask] = float(params.opacity[index, 0]) * np.exp(-0.5 * dist2[mask] / (sigma * sigma))
        alpha = np.where(alpha > (1.0 / 255.0), np.minimum(alpha, 0.999), 0.0)
        contribution = transmittance * alpha
        rgba[..., :3] += contribution[..., None] * colors[index]
        rgba[..., 3] += contribution
        depth_weighted += contribution * float(z[index])
        transmittance *= 1.0 - alpha

    alpha_mask = rgba[..., 3] > 1e-6
    rgba[..., :3] = np.where(
        alpha_mask[..., None],
        rgba[..., :3] / np.maximum(rgba[..., 3:4], 1e-6),
        0.0,
    )
    depth = np.where(alpha_mask, depth_weighted / np.maximum(rgba[..., 3], 1e-6), 0.0).astype(_RENDER_DTYPE)
    return LitoRenderResult(
        image=rgba[None, None, :, :, :3],
        alpha=rgba[None, None, :, :, 3:4],
        rgba=rgba[None, None, :, :, :],
        depth=depth,
        metadata={
            **metadata,
            "gaussian_count": int(params.xyz_w.shape[0]),
            "visible_gaussian_count": int(indices.shape[0]),
            "image_size": (int(height), int(width)),
        },
    )


def _coerce_camera(
    gaussians: Mapping[str, Any],
    camera: Mapping[str, Any] | None,
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    source = camera or gaussians
    if "intrinsics" in source:
        intrinsic = source["intrinsics"]
    elif "intrinsic" in source:
        intrinsic = source["intrinsic"]
    else:
        raise ValueError("camera must include intrinsic/intrinsics")

    if "view_matrix" in source:
        view_matrix = _as_render_np(source["view_matrix"], "view_matrix")
    elif "H_w2c" in source:
        view_matrix = _as_render_np(source["H_w2c"], "H_w2c")
    elif "H_c2w" in source:
        h_c2w = _as_render_np(source["H_c2w"], "H_c2w")
        h_c2w = h_c2w.reshape((-1, 4, 4))[0]
        view_matrix = np.linalg.inv(h_c2w).astype(_RENDER_DTYPE)
    elif "camera_to_world" in source:
        h_c2w = _as_render_np(source["camera_to_world"], "camera_to_world")
        h_c2w = h_c2w.reshape((-1, 4, 4))[0]
        view_matrix = np.linalg.inv(h_c2w).astype(_RENDER_DTYPE)
    else:
        raise ValueError("camera must include view_matrix, H_w2c, H_c2w, or camera_to_world")

    intrinsic = _as_render_np(intrinsic, "intrinsic").reshape((-1, 3, 3))[0].astype(_RENDER_DTYPE)
    view_matrix = view_matrix.reshape((-1, 4, 4))[0].astype(_RENDER_DTYPE)
    return {"intrinsics": intrinsic, "view_matrix": view_matrix}, _image_size_from_gaussians(gaussians, camera)


def _image_size_from_gaussians(
    gaussians: Mapping[str, Any],
    camera: Mapping[str, Any] | None = None,
) -> tuple[int, int]:
    source = camera or gaussians
    if "image_size" in source:
        value = np.asarray(source["image_size"]).reshape(-1)
        return int(value[0]), int(value[1])
    if "height_px" in source and "width_px" in source:
        return (
            int(np.asarray(source["height_px"]).reshape(-1)[0]),
            int(np.asarray(source["width_px"]).reshape(-1)[0]),
        )
    raise ValueError("image size must be provided as image_size or height_px/width_px")


def _rgb_from_sh(rgb_sh: np.ndarray) -> np.ndarray:
    if rgb_sh.ndim == 2:
        return np.clip(rgb_sh, 0.0, 1.0)
    if rgb_sh.shape[-2] == 1:
        return np.clip(rgb_sh[:, 0, :], 0.0, 1.0)
    return np.clip(rgb_sh[:, 0, :] * 0.28209479177387814 + 0.5, 0.0, 1.0).astype(_RENDER_DTYPE)


def _sh_degree_from_coeffs(coeffs: int) -> int:
    degree = int(round(np.sqrt(coeffs) - 1))
    return degree if (degree + 1) ** 2 == coeffs else 0


def _flatten_gaussian_tensor(values: np.ndarray, name: str, *, trailing: int) -> np.ndarray:
    if values.shape[-1] != trailing:
        raise ValueError(f"{name} trailing dimension must be {trailing}, got {values.shape}")
    return values.reshape((-1, trailing)).astype(_RENDER_DTYPE, copy=False)


def _normalize_quaternion(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    fallback = np.zeros_like(values, dtype=_RENDER_DTYPE)
    fallback[..., 3] = 1.0
    return np.where(norm > 1e-8, values / np.maximum(norm, 1e-8), fallback).astype(_RENDER_DTYPE)


def _as_render_np(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=_RENDER_DTYPE)
    except Exception:
        array = np.asarray(np.array(values), dtype=_RENDER_DTYPE)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _mlx_runtime_available() -> bool:
    try:
        probe = mx.array([0.0], dtype=_MLX_RENDER_DTYPE)
        mx.eval(probe)
        return True
    except RuntimeError as exc:
        if "No Metal device available" in str(exc) or "metal::load_device" in str(exc):
            return False
        raise


def _safe_peak_memory_gb() -> float:
    metal = getattr(mx, "metal", None)
    if metal is None or not hasattr(metal, "get_peak_memory"):
        return 0.0
    try:
        return float(metal.get_peak_memory()) / float(1024**3)
    except Exception:
        return 0.0


def _reset_peak_memory() -> None:
    metal = getattr(mx, "metal", None)
    if metal is None or not hasattr(metal, "reset_peak_memory"):
        return
    try:
        metal.reset_peak_memory()
    except Exception:
        return


def _sync_mlx() -> None:
    try:
        mx.synchronize()
    except Exception:
        return


def _eval_render_result(result: LitoRenderResult) -> None:
    arrays = [value for value in (result.image, result.alpha, result.rgba, result.depth) if isinstance(value, mx.array)]
    if not arrays:
        return
    try:
        mx.eval(*arrays)
    except Exception:
        return
