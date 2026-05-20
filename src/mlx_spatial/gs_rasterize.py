"""Mac-native Gaussian splat projection and rasterization.

This module provides the Slice 8 correctness-first renderer. Projection,
visibility filtering, and depth sorting are done deterministically on the CPU;
the Metal path renders one pixel per thread and loops over the projected,
front-to-back splats. That shape deliberately avoids float atomics, which are
not portable enough for the M1 baseline, and is intended for small/test-size
texture baking before a tiled/binning renderer is added.

The renderer is inference-only and has no runtime PyTorch, gsplat,
diff-gaussian-rasterization, or CUDA dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import numpy as np

_SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class GaussianRasterizeResult:
    """Result of Gaussian splat rasterization.

    Attributes:
        rgba: ``(H, W, 4)`` float32 MLX array with straight RGB and accumulated
            alpha in ``[0, 1]``. Transparent pixels have zero RGB.
        depth: ``(H, W)`` float32 MLX array containing alpha-weighted camera
            depth for contributing pixels, or ``0`` where alpha is zero.
        pixel_count: Number of pixels with alpha greater than zero.
        metadata: Deterministic render metadata, including backend and visible
            Gaussian count.
    """

    rgba: mx.array
    depth: mx.array
    pixel_count: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GaussianCameraParams:
    """Pinhole camera parameters for ``rasterize_gaussians``.

    ``view_matrix`` is world-to-camera. As a convenience, callers may pass
    ``camera_to_world`` instead through the mapping API; it will be inverted.
    """

    intrinsics: np.ndarray
    view_matrix: np.ndarray


@dataclass(frozen=True)
class _ProjectedGaussians:
    centers_xy: np.ndarray
    depths: np.ndarray
    conics: np.ndarray
    radii: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    mahalanobis_clip_sq: float
    metadata: dict[str, object]


class GaussianSplatRenderer:
    """Small deterministic Gaussian splat renderer with an optional Metal path.

    The Metal kernel is cached per renderer instance. If ``allow_cpu_fallback``
    is true, runtime Metal failures fall back to the NumPy reference path and
    mark ``metadata["backend"]`` as ``"cpu"`` with a ``metal_error`` note.
    """

    def __init__(
        self,
        *,
        use_metal: bool = True,
        allow_cpu_fallback: bool = True,
        threadgroup_size: int = 256,
    ) -> None:
        self.use_metal = bool(use_metal)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.threadgroup_size = int(threadgroup_size)
        self._kernel: Any | None = None

    def rasterize_projected(
        self,
        projected: _ProjectedGaussians,
        image_size: tuple[int, int],
        *,
        min_alpha: float = 1.0 / 255.0,
    ) -> GaussianRasterizeResult:
        """Render already projected/sorted Gaussians."""

        if self.use_metal:
            try:
                return self._rasterize_projected_metal(projected, image_size, min_alpha=min_alpha)
            except Exception as exc:
                if not self.allow_cpu_fallback:
                    raise
                result = _rasterize_projected_cpu(projected, image_size, min_alpha=min_alpha)
                metadata = {
                    **result.metadata,
                    "backend": "cpu",
                    "metal_error": f"{type(exc).__name__}: {exc}",
                    "metal_fallback": True,
                }
                return GaussianRasterizeResult(
                    rgba=result.rgba,
                    depth=result.depth,
                    pixel_count=result.pixel_count,
                    metadata=metadata,
                )

        return _rasterize_projected_cpu(projected, image_size, min_alpha=min_alpha)

    def __call__(
        self,
        means_3d: Any,
        quaternions: Any,
        scales: Any,
        opacities: Any,
        sh_features: Any,
        camera_params: Mapping[str, Any] | GaussianCameraParams,
        image_size: tuple[int, int],
        **kwargs: Any,
    ) -> GaussianRasterizeResult:
        return rasterize_gaussians(
            means_3d,
            quaternions,
            scales,
            opacities,
            sh_features,
            camera_params,
            image_size,
            renderer=self,
            **kwargs,
        )

    def _rasterize_projected_metal(
        self,
        projected: _ProjectedGaussians,
        image_size: tuple[int, int],
        *,
        min_alpha: float,
    ) -> GaussianRasterizeResult:
        height, width = _validate_image_size(image_size)
        kernel = self._metal_kernel()
        gaussian_count = int(projected.centers_xy.shape[0])

        inputs = [
            mx.array(projected.centers_xy, dtype=mx.float32),
            mx.array(projected.depths, dtype=mx.float32),
            mx.array(projected.conics, dtype=mx.float32),
            mx.array(projected.radii, dtype=mx.float32),
            mx.array(projected.colors, dtype=mx.float32),
            mx.array(projected.opacities, dtype=mx.float32),
            gaussian_count,
            int(width),
            int(height),
            float(min_alpha),
            float(projected.mahalanobis_clip_sq),
        ]
        rgba, depth = kernel(
            inputs=inputs,
            grid=(height * width, 1, 1),
            threadgroup=(self.threadgroup_size, 1, 1),
            output_shapes=[(height, width, 4), (height, width)],
            output_dtypes=[mx.float32, mx.float32],
        )
        mx.eval(rgba, depth)
        rgba_np = np.asarray(rgba)
        pixel_count = int(np.count_nonzero(rgba_np[..., 3] > 0.0))
        metadata = {
            **projected.metadata,
            "backend": "metal",
            "pixel_count": pixel_count,
            "m1_limitations": (
                "per-pixel loop avoids float atomics; intended for correctness/test-size renders "
                "before tiled binning optimization"
            ),
        }
        return GaussianRasterizeResult(rgba=rgba, depth=depth, pixel_count=pixel_count, metadata=metadata)

    def _metal_kernel(self) -> Any:
        if self._kernel is None:
            source = _load_metal_source()
            self._kernel = mx.fast.metal_kernel(
                name="mlx_spatial_gs_rasterize",
                input_names=[
                    "centers_xy",
                    "depths",
                    "conics",
                    "radii",
                    "colors",
                    "opacities",
                    "gaussian_count",
                    "width",
                    "height",
                    "min_alpha",
                    "mahalanobis_clip_sq",
                ],
                output_names=["rgba", "depth"],
                source=source,
                ensure_row_contiguous=True,
                atomic_outputs=False,
            )
        return self._kernel


def rasterize_gaussians(
    means_3d: Any,
    quaternions: Any,
    scales: Any,
    opacities: Any,
    sh_features: Any,
    camera_params: Mapping[str, Any] | GaussianCameraParams,
    image_size: tuple[int, int],
    *,
    renderer: GaussianSplatRenderer | None = None,
    use_metal: bool = True,
    allow_cpu_fallback: bool = True,
    sh_degree: int = 0,
    sh_features_are_rgb: bool | None = None,
    scale_multiplier: float = 1.0,
    radius_clip: float = 3.0,
    min_radius_px: float = 0.5,
    min_alpha: float = 1.0 / 255.0,
) -> GaussianRasterizeResult:
    """Project and rasterize 3D Gaussian splats.

    Args:
        means_3d: ``(N, 3)`` Gaussian centers in world coordinates.
        quaternions: ``(N, 4)`` Gaussian rotations in XYZW scalar-last
            convention. They are applied to ``diag(scales ** 2)`` before the
            3D covariance is projected into screen space.
        scales: ``(N, 3)`` world-space Gaussian scales. Values are interpreted
            directly, not exponentiated.
        opacities: ``(N,)`` or ``(N, 1)`` alpha values in ``[0, 1]``.
        sh_features: ``(N, 3)`` direct RGB colors, ``(N, 1, 3)`` degree-0 SH
            coefficients by default. Higher-degree SH should be pre-evaluated
            to RGB and passed with ``sh_features_are_rgb=True`` in this
            correctness-first renderer.
        camera_params: Mapping or ``GaussianCameraParams``. Mappings must
            include ``intrinsics`` and either ``view_matrix`` (world-to-camera)
            or ``camera_to_world``.
        image_size: ``(height, width)``.
        renderer: Optional renderer instance to reuse the cached Metal kernel.
        use_metal: Use the Metal kernel by default.
        allow_cpu_fallback: Fall back to the CPU reference if Metal is
            unavailable or compilation fails.

    Returns:
        ``GaussianRasterizeResult`` with MLX arrays and render metadata.
    """

    height, width = _validate_image_size(image_size)
    camera = _coerce_camera_params(camera_params)
    means_np = _as_float32_np(means_3d, "means_3d", ndim=2)
    if means_np.shape[1] != 3:
        raise ValueError(f"means_3d must have shape (N, 3), got {means_np.shape}")
    gaussian_count = int(means_np.shape[0])

    quats_np = _as_float32_np(quaternions, "quaternions", ndim=2)
    if quats_np.shape != (gaussian_count, 4):
        raise ValueError(f"quaternions must have shape ({gaussian_count}, 4), got {quats_np.shape}")
    scales_np = _as_float32_np(scales, "scales", ndim=2)
    if scales_np.shape != (gaussian_count, 3):
        raise ValueError(f"scales must have shape ({gaussian_count}, 3), got {scales_np.shape}")
    opacities_np = _as_float32_np(opacities, "opacities").reshape(-1)
    if opacities_np.shape != (gaussian_count,):
        raise ValueError(f"opacities must flatten to shape ({gaussian_count},), got {opacities_np.shape}")

    projected = _project_gaussians(
        means_np,
        quats_np,
        scales_np,
        np.clip(opacities_np, 0.0, 1.0),
        _colors_from_sh_features(
            sh_features,
            camera.view_matrix,
            means_np,
            sh_degree=sh_degree,
            sh_features_are_rgb=sh_features_are_rgb,
        ),
        camera,
        (height, width),
        scale_multiplier=scale_multiplier,
        radius_clip=radius_clip,
        min_radius_px=min_radius_px,
    )
    render = renderer or GaussianSplatRenderer(use_metal=use_metal, allow_cpu_fallback=allow_cpu_fallback)
    return render.rasterize_projected(projected, (height, width), min_alpha=min_alpha)


def rasterize_gaussians_cpu_reference(
    means_3d: Any,
    quaternions: Any,
    scales: Any,
    opacities: Any,
    sh_features: Any,
    camera_params: Mapping[str, Any] | GaussianCameraParams,
    image_size: tuple[int, int],
    **kwargs: Any,
) -> GaussianRasterizeResult:
    """CPU/NumPy reference path used for deterministic parity tests."""

    renderer = GaussianSplatRenderer(use_metal=False)
    return rasterize_gaussians(
        means_3d,
        quaternions,
        scales,
        opacities,
        sh_features,
        camera_params,
        image_size,
        renderer=renderer,
        **kwargs,
    )


def _project_gaussians(
    means: np.ndarray,
    quaternions: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    camera: GaussianCameraParams,
    image_size: tuple[int, int],
    *,
    scale_multiplier: float,
    radius_clip: float,
    min_radius_px: float,
) -> _ProjectedGaussians:
    height, width = image_size
    means_h = np.concatenate([means, np.ones((means.shape[0], 1), dtype=np.float32)], axis=1)
    camera_xyz = (camera.view_matrix @ means_h.T).T[:, :3]
    z = camera_xyz[:, 2]
    positive_depth = z > 1e-6

    fx = float(camera.intrinsics[0, 0])
    fy = float(camera.intrinsics[1, 1])
    cx = float(camera.intrinsics[0, 2])
    cy = float(camera.intrinsics[1, 2])
    x = fx * (camera_xyz[:, 0] / np.maximum(z, 1e-6)) + cx
    y = fy * (camera_xyz[:, 1] / np.maximum(z, 1e-6)) + cy
    centers = np.stack([x, y], axis=-1).astype(np.float32)

    conics, radii = _project_covariances_to_conics(
        camera_xyz,
        quaternions,
        scales * float(scale_multiplier),
        camera.view_matrix[:3, :3],
        fx=fx,
        fy=fy,
        min_radius_px=float(min_radius_px),
        radius_clip=float(radius_clip),
    )
    visible = (
        positive_depth
        & (opacities > 0.0)
        & (radii > 0.0)
        & (centers[:, 0] + radii >= 0.0)
        & (centers[:, 0] - radii < float(width))
        & (centers[:, 1] + radii >= 0.0)
        & (centers[:, 1] - radii < float(height))
    )
    order = np.argsort(z[visible], kind="stable")
    visible_idx = np.nonzero(visible)[0][order]

    metadata: dict[str, object] = {
        "gaussian_count": int(means.shape[0]),
        "visible_gaussian_count": int(visible_idx.shape[0]),
        "image_size": (int(height), int(width)),
        "depth_order": "front-to-back",
        "projection": "pinhole",
        "footprint_model": "anisotropic projected 3D covariance",
        "quaternion_convention": "XYZW scalar-last",
        "quaternion_rotation_applied": True,
        "radius_clip": float(radius_clip),
    }
    return _ProjectedGaussians(
        centers_xy=np.ascontiguousarray(centers[visible_idx], dtype=np.float32),
        depths=np.ascontiguousarray(z[visible_idx], dtype=np.float32),
        conics=np.ascontiguousarray(conics[visible_idx], dtype=np.float32),
        radii=np.ascontiguousarray(radii[visible_idx], dtype=np.float32),
        colors=np.ascontiguousarray(np.clip(colors[visible_idx], 0.0, 1.0), dtype=np.float32),
        opacities=np.ascontiguousarray(opacities[visible_idx], dtype=np.float32),
        mahalanobis_clip_sq=float(radius_clip * radius_clip),
        metadata=metadata,
    )


def _rasterize_projected_cpu(
    projected: _ProjectedGaussians,
    image_size: tuple[int, int],
    *,
    min_alpha: float,
) -> GaussianRasterizeResult:
    height, width = _validate_image_size(image_size)
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    depth_weighted = np.zeros((height, width), dtype=np.float32)
    transmittance = np.ones((height, width), dtype=np.float32)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    px = xx + 0.5
    py = yy + 0.5
    for center, depth, conic, radius, color, opacity in zip(
        projected.centers_xy,
        projected.depths,
        projected.conics,
        projected.radii,
        projected.colors,
        projected.opacities,
        strict=True,
    ):
        if radius <= 0.0:
            continue
        dx = px - center[0]
        dy = py - center[1]
        dist2 = dx * dx + dy * dy
        in_bounds = dist2 <= radius * radius
        mahalanobis = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
        mask = in_bounds & (mahalanobis <= projected.mahalanobis_clip_sq)
        alpha = np.zeros((height, width), dtype=np.float32)
        alpha[mask] = float(opacity) * np.exp(-0.5 * mahalanobis[mask])
        alpha = np.where(alpha > float(min_alpha), np.minimum(alpha, 0.999), 0.0)
        contribution = transmittance * alpha
        rgba[..., :3] += contribution[..., None] * color
        rgba[..., 3] += contribution
        depth_weighted += contribution * float(depth)
        transmittance *= 1.0 - alpha

    alpha_mask = rgba[..., 3] > 1e-6
    rgba[..., :3] = np.where(alpha_mask[..., None], rgba[..., :3] / np.maximum(rgba[..., 3:4], 1e-6), 0.0)
    depth = np.where(alpha_mask, depth_weighted / np.maximum(rgba[..., 3], 1e-6), 0.0)
    pixel_count = int(np.count_nonzero(rgba[..., 3] > 0.0))
    metadata = {
        **projected.metadata,
        "backend": "cpu",
        "pixel_count": pixel_count,
        "m1_limitations": (
            "CPU reference mirrors the M1-safe per-pixel Metal algorithm without float atomics"
        ),
    }
    return GaussianRasterizeResult(
        rgba=mx.array(rgba, dtype=mx.float32),
        depth=mx.array(depth.astype(np.float32), dtype=mx.float32),
        pixel_count=pixel_count,
        metadata=metadata,
    )


def _project_covariances_to_conics(
    camera_xyz: np.ndarray,
    quaternions: np.ndarray,
    scales: np.ndarray,
    view_rotation: np.ndarray,
    *,
    fx: float,
    fy: float,
    min_radius_px: float,
    radius_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotations = _quat_xyzw_to_rotmat_np(quaternions)
    scale_sq = np.maximum(np.abs(scales), 1e-6) ** 2
    conics = np.zeros((camera_xyz.shape[0], 3), dtype=np.float32)
    radii = np.zeros((camera_xyz.shape[0],), dtype=np.float32)
    min_var = float(min_radius_px) ** 2

    for index, (center, rotation, diag_scale_sq) in enumerate(zip(camera_xyz, rotations, scale_sq, strict=True)):
        z = float(center[2])
        if z <= 1e-6:
            continue
        scaled_rotation = rotation * np.sqrt(diag_scale_sq.astype(np.float64))[None, :]
        world_cov = scaled_rotation @ scaled_rotation.T
        camera_cov = view_rotation.astype(np.float64) @ world_cov @ view_rotation.astype(np.float64).T
        jacobian = np.array(
            [
                [float(fx) / z, 0.0, -float(fx) * float(center[0]) / (z * z)],
                [0.0, float(fy) / z, -float(fy) * float(center[1]) / (z * z)],
            ],
            dtype=np.float64,
        )
        cov2d = jacobian @ camera_cov @ jacobian.T
        cov2d = 0.5 * (cov2d + cov2d.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov2d)
        eigenvalues = np.maximum(eigenvalues, min_var)
        cov2d = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        inv_cov = np.linalg.inv(cov2d)
        conics[index] = np.array([inv_cov[0, 0], inv_cov[0, 1], inv_cov[1, 1]], dtype=np.float32)
        radii[index] = np.float32(radius_clip * np.sqrt(float(np.max(eigenvalues))))

    return conics, radii


def _quat_xyzw_to_rotmat_np(quaternions: np.ndarray) -> np.ndarray:
    q = quaternions.astype(np.float64, copy=False)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    qx = q[:, 0]
    qy = q[:, 1]
    qz = q[:, 2]
    qw = q[:, 3]

    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1.0 - 2.0 * (qy * qy + qz * qz)
    rot[:, 0, 1] = 2.0 * (qx * qy - qz * qw)
    rot[:, 0, 2] = 2.0 * (qx * qz + qy * qw)
    rot[:, 1, 0] = 2.0 * (qx * qy + qz * qw)
    rot[:, 1, 1] = 1.0 - 2.0 * (qx * qx + qz * qz)
    rot[:, 1, 2] = 2.0 * (qy * qz - qx * qw)
    rot[:, 2, 0] = 2.0 * (qx * qz - qy * qw)
    rot[:, 2, 1] = 2.0 * (qy * qz + qx * qw)
    rot[:, 2, 2] = 1.0 - 2.0 * (qx * qx + qy * qy)
    return rot


def _colors_from_sh_features(
    sh_features: Any,
    view_matrix: np.ndarray,
    means: np.ndarray,
    *,
    sh_degree: int,
    sh_features_are_rgb: bool | None,
) -> np.ndarray:
    values = _as_float32_np(sh_features, "sh_features")
    if values.shape[0] != means.shape[0]:
        raise ValueError(f"sh_features first dimension must match means_3d, got {values.shape} and {means.shape}")
    if values.ndim == 2:
        if values.shape[1] != 3:
            raise ValueError(f"2D sh_features must have shape (N, 3), got {values.shape}")
        return values.astype(np.float32, copy=False)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"sh_features must have shape (N, 3) or (N, K, 3), got {values.shape}")

    if sh_features_are_rgb is True:
        return values[:, 0, :].astype(np.float32, copy=False)
    if sh_features_are_rgb is None and sh_degree == 0 and values.shape[1] == 1:
        return (values[:, 0, :] * _SH_C0 + 0.5).astype(np.float32, copy=False)
    if sh_features_are_rgb is False and sh_degree == 0:
        return (values[:, 0, :] * _SH_C0 + 0.5).astype(np.float32, copy=False)
    if sh_degree > 0:
        raise NotImplementedError(
            "higher-degree SH rasterization is pending; pre-evaluate colors and pass sh_features_are_rgb=True"
        )
    raise ValueError(f"degree-0 sh_features must have shape (N, 1, 3), got {values.shape}")


def _coerce_camera_params(camera_params: Mapping[str, Any] | GaussianCameraParams) -> GaussianCameraParams:
    if isinstance(camera_params, GaussianCameraParams):
        intrinsics = _as_float32_np(camera_params.intrinsics, "intrinsics", ndim=2)
        view_matrix = _as_float32_np(camera_params.view_matrix, "view_matrix", ndim=2)
    else:
        if "intrinsics" not in camera_params:
            raise ValueError("camera_params must include intrinsics")
        intrinsics = _as_float32_np(camera_params["intrinsics"], "intrinsics", ndim=2)
        if "view_matrix" in camera_params:
            view_matrix = _as_float32_np(camera_params["view_matrix"], "view_matrix", ndim=2)
        elif "camera_to_world" in camera_params:
            camera_to_world = _as_float32_np(camera_params["camera_to_world"], "camera_to_world", ndim=2)
            if camera_to_world.shape != (4, 4):
                raise ValueError(f"camera_to_world must have shape (4, 4), got {camera_to_world.shape}")
            view_matrix = np.linalg.inv(camera_to_world).astype(np.float32)
        else:
            raise ValueError("camera_params must include view_matrix or camera_to_world")

    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
    if view_matrix.shape != (4, 4):
        raise ValueError(f"view_matrix must have shape (4, 4), got {view_matrix.shape}")
    return GaussianCameraParams(intrinsics=intrinsics, view_matrix=view_matrix)


def _as_float32_np(values: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    return array


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError(f"image_size must be (height, width), got {image_size}")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size dimensions must be positive, got {image_size}")
    return height, width


def _load_metal_source() -> str:
    return (Path(__file__).with_name("metal") / "gs_rasterize.metal").read_text(encoding="utf-8")
