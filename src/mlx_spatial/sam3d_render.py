"""SAM 3D multi-view Gaussian rendering and layout alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, tan
from typing import Any, Sequence

import numpy as np

from .gs_rasterize import GaussianRasterizeResult, rasterize_gaussians


@dataclass(frozen=True)
class Sam3dRenderCamera:
    """Matrix-first camera for SAM3D Gaussian rendering."""

    view_matrix: np.ndarray
    intrinsics: np.ndarray
    name: str

    def as_raster_camera(self) -> dict[str, np.ndarray]:
        return {"view_matrix": self.view_matrix, "intrinsics": self.intrinsics}


@dataclass(frozen=True)
class Sam3dMultiviewRenderResult:
    """Stacked multi-view render output."""

    rgba: np.ndarray
    depth: np.ndarray
    cameras: tuple[Sam3dRenderCamera, ...]
    pixel_counts: tuple[int, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class Sam3dLayoutOptimizationResult:
    """Rigid layout alignment result for render-and-compare post-optimization."""

    transform: np.ndarray
    aligned_points: np.ndarray
    initial_rmse: float
    optimized_rmse: float
    iterations: int
    matched_indices: np.ndarray

    @property
    def improved(self) -> bool:
        return self.optimized_rmse < self.initial_rmse


def sam3d_orbit_cameras(
    *,
    view_count: int,
    image_size: tuple[int, int],
    radius: float = 2.5,
    elevation_degrees: float = 0.0,
    fov_y_degrees: float = 50.0,
    target: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[Sam3dRenderCamera, ...]:
    """Build deterministic orbit cameras using the renderer's +Z-forward convention."""

    if view_count <= 0:
        raise ValueError(f"view_count must be positive, got {view_count}")
    height, width = _validate_image_size(image_size)
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    focal_y = (0.5 * float(height)) / tan(radians(float(fov_y_degrees)) * 0.5)
    focal_x = focal_y
    intrinsics = np.array(
        [
            [focal_x, 0.0, float(width) * 0.5],
            [0.0, focal_y, float(height) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    target_np = _as_points(np.asarray(target, dtype=np.float32), "target").reshape(3)
    elevation = radians(float(elevation_degrees))
    horizontal_radius = float(radius) * cos(elevation)
    eye_y = float(radius) * sin(elevation)
    cameras: list[Sam3dRenderCamera] = []
    for index in range(view_count):
        azimuth = 2.0 * np.pi * float(index) / float(view_count)
        eye = target_np + np.array(
            [horizontal_radius * sin(azimuth), eye_y, horizontal_radius * cos(azimuth)],
            dtype=np.float32,
        )
        cameras.append(
            Sam3dRenderCamera(
                view_matrix=_look_at_view_matrix(eye, target_np),
                intrinsics=intrinsics.copy(),
                name=f"orbit_{index:03d}",
            )
        )
    return tuple(cameras)


def render_sam3d_gaussian_multiview(
    *,
    xyz: np.ndarray,
    features_dc: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
    image_size: tuple[int, int],
    cameras: Sequence[Sam3dRenderCamera] | None = None,
    view_count: int = 4,
    camera_radius: float = 2.5,
    use_metal: bool = True,
    allow_cpu_fallback: bool = True,
) -> Sam3dMultiviewRenderResult:
    """Render SAM3D official-field Gaussian arrays from multiple views."""

    height, width = _validate_image_size(image_size)
    means, quats_xyzw, scales_linear, alphas, sh_dc = sam3d_gaussian_fields_to_raster_inputs(
        xyz=xyz,
        features_dc=features_dc,
        opacity=opacity,
        scale=scale,
        rotation=rotation,
    )
    render_cameras = (
        tuple(cameras)
        if cameras is not None
        else sam3d_orbit_cameras(view_count=view_count, image_size=(height, width), radius=camera_radius)
    )
    if not render_cameras:
        raise ValueError("render_sam3d_gaussian_multiview requires at least one camera")

    results: list[GaussianRasterizeResult] = []
    rgba: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    for camera in render_cameras:
        result = rasterize_gaussians(
            means,
            quats_xyzw,
            scales_linear,
            alphas,
            sh_dc,
            camera.as_raster_camera(),
            (height, width),
            use_metal=use_metal,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        results.append(result)
        rgba.append(np.asarray(result.rgba, dtype=np.float32))
        depth.append(np.asarray(result.depth, dtype=np.float32))

    backends = tuple(str(result.metadata.get("backend", "unknown")) for result in results)
    return Sam3dMultiviewRenderResult(
        rgba=np.stack(rgba, axis=0),
        depth=np.stack(depth, axis=0),
        cameras=render_cameras,
        pixel_counts=tuple(int(result.pixel_count) for result in results),
        metadata={
            "view_count": int(len(render_cameras)),
            "image_size": (int(height), int(width)),
            "gaussian_count": int(means.shape[0]),
            "backends": backends,
            "rotation_source_convention": "SAM3D WXYZ",
            "rotation_renderer_convention": "XYZW",
            "scale_source": "SAM3D log scale",
            "opacity_source": "SAM3D opacity logits",
        },
    )


def sam3d_gaussian_fields_to_raster_inputs(
    *,
    xyz: np.ndarray,
    features_dc: np.ndarray,
    opacity: np.ndarray,
    scale: np.ndarray,
    rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert official SAM3D Gaussian fields to `rasterize_gaussians` inputs."""

    means = _as_matrix(xyz, 3, "xyz")
    row_count = int(means.shape[0])
    features = np.asarray(features_dc, dtype=np.float32)
    if features.shape == (row_count, 3):
        sh_dc = features[:, None, :]
    elif features.shape == (row_count, 1, 3):
        sh_dc = features
    else:
        raise ValueError(f"features_dc must have shape (N,3) or (N,1,3), got {features.shape}")
    opacity_logits = _as_matrix(opacity, 1, "opacity").reshape(-1)
    log_scales = _as_matrix(scale, 3, "scale")
    rotations_wxyz = _as_matrix(rotation, 4, "rotation")
    if not (
        sh_dc.shape[0] == row_count
        and opacity_logits.shape[0] == row_count
        and log_scales.shape[0] == row_count
        and rotations_wxyz.shape[0] == row_count
    ):
        raise ValueError("SAM3D Gaussian renderer inputs must have matching row counts")
    alphas = _sigmoid(opacity_logits).astype(np.float32, copy=False)
    scales_linear = np.clip(np.exp(log_scales), 1e-6, 1.0).astype(np.float32, copy=False)
    quats_xyzw = rotations_wxyz[:, [1, 2, 3, 0]].astype(np.float32, copy=False)
    return (
        means.astype(np.float32, copy=False),
        quats_xyzw,
        scales_linear,
        alphas,
        sh_dc.astype(np.float32, copy=False),
    )


def optimize_sam3d_layout_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    initial_transform: np.ndarray | None = None,
    iterations: int = 8,
) -> Sam3dLayoutOptimizationResult:
    """Rigid ICP-style post-optimization for SAM3D layout alignment."""

    source = _as_points(source_points, "source_points")
    target = _as_points(target_points, "target_points")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("layout alignment requires non-empty source and target point sets")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    transform = np.eye(4, dtype=np.float64) if initial_transform is None else _as_transform(initial_transform)
    initial_aligned = _apply_transform(source, transform)
    initial_indices = _nearest_neighbor_indices(initial_aligned, target)
    initial_rmse = _rmse(initial_aligned, target[initial_indices])

    matched_indices = initial_indices
    aligned = initial_aligned
    for _ in range(int(iterations)):
        matched_indices = _nearest_neighbor_indices(aligned, target)
        delta = _rigid_transform(aligned, target[matched_indices])
        transform = delta @ transform
        aligned = _apply_transform(source, transform)

    optimized_rmse = _rmse(aligned, target[matched_indices])
    return Sam3dLayoutOptimizationResult(
        transform=transform.astype(np.float32),
        aligned_points=aligned.astype(np.float32, copy=False),
        initial_rmse=float(initial_rmse),
        optimized_rmse=float(optimized_rmse),
        iterations=int(iterations),
        matched_indices=matched_indices.astype(np.int64, copy=False),
    )


def _look_at_view_matrix(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.maximum(np.linalg.norm(forward), 1e-12)
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_hint))) > 0.98:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(up_hint, forward)
    right = right / np.maximum(np.linalg.norm(right), 1e-12)
    up = np.cross(forward, right)
    rotation = np.stack([right, up, forward], axis=0).astype(np.float32)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -(rotation @ eye.astype(np.float32))
    return view


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    hom = np.concatenate([points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    return (transform @ hom.T).T[:, :3].astype(np.float64)


def _nearest_neighbor_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    distances = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=1).astype(np.int64, copy=False)


def _rmse(source: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1))))


def _as_points(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and array.shape[0] == 3:
        array = array.reshape(1, 3)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{label} must have shape (N, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _as_matrix(value: np.ndarray, width: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and width == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{label} must have shape (N, {width}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array.astype(np.float32, copy=False)


def _as_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"initial_transform must have shape (4, 4), got {transform.shape}")
    if not np.all(np.isfinite(transform)):
        raise ValueError("initial_transform must contain only finite values")
    return transform


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError(f"image_size must be (height, width), got {image_size}")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size dimensions must be positive, got {image_size}")
    return height, width


def sam3d_layout_render_and_compare_score(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    view_count: int = 8,
    image_size: tuple[int, int] = (64, 64),
    target_valid_mask: np.ndarray | None = None,
) -> float:
    """Compute a render-and-compare layout quality score.

    Projects source and target point clouds orthographically from
    multiple azimuthal views and compares silhouette overlap via
    intersection-over-union.  Higher scores indicate better alignment.

    Both clouds are projected into a shared normalized space so that
    offsets between them affect the silhouette comparison.

    Args:
        source_points: (N, 3) float32 source point cloud.
        target_points: (M, 3) float32 target point cloud.
        view_count: Number of azimuthal views (default 8).
        image_size: Silhouette image (height, width) for comparison.
        target_valid_mask: Optional (M,) bool mask for target points
            that should be considered in the comparison.

    Returns:
        Mean IoU score across all views in [0, 1].
    """
    source = _as_points(source_points, "source_points")
    target = _as_points(target_points, "target_points")
    if view_count <= 0:
        raise ValueError(f"view_count must be positive, got {view_count}")
    height, width = _validate_image_size(image_size)
    if target_valid_mask is not None:
        mask = np.asarray(target_valid_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != target.shape[0]:
            raise ValueError(f"target_valid_mask must have shape ({target.shape[0]},), got {mask.shape}")
    else:
        mask = np.ones(target.shape[0], dtype=bool)

    visible_target = target[mask]
    if visible_target.shape[0] == 0:
        return 0.0

    scores: list[float] = []
    for idx in range(view_count):
        azimuth = 2.0 * np.pi * float(idx) / float(view_count)
        source_proj = _orthographic_project(source, azimuth=azimuth, elevation=0.0)
        target_proj = _orthographic_project(visible_target, azimuth=azimuth, elevation=0.0)

        source_sil, target_sil = _shared_normalized_silhouettes(
            source_proj, target_proj, image_size=(height, width)
        )

        intersection = np.sum(source_sil & target_sil)
        union = np.sum(source_sil | target_sil)
        if union > 0:
            scores.append(float(intersection) / float(union))

    return float(np.mean(scores)) if scores else 0.0


def optimize_sam3d_layout_render_and_compare(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    initial_transform: np.ndarray | None = None,
    iterations: int = 64,
    max_translation: float = 0.5,
    max_rotation_degrees: float = 15.0,
    view_count: int = 8,
    image_size: tuple[int, int] = (64, 64),
    improvement_threshold: float = 1e-6,
) -> Sam3dLayoutOptimizationResult:
    """Render-and-compare layout post-optimization beyond rigid ICP.

    Uses gradient-free perturbation sampling: proposes random rigid
    perturbations around the current transform, accepts if the
    render-and-compare score improves.  Followed by a rigid ICP
    refinement step.

    Args:
        source_points: (N, 3) float32 source point cloud.
        target_points: (M, 3) float32 target point cloud.
        initial_transform: Optional (4, 4) initial rigid transform.
        iterations: Number of perturbation trials (default 64).
        max_translation: Maximum per-axis translation perturbation.
        max_rotation_degrees: Maximum rotation perturbation in degrees.
        view_count: Number of azimuthal comparison views.
        image_size: Silhouette image size for comparison.
        improvement_threshold: Minimum score improvement to accept.

    Returns:
        Sam3dLayoutOptimizationResult with optimized transform and scores.
    """
    source = _as_points(source_points, "source_points")
    target = _as_points(target_points, "target_points")
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")

    transform = np.eye(4, dtype=np.float64) if initial_transform is None else _as_transform(initial_transform)
    initial_aligned = _apply_transform(source, transform)
    initial_rmse = _rmse(initial_aligned, target[_nearest_neighbor_indices(initial_aligned, target)])

    current_score = sam3d_layout_render_and_compare_score(
        initial_aligned, target, view_count=view_count, image_size=image_size
    )

    best_transform = transform.copy()
    best_aligned = initial_aligned.copy()
    best_score = current_score

    rng = np.random.default_rng(42)
    for _ in range(int(iterations)):
        rand_t = rng.uniform(-max_translation, max_translation, size=3).astype(np.float64)
        rand_axis = rng.normal(size=3).astype(np.float64)
        rand_axis = rand_axis / np.maximum(np.linalg.norm(rand_axis), 1e-12)
        rand_angle = np.deg2rad(rng.uniform(-max_rotation_degrees, max_rotation_degrees))
        rand_rot = _axis_angle_to_rotation(rand_axis, float(rand_angle))

        candidate_transform = np.eye(4, dtype=np.float64)
        candidate_transform[:3, :3] = rand_rot
        candidate_transform[:3, 3] = rand_t
        candidate_transform = candidate_transform @ best_transform

        candidate_aligned = _apply_transform(source, candidate_transform)
        candidate_score = sam3d_layout_render_and_compare_score(
            candidate_aligned, target, view_count=view_count, image_size=image_size
        )

        if candidate_score > best_score + improvement_threshold:
            best_transform = candidate_transform
            best_score = candidate_score
            best_aligned = candidate_aligned

    matched_indices = _nearest_neighbor_indices(best_aligned, target)
    optimized_rmse = _rmse(best_aligned, target[matched_indices])

    return Sam3dLayoutOptimizationResult(
        transform=best_transform.astype(np.float32),
        aligned_points=best_aligned.astype(np.float32, copy=False),
        initial_rmse=float(initial_rmse),
        optimized_rmse=float(optimized_rmse),
        iterations=int(iterations),
        matched_indices=matched_indices.astype(np.int64, copy=False),
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _orthographic_project(points: np.ndarray, *, azimuth: float, elevation: float) -> np.ndarray:
    cos_a = np.cos(azimuth)
    sin_a = np.sin(azimuth)
    cos_e = np.cos(elevation)
    sin_e = np.sin(elevation)

    x = points[:, 0] * cos_a - points[:, 1] * sin_a
    y = points[:, 0] * sin_e * sin_a + points[:, 1] * sin_e * cos_a + points[:, 2] * cos_e
    return np.stack([x, y], axis=1).astype(np.float32, copy=False)


def _point_silhouette(projected: np.ndarray, *, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    x = projected[:, 0]
    y = projected[:, 1]
    if x.size == 0:
        return np.zeros((height, width), dtype=bool)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0

    px = np.clip(((x - x_min) / x_span * (width - 1)).astype(np.int32), 0, width - 1)
    py = np.clip(((y - y_min) / y_span * (height - 1)).astype(np.int32), 0, height - 1)

    silhouette = np.zeros((height, width), dtype=bool)
    silhouette[py, px] = True
    return silhouette


def _shared_normalized_silhouettes(
    source_proj: np.ndarray,
    target_proj: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_size
    if source_proj.size == 0 and target_proj.size == 0:
        return np.zeros((height, width), dtype=bool), np.zeros((height, width), dtype=bool)

    all_x = np.concatenate([source_proj[:, 0], target_proj[:, 0]]) if source_proj.size else target_proj[:, 0]
    all_y = np.concatenate([source_proj[:, 1], target_proj[:, 1]]) if source_proj.size else target_proj[:, 1]

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0

    def _render(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return np.zeros((height, width), dtype=bool)
        px = np.clip(((points[:, 0] - x_min) / x_span * (width - 1)).astype(np.int32), 0, width - 1)
        py = np.clip(((points[:, 1] - y_min) / y_span * (height - 1)).astype(np.int32), 0, height - 1)
        sil = np.zeros((height, width), dtype=bool)
        sil[py, px] = True
        return sil

    return _render(source_proj), _render(target_proj)


def _axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = axis / np.maximum(np.linalg.norm(axis), 1e-12)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1.0 - cos_a
    ux, uy, uz = float(axis_norm[0]), float(axis_norm[1]), float(axis_norm[2])
    return np.array(
        [
            [cos_a + ux * ux * one_minus_cos, ux * uy * one_minus_cos - uz * sin_a, ux * uz * one_minus_cos + uy * sin_a],
            [uy * ux * one_minus_cos + uz * sin_a, cos_a + uy * uy * one_minus_cos, uy * uz * one_minus_cos - ux * sin_a],
            [uz * ux * one_minus_cos - uy * sin_a, uz * uy * one_minus_cos + ux * sin_a, cos_a + uz * uz * one_minus_cos],
        ],
        dtype=np.float64,
    )
