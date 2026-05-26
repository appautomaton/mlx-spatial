"""MapAnything scene postprocess and geometry helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from numbers import Number
from typing import Mapping, Sequence

import mlx.core as mx
import numpy as np

from .mapanything_heads import MapAnythingHeadsOutput
from .mapanything_preprocess import (
    MAPANYTHING_DINOV2_IMAGE_MEAN,
    MAPANYTHING_DINOV2_IMAGE_STD,
    MapAnythingPreprocessedInput,
)


MAPANYTHING_FINAL_OUTPUT_KEYS = (
    "pts3d",
    "pts3d_cam",
    "ray_directions",
    "depth_along_ray",
    "depth_z",
    "cam_trans",
    "cam_quats",
    "camera_poses",
    "intrinsics",
    "conf",
    "non_ambiguous_mask",
    "non_ambiguous_mask_logits",
    "metric_scaling_factor",
    "mask",
    "img_no_norm",
)
MAPANYTHING_SCENE_PAYLOAD_KEYS = (
    "images",
    "depth",
    "confidence",
    "masks",
    "intrinsics",
    "camera_poses",
    "extrinsics",
    "world_points",
)


@dataclass(frozen=True)
class MapAnythingPostprocessConfig:
    """Inference-time postprocess settings for image-only MapAnything output."""

    apply_mask: bool = True
    mask_edges: bool = True
    edge_normal_threshold: float = 5.0
    edge_depth_threshold: float = 0.03
    apply_confidence_mask: bool = False
    confidence_percentile: float = 10.0
    apply_valid_depth_mask: bool = True


@dataclass(frozen=True)
class MapAnythingPostprocessResult:
    """Postprocessed per-view predictions plus scene-payload helpers."""

    views: tuple[dict[str, np.ndarray], ...]
    trace: dict[str, object] = field(default_factory=dict)

    @property
    def parity_tensors(self) -> dict[str, np.ndarray]:
        tensors: dict[str, np.ndarray] = {}
        for index, view in enumerate(self.views):
            for key in MAPANYTHING_FINAL_OUTPUT_KEYS:
                if key in view:
                    tensors[f"final.{key}.{index}"] = view[key]
        scene_payload = mapanything_scene_payload_from_postprocessed_views(self.views)
        scene_key_map = {
            "images": "scene.images",
            "depth": "scene.depth",
            "confidence": "scene.conf",
            "masks": "scene.final_masks",
            "intrinsics": "scene.intrinsics",
            "camera_poses": "scene.camera_poses",
            "world_points": "scene.world_points",
        }
        for payload_key, parity_key in scene_key_map.items():
            tensors[parity_key] = scene_payload[payload_key]
        return tensors

    @property
    def scene_payload(self) -> dict[str, np.ndarray]:
        return mapanything_scene_payload_from_postprocessed_views(self.views)


def postprocess_mapanything_heads_output(
    heads_output: MapAnythingHeadsOutput,
    preprocessed: MapAnythingPreprocessedInput,
    *,
    config: MapAnythingPostprocessConfig | None = None,
) -> MapAnythingPostprocessResult:
    """Convert MLX head output into the vendored image-only inference contract."""

    cfg = config or MapAnythingPostprocessConfig()
    raw_views = mapanything_heads_output_to_raw_views(heads_output, view_count=preprocessed.frame_count)
    processed_views: list[dict[str, np.ndarray]] = []

    for raw_view, input_view in zip(raw_views, preprocessed.views):
        processed = {key: value.copy() for key, value in raw_view.items()}
        processed["img_no_norm"] = denormalize_mapanything_image(input_view.img, input_view.data_norm_type)

        if "pts3d_cam" in processed:
            processed["depth_z"] = processed["pts3d_cam"][..., 2:3].astype(np.float32, copy=False)

        if "ray_directions" in processed:
            processed["intrinsics"] = mapanything_recover_pinhole_intrinsics_from_ray_directions(
                processed["ray_directions"]
            )

        if "cam_trans" in processed and "cam_quats" in processed:
            processed["camera_poses"] = mapanything_camera_poses_from_trans_quats(
                processed["cam_trans"],
                processed["cam_quats"],
            )

        if cfg.apply_mask:
            _apply_mapanything_final_mask(processed, cfg)

        processed_views.append(processed)

    return MapAnythingPostprocessResult(
        views=tuple(processed_views),
        trace={
            "stage": "scene-postprocess",
            "runtime_depends_on_torch": False,
            "view_count": preprocessed.frame_count,
            "apply_mask": cfg.apply_mask,
            "mask_edges": cfg.mask_edges,
            "apply_valid_depth_mask": cfg.apply_valid_depth_mask,
            "edge_mask_numeric_boundary": (
                "NumPy edge-mask logic mirrors the vendored algorithm; pixels exactly at normal/depth "
                "thresholds may differ from the Torch reference by implementation precision."
            ),
            "scene_payload_keys": list(MAPANYTHING_SCENE_PAYLOAD_KEYS),
        },
    )


def mapanything_heads_output_to_raw_views(
    heads_output: MapAnythingHeadsOutput,
    *,
    view_count: int,
) -> tuple[dict[str, np.ndarray], ...]:
    """Assemble vendored raw per-view output dictionaries from head tensors."""

    if view_count <= 0:
        raise ValueError("view_count must be positive")

    dense_value = _to_numpy_float32(heads_output.dense.value)
    if dense_value.ndim != 4 or dense_value.shape[1] != 4:
        raise ValueError(f"dense.value must have shape [B*V, 4, H, W], got {dense_value.shape}")
    dense_batch = dense_value.shape[0]
    if dense_batch % view_count != 0:
        raise ValueError(f"dense batch {dense_batch} is not divisible by view_count {view_count}")
    batch_size = dense_batch // view_count

    dense_nhwc = np.transpose(dense_value, (0, 2, 3, 1)).copy()
    ray_directions = dense_nhwc[..., :3]
    depth_along_ray = dense_nhwc[..., 3:4]
    pose_value = _to_numpy_float32(heads_output.pose_value)
    scale_value = _to_numpy_float32(heads_output.scale_value)
    if pose_value.shape != (dense_batch, 7):
        raise ValueError(f"pose_value must have shape {(dense_batch, 7)}, got {pose_value.shape}")
    if scale_value.ndim == 1:
        scale_value = scale_value[:, None]
    if scale_value.shape[0] != batch_size:
        raise ValueError(f"scale_value batch must be {batch_size}, got {scale_value.shape[0]}")

    cam_trans = pose_value[:, :3]
    cam_quats = pose_value[:, 3:7]
    pts3d = mapanything_convert_ray_dirs_depth_pose_to_pointmap(
        ray_directions,
        depth_along_ray,
        cam_trans,
        cam_quats,
    )
    pts3d_cam = ray_directions * depth_along_ray

    confidence = np.transpose(_to_numpy_float32(heads_output.dense.confidence), (0, 2, 3, 1))[..., 0]
    mask_prob = np.transpose(_to_numpy_float32(heads_output.dense.mask), (0, 2, 3, 1))[..., 0]
    mask_logits = np.transpose(_to_numpy_float32(heads_output.dense.logits), (0, 2, 3, 1))[..., 0]
    non_ambiguous_mask = mask_prob > 0.5

    scale_dense = scale_value[:, None, None, :]
    raw_views = []
    for view_index in range(view_count):
        start = view_index * batch_size
        stop = start + batch_size
        raw_views.append(
            {
                "pts3d": (pts3d[start:stop] * scale_dense).astype(np.float32, copy=False),
                "pts3d_cam": (pts3d_cam[start:stop] * scale_dense).astype(np.float32, copy=False),
                "ray_directions": ray_directions[start:stop].astype(np.float32, copy=False),
                "depth_along_ray": (depth_along_ray[start:stop] * scale_dense).astype(np.float32, copy=False),
                "cam_trans": (cam_trans[start:stop] * scale_value).astype(np.float32, copy=False),
                "cam_quats": cam_quats[start:stop].astype(np.float32, copy=False),
                "metric_scaling_factor": scale_value.astype(np.float32, copy=False),
                "conf": confidence[start:stop].astype(np.float32, copy=False),
                "non_ambiguous_mask": non_ambiguous_mask[start:stop],
                "non_ambiguous_mask_logits": mask_logits[start:stop].astype(np.float32, copy=False),
            }
        )
    return tuple(raw_views)


def mapanything_postprocess_outputs_for_parity(
    result: MapAnythingPostprocessResult,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return named postprocess tensors for comparison with a reference bundle."""

    tensors = result.parity_tensors
    if names is None:
        return tensors
    return {name: tensors[name] for name in names if name in tensors}


def mapanything_scene_payload_from_postprocessed_views(
    views: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build the stable view-first scene payload from postprocessed predictions."""

    images = []
    depths = []
    confidences = []
    masks = []
    intrinsics = []
    camera_poses = []
    extrinsics = []
    world_points = []
    for view in views:
        depth_z = np.asarray(view["depth_z"])
        if depth_z.ndim != 4 or depth_z.shape[0] != 1 or depth_z.shape[-1] != 1:
            raise ValueError(f"scene payload expects single-batch depth_z [1,H,W,1], got {depth_z.shape}")
        intrinsic = np.asarray(view["intrinsics"])[0]
        camera_pose = np.asarray(view["camera_poses"])[0]
        depth = depth_z[0, ..., 0]
        points, valid_mask = mapanything_depthmap_to_world_frame(depth, intrinsic, camera_pose)
        if "mask" in view:
            final_mask = np.asarray(view["mask"])[0, ..., 0].astype(bool)
        else:
            final_mask = np.ones_like(depth, dtype=bool)
        final_mask = final_mask & valid_mask.astype(bool)

        images.append(np.asarray(view["img_no_norm"])[0].astype(np.float32, copy=False))
        depths.append(depth.astype(np.float32, copy=False))
        confidences.append(np.asarray(view["conf"])[0].astype(np.float32, copy=False))
        masks.append(final_mask.astype(np.float32, copy=False))
        intrinsics.append(intrinsic.astype(np.float32, copy=False))
        camera_poses.append(camera_pose.astype(np.float32, copy=False))
        extrinsics.append(np.linalg.inv(camera_pose).astype(np.float32, copy=False))
        world_points.append(points.astype(np.float32, copy=False))

    return {
        "images": np.stack(images),
        "depth": np.stack(depths),
        "confidence": np.stack(confidences),
        "masks": np.stack(masks),
        "intrinsics": np.stack(intrinsics),
        "camera_poses": np.stack(camera_poses),
        "extrinsics": np.stack(extrinsics),
        "world_points": np.stack(world_points),
    }


def denormalize_mapanything_image(value: object, data_norm_type: str | Sequence[str]) -> np.ndarray:
    """Convert normalized MapAnything image tensors to clipped RGB NHWC arrays."""

    norm_type = data_norm_type[0] if isinstance(data_norm_type, (tuple, list)) else data_norm_type
    if norm_type != "dinov2":
        raise ValueError(f"only dinov2 MapAnything normalization is supported, got {norm_type!r}")
    tensor = _to_numpy_float32(value)
    if tensor.ndim == 3 and tensor.shape[0] == 3:
        tensor = np.transpose(tensor, (1, 2, 0))
    elif tensor.ndim == 4 and tensor.shape[1] == 3:
        tensor = np.transpose(tensor, (0, 2, 3, 1))
    else:
        raise ValueError(f"image tensor must have shape [3,H,W] or [B,3,H,W], got {tensor.shape}")
    mean = np.asarray(MAPANYTHING_DINOV2_IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(MAPANYTHING_DINOV2_IMAGE_STD, dtype=np.float32)
    return np.clip(tensor * std + mean, 0.0, 1.0).astype(np.float32, copy=False)


def mapanything_quaternion_to_rotation_matrix(quat: object) -> np.ndarray:
    """Convert XYZW quaternions to rotation matrices."""

    quats = _to_numpy_float32(quat)
    squeeze = quats.ndim == 1
    if squeeze:
        quats = quats[None, :]
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"quat must have shape [4] or [B,4], got {quats.shape}")
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-8)
    x, y, z, w = [quats[:, index] for index in range(4)]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    matrix = np.stack(
        (
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    if squeeze:
        matrix = matrix[0]
    return matrix.astype(np.float32, copy=False)


def mapanything_camera_poses_from_trans_quats(cam_trans: object, cam_quats: object) -> np.ndarray:
    """Build camera-to-world pose matrices from translation and XYZW quaternions."""

    trans = _to_numpy_float32(cam_trans)
    quats = _to_numpy_float32(cam_quats)
    if trans.ndim == 1:
        trans = trans[None, :]
    if quats.ndim == 1:
        quats = quats[None, :]
    if trans.shape[0] != quats.shape[0] or trans.shape[1:] != (3,) or quats.shape[1:] != (4,):
        raise ValueError(f"cam_trans/cam_quats have incompatible shapes: {trans.shape}, {quats.shape}")
    rotation = mapanything_quaternion_to_rotation_matrix(quats)
    pose = np.tile(np.eye(4, dtype=np.float32), (trans.shape[0], 1, 1))
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = trans
    return pose


def mapanything_convert_ray_dirs_depth_pose_to_pointmap(
    ray_directions: object,
    depth_along_ray: object,
    pose_trans: object,
    pose_quats: object,
) -> np.ndarray:
    """Convert ray directions, ray depth, and pose to a world-frame pointmap."""

    rays = _to_numpy_float32(ray_directions)
    depth = _to_numpy_float32(depth_along_ray)
    trans = _to_numpy_float32(pose_trans)
    quats = _to_numpy_float32(pose_quats)
    squeeze = rays.ndim == 3
    if squeeze:
        rays = rays[None, ...]
        depth = depth[None, ...]
        trans = trans[None, ...]
        quats = quats[None, ...]
    if rays.ndim != 4 or rays.shape[-1] != 3:
        raise ValueError(f"ray_directions must have shape [B,H,W,3], got {rays.shape}")
    if depth.shape != rays.shape[:-1] + (1,):
        raise ValueError(f"depth_along_ray must have shape {rays.shape[:-1] + (1,)}, got {depth.shape}")
    if trans.shape != (rays.shape[0], 3) or quats.shape != (rays.shape[0], 4):
        raise ValueError(f"pose shapes must be [B,3]/[B,4], got {trans.shape}/{quats.shape}")
    rotation = mapanything_quaternion_to_rotation_matrix(quats)
    pts3d_local = depth * rays
    pts3d_world = np.einsum("bij,bhwj->bhwi", rotation, pts3d_local) + trans[:, None, None, :]
    if squeeze:
        pts3d_world = pts3d_world[0]
    return pts3d_world.astype(np.float32, copy=False)


def mapanything_depthmap_to_camera_frame(depthmap: object, intrinsics: object) -> tuple[np.ndarray, np.ndarray]:
    """Convert Z depth to camera-frame points and a positive-depth validity mask."""

    depth = _to_numpy_float32(depthmap)
    intrinsic = _to_numpy_float32(intrinsics)
    squeeze = depth.ndim == 2
    if squeeze:
        depth = depth[None, ...]
        intrinsic = intrinsic[None, ...]
    if depth.ndim != 3 or intrinsic.shape != (depth.shape[0], 3, 3):
        raise ValueError(f"depth/intrinsics must be [B,H,W]/[B,3,3], got {depth.shape}/{intrinsic.shape}")
    batch, height, width = depth.shape
    x_grid, y_grid = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    x_grid = np.broadcast_to(x_grid, (batch, height, width))
    y_grid = np.broadcast_to(y_grid, (batch, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    xx = (x_grid - cx) * depth / fx
    yy = (y_grid - cy) * depth / fy
    pts3d_cam = np.stack((xx, yy, depth), axis=-1)
    valid_mask = depth > 0.0
    if squeeze:
        pts3d_cam = pts3d_cam[0]
        valid_mask = valid_mask[0]
    return pts3d_cam.astype(np.float32, copy=False), valid_mask


def mapanything_depthmap_to_world_frame(
    depthmap: object,
    intrinsics: object,
    camera_pose: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Z depth to world-frame points using camera-to-world poses."""

    pts3d_cam, valid_mask = mapanything_depthmap_to_camera_frame(depthmap, intrinsics)
    if camera_pose is None:
        return pts3d_cam, valid_mask
    pose = _to_numpy_float32(camera_pose)
    squeeze = pts3d_cam.ndim == 3
    if squeeze:
        pts3d_cam = pts3d_cam[None, ...]
        pose = pose[None, ...]
    if pose.shape != (pts3d_cam.shape[0], 4, 4):
        raise ValueError(f"camera_pose must have shape [B,4,4], got {pose.shape}")
    ones = np.ones_like(pts3d_cam[..., :1])
    pts3d_homo = np.concatenate((pts3d_cam, ones), axis=-1)
    pts3d_world = np.einsum("bij,bhwj->bhwi", pose, pts3d_homo)[..., :3]
    if squeeze:
        pts3d_world = pts3d_world[0]
    return pts3d_world.astype(np.float32, copy=False), valid_mask


def mapanything_recover_pinhole_intrinsics_from_ray_directions(
    ray_directions: object,
    *,
    use_geometric_calculation: bool = False,
) -> np.ndarray:
    """Recover pinhole intrinsics from unit ray directions."""

    rays = _to_numpy_float32(ray_directions)
    squeeze = rays.ndim == 3
    if squeeze:
        rays = rays[None, ...]
    if rays.ndim != 4 or rays.shape[-1] != 3:
        raise ValueError(f"ray_directions must have shape [H,W,3] or [B,H,W,3], got {rays.shape}")
    batch_size, height, width, _ = rays.shape

    if height * width > 1_000_000 or use_geometric_calculation:
        center_h, center_w = height // 2, width // 2
        quarter_w, three_quarter_w = width // 4, 3 * width // 4
        quarter_h, three_quarter_h = height // 4, 3 * height // 4
        center_rays = rays[:, center_h, center_w, :] / rays[:, center_h, center_w, 2:3]
        left_rays = rays[:, center_h, quarter_w, :] / rays[:, center_h, quarter_w, 2:3]
        right_rays = rays[:, center_h, three_quarter_w, :] / rays[:, center_h, three_quarter_w, 2:3]
        top_rays = rays[:, quarter_h, center_w, :] / rays[:, quarter_h, center_w, 2:3]
        bottom_rays = rays[:, three_quarter_h, center_w, :] / rays[:, three_quarter_h, center_w, 2:3]
        fx_left = (quarter_w - center_w) / (left_rays[:, 0] - center_rays[:, 0])
        fx_right = (three_quarter_w - center_w) / (right_rays[:, 0] - center_rays[:, 0])
        fx = (fx_left + fx_right) / 2
        cx = center_w - fx * center_rays[:, 0]
        fy_top = (quarter_h - center_h) / (top_rays[:, 1] - center_rays[:, 1])
        fy_bottom = (three_quarter_h - center_h) / (bottom_rays[:, 1] - center_rays[:, 1])
        fy = (fy_top + fy_bottom) / 2
        cy = center_h - fy * center_rays[:, 1]
    else:
        step_h = max(1, height // 50)
        step_w = max(1, width // 50)
        h_indices = np.arange(0, height, step_h)
        w_indices = np.arange(0, width, step_w)
        x_grid, y_grid = np.meshgrid(
            np.arange(width, dtype=np.float32),
            np.arange(height, dtype=np.float32),
            indexing="xy",
        )
        x_flat = x_grid[h_indices[:, None], w_indices[None, :]].reshape(1, -1)
        y_flat = y_grid[h_indices[:, None], w_indices[None, :]].reshape(1, -1)
        x_flat = np.broadcast_to(x_flat, (batch_size, x_flat.shape[1]))
        y_flat = np.broadcast_to(y_flat, (batch_size, y_flat.shape[1]))
        sampled = rays[:, h_indices[:, None], w_indices[None, :], :]
        dx = sampled[..., 0].reshape(batch_size, -1)
        dy = sampled[..., 1].reshape(batch_size, -1)
        dz = sampled[..., 2].reshape(batch_size, -1)
        ratio_x = dx / dz
        ratio_y = dy / dz
        ones = np.ones_like(x_flat)
        cx = np.empty((batch_size,), dtype=np.float32)
        fx = np.empty((batch_size,), dtype=np.float32)
        cy = np.empty((batch_size,), dtype=np.float32)
        fy = np.empty((batch_size,), dtype=np.float32)
        for batch_index in range(batch_size):
            a_x = np.stack((ones[batch_index], ratio_x[batch_index]), axis=1)
            solution_x = np.linalg.solve(a_x.T @ a_x, a_x.T @ x_flat[batch_index])
            a_y = np.stack((ones[batch_index], ratio_y[batch_index]), axis=1)
            solution_y = np.linalg.solve(a_y.T @ a_y, a_y.T @ y_flat[batch_index])
            cx[batch_index], fx[batch_index] = solution_x
            cy[batch_index], fy[batch_index] = solution_y

    intrinsics = np.zeros((batch_size, 3, 3), dtype=np.float32)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0
    if squeeze:
        intrinsics = intrinsics[0]
    return intrinsics


def _apply_mapanything_final_mask(
    processed: dict[str, np.ndarray],
    config: MapAnythingPostprocessConfig,
) -> None:
    final_mask = None
    if "non_ambiguous_mask" in processed:
        final_mask = processed["non_ambiguous_mask"].astype(bool)

    if config.apply_valid_depth_mask and "depth_z" in processed:
        valid_depth = processed["depth_z"][..., 0] > 0.0
        final_mask = valid_depth if final_mask is None else final_mask & valid_depth

    if config.apply_confidence_mask and "conf" in processed:
        confidences = processed["conf"]
        thresholds = np.quantile(
            confidences.reshape(confidences.shape[0], -1),
            config.confidence_percentile / 100.0,
            axis=1,
        )[:, None, None]
        confidence_mask = confidences > thresholds
        final_mask = confidence_mask if final_mask is None else final_mask & confidence_mask

    if config.mask_edges and final_mask is not None and "pts3d" in processed:
        edge_masks = []
        pred_pts3d = processed["pts3d"]
        depth_z = processed["depth_z"][..., 0]
        for batch_index in range(final_mask.shape[0]):
            batch_mask = final_mask[batch_index]
            if batch_mask.any():
                normals, normals_mask = mapanything_points_to_normals(pred_pts3d[batch_index], mask=batch_mask)
                normal_edges = mapanything_normals_edge(
                    normals,
                    tol=config.edge_normal_threshold,
                    mask=normals_mask,
                )
                depth_edges = mapanything_depth_edge(
                    depth_z[batch_index],
                    rtol=config.edge_depth_threshold,
                    mask=batch_mask,
                )
                edge_masks.append(~(depth_edges & normal_edges))
            else:
                edge_masks.append(np.zeros_like(batch_mask, dtype=bool))
        final_mask = final_mask & np.stack(edge_masks, axis=0)

    if final_mask is None:
        return

    final_mask = final_mask.astype(bool, copy=False)
    final_mask_expanded = final_mask[..., None]
    for key in ("pts3d", "pts3d_cam", "depth_along_ray", "depth_z"):
        if key in processed:
            processed[key] = (processed[key] * final_mask_expanded).astype(np.float32, copy=False)
    processed["mask"] = final_mask_expanded


def mapanything_points_to_normals(
    point: np.ndarray,
    mask: np.ndarray | None = None,
    edge_threshold: float | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calculate a normal map from a point map."""

    point = np.asarray(point, dtype=np.float32)
    height, width = point.shape[-3:-1]
    has_mask = mask is not None
    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask.astype(bool)

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack(
        (
            np.cross(up, left, axis=-1),
            np.cross(left, down, axis=-1),
            np.cross(down, right, axis=-1),
            np.cross(right, up, axis=-1),
        )
    )
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    valid = (
        np.stack(
            (
                mask_pad[:-2, 1:-1] & mask_pad[1:-1, :-2],
                mask_pad[1:-1, :-2] & mask_pad[2:, 1:-1],
                mask_pad[2:, 1:-1] & mask_pad[1:-1, 2:],
                mask_pad[1:-1, 2:] & mask_pad[:-2, 1:-1],
            )
        )
        & mask_pad[None, 1:-1, 1:-1]
    )
    if edge_threshold is not None:
        view_angle = _angle_diff_vec3(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))

    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)

    if has_mask:
        normal_mask = valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal.astype(np.float32, copy=False), normal_mask
    return normal.astype(np.float32, copy=False)


def mapanything_depth_edge(
    depth: np.ndarray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    kernel_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a depth discontinuity mask."""

    depth = np.asarray(depth, dtype=np.float32)
    if mask is None:
        diff = _max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + _max_pool_2d(
            -depth,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
    else:
        mask = mask.astype(bool)
        diff = _max_pool_2d(
            np.where(mask, depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ) + _max_pool_2d(
            np.where(mask, -depth, -np.inf),
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


def mapanything_normals_edge(
    normals: np.ndarray,
    *,
    tol: float,
    kernel_size: int = 3,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a normal discontinuity mask."""

    normals = np.asarray(normals, dtype=np.float32)
    if normals.ndim < 3 or normals.shape[-1] != 3:
        raise ValueError(f"normals must have shape [...,H,W,3], got {normals.shape}")
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    padding = kernel_size // 2
    normals_window = _sliding_window_2d(
        np.pad(
            normals,
            (
                *([(0, 0)] * (normals.ndim - 3)),
                (padding, padding),
                (padding, padding),
                (0, 0),
            ),
            mode="edge",
        ),
        window_size=kernel_size,
        stride=1,
        axis=(-3, -2),
    )
    cosine = (normals[..., None, None] * normals_window).sum(axis=-3)
    with np.errstate(invalid="ignore"):
        angle_values = np.arccos(cosine)
    if mask is None:
        angle_diff = angle_values.max(axis=(-2, -1))
    else:
        mask_window = _sliding_window_2d(
            np.pad(mask.astype(bool), ((padding, padding), (padding, padding)), mode="edge"),
            window_size=kernel_size,
            stride=1,
            axis=(-2, -1),
        )
        angle_diff = np.where(mask_window, angle_values, 0).max(axis=(-2, -1))
    angle_diff = _max_pool_2d(angle_diff, kernel_size, stride=1, padding=kernel_size // 2)
    return angle_diff > np.deg2rad(tol)


def _angle_diff_vec3(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.arctan2(
        np.linalg.norm(np.cross(v1, v2, axis=-1), axis=-1) + eps,
        (v1 * v2).sum(axis=-1),
    )


def _sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int = -1) -> np.ndarray:
    if x.shape[axis] < window_size:
        raise ValueError(f"window_size {window_size} is larger than axis size {x.shape[axis]}")
    axis = axis % x.ndim
    shape = (
        *x.shape[:axis],
        (x.shape[axis] - window_size + 1) // stride,
        *x.shape[axis + 1 :],
        window_size,
    )
    strides = (
        *x.strides[:axis],
        stride * x.strides[axis],
        *x.strides[axis + 1 :],
        x.strides[axis],
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _sliding_window_nd(
    x: np.ndarray,
    window_size: tuple[int, ...],
    stride: tuple[int, ...],
    axis: tuple[int, ...],
) -> np.ndarray:
    axis = tuple(item % x.ndim for item in axis)
    for index, axis_item in enumerate(axis):
        x = _sliding_window_1d(x, window_size[index], stride[index], axis_item)
    return x


def _sliding_window_2d(
    x: np.ndarray,
    window_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    axis: tuple[int, int] = (-2, -1),
) -> np.ndarray:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return _sliding_window_nd(x, window_size, stride, axis)


def _max_pool_1d(
    x: np.ndarray,
    kernel_size: int,
    stride: int,
    padding: int = 0,
    axis: int = -1,
) -> np.ndarray:
    axis = axis % x.ndim
    if padding > 0:
        fill_value = np.nan if x.dtype.kind == "f" else np.iinfo(x.dtype).min
        padding_arr = np.full(
            (*x.shape[:axis], padding, *x.shape[axis + 1 :]),
            fill_value=fill_value,
            dtype=x.dtype,
        )
        x = np.concatenate((padding_arr, x, padding_arr), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmax(_sliding_window_1d(x, kernel_size, stride, axis), axis=-1)


def _max_pool_nd(
    x: np.ndarray,
    kernel_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    axis: tuple[int, ...],
) -> np.ndarray:
    for index, axis_item in enumerate(axis):
        x = _max_pool_1d(x, kernel_size[index], stride[index], padding[index], axis_item)
    return x


def _max_pool_2d(
    x: np.ndarray,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    axis: tuple[int, int] = (-2, -1),
) -> np.ndarray:
    if isinstance(kernel_size, Number):
        kernel_size = (int(kernel_size), int(kernel_size))
    if isinstance(stride, Number):
        stride = (int(stride), int(stride))
    if isinstance(padding, Number):
        padding = (int(padding), int(padding))
    return _max_pool_nd(x, kernel_size, stride, padding, axis)


def _to_numpy_float32(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, mx.array):
        mx.eval(value)
        return np.asarray(value).astype(np.float32, copy=False)
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)
