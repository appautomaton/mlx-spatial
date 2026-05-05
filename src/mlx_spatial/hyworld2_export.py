"""Deterministic HY-World-2.0 WorldMirror export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image


def export_hyworld2_depth(
    output_dir: str | Path,
    depth: mx.array,
    confidence: mx.array,
) -> tuple[dict[str, object], ...]:
    root = Path(output_dir) / "depth"
    root.mkdir(parents=True, exist_ok=True)
    depth_np = _array(depth).astype(np.float32)
    confidence_np = _array(confidence).astype(np.float32)
    depth_path = root / "depth.npy"
    confidence_path = root / "confidence.npy"
    np.save(depth_path, depth_np)
    np.save(confidence_path, confidence_np)

    previews = _write_scalar_previews(root, "depth", np.squeeze(depth_np, axis=-1))
    return (
        {"name": "depth", "path": depth_path, "kind": "npy"},
        {"name": "depth-confidence", "path": confidence_path, "kind": "npy"},
        *(
            {"name": f"depth-preview-{index:03d}", "path": path, "kind": "png"}
            for index, path in enumerate(previews)
        ),
    )


def export_hyworld2_normals(
    output_dir: str | Path,
    normals: mx.array,
    confidence: mx.array,
) -> tuple[dict[str, object], ...]:
    root = Path(output_dir) / "normal"
    root.mkdir(parents=True, exist_ok=True)
    normal_np = _array(normals).astype(np.float32)
    confidence_np = _array(confidence).astype(np.float32)
    normal_path = root / "normal.npy"
    confidence_path = root / "confidence.npy"
    np.save(normal_path, normal_np)
    np.save(confidence_path, confidence_np)

    previews = []
    preview_values = np.clip((normal_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
    for batch_index, frame_index, frame in _iter_frames(preview_values):
        path = root / f"normal_b{batch_index:02d}_f{frame_index:03d}.png"
        Image.fromarray(frame, mode="RGB").save(path)
        previews.append(path)
    return (
        {"name": "normal", "path": normal_path, "kind": "npy"},
        {"name": "normal-confidence", "path": confidence_path, "kind": "npy"},
        *(
            {"name": f"normal-preview-{index:03d}", "path": path, "kind": "png"}
            for index, path in enumerate(previews)
        ),
    )


def export_hyworld2_cameras(
    output_dir: str | Path,
    camera_params: mx.array,
    *,
    image_size: tuple[int, int],
    image_paths: Sequence[str | Path] = (),
) -> tuple[dict[str, object], ...]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    params = _array(camera_params).astype(np.float32)
    extrinsics, intrinsics = camera_params_to_matrices(params, image_size=image_size)
    flat_extrinsics = extrinsics.reshape((-1, 4, 4))
    flat_intrinsics = intrinsics.reshape((-1, 3, 3))
    payload = {
        "num_cameras": int(flat_extrinsics.shape[0]),
        "extrinsics": [
            {"camera_id": index, "matrix": flat_extrinsics[index].tolist()}
            for index in range(flat_extrinsics.shape[0])
        ],
        "intrinsics": [
            {"camera_id": index, "matrix": flat_intrinsics[index].tolist()}
            for index in range(flat_intrinsics.shape[0])
        ],
    }
    if image_paths:
        payload["image_paths"] = [str(path) for path in image_paths]
    path = root / "camera_params.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return ({"name": "camera-params", "path": path, "kind": "json"},)


def export_hyworld2_points_ply(
    output_dir: str | Path,
    points: mx.array,
    image_tensor: mx.array,
) -> tuple[dict[str, object], ...]:
    root = Path(output_dir) / "points"
    root.mkdir(parents=True, exist_ok=True)
    points_np = _array(points).astype(np.float32)
    colors = _image_colors(image_tensor, points_np.shape[:4])
    vertex_count = int(np.prod(points_np.shape[:4]))
    path = root / "points.ply"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertex_count}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for batch_index in range(points_np.shape[0]):
            for frame_index in range(points_np.shape[1]):
                for y_index in range(points_np.shape[2]):
                    for x_index in range(points_np.shape[3]):
                        x, y, z = points_np[batch_index, frame_index, y_index, x_index]
                        red, green, blue = colors[batch_index, frame_index, y_index, x_index]
                        handle.write(
                            f"{x:.6f} {y:.6f} {z:.6f} "
                            f"{int(red)} {int(green)} {int(blue)}\n"
                        )
    return ({"name": "points", "path": path, "kind": "ply"},)


def export_hyworld2_gaussian_attributes(
    output_dir: str | Path,
    *,
    features: mx.array,
    depth: mx.array,
    confidence: mx.array,
    raw_params: mx.array | None = None,
    image_tensor: mx.array | None = None,
    camera_params: mx.array | None = None,
    points: mx.array | None = None,
    depth_mask_logits: mx.array | None = None,
) -> tuple[dict[str, object], ...]:
    root = Path(output_dir) / "gaussian"
    root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "features": _array(features).astype(np.float32),
        "depth": _array(depth).astype(np.float32),
        "confidence": _array(confidence).astype(np.float32),
    }
    if depth_mask_logits is not None:
        arrays["depth_mask_logits"] = _array(depth_mask_logits).astype(np.float32)
    attributes_path = root / "attributes.npz"
    np.savez(attributes_path, **arrays)

    records: list[dict[str, object]] = [
        {"name": "gaussian-attributes", "path": attributes_path, "kind": "npz"}
    ]
    splat_stats: dict[str, object] | None = None
    if raw_params is not None and image_tensor is not None:
        splats = build_hyworld2_gaussian_splats(
            raw_params=raw_params,
            image_tensor=image_tensor,
            depth=depth,
            camera_params=camera_params,
            points=points,
        )
        ply_path = Path(output_dir) / "gaussians.ply"
        splat_stats = _write_gaussians_ply(ply_path, splats)
        records.append({"name": "gaussians", "path": ply_path, "kind": "ply"})

    metadata = {
        "features_shape": list(arrays["features"].shape),
        "depth_shape": list(arrays["depth"].shape),
        "confidence_shape": list(arrays["confidence"].shape),
        "depth_mask_logits_shape": list(arrays["depth_mask_logits"].shape)
        if "depth_mask_logits" in arrays
        else None,
        "gaussians_ply": splat_stats,
        "rendering": {"status": "not requested", "requires_cuda_gsplat": False},
    }
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    records.append({"name": "gaussian-metadata", "path": metadata_path, "kind": "json"})
    return tuple(records)


def camera_params_to_matrices(
    camera_params: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(camera_params, dtype=np.float32)
    height, width = (int(value) for value in image_size)
    rotations = _quat_to_rotmat(params[..., 3:7])
    w2c = np.zeros((*params.shape[:-1], 4, 4), dtype=np.float32)
    w2c[..., :3, :3] = rotations
    w2c[..., :3, 3] = params[..., :3]
    w2c[..., 3, 3] = 1.0
    c2w = np.linalg.inv(w2c)

    fov_v = np.maximum(params[..., 7], 1e-6)
    fov_u = np.maximum(params[..., 8], 1e-6)
    intrinsics = np.zeros((*params.shape[:-1], 3, 3), dtype=np.float32)
    intrinsics[..., 0, 0] = width * 0.5 / np.tan(fov_u * 0.5)
    intrinsics[..., 1, 1] = height * 0.5 / np.tan(fov_v * 0.5)
    intrinsics[..., 0, 2] = width * 0.5
    intrinsics[..., 1, 2] = height * 0.5
    intrinsics[..., 2, 2] = 1.0
    return c2w, intrinsics


def build_hyworld2_gaussian_splats(
    *,
    raw_params: mx.array,
    image_tensor: mx.array,
    depth: mx.array,
    camera_params: mx.array | None = None,
    points: mx.array | None = None,
) -> dict[str, np.ndarray | str]:
    raw = np.transpose(_array(raw_params).astype(np.float32), (0, 1, 3, 4, 2))
    images = np.transpose(_array(image_tensor).astype(np.float32), (0, 1, 3, 4, 2))
    quats = raw[..., 0:4]
    scales = np.minimum(np.exp(raw[..., 4:7]), 0.3)
    opacities = _sigmoid(raw[..., 7])
    residual_sh = raw[..., 8:11][..., None, :]
    weights = _sigmoid(raw[..., 11])
    sh = np.zeros_like(residual_sh, dtype=np.float32)
    sh[..., 0, :] = (images - 0.5) / 0.28209479177387814
    sh = sh + residual_sh
    quat_norm = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / np.maximum(quat_norm, 1e-8)

    if camera_params is not None:
        means = _depth_to_world_points(
            _array(depth).astype(np.float32)[..., 0],
            _array(camera_params).astype(np.float32),
            image_size=tuple(int(value) for value in images.shape[2:4]),
        )
        means_source = "gsdepth+predcamera"
    elif points is not None:
        means = _array(points).astype(np.float32)
        means_source = "points-head"
    else:
        means = _depth_to_grid_points(_array(depth).astype(np.float32)[..., 0])
        means_source = "gsdepth+pixel-grid"

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh": sh,
        "weights": weights,
        "means_source": means_source,
    }


def write_hyworld2_trace(output_dir: str | Path, payload: dict[str, object]) -> dict[str, object]:
    path = Path(output_dir) / "trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"name": "trace", "path": path, "kind": "json"}


def _write_scalar_previews(root: Path, prefix: str, values: np.ndarray) -> tuple[Path, ...]:
    previews = []
    for batch_index, frame_index, frame in _iter_frames(values):
        frame_min = float(np.min(frame))
        frame_max = float(np.max(frame))
        if frame_max > frame_min:
            preview = (frame - frame_min) / (frame_max - frame_min)
        else:
            preview = np.zeros_like(frame, dtype=np.float32)
        path = root / f"{prefix}_b{batch_index:02d}_f{frame_index:03d}.png"
        Image.fromarray(np.clip(preview * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)
        previews.append(path)
    return tuple(previews)


def _iter_frames(values: np.ndarray):
    for batch_index in range(values.shape[0]):
        for frame_index in range(values.shape[1]):
            yield batch_index, frame_index, values[batch_index, frame_index]


def _image_colors(image_tensor: mx.array, target_shape: tuple[int, int, int, int]) -> np.ndarray:
    colors = _array(image_tensor).astype(np.float32)
    colors = np.transpose(colors, (0, 1, 3, 4, 2))
    if colors.shape[:4] != target_shape:
        raise ValueError(f"image tensor shape {colors.shape[:4]} does not match point grid {target_shape}")
    return np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)


def _write_gaussians_ply(path: Path, splats: dict[str, np.ndarray | str]) -> dict[str, object]:
    means = np.asarray(splats["means"], dtype=np.float32)[0].reshape((-1, 3))
    scales = np.asarray(splats["scales"], dtype=np.float32)[0].reshape((-1, 3))
    quats = np.asarray(splats["quats"], dtype=np.float32)[0].reshape((-1, 4))
    sh = np.asarray(splats["sh"], dtype=np.float32)[0].reshape((-1, 1, 3))
    opacities = np.asarray(splats["opacities"], dtype=np.float32)[0].reshape((-1,))
    weights = np.asarray(splats["weights"], dtype=np.float32)[0].reshape((-1,))
    raw_count = int(means.shape[0])
    finite = (
        np.isfinite(means).all(axis=1)
        & (np.max(np.abs(means), axis=1) < 1e12)
        & np.isfinite(scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(sh[:, 0, :]).all(axis=1)
        & np.isfinite(opacities)
        & (weights > 1e-6)
    )
    if np.any(finite):
        means = means[finite]
        scales = scales[finite]
        quats = quats[finite]
        sh = sh[finite]
        opacities = opacities[finite]
        weights = weights[finite]
    else:
        means = means[:0]
        scales = scales[:0]
        quats = quats[:0]
        sh = sh[:0]
        opacities = opacities[:0]
        weights = weights[:0]
    finite_count = int(means.shape[0])
    means, scales, quats, sh, opacities = _voxel_prune_gaussians_np(
        means,
        scales,
        quats,
        sh,
        opacities,
        weights,
        voxel_size=0.002,
    )
    voxel_count = int(means.shape[0])
    if means.shape[0] > 0:
        threshold = np.quantile(np.max(scales, axis=1), 0.98)
        keep = np.max(scales, axis=1) <= threshold
        means = means[keep]
        scales = scales[keep]
        quats = quats[keep]
        sh = sh[keep]
        opacities = opacities[keep]

    fields = (
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    log_scales = np.log(np.maximum(scales, 1e-12))
    rows = np.concatenate(
        (
            means,
            np.zeros((means.shape[0], 3), dtype=np.float32),
            sh[:, 0, :],
            opacities[:, None],
            log_scales,
            quats,
        ),
        axis=1,
    ).astype("<f4", copy=False)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {int(means.shape[0])}\n"
        + "".join(f"property float {name}\n" for name in fields)
        + "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(rows.tobytes(order="C"))
    return {
        "path": str(path),
        "format": "3dgs-ply",
        "ply_encoding": "binary_little_endian",
        "vertex_count": int(means.shape[0]),
        "raw_vertex_count": raw_count,
        "finite_vertex_count": finite_count,
        "voxel_pruned_vertex_count": voxel_count,
        "voxel_size": 0.002,
        "means_source": str(splats["means_source"]),
        "scale_quantile": 0.98,
        "fields": fields,
    }


def _voxel_prune_gaussians_np(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    sh: np.ndarray,
    opacities: np.ndarray,
    weights: np.ndarray,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if means.shape[0] == 0 or voxel_size <= 0:
        return means, scales, quats, sh, opacities
    voxel = np.floor(means / voxel_size).astype(np.int64)
    voxel = voxel - voxel.min(axis=0, keepdims=True)
    _, inverse = np.unique(voxel, axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    if count == means.shape[0]:
        return means, scales, quats, sh, opacities
    weight_sums = np.maximum(np.bincount(inverse, weights=weights, minlength=count).astype(np.float32), 1e-8)

    def weighted_average(values: np.ndarray) -> np.ndarray:
        flat = values.reshape((values.shape[0], -1))
        merged = np.stack(
            [
                np.bincount(inverse, weights=flat[:, dim] * weights, minlength=count)
                for dim in range(flat.shape[1])
            ],
            axis=1,
        ).astype(np.float32)
        return (merged / weight_sums[:, None]).reshape((count, *values.shape[1:]))

    merged_quats = weighted_average(quats)
    merged_quats = merged_quats / np.maximum(np.linalg.norm(merged_quats, axis=1, keepdims=True), 1e-8)
    merged_opacities = np.bincount(inverse, weights=weights * weights, minlength=count).astype(np.float32) / weight_sums
    return (
        weighted_average(means),
        weighted_average(scales),
        merged_quats,
        weighted_average(sh),
        merged_opacities,
    )


def _depth_to_world_points(
    depth: np.ndarray,
    camera_params: np.ndarray,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    c2w, intrinsics = camera_params_to_matrices(camera_params, image_size=image_size)
    batch, frames, height, width = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    output = np.empty((batch, frames, height, width, 3), dtype=np.float32)
    for batch_index in range(batch):
        for frame_index in range(frames):
            z = depth[batch_index, frame_index]
            intr = intrinsics[batch_index, frame_index]
            x = (u_grid - intr[0, 2]) * z / intr[0, 0]
            y = (v_grid - intr[1, 2]) * z / intr[1, 1]
            camera_points = np.stack((x, y, z), axis=-1)
            pose = c2w[batch_index, frame_index]
            output[batch_index, frame_index] = camera_points @ pose[:3, :3].T + pose[:3, 3]
    return output


def _depth_to_grid_points(depth: np.ndarray) -> np.ndarray:
    batch, frames, height, width = depth.shape
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    output = np.empty((batch, frames, height, width, 3), dtype=np.float32)
    output[..., 0] = grid_x[None, None, :, :]
    output[..., 1] = grid_y[None, None, :, :]
    output[..., 2] = depth
    return output


def _quat_to_rotmat(quaternions: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternions, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = np.where(norm > 1e-8, quat / np.maximum(norm, 1e-8), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    i, j, k, r = np.moveaxis(quat, -1, 0)
    two_s = 2.0 / np.maximum(np.sum(quat * quat, axis=-1), 1e-8)
    stacked = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return stacked.reshape((*quat.shape[:-1], 3, 3)).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _array(values: mx.array) -> np.ndarray:
    return np.array(values)
