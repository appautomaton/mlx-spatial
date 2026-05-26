"""Pixal3D intermediate artifact writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .pixal3d_projection import Pixal3DProjectionConditioning


@dataclass(frozen=True)
class Pixal3DProjectionArtifact:
    """Written Pixal3D projection-conditioning bundle."""

    path: Path
    global_shape: tuple[int, ...]
    projected_shape: tuple[int, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DSparseStructureArtifact:
    """Written Pixal3D sparse-structure coordinate bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    decoded_shape: tuple[int, ...]
    target_resolution: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DShapeSLatArtifact:
    """Written Pixal3D shape SLat feature bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    features_shape: tuple[int, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DTextureSLatArtifact:
    """Written Pixal3D texture SLat feature bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    features_shape: tuple[int, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DShapeDecoderArtifact:
    """Written Pixal3D decoded FlexiDualGrid field bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    fields_shape: tuple[int, int]
    subdivision_shapes: tuple[tuple[int, int], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DTextureDecoderArtifact:
    """Written Pixal3D decoded PBR voxel bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    attributes_shape: tuple[int, int]
    spatial_shape: tuple[int, int, int]
    batch_size: int
    decode_resolution: int | None
    voxel_size: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Pixal3DShapeHRCoordinatesArtifact:
    """Written Pixal3D HR shape-cascade coordinate bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    requested_hr_resolution: int
    actual_hr_resolution: int
    actual_hr_grid_resolution: int
    token_count: int
    metadata: dict[str, Any]


def write_pixal3d_projection_npz(
    path: str | Path,
    conditioning: Pixal3DProjectionConditioning,
    *,
    metadata: dict[str, Any] | None = None,
) -> Pixal3DProjectionArtifact:
    """Write completed Pixal3D projection conditioning to a compact NPZ bundle."""

    if conditioning.global_tokens is None or conditioning.projected_features is None:
        raise ValueError("projection conditioning must contain global and projected features")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        "stage": conditioning.stage.name,
        "image_size": conditioning.stage.image_size,
        "grid_resolution": conditioning.stage.grid_resolution,
        "use_naf_upsample": conditioning.stage.use_naf_upsample,
        **(metadata or {}),
    }
    global_tokens = _array(conditioning.global_tokens)
    projected = _array(conditioning.projected_features)
    np.savez_compressed(
        output,
        global_tokens=global_tokens,
        projected_features=projected,
        metadata_json=json.dumps(payload_metadata, sort_keys=True, default=str),
    )
    return Pixal3DProjectionArtifact(
        path=output,
        global_shape=tuple(int(dim) for dim in global_tokens.shape),
        projected_shape=tuple(int(dim) for dim in projected.shape),
        metadata=payload_metadata,
    )


def write_pixal3d_shape_slat_npz(
    path: str | Path,
    coordinates: mx.array,
    features: mx.array,
    *,
    metadata: dict[str, Any] | None = None,
) -> Pixal3DShapeSLatArtifact:
    """Write Pixal3D shape SLat coordinates/features to an inspectable NPZ bundle."""

    coordinates_array = _array(coordinates)
    features_array = _array(features)
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 4:
        raise ValueError(f"shape SLat coordinates must have shape (n, 4), got {coordinates_array.shape}")
    if features_array.ndim != 2:
        raise ValueError(f"shape SLat features must have shape (n, channels), got {features_array.shape}")
    if features_array.shape[0] != coordinates_array.shape[0]:
        raise ValueError(
            "shape SLat coordinate/feature token mismatch: "
            f"coordinates={coordinates_array.shape[0]} features={features_array.shape[0]}"
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        "stage": "shape_slat_lr",
        "coordinate_order": "batch,z,y,x",
        "coordinates_shape": tuple(int(dim) for dim in coordinates_array.shape),
        "features_shape": tuple(int(dim) for dim in features_array.shape),
        **(metadata or {}),
    }
    np.savez_compressed(
        output,
        coordinates=coordinates_array.astype(np.int32, copy=False),
        features=features_array.astype(np.float32, copy=False),
        metadata_json=json.dumps(payload_metadata, sort_keys=True, default=str),
    )
    return Pixal3DShapeSLatArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        features_shape=tuple(int(dim) for dim in features_array.shape),
        metadata=payload_metadata,
    )


def write_pixal3d_texture_slat_npz(
    path: str | Path,
    coordinates: mx.array,
    features: mx.array,
    *,
    metadata: dict[str, Any] | None = None,
) -> Pixal3DTextureSLatArtifact:
    """Write Pixal3D texture SLat coordinates/features to an inspectable NPZ bundle."""

    coordinates_array = _array(coordinates)
    features_array = _array(features)
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 4:
        raise ValueError(f"texture SLat coordinates must have shape (n, 4), got {coordinates_array.shape}")
    if features_array.ndim != 2:
        raise ValueError(f"texture SLat features must have shape (n, channels), got {features_array.shape}")
    if features_array.shape[0] != coordinates_array.shape[0]:
        raise ValueError(
            "texture SLat coordinate/feature token mismatch: "
            f"coordinates={coordinates_array.shape[0]} features={features_array.shape[0]}"
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        "stage": "texture_slat",
        "coordinate_order": "batch,z,y,x",
        "coordinates_shape": tuple(int(dim) for dim in coordinates_array.shape),
        "features_shape": tuple(int(dim) for dim in features_array.shape),
        **(metadata or {}),
    }
    np.savez_compressed(
        output,
        coordinates=coordinates_array.astype(np.int32, copy=False),
        features=features_array.astype(np.float32, copy=False),
        metadata_json=json.dumps(payload_metadata, sort_keys=True, default=str),
    )
    return Pixal3DTextureSLatArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        features_shape=tuple(int(dim) for dim in features_array.shape),
        metadata=payload_metadata,
    )


def write_pixal3d_shape_decoder_npz(
    path: str | Path,
    coordinates: mx.array,
    fields: mx.array,
    *,
    subdivisions: tuple[mx.array, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> Pixal3DShapeDecoderArtifact:
    """Write decoded Pixal3D FlexiDualGrid coordinates/fields to an NPZ bundle."""

    coordinates_array = _array(coordinates)
    fields_array = _array(fields)
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 4:
        raise ValueError(f"shape decoder coordinates must have shape (n, 4), got {coordinates_array.shape}")
    if fields_array.ndim != 2 or fields_array.shape[1] != 7:
        raise ValueError(f"shape decoder fields must have shape (n, 7), got {fields_array.shape}")
    if fields_array.shape[0] != coordinates_array.shape[0]:
        raise ValueError(
            "shape decoder coordinate/field token mismatch: "
            f"coordinates={coordinates_array.shape[0]} fields={fields_array.shape[0]}"
        )

    subdivision_arrays = tuple(_array(subdivision) for subdivision in subdivisions)
    subdivision_shapes = tuple(tuple(int(dim) for dim in subdivision.shape) for subdivision in subdivision_arrays)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        "stage": "shape_decoder_fields",
        "coordinate_order": "batch,z,y,x",
        "coordinates_shape": tuple(int(dim) for dim in coordinates_array.shape),
        "fields_shape": tuple(int(dim) for dim in fields_array.shape),
        "field_channels": "FlexiDualGrid decoder output, 7 channels",
        "subdivision_shapes": subdivision_shapes,
        **(metadata or {}),
    }
    payload = {
        "coordinates": coordinates_array.astype(np.int32, copy=False),
        "fields": fields_array.astype(np.float32, copy=False),
        "metadata_json": json.dumps(payload_metadata, sort_keys=True, default=str),
    }
    payload.update(
        {
            f"subdivision_{index}": subdivision.astype(np.float32, copy=False)
            for index, subdivision in enumerate(subdivision_arrays)
        }
    )
    np.savez_compressed(output, **payload)
    return Pixal3DShapeDecoderArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        fields_shape=tuple(int(dim) for dim in fields_array.shape),
        subdivision_shapes=subdivision_shapes,
        metadata=payload_metadata,
    )


def write_pixal3d_texture_decoder_npz(
    path: str | Path,
    coordinates: mx.array,
    attributes: mx.array,
    *,
    spatial_shape: tuple[int, int, int],
    batch_size: int,
    decode_resolution: int | None,
    voxel_size: float | None,
    metadata: dict[str, Any] | None = None,
) -> Pixal3DTextureDecoderArtifact:
    """Write decoded Pixal3D texture PBR voxels to an NPZ bundle."""

    coordinates_array = _array(coordinates)
    attributes_array = _array(attributes)
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 4:
        raise ValueError(f"texture decoder coordinates must have shape (n, 4), got {coordinates_array.shape}")
    if attributes_array.ndim != 2 or attributes_array.shape[1] != 6:
        raise ValueError(f"texture decoder attributes must have shape (n, 6), got {attributes_array.shape}")
    if attributes_array.shape[0] != coordinates_array.shape[0]:
        raise ValueError(
            "texture decoder coordinate/attribute token mismatch: "
            f"coordinates={coordinates_array.shape[0]} attributes={attributes_array.shape[0]}"
        )
    normalized_spatial_shape = tuple(int(dim) for dim in spatial_shape)
    if len(normalized_spatial_shape) != 3 or any(dim <= 0 for dim in normalized_spatial_shape):
        raise ValueError(f"texture decoder spatial_shape must contain three positive dims, got {spatial_shape}")
    if batch_size <= 0:
        raise ValueError("texture decoder batch_size must be positive")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        "stage": "texture_decoder_pbr",
        "coordinate_order": "batch,z,y,x",
        "coordinates_shape": tuple(int(dim) for dim in coordinates_array.shape),
        "attributes_shape": tuple(int(dim) for dim in attributes_array.shape),
        "attribute_channels": ("base_color_r", "base_color_g", "base_color_b", "metallic", "roughness", "alpha"),
        "spatial_shape": normalized_spatial_shape,
        "batch_size": int(batch_size),
        "decode_resolution": int(decode_resolution) if decode_resolution is not None else None,
        "voxel_size": float(voxel_size) if voxel_size is not None else None,
        **(metadata or {}),
    }
    np.savez_compressed(
        output,
        coordinates=coordinates_array.astype(np.int32, copy=False),
        attributes=attributes_array.astype(np.float32, copy=False),
        spatial_shape=np.array(normalized_spatial_shape, dtype=np.int32),
        batch_size=np.array(int(batch_size), dtype=np.int32),
        decode_resolution=np.array(-1 if decode_resolution is None else int(decode_resolution), dtype=np.int32),
        voxel_size=np.array(np.nan if voxel_size is None else float(voxel_size), dtype=np.float32),
        metadata_json=json.dumps(payload_metadata, sort_keys=True, default=str),
    )
    return Pixal3DTextureDecoderArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        attributes_shape=tuple(int(dim) for dim in attributes_array.shape),
        spatial_shape=normalized_spatial_shape,
        batch_size=int(batch_size),
        decode_resolution=int(decode_resolution) if decode_resolution is not None else None,
        voxel_size=float(voxel_size) if voxel_size is not None else None,
        metadata=payload_metadata,
    )


def write_pixal3d_shape_hr_coordinates_npz(
    path: str | Path,
    coordinates: mx.array,
    *,
    requested_hr_resolution: int,
    actual_hr_resolution: int,
    actual_hr_grid_resolution: int,
    max_num_tokens: int,
    raw_upsampled_shape: tuple[int, int],
    metadata: dict[str, Any] | None = None,
) -> Pixal3DShapeHRCoordinatesArtifact:
    """Write guarded Pixal3D HR cascade coordinates to an inspectable NPZ bundle."""

    coordinates_array = _array(coordinates)
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 4:
        raise ValueError(f"HR shape coordinates must have shape (n, 4), got {coordinates_array.shape}")
    if requested_hr_resolution <= 0 or actual_hr_resolution <= 0 or actual_hr_grid_resolution <= 0:
        raise ValueError("HR resolutions and grid resolution must be positive")
    if max_num_tokens <= 0:
        raise ValueError("max_num_tokens must be positive")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_raw_shape = tuple(int(dim) for dim in raw_upsampled_shape)
    payload_metadata = {
        "stage": "shape_slat_hr_coordinates",
        "coordinate_order": "batch,z,y,x",
        "coordinates_shape": tuple(int(dim) for dim in coordinates_array.shape),
        "raw_upsampled_shape": normalized_raw_shape,
        "requested_hr_resolution": int(requested_hr_resolution),
        "actual_hr_resolution": int(actual_hr_resolution),
        "actual_hr_grid_resolution": int(actual_hr_grid_resolution),
        "max_num_tokens": int(max_num_tokens),
        "token_count": int(coordinates_array.shape[0]),
        **(metadata or {}),
    }
    np.savez_compressed(
        output,
        coordinates=coordinates_array.astype(np.int32, copy=False),
        requested_hr_resolution=np.array(int(requested_hr_resolution), dtype=np.int32),
        actual_hr_resolution=np.array(int(actual_hr_resolution), dtype=np.int32),
        actual_hr_grid_resolution=np.array(int(actual_hr_grid_resolution), dtype=np.int32),
        max_num_tokens=np.array(int(max_num_tokens), dtype=np.int32),
        metadata_json=json.dumps(payload_metadata, sort_keys=True, default=str),
    )
    return Pixal3DShapeHRCoordinatesArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        requested_hr_resolution=int(requested_hr_resolution),
        actual_hr_resolution=int(actual_hr_resolution),
        actual_hr_grid_resolution=int(actual_hr_grid_resolution),
        token_count=int(coordinates_array.shape[0]),
        metadata=payload_metadata,
    )


def write_pixal3d_sparse_structure_npz(
    path: str | Path,
    coordinates: mx.array,
    *,
    decoded_shape: tuple[int, ...],
    target_resolution: int,
    metadata: dict[str, Any] | None = None,
) -> Pixal3DSparseStructureArtifact:
    """Write sparse decoder coordinates to an inspectable NPZ bundle."""

    coordinates_array = _array(coordinates)
    if coordinates_array.ndim != 2 or coordinates_array.shape[1] != 4:
        raise ValueError(f"sparse coordinates must have shape (n, 4), got {coordinates_array.shape}")
    if target_resolution <= 0:
        raise ValueError("target_resolution must be positive")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized_decoded_shape = tuple(int(dim) for dim in decoded_shape)
    payload_metadata = {
        "stage": "sparse_structure",
        "coordinate_order": "batch,z,y,x",
        "decoded_shape": normalized_decoded_shape,
        "coordinates_shape": tuple(int(dim) for dim in coordinates_array.shape),
        "target_resolution": int(target_resolution),
        **(metadata or {}),
    }
    np.savez_compressed(
        output,
        coordinates=coordinates_array.astype(np.int32, copy=False),
        decoded_shape=np.array(normalized_decoded_shape, dtype=np.int32),
        target_resolution=np.array(int(target_resolution), dtype=np.int32),
        metadata_json=json.dumps(payload_metadata, sort_keys=True, default=str),
    )
    return Pixal3DSparseStructureArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        decoded_shape=normalized_decoded_shape,
        target_resolution=int(target_resolution),
        metadata=payload_metadata,
    )


def _array(value: mx.array) -> np.ndarray:
    return np.array(value)
