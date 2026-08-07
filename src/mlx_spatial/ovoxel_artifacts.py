"""Model-neutral decoded O-Voxel artifact writers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class DecodedOVoxelShapeArtifact:
    """Written decoded FlexiDualGrid field bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    fields_shape: tuple[int, int]
    subdivision_shapes: tuple[tuple[int, int], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DecodedOVoxelTextureArtifact:
    """Written decoded PBR voxel bundle."""

    path: Path
    coordinates_shape: tuple[int, int]
    attributes_shape: tuple[int, int]
    spatial_shape: tuple[int, int, int]
    batch_size: int
    decode_resolution: int | None
    voxel_size: float | None
    metadata: dict[str, Any]


def write_decoded_ovoxel_shape_npz(
    path: str | Path,
    coordinates: mx.array,
    fields: mx.array,
    *,
    subdivisions: tuple[mx.array, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> DecodedOVoxelShapeArtifact:
    """Write decoded FlexiDualGrid coordinates and fields to an NPZ bundle."""

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
    payload: dict[str, Any] = {
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
    _write_npz_atomic(output, payload)
    return DecodedOVoxelShapeArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        fields_shape=tuple(int(dim) for dim in fields_array.shape),
        subdivision_shapes=subdivision_shapes,
        metadata=payload_metadata,
    )


def write_decoded_ovoxel_texture_npz(
    path: str | Path,
    coordinates: mx.array,
    attributes: mx.array,
    *,
    spatial_shape: tuple[int, int, int],
    batch_size: int,
    decode_resolution: int | None,
    voxel_size: float | None,
    metadata: dict[str, Any] | None = None,
) -> DecodedOVoxelTextureArtifact:
    """Write decoded PBR voxel coordinates and attributes to an NPZ bundle."""

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
    _write_npz_atomic(
        output,
        {
            "coordinates": coordinates_array.astype(np.int32, copy=False),
            "attributes": attributes_array.astype(np.float32, copy=False),
            "spatial_shape": np.array(normalized_spatial_shape, dtype=np.int32),
            "batch_size": np.array(int(batch_size), dtype=np.int32),
            "decode_resolution": np.array(
                -1 if decode_resolution is None else int(decode_resolution),
                dtype=np.int32,
            ),
            "voxel_size": np.array(
                np.nan if voxel_size is None else float(voxel_size),
                dtype=np.float32,
            ),
            "metadata_json": json.dumps(payload_metadata, sort_keys=True, default=str),
        },
    )
    return DecodedOVoxelTextureArtifact(
        path=output,
        coordinates_shape=tuple(int(dim) for dim in coordinates_array.shape),
        attributes_shape=tuple(int(dim) for dim in attributes_array.shape),
        spatial_shape=normalized_spatial_shape,
        batch_size=int(batch_size),
        decode_resolution=int(decode_resolution) if decode_resolution is not None else None,
        voxel_size=float(voxel_size) if voxel_size is not None else None,
        metadata=payload_metadata,
    )


def _array(value: mx.array | np.ndarray) -> np.ndarray:
    return np.asarray(value)


def _write_npz_atomic(output: Path, payload: dict[str, Any]) -> None:
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "DecodedOVoxelShapeArtifact",
    "DecodedOVoxelTextureArtifact",
    "write_decoded_ovoxel_shape_npz",
    "write_decoded_ovoxel_texture_npz",
]
