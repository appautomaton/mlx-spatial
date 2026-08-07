"""Decoded O-Voxel artifact contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..coordinate_systems import (
    GLTF_Y_UP,
    SUPPORTED_SOURCE_COORDINATE_SYSTEMS,
    TRELLIS_Z_UP,
)
from ._native import (
    validate_pixal3d_shape_fields as validate_ovoxel_shape_fields,
    validate_pixal3d_texture_attributes as validate_ovoxel_texture_attributes,
)


@dataclass(frozen=True)
class DecodedOVoxelInputs:
    """Decoded O-Voxel arrays validated at the native boundary."""

    shape_coordinates: np.ndarray
    shape_fields: np.ndarray
    texture_coordinates: np.ndarray
    texture_attributes: np.ndarray
    contracts: dict[str, Any]
    shape_metadata: dict[str, Any]
    texture_metadata: dict[str, Any]
    texture_spatial_shape: tuple[int, int, int] | None
    texture_batch_size: int | None
    texture_decode_resolution: int | None
    texture_voxel_size: float | None


def validate_decoded_ovoxel(
    shape_coordinates: np.ndarray,
    shape_fields: np.ndarray,
    texture_coordinates: np.ndarray,
    texture_attributes: np.ndarray,
) -> dict[str, Any]:
    """Validate decoded O-Voxel arrays through native contract checks."""

    shape_contract = validate_ovoxel_shape_fields(shape_coordinates, shape_fields)
    texture_contract = validate_ovoxel_texture_attributes(texture_coordinates, texture_attributes)
    return {"shape": shape_contract, "texture": texture_contract}


def load_decoded_ovoxel_npz(
    shape_decoder_path: str | Path,
    texture_decoder_path: str | Path,
) -> DecodedOVoxelInputs:
    """Load decoded O-Voxel NPZ files and validate their native contracts."""

    shape_path = Path(shape_decoder_path)
    texture_path = Path(texture_decoder_path)
    with np.load(shape_path) as shape_payload:
        shape_coordinates = _load_npz_array(shape_payload, "coordinates", shape_path)
        shape_fields = _load_npz_array(shape_payload, "fields", shape_path)
        shape_metadata = _load_npz_metadata(shape_payload, shape_path)
    with np.load(texture_path) as texture_payload:
        texture_coordinates = _load_npz_array(texture_payload, "coordinates", texture_path)
        texture_attributes = _load_npz_array(texture_payload, "attributes", texture_path)
        texture_metadata = _load_npz_metadata(texture_payload, texture_path)
        texture_spatial_shape = (
            tuple(int(dim) for dim in _load_npz_array(texture_payload, "spatial_shape", texture_path))
            if "spatial_shape" in texture_payload.files
            else None
        )
        texture_batch_size = _load_optional_scalar(texture_payload, "batch_size", texture_path)
        texture_decode_resolution = _load_optional_scalar(texture_payload, "decode_resolution", texture_path)
        texture_voxel_size = _load_optional_scalar(texture_payload, "voxel_size", texture_path)
    contracts = validate_decoded_ovoxel(
        shape_coordinates,
        shape_fields,
        texture_coordinates,
        texture_attributes,
    )
    return DecodedOVoxelInputs(
        shape_coordinates=shape_coordinates,
        shape_fields=shape_fields,
        texture_coordinates=texture_coordinates,
        texture_attributes=texture_attributes,
        contracts=contracts,
        shape_metadata=shape_metadata,
        texture_metadata=texture_metadata,
        texture_spatial_shape=texture_spatial_shape,
        texture_batch_size=int(texture_batch_size) if texture_batch_size is not None else None,
        texture_decode_resolution=(
            None
            if texture_decode_resolution is None or int(texture_decode_resolution) < 0
            else int(texture_decode_resolution)
        ),
        texture_voxel_size=(
            None
            if texture_voxel_size is None or not np.isfinite(float(texture_voxel_size))
            else float(texture_voxel_size)
        ),
    )


def resolve_source_coordinate_system(
    requested: str,
    shape_metadata: dict[str, Any],
    texture_metadata: dict[str, Any],
) -> str:
    """Resolve the decoded model space before glTF serialization."""

    if requested != "auto":
        return requested

    explicit_values = {
        str(metadata["source_coordinate_system"]).strip().lower()
        for metadata in (shape_metadata, texture_metadata)
        if metadata.get("source_coordinate_system")
    }
    if len(explicit_values) > 1:
        raise ValueError(
            "decoded shape and texture artifacts disagree on source_coordinate_system: "
            f"{tuple(sorted(explicit_values))}"
        )
    if explicit_values:
        resolved = next(iter(explicit_values))
        if resolved not in SUPPORTED_SOURCE_COORDINATE_SYSTEMS:
            raise ValueError(f"decoded artifacts declare unsupported source_coordinate_system {resolved!r}")
        return resolved

    source_models = {
        str(value).strip().lower()
        for metadata in (shape_metadata, texture_metadata)
        if (value := metadata.get("model_family") or metadata.get("source_model"))
    }
    if len(source_models) > 1:
        raise ValueError(
            "decoded shape and texture artifacts disagree on model family: "
            f"{tuple(sorted(source_models))}"
        )
    if source_models and next(iter(source_models)).startswith("trellis"):
        return TRELLIS_Z_UP
    return GLTF_Y_UP


def resolve_model_identity(
    shape_metadata: dict[str, Any],
    texture_metadata: dict[str, Any],
) -> dict[str, str]:
    """Resolve stable GLB labels from decoded artifact metadata."""

    declared = {
        str(value).strip().lower()
        for metadata in (shape_metadata, texture_metadata)
        if (value := metadata.get("model_family") or metadata.get("source_model"))
    }
    normalized = {
        "trellis2" if value.startswith("trellis") else "pixal3d" if value.startswith("pixal") else value
        for value in declared
    }
    if len(normalized) > 1:
        raise ValueError(f"decoded shape and texture artifacts disagree on model family: {tuple(sorted(normalized))}")
    family = next(iter(normalized), "ovoxel")
    if family == "trellis2":
        return {"family": family, "label": "TRELLIS.2", "asset_prefix": "TRELLIS2"}
    if family == "pixal3d":
        return {"family": family, "label": "Pixal3D", "asset_prefix": "Pixal3D"}
    return {"family": family, "label": "O-Voxel", "asset_prefix": "OVoxel"}


def _load_npz_array(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in payload.files:
        raise ValueError(f"{path} is missing required array {key!r}")
    return np.asarray(payload[key])


def _load_npz_metadata(payload: np.lib.npyio.NpzFile, path: Path) -> dict[str, Any]:
    if "metadata_json" not in payload.files:
        return {}
    raw = payload["metadata_json"]
    try:
        text = str(raw.item() if raw.shape == () else raw.tolist())
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} contains invalid metadata_json") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} metadata_json must decode to an object")
    return value


def _load_optional_scalar(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> Any:
    if key not in payload.files:
        return None
    value = payload[key]
    if value.shape != ():
        raise ValueError(f"{path} optional scalar {key!r} must be rank 0")
    return value.item()


__all__ = [
    "DecodedOVoxelInputs",
    "load_decoded_ovoxel_npz",
    "resolve_model_identity",
    "resolve_source_coordinate_system",
    "validate_decoded_ovoxel",
    "validate_ovoxel_shape_fields",
    "validate_ovoxel_texture_attributes",
]
