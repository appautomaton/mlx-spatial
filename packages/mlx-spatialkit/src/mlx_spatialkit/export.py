"""Thin Python entry points for native export functionality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._native import (
    backend_info,
    validate_pixal3d_shape_fields,
    validate_pixal3d_texture_attributes,
)


@dataclass(frozen=True)
class Pixal3DDecodedInputs:
    """Decoded Pixal3D model-stage arrays validated at the native boundary."""

    shape_coordinates: np.ndarray
    shape_fields: np.ndarray
    texture_coordinates: np.ndarray
    texture_attributes: np.ndarray
    contracts: dict[str, Any]


def validate_pixal3d_decoded(
    shape_coordinates: np.ndarray,
    shape_fields: np.ndarray,
    texture_coordinates: np.ndarray,
    texture_attributes: np.ndarray,
) -> dict[str, Any]:
    """Validate Pixal3D decoded arrays through native contract checks."""

    shape_contract = validate_pixal3d_shape_fields(shape_coordinates, shape_fields)
    texture_contract = validate_pixal3d_texture_attributes(texture_coordinates, texture_attributes)
    return {"shape": shape_contract, "texture": texture_contract}


def _load_npz_array(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> np.ndarray:
    if key not in payload.files:
        raise ValueError(f"{path} is missing required array {key!r}")
    return np.asarray(payload[key])


def load_pixal3d_decoded_npz(
    shape_decoder_path: str | Path,
    texture_decoder_path: str | Path,
) -> Pixal3DDecodedInputs:
    """Load Pixal3D decoded NPZ files and validate their native contracts."""

    shape_path = Path(shape_decoder_path)
    texture_path = Path(texture_decoder_path)
    with np.load(shape_path) as shape_payload:
        shape_coordinates = _load_npz_array(shape_payload, "coordinates", shape_path)
        shape_fields = _load_npz_array(shape_payload, "fields", shape_path)
    with np.load(texture_path) as texture_payload:
        texture_coordinates = _load_npz_array(texture_payload, "coordinates", texture_path)
        texture_attributes = _load_npz_array(texture_payload, "attributes", texture_path)
    contracts = validate_pixal3d_decoded(
        shape_coordinates,
        shape_fields,
        texture_coordinates,
        texture_attributes,
    )
    return Pixal3DDecodedInputs(
        shape_coordinates=shape_coordinates,
        shape_fields=shape_fields,
        texture_coordinates=texture_coordinates,
        texture_attributes=texture_attributes,
        contracts=contracts,
    )

__all__ = [
    "Pixal3DDecodedInputs",
    "backend_info",
    "load_pixal3d_decoded_npz",
    "validate_pixal3d_decoded",
]
