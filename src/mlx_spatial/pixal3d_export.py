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


def _array(value: mx.array) -> np.ndarray:
    return np.array(value)
