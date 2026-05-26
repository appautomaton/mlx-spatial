"""Dev-only Pixal3D Torch reference bundle helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np


PIXAL3D_TORCH_PARITY_ENV = "PIXAL3D_TORCH_REF"
PIXAL3D_PARITY_BUNDLE_VERSION = 1
PIXAL3D_PARITY_REFERENCE_SOURCE = "Pixal3D PyTorch reference"


@dataclass(frozen=True)
class Pixal3DParityReference:
    """Loaded Pixal3D parity reference bundle."""

    path: Path
    metadata: dict[str, object]
    tensors: dict[str, np.ndarray]


def require_pixal3d_torch_parity_enabled() -> None:
    """Require an explicit opt-in before running Torch/vendor Pixal3D reference code."""

    if os.environ.get(PIXAL3D_TORCH_PARITY_ENV) != "1":
        raise RuntimeError(f"set {PIXAL3D_TORCH_PARITY_ENV}=1 to run Pixal3D Torch reference capture")


def write_pixal3d_parity_bundle(
    path: str | Path,
    tensors: dict[str, np.ndarray | mx.array],
    *,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a small `.npz` tensor bundle with JSON metadata."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = {
        name: np.array(value)
        for name, value in tensors.items()
    }
    payload_metadata = {
        "bundle_version": PIXAL3D_PARITY_BUNDLE_VERSION,
        "source": PIXAL3D_PARITY_REFERENCE_SOURCE,
        "runtime_depends_on_torch": False,
        **(metadata or {}),
    }
    np.savez(output, __metadata__=json.dumps(payload_metadata, sort_keys=True), **normalized)
    return output


def load_pixal3d_parity_bundle(path: str | Path) -> Pixal3DParityReference:
    """Load a Pixal3D `.npz` parity reference bundle."""

    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        metadata = json.loads(str(data["__metadata__"].item())) if "__metadata__" in data else {}
        tensors = {name: np.array(data[name]) for name in data.files if name != "__metadata__"}
    return Pixal3DParityReference(path=bundle_path, metadata=metadata, tensors=tensors)


def pixal3d_parity_trace_metadata(*, verified: bool = False) -> dict[str, object]:
    """Return trace metadata for Pixal3D parity status."""

    return {
        "runtime_depends_on_torch": False,
        "numeric_parity_verified": bool(verified),
        "status": "verified" if verified else "unverified",
        "dev_reference_env": PIXAL3D_TORCH_PARITY_ENV,
    }
