"""Local RMBG-2.0 checkpoint inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn

from .checkpoint import CheckpointTensorInfo, inspect_checkpoint, load_checkpoint_tensors
from .model_assets import RMBG2_ASSETS
from .trellis2 import validate_rmbg2_assets


@dataclass(frozen=True)
class Rmbg2TensorProbe:
    """Loaded RMBG-2.0 tensor and its local checkpoint context."""

    checkpoint_path: str
    name: str
    array: mx.array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def dtype(self) -> str:
        return str(self.array.dtype).removeprefix("mlx.core.")


@dataclass(frozen=True)
class Rmbg2KeyInventory:
    """Small deterministic summary of a local RMBG-2.0 safetensors key set."""

    checkpoint_path: Path
    tensor_count: int
    top_level_prefixes: tuple[str, ...]
    sample_keys: tuple[str, ...]


@dataclass(frozen=True)
class Rmbg2PortBlocker:
    stage: str
    operation: str
    reference: str
    reason: str
    next_slice: str


@dataclass(frozen=True)
class Rmbg2PortAssessment:
    root: Path
    tensor_count: int
    top_level_prefixes: tuple[str, ...]
    blocker: Rmbg2PortBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.blocker is None


def inspect_rmbg2_checkpoint(
    root: str | Path = RMBG2_ASSETS.root_hint,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> tuple[CheckpointTensorInfo, ...]:
    """Inspect local RMBG-2.0 safetensors metadata without loading full tensors."""

    return inspect_checkpoint(_checkpoint_path(root), names=names, prefixes=prefixes)


def load_rmbg2_tensors(
    root: str | Path = RMBG2_ASSETS.root_hint,
    *,
    names: Iterable[str] | None = None,
    prefixes: Iterable[str] | None = None,
) -> tuple[Rmbg2TensorProbe, ...]:
    """Load selected RMBG-2.0 tensors from local safetensors as MLX arrays."""

    tensors = load_checkpoint_tensors(_checkpoint_path(root), names=names, prefixes=prefixes)
    return tuple(
        Rmbg2TensorProbe(
            checkpoint_path="model.safetensors",
            name=name,
            array=tensors[name],
        )
        for name in sorted(tensors)
    )


def inspect_rmbg2_key_inventory(root: str | Path = RMBG2_ASSETS.root_hint) -> Rmbg2KeyInventory:
    """Return a compact key inventory for planning the MLX BiRefNet mapping."""

    infos = inspect_rmbg2_checkpoint(root)
    prefixes = sorted({info.name.split(".", 1)[0] for info in infos})
    sample_keys = tuple(info.name for info in infos[:20])
    return Rmbg2KeyInventory(
        checkpoint_path=_checkpoint_path(root),
        tensor_count=len(infos),
        top_level_prefixes=tuple(prefixes),
        sample_keys=sample_keys,
    )


def assess_rmbg2_mlx_port(root: str | Path = RMBG2_ASSETS.root_hint) -> Rmbg2PortAssessment:
    """Assess whether local RMBG-2.0 assets can run through the current MLX port."""

    root_path = Path(root)
    inventory = inspect_rmbg2_key_inventory(root_path)
    architecture_path = root_path / "birefnet.py"
    architecture = architecture_path.read_text(encoding="utf-8")

    if "deform_conv2d" in architecture and not hasattr(nn, "DeformConv2d"):
        return Rmbg2PortAssessment(
            root=root_path,
            tensor_count=inventory.tensor_count,
            top_level_prefixes=inventory.top_level_prefixes,
            blocker=Rmbg2PortBlocker(
                stage="image-preprocessing-background",
                operation="MLX BiRefNet deformable convolution",
                reference="weights/rmbg2/birefnet.py:1230-1295",
                reason="RMBG-2.0 BiRefNet imports torchvision.ops.deform_conv2d, but mlx.nn has no DeformConv2d implementation",
                next_slice="implement or replace deformable convolution for the RMBG-2.0 ASPPDeformable decoder path",
            ),
        )

    required_prefixes = {"bb", "decoder", "squeeze_module"}
    missing_prefixes = sorted(required_prefixes.difference(inventory.top_level_prefixes))
    if missing_prefixes:
        return Rmbg2PortAssessment(
            root=root_path,
            tensor_count=inventory.tensor_count,
            top_level_prefixes=inventory.top_level_prefixes,
            blocker=Rmbg2PortBlocker(
                stage="image-preprocessing-background",
                operation="MLX BiRefNet checkpoint key mapping",
                reference=str(inventory.checkpoint_path),
                reason=f"RMBG-2.0 checkpoint is missing expected key prefixes: {missing_prefixes}",
                next_slice="map the actual RMBG checkpoint key structure before implementing model construction",
            ),
        )

    return Rmbg2PortAssessment(
        root=root_path,
        tensor_count=inventory.tensor_count,
        top_level_prefixes=inventory.top_level_prefixes,
    )


def _checkpoint_path(root: str | Path) -> Path:
    validation = validate_rmbg2_assets(root)
    if "model.safetensors" in validation.missing:
        raise FileNotFoundError(f"RMBG-2.0 checkpoint file not found: {validation.root / 'model.safetensors'}")
    return validation.root / "model.safetensors"
