"""MapAnything MLX prefix smoke pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .mapanything_assets import MAPANYTHING_DEFAULT_ROOT, inspect_mapanything_model_assets
from .mapanything_model import (
    MapAnythingEncoderPrefixOutput,
    load_mapanything_encoder_prefix_weights,
    mapanything_encoder_prefix_config_from_model_config,
    run_mapanything_encoder_prefix,
)
from .mapanything_parity import mapanything_parity_trace_metadata
from .mapanything_preprocess import MapAnythingPreprocessedInput, preprocess_mapanything_images


MAPANYTHING_PREFIX_PARITY_ATOL = 1.2e-2
MAPANYTHING_PREFIX_PARITY_RTOL = 1e-3


@dataclass(frozen=True)
class MapAnythingPrefixBlocker:
    """Structured blocker for the current MapAnything prefix smoke path."""

    stage: str
    operation: str
    reason: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MapAnythingTensorSummary:
    """Compact tensor metadata for smoke results."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    mean: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class MapAnythingPrefixTrace:
    """Trace for the implemented MapAnything MLX prefix boundary."""

    completed_stages: tuple[str, ...]
    model_root: Path
    input_path: Path
    frame_count: int
    target_size: tuple[int, int] | None
    patch_grid: tuple[int, int] | None
    tensor_summaries: tuple[MapAnythingTensorSummary, ...]
    metadata: dict[str, object] = field(default_factory=dict)
    blocker: MapAnythingPrefixBlocker | None = None

    @property
    def ready(self) -> bool:
        return self.blocker is None


@dataclass(frozen=True)
class MapAnythingPrefixResult:
    """Result wrapper for the MapAnything prefix smoke pipeline."""

    trace: MapAnythingPrefixTrace
    preprocessed: MapAnythingPreprocessedInput | None = None
    prefix: MapAnythingEncoderPrefixOutput | None = None

    @property
    def ready(self) -> bool:
        return self.trace.ready and self.prefix is not None


class MapAnythingPrefixPipeline:
    """Validate assets, preprocess images, and run the MLX encoder prefix."""

    def __init__(self, root: str | Path = MAPANYTHING_DEFAULT_ROOT):
        self.root = Path(root)

    def run(
        self,
        input_path: str | Path,
        *,
        resize_mode: str = "fixed_mapping",
        size: int | tuple[int, int] | None = None,
        stride: int = 1,
    ) -> MapAnythingPrefixResult:
        completed: list[str] = []
        source = Path(input_path)

        inspection = inspect_mapanything_model_assets(self.root, required_groups=("encoder",))
        if inspection.blocker is not None or inspection.config is None:
            blocker = _blocker_from_asset_inspection(inspection.blocker)
            return MapAnythingPrefixResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    blocker=blocker,
                )
            )
        completed.append("asset-config-validation")
        config = mapanything_encoder_prefix_config_from_model_config(inspection.config)

        try:
            preprocessed = preprocess_mapanything_images(
                source,
                resize_mode=resize_mode,
                size=size,
                patch_size=config.patch_size,
                stride=stride,
            )
        except (FileNotFoundError, ValueError, OSError) as error:
            return MapAnythingPrefixResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    blocker=MapAnythingPrefixBlocker(
                        stage="image-preprocessing",
                        operation="prepare MapAnything image-only views",
                        reason=str(error),
                    ),
                )
            )
        completed.append("image-preprocessing")

        try:
            weights = load_mapanything_encoder_prefix_weights(self.root, config=config)
        except (FileNotFoundError, ValueError, OSError) as error:
            return MapAnythingPrefixResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingPrefixBlocker(
                        stage="checkpoint-loading",
                        operation="load MapAnything encoder-prefix tensors",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("checkpoint-loading")

        images = mx.concatenate([view.img for view in preprocessed.views], axis=0)
        try:
            prefix = run_mapanything_encoder_prefix(images, weights, config=config)
            mx.eval(prefix.patch_embeddings, prefix.tokens_with_position, prefix.block0)
        except (RuntimeError, ValueError) as error:
            return MapAnythingPrefixResult(
                trace=_trace(
                    completed,
                    root=self.root,
                    input_path=source,
                    target_size=preprocessed.target_size,
                    blocker=MapAnythingPrefixBlocker(
                        stage="encoder-prefix",
                        operation="run MLX MapAnything encoder prefix",
                        reason=str(error),
                    ),
                ),
                preprocessed=preprocessed,
            )
        completed.append("encoder-prefix")

        summaries = tuple(
            _tensor_summary(name, tensor)
            for name, tensor in (
                ("encoder.patch_embed", prefix.patch_embeddings),
                ("encoder.tokens", prefix.tokens_with_position),
                ("encoder.block0", prefix.block0),
            )
        )
        return MapAnythingPrefixResult(
            trace=_trace(
                completed,
                root=self.root,
                input_path=source,
                target_size=preprocessed.target_size,
                patch_grid=prefix.patch_grid,
                frame_count=preprocessed.frame_count,
                summaries=summaries,
                metadata={
                    "runtime_depends_on_torch": False,
                    "implemented_boundary": "encoder-prefix",
                    "parity": {
                        **mapanything_parity_trace_metadata(),
                        "documented_atol": MAPANYTHING_PREFIX_PARITY_ATOL,
                        "documented_rtol": MAPANYTHING_PREFIX_PARITY_RTOL,
                        "checked_tensors": [
                            "encoder.patch_embed",
                            "encoder.tokens",
                            "encoder.block0",
                        ],
                    },
                },
            ),
            preprocessed=preprocessed,
            prefix=prefix,
        )


def _trace(
    completed: list[str],
    *,
    root: Path,
    input_path: Path,
    target_size: tuple[int, int] | None = None,
    patch_grid: tuple[int, int] | None = None,
    frame_count: int = 0,
    summaries: tuple[MapAnythingTensorSummary, ...] = (),
    metadata: dict[str, object] | None = None,
    blocker: MapAnythingPrefixBlocker | None = None,
) -> MapAnythingPrefixTrace:
    return MapAnythingPrefixTrace(
        completed_stages=tuple(completed),
        model_root=root,
        input_path=input_path,
        frame_count=frame_count,
        target_size=target_size,
        patch_grid=patch_grid,
        tensor_summaries=summaries,
        metadata=dict(metadata or {"runtime_depends_on_torch": False}),
        blocker=blocker,
    )


def _blocker_from_asset_inspection(blocker: object | None) -> MapAnythingPrefixBlocker:
    if blocker is None:
        return MapAnythingPrefixBlocker(
            stage="asset-validation",
            operation="validate MapAnything encoder-prefix assets",
            reason="MapAnything asset inspection returned no config",
        )
    return MapAnythingPrefixBlocker(
        stage=str(getattr(blocker, "stage", "asset-validation")),
        operation=str(getattr(blocker, "operation", "validate MapAnything encoder-prefix assets")),
        reason=str(getattr(blocker, "reason", "unknown MapAnything asset blocker")),
        metadata=dict(getattr(blocker, "metadata", {})),
    )


def _tensor_summary(name: str, tensor: mx.array) -> MapAnythingTensorSummary:
    mean = mx.mean(tensor.astype(mx.float32))
    minimum = mx.min(tensor.astype(mx.float32))
    maximum = mx.max(tensor.astype(mx.float32))
    mx.eval(mean, minimum, maximum)
    return MapAnythingTensorSummary(
        name=name,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=str(tensor.dtype).removeprefix("mlx.core."),
        mean=float(np.asarray(mean)),
        minimum=float(np.asarray(minimum)),
        maximum=float(np.asarray(maximum)),
    )
