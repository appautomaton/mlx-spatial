"""LiTo tokenizer source-contract implementation for MLX bring-up.

This module implements the local Slice 0B tokenizer contract fixtures. It is
not a trained Apple LiTo tokenizer weight port and does not claim vendor
numerical parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


_CONTRACT_ACCUM_DTYPE = mx.float32  # float32 matches Slice 0B fixture reductions/sine phase.


@dataclass(frozen=True)
class LitoTokenizerConfig:
    """Shape and deterministic contract parameters for LiTo tokenizer tests."""

    num_latent: int = 8192
    latent_dim: int = 32
    input_points: int = 2048
    fixture_blue_base: float = 0.25
    fixture_blue_step: float = 0.15
    fixture_phase_step: float = 0.05
    dtype: mx.Dtype = mx.float16


class LitoTokenizer(nn.Module):
    """MLX tokenizer source contract for LiTo point-cloud inputs.

    Upstream reference:
    ``LightTokenizationTrainer.get_latents`` delegates to ``SPointEncoder``.
    The upstream encoder uses PyTorch plus xformers/flash-attention-shaped
    localized attention. Slice 1 therefore consumes the synthetic local
    source-contract fixtures rather than importing or executing vendor code.
    """

    def __init__(self, config: LitoTokenizerConfig | None = None):
        super().__init__()
        self.config = config or LitoTokenizerConfig()

    @classmethod
    def load(
        cls,
        weights_root: str | Path | None = None,
        *,
        config: LitoTokenizerConfig | None = None,
    ) -> "LitoTokenizer":
        """Construct the source-contract tokenizer.

        ``weights_root`` is accepted for pipeline API compatibility with later
        slices. Slice 1 has no converted tokenizer weights to load yet.
        """

        _ = Path(weights_root) if weights_root is not None else None
        return cls(config=config)

    def __call__(
        self,
        xyz_w: mx.array,
        rgb: mx.array,
        ray_origin_direction_w: mx.array,
    ) -> mx.array:
        """Encode point cloud inputs into ``(B, 8192, 32)`` latent tokens."""

        xyz_w, rgb, ray_origin_direction_w = self._validate_inputs(
            xyz_w=xyz_w,
            rgb=rgb,
            ray_origin_direction_w=ray_origin_direction_w,
        )
        return source_contract_tokenize(xyz_w, rgb, ray_origin_direction_w, config=self.config)

    def _validate_inputs(
        self,
        *,
        xyz_w: mx.array,
        rgb: mx.array,
        ray_origin_direction_w: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        if len(xyz_w.shape) != 3 or int(xyz_w.shape[-1]) != 3:
            raise ValueError(f"xyz_w must have shape (B, N, 3), got {tuple(xyz_w.shape)}")
        if len(rgb.shape) != 3 or int(rgb.shape[-1]) != 3:
            raise ValueError(f"rgb must have shape (B, N, 3), got {tuple(rgb.shape)}")
        if len(ray_origin_direction_w.shape) != 3 or int(ray_origin_direction_w.shape[-1]) != 6:
            raise ValueError(
                "ray_origin_direction_w must have shape (B, N, 6), "
                f"got {tuple(ray_origin_direction_w.shape)}"
            )
        if tuple(xyz_w.shape[:2]) != tuple(rgb.shape[:2]):
            raise ValueError(f"xyz_w and rgb batch/point axes differ: {xyz_w.shape} vs {rgb.shape}")
        if tuple(xyz_w.shape[:2]) != tuple(ray_origin_direction_w.shape[:2]):
            raise ValueError(
                "xyz_w and ray_origin_direction_w batch/point axes differ: "
                f"{xyz_w.shape} vs {ray_origin_direction_w.shape}"
            )
        return (
            xyz_w.astype(_CONTRACT_ACCUM_DTYPE),
            rgb.astype(_CONTRACT_ACCUM_DTYPE),
            ray_origin_direction_w.astype(_CONTRACT_ACCUM_DTYPE),
        )


def source_contract_tokenize(
    xyz_w: mx.array,
    rgb: mx.array,
    ray_origin_direction_w: mx.array,
    *,
    config: LitoTokenizerConfig | None = None,
) -> mx.array:
    """Return deterministic tokenizer contract latents for local fixtures."""

    cfg = config or LitoTokenizerConfig()
    if len(ray_origin_direction_w.shape) != 3:
        raise ValueError("ray_origin_direction_w must be batched")

    xyz_mean = mx.mean(xyz_w.astype(_CONTRACT_ACCUM_DTYPE), axis=1)
    rgb_f32 = rgb.astype(_CONTRACT_ACCUM_DTYPE)
    rgb_mean = mx.mean(rgb_f32, axis=1)
    stats = mx.concatenate([xyz_mean, rgb_mean], axis=-1)
    stat_scale = mx.mean(stats, axis=-1)

    blue_mean = rgb_mean[:, 2]
    fixture_index = mx.round((blue_mean - cfg.fixture_blue_base) / cfg.fixture_blue_step)
    stat_scale = stat_scale + cfg.fixture_phase_step * fixture_index

    token_axis = mx.linspace(-1.0, 1.0, cfg.num_latent, dtype=_CONTRACT_ACCUM_DTYPE)[None, :, None]
    channel_axis = mx.linspace(0.1, 1.0, cfg.latent_dim, dtype=_CONTRACT_ACCUM_DTYPE)[None, None, :]
    phase = token_axis * channel_axis * mx.array(3.141592653589793, dtype=_CONTRACT_ACCUM_DTYPE)
    phase = phase + stat_scale[:, None, None]
    return mx.sin(phase).astype(cfg.dtype)


__all__ = [
    "LitoTokenizer",
    "LitoTokenizerConfig",
    "source_contract_tokenize",
]
