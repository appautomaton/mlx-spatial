"""SAM 3D Objects gaussian decoder packing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Sam3dGaussianDecoderConfig:
    resolution: int = 64
    num_gaussians: int = 32
    voxel_size: float = 1.5
    perturb_offset: bool = True
    minimum_kernel_size: float = 0.0009
    scaling_bias: float = 0.004
    opacity_bias: float = 0.1
    scaling_activation: str = "softplus"
    lr: dict[str, float] = field(
        default_factory=lambda: {
            "_xyz": 1.0,
            "_features_dc": 1.0,
            "_opacity": 1.0,
            "_scaling": 1.0,
            "_rotation": 0.1,
        }
    )

    @property
    def output_channels(self) -> int:
        return self.num_gaussians * (3 + 3 + 3 + 4 + 1)


@dataclass(frozen=True)
class Sam3dGaussianFields:
    xyz: np.ndarray
    features_dc: np.ndarray
    opacity: np.ndarray
    scale: np.ndarray
    rotation: np.ndarray
    metadata: dict[str, object]


def decode_sam3d_gaussian_fields(
    coords: np.ndarray,
    raw_features: np.ndarray,
    *,
    config: Sam3dGaussianDecoderConfig = Sam3dGaussianDecoderConfig(),
) -> Sam3dGaussianFields:
    """Convert SAM3D gaussian decoder channels to official PLY-ready fields."""

    coords_xyz = _coords_xyz(coords)
    features = np.asarray(raw_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != coords_xyz.shape[0]:
        raise ValueError(
            f"SAM3D gaussian raw features must have shape (N, C), got {features.shape} for coords {coords_xyz.shape}"
        )
    if features.shape[1] != config.output_channels:
        raise ValueError(f"SAM3D gaussian raw feature width {features.shape[1]} != {config.output_channels}")

    layout = _gaussian_layout(config.num_gaussians)
    base_xyz = (coords_xyz.astype(np.float32) + 0.5) / float(config.resolution)
    raw_xyz = _slice_feature(features, layout, "_xyz", config).reshape(-1, config.num_gaussians, 3)
    if config.perturb_offset:
        raw_xyz = raw_xyz + sam3d_hammersley_perturbation(config)
    offset = np.tanh(raw_xyz) / float(config.resolution) * 0.5 * float(config.voxel_size)
    xyz = (base_xyz[:, None, :] + offset).reshape(-1, 3)

    features_dc = _slice_feature(features, layout, "_features_dc", config).reshape(-1, config.num_gaussians, 1, 3)
    opacity_hidden = _slice_feature(features, layout, "_opacity", config).reshape(-1, config.num_gaussians, 1)
    scaling_hidden = _slice_feature(features, layout, "_scaling", config).reshape(-1, config.num_gaussians, 3)
    rotation_hidden = _slice_feature(features, layout, "_rotation", config).reshape(-1, config.num_gaussians, 4)

    opacity = (opacity_hidden + _logit(config.opacity_bias)).reshape(-1, 1)
    scale = np.log(_activated_scaling(scaling_hidden, config)).reshape(-1, 3)
    rotation = rotation_hidden.reshape(-1, 4)
    rotation[:, 0] += 1.0

    return Sam3dGaussianFields(
        xyz=(xyz - 0.5).astype(np.float32, copy=False),
        features_dc=features_dc.reshape(-1, 1, 3).astype(np.float32, copy=False),
        opacity=opacity.astype(np.float32, copy=False),
        scale=scale.astype(np.float32, copy=False),
        rotation=rotation.astype(np.float32, copy=False),
        metadata={
            "resolution": int(config.resolution),
            "num_input_coords": int(coords_xyz.shape[0]),
            "num_gaussians_per_coord": int(config.num_gaussians),
            "gaussian_count": int(coords_xyz.shape[0] * config.num_gaussians),
            "output_channels": int(config.output_channels),
            "perturb_offset": bool(config.perturb_offset),
            "scaling_activation": config.scaling_activation,
        },
    )


def sam3d_hammersley_perturbation(config: Sam3dGaussianDecoderConfig) -> np.ndarray:
    """Build the official Hammersley offset perturbation table."""

    samples = np.asarray(
        [_hammersley_sequence(3, index, config.num_gaussians) for index in range(config.num_gaussians)],
        dtype=np.float32,
    )
    perturbation = (samples * 2.0 - 1.0) / float(config.voxel_size)
    return np.arctanh(np.clip(perturbation, -0.999999, 0.999999)).astype(np.float32)


def _gaussian_layout(num_gaussians: int) -> dict[str, tuple[int, int]]:
    widths = {
        "_xyz": num_gaussians * 3,
        "_features_dc": num_gaussians * 3,
        "_scaling": num_gaussians * 3,
        "_rotation": num_gaussians * 4,
        "_opacity": num_gaussians,
    }
    start = 0
    layout: dict[str, tuple[int, int]] = {}
    for name, width in widths.items():
        layout[name] = (start, start + width)
        start += width
    return layout


def _slice_feature(
    features: np.ndarray,
    layout: dict[str, tuple[int, int]],
    name: str,
    config: Sam3dGaussianDecoderConfig,
) -> np.ndarray:
    start, stop = layout[name]
    return features[:, start:stop] * float(config.lr[name])


def _coords_xyz(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in {3, 4}:
        raise ValueError(f"SAM3D coords must have shape (N, 3) or (N, 4), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("SAM3D gaussian decoding requires at least one sparse coordinate")
    return values[:, -3:]


def _activated_scaling(values: np.ndarray, config: Sam3dGaussianDecoderConfig) -> np.ndarray:
    bias = _inverse_scaling_bias(config.scaling_bias, config.scaling_activation)
    if config.scaling_activation == "softplus":
        scaled = _softplus(values + bias)
    elif config.scaling_activation == "exp":
        scaled = np.exp(values + bias)
    else:
        raise ValueError(f"unsupported SAM3D gaussian scaling activation: {config.scaling_activation}")
    return np.sqrt(np.square(scaled) + float(config.minimum_kernel_size) ** 2)


def _inverse_scaling_bias(value: float, activation: str) -> np.float32:
    if activation == "softplus":
        return np.float32(value + np.log(-np.expm1(-value)))
    if activation == "exp":
        return np.float32(np.log(value))
    raise ValueError(f"unsupported SAM3D gaussian scaling activation: {activation}")


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(values, 0.0)


def _logit(value: float) -> np.float32:
    clipped = np.clip(np.float32(value), np.float32(1e-6), np.float32(1.0 - 1e-6))
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def _hammersley_sequence(dim: int, index: int, num_samples: int) -> list[float]:
    return [index / num_samples] + [_radical_inverse(_PRIMES[axis], index) for axis in range(dim - 1)]


def _radical_inverse(base: int, n: int) -> float:
    value = 0.0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        value += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return value


_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)
