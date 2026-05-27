"""Native spatial export primitives for mlx-spatial."""

from __future__ import annotations

from ._native import backend_info, metal_device_available

__all__ = [
    "backend_info",
    "metal_device_available",
]
