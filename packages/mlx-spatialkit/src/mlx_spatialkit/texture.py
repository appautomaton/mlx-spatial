"""Thin Python entry points for native texture functionality."""

from __future__ import annotations

from ._native import metal_device_available, validate_pixal3d_texture_attributes

__all__ = ["metal_device_available", "validate_pixal3d_texture_attributes"]
