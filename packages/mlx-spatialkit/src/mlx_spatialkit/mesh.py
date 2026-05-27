"""Thin Python entry points for native mesh functionality."""

from __future__ import annotations

from ._native import backend_info, validate_pixal3d_shape_fields

__all__ = ["backend_info", "validate_pixal3d_shape_fields"]
