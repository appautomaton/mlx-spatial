from __future__ import annotations

from pathlib import Path

from mlx_spatial import spatialkit as mlx_spatialkit
import mlx_spatial.spatialkit._native as _native


def test_backend_info_reports_native_backend() -> None:
    info = mlx_spatialkit.backend_info()

    assert info["native"] is True
    assert "metal_available" in info
    assert "metal_device" in info


def test_texture_bake_metal_source_is_packaged() -> None:
    resource = Path(_native.__file__).with_name("texture_bake.metal")

    assert resource.is_file(), resource
    assert resource.read_bytes()
