from __future__ import annotations

import mlx_spatialkit


def test_backend_info_reports_native_backend() -> None:
    info = mlx_spatialkit.backend_info()

    assert info["native"] is True
    assert "metal_available" in info
    assert "metal_device" in info
