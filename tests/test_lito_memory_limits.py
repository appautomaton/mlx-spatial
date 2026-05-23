from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlx_spatial import lito_inference
from mlx_spatial.lito_inference import (
    LITO_HARD_MEMORY_LIMIT_GB,
    LITO_SOFT_MEMORY_LIMIT_GB,
    LitoInferencePipeline,
    LitoMemoryLimitExceeded,
)


def test_soft_threshold_warning_at_90gb(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr(
        lito_inference,
        "_memory_bytes",
        lambda kind: int((LITO_SOFT_MEMORY_LIMIT_GB + 0.25) * 1024**3) if kind in {"active", "peak"} else 0,
    )

    observed = lito_inference._check_memory("synthetic")

    assert observed >= LITO_SOFT_MEMORY_LIMIT_GB
    assert "soft threshold" in caplog.text


def test_hard_ceiling_raises_at_100gb(monkeypatch):
    monkeypatch.setattr(
        lito_inference,
        "_memory_bytes",
        lambda kind: int((LITO_HARD_MEMORY_LIMIT_GB + 0.25) * 1024**3) if kind in {"active", "peak"} else 0,
    )

    with pytest.raises(LitoMemoryLimitExceeded, match="exceeded"):
        lito_inference._check_memory("synthetic")


def test_pipeline_aborts_cleanly_when_memory_ceiling_is_reached(tmp_path, monkeypatch):
    image = _write_synthetic_image(tmp_path / "input.png")
    calls = {"count": 0}

    def fake_memory_bytes(kind: str) -> int:
        if kind not in {"active", "peak"}:
            return 0
        calls["count"] += 1
        return int((LITO_HARD_MEMORY_LIMIT_GB + 1.0) * 1024**3)

    monkeypatch.setattr(lito_inference, "_memory_bytes", fake_memory_bytes)

    with pytest.raises(LitoMemoryLimitExceeded):
        LitoInferencePipeline(
            tmp_path / "weights",
            memory_profile="safe",
            source_contract_smoke=True,
        ).generate(
            image,
            output_path=tmp_path / "out.ply",
            resolution=24,
            render_size=12,
        )

    assert calls["count"] > 0
    assert not (tmp_path / "out.ply").exists()


def _write_synthetic_image(path: Path) -> Path:
    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., :3] = (80, 120, 200)
    rgba[4:28, 4:28, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path
