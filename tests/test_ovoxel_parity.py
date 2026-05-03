import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_spatial.ovoxel import flatten_coordinates

LOCAL_PYTORCH = Path("/Users/ac/dev/ai/ai-frameworks/pytorch")

pytestmark = pytest.mark.torch_parity


def _load_local_torch():
    if os.environ.get("MLX_SPATIAL_RUN_TORCH_PARITY") != "1":
        pytest.skip("set MLX_SPATIAL_RUN_TORCH_PARITY=1 to run local PyTorch parity checks")
    if not LOCAL_PYTORCH.exists():
        pytest.skip(f"local PyTorch checkout not found: {LOCAL_PYTORCH}")

    sys.path.insert(0, str(LOCAL_PYTORCH))
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional local dependency path
        pytest.skip(f"local PyTorch import failed: {exc}")
    return torch


def test_flatten_coordinates_matches_local_torch_row_major_reference():
    torch = _load_local_torch()

    coords = [[0, 0, 0], [0, 1, 2], [1, 2, 3]]
    shape = (2, 3, 4)
    mlx_flat = flatten_coordinates(mx.array(coords, dtype=mx.int32), shape).tolist()

    torch_coords = torch.tensor(coords, dtype=torch.int64)
    strides = torch.tensor([12, 4, 1], dtype=torch.int64)
    torch_flat = (torch_coords * strides).sum(dim=-1).tolist()

    assert mlx_flat == torch_flat
