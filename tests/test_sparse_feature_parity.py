import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_spatial.sparse_conv import gather_sparse_features, scatter_sparse_features

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


def test_sparse_feature_gather_and_scatter_match_local_torch_reference():
    torch = _load_local_torch()

    source_features = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
    map_rows = [[0, 2, 4], [1, 0, 5], [0, 1, 6], [2, 2, 7]]
    mlx_rows = mx.array(map_rows, dtype=mx.int32)

    mlx_gathered = gather_sparse_features(mx.array(source_features, dtype=mx.float32), mlx_rows)
    mlx_scattered = scatter_sparse_features(mlx_gathered, mlx_rows, target_count=4)

    torch_features = torch.tensor(source_features, dtype=torch.float32)
    torch_rows = torch.tensor(map_rows, dtype=torch.int64)
    torch_gathered = torch_features[torch_rows[:, 1]]
    torch_scattered = torch.zeros((4, 2), dtype=torch.float32)
    for row_index, target_index in enumerate(torch_rows[:, 0].tolist()):
        torch_scattered[target_index] += torch_gathered[row_index]

    assert mlx_gathered.tolist() == torch_gathered.tolist()
    assert mlx_scattered.tolist() == torch_scattered.tolist()
