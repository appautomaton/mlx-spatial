import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_spatial.sparse_conv import weighted_sparse_conv

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


def test_weighted_sparse_conv_matches_local_torch_reference():
    torch = _load_local_torch()

    source_features = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    map_rows = [[0, 0, 0], [1, 1, 1], [0, 2, 1]]
    kernel_weights = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ]

    mlx_output = weighted_sparse_conv(
        mx.array(source_features, dtype=mx.float32),
        mx.array(map_rows, dtype=mx.int32),
        mx.array(kernel_weights, dtype=mx.float32),
        target_count=3,
    )

    torch_features = torch.tensor(source_features, dtype=torch.float32)
    torch_rows = torch.tensor(map_rows, dtype=torch.int64)
    torch_weights = torch.tensor(kernel_weights, dtype=torch.float32)
    torch_output = torch.zeros((3, 2), dtype=torch.float32)
    for target_index, source_index, kernel_index in torch_rows.tolist():
        torch_output[target_index] += torch_features[source_index] @ torch_weights[kernel_index]

    assert mlx_output.tolist() == torch_output.tolist()
