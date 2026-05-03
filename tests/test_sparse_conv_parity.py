import os
import sys
from pathlib import Path

import pytest

from mlx_spatial.sparse_conv import kernel_offsets

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


def test_kernel_offsets_matches_local_torch_reference():
    torch = _load_local_torch()

    expected = torch.tensor(
        [[dz, dy, dx] for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
        dtype=torch.int32,
    )

    assert kernel_offsets((3, 3, 3)).tolist() == expected.tolist()
