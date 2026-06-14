#!/usr/bin/env python3
"""Convert Valeo NAF PyTorch release weights to runtime safetensors.

This is a setup utility for local development. The `mlx_spatial` runtime loads
the resulting safetensors file and does not import Torch.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

NAF_RELEASE_URL = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="existing naf_release.pth path; downloaded when omitted")
    parser.add_argument("--output", type=Path, default=Path("weights/naf/naf_release.safetensors"))
    args = parser.parse_args()

    source = args.input
    with tempfile.TemporaryDirectory(prefix="mlx-spatial-naf-") as tmp:
        if source is None:
            source = Path(tmp) / "naf_release.pth"
            urllib.request.urlretrieve(NAF_RELEASE_URL, source)
        tensors = _load_torch_state_dict(source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, args.output)
    print(args.output)
    return 0


def _load_torch_state_dict(path: Path) -> dict[str, np.ndarray]:
    import torch

    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"expected NAF checkpoint state dict, got {type(state).__name__}")
    tensors: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if not hasattr(value, "detach"):
            continue
        tensors[str(name)] = value.detach().cpu().numpy()
    if not tensors:
        raise ValueError(f"no tensors found in NAF checkpoint: {path}")
    return tensors


if __name__ == "__main__":
    raise SystemExit(main())
