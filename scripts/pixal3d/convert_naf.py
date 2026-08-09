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

from mlx_spatial.safetensors_io import save_safetensors
from mlx_spatial.torch_checkpoint import load_torch_zip_state_dict

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
        tensors = load_torch_zip_state_dict(source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_safetensors(args.output, tensors)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
