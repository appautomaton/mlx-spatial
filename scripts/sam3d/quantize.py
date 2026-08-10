#!/usr/bin/env python3
"""Create a same-layout selectively quantized SAM3D MLX weights root."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))


if __name__ == "__main__":
    from mlx_spatial.sam3d_quantization import main

    raise SystemExit(main())
