#!/usr/bin/env python3
"""Capture Pixal3D PyTorch reference metadata for dev-only parity work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

from mlx_spatial.pixal3d_parity import (  # noqa: E402
    PIXAL3D_TORCH_PARITY_ENV,
    require_pixal3d_torch_parity_enabled,
    write_pixal3d_parity_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="local TencentARC/Pixal3D weight root")
    parser.add_argument("image", type=Path, help="input image for reference capture")
    parser.add_argument("--output", type=Path, required=True, help="output .npz reference bundle")
    parser.add_argument("--vendor-root", type=Path, default=ROOT / "vendors/Pixal3D")
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="write config/input metadata without running the heavy PyTorch pipeline",
    )
    try:
        require_pixal3d_torch_parity_enabled()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"Pixal3D root not found: {args.root}", file=sys.stderr)
        return 1
    if not args.image.is_file():
        print(f"input image not found: {args.image}", file=sys.stderr)
        return 1

    pipeline_path = args.root / "pipeline.json"
    metadata: dict[str, object] = {
        "root": str(args.root),
        "image": str(args.image),
        "vendor_root": str(args.vendor_root),
        "device": args.device,
        "torch_reference_env": PIXAL3D_TORCH_PARITY_ENV,
        "metadata_only": args.metadata_only,
    }
    if pipeline_path.is_file():
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        pipeline_args = pipeline.get("args", {})
        metadata["default_pipeline_type"] = pipeline_args.get("default_pipeline_type")
        metadata["model_keys"] = sorted((pipeline_args.get("models") or {}).keys())

    # Slice 1 only establishes the guarded boundary. Heavy tensor capture is added
    # by later parity slices once the MLX projection/model boundary is ready.
    write_pixal3d_parity_bundle(
        args.output,
        {},
        metadata={
            **metadata,
            "runtime_depends_on_torch": False,
            "capture_boundary": "metadata",
        },
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
