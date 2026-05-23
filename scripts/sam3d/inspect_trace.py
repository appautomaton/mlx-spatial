#!/usr/bin/env python3
"""Print a compact SAM3D reconstruction trace summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="SAM3D trace.json path")
    args = parser.parse_args(argv)

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    sparse = metadata.get("sparse_structure", {})
    gaussian = metadata.get("gaussian_decoder", {})

    print(f"trace={args.trace}")
    print(f"blocker={_blocker(payload)}")
    print(f"completed={','.join(payload.get('completed_stages', []))}")
    print(f"output={payload.get('output_path')}")
    _print_sparse(sparse)
    _print_gaussian(gaussian)
    return 0


def _blocker(payload: dict[str, object]) -> str:
    blocker = payload.get("blocker")
    if not isinstance(blocker, dict):
        return "none"
    stage = blocker.get("stage", "unknown")
    reason = blocker.get("reason", "unknown")
    return f"{stage}: {reason}"


def _print_sparse(sparse: object) -> None:
    if not isinstance(sparse, dict):
        return
    decoder = sparse.get("decoder", {})
    geometry = sparse.get("geometry_quality", {})
    occupancy = sparse.get("occupancy_quality", {})
    if isinstance(decoder, dict):
        print(f"ss_coords={decoder.get('coords_count')}")
        print(f"ss_occupancy_positive_fraction={decoder.get('occupancy_positive_fraction')}")
    if isinstance(geometry, dict):
        print(f"ss_axis_range={geometry.get('axis_range')}")
        print(f"ss_geometry_status={geometry.get('status')}")
    if isinstance(occupancy, dict):
        print(f"ss_occupancy_status={occupancy.get('status')}")


def _print_gaussian(gaussian: object) -> None:
    if not isinstance(gaussian, dict):
        return
    fields = gaussian.get("fields", {})
    geometry = gaussian.get("geometry_quality", {})
    opacity = gaussian.get("opacity_quality", {})
    if isinstance(fields, dict):
        print(f"gaussian_count={fields.get('gaussian_count')}")
    if isinstance(geometry, dict):
        print(f"gaussian_xyz_range={geometry.get('axis_range')}")
        print(f"gaussian_geometry_status={geometry.get('status')}")
    if isinstance(opacity, dict):
        print(f"gaussian_alpha_gt_0_5_fraction={opacity.get('alpha_gt_0_5_fraction')}")
        print(f"gaussian_opacity_status={opacity.get('status')}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
