"""LiTo CLI and import surface."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .lito_assets import LITO_DEFAULT_ROOT, LITO_RAW_DEFAULT_ROOT, download_command, inspect, validate
from .lito_inference import (
    LITO_DEFAULT_MEMORY_PROFILE,
    LITO_MEMORY_PROFILES,
    LITO_RECOMMENDED_CFG_SCALE,
    LITO_RECOMMENDED_NUM_STEPS,
    LITO_RECOMMENDED_RESOLUTION,
    LitoBackendUnavailable,
    LitoRealGenerationNotImplemented,
    LitoGenerationResult,
    LitoInferencePipeline,
    metrics_to_json,
    normalize_lito_init_coord_cap,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apple LiTo MLX inference")
    parser.add_argument("--root", dest="global_root", default=LITO_DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the local LiTo asset layout")
    validate_parser.add_argument("root_path", nargs="?")
    validate_parser.add_argument("--root", dest="command_root")

    inspect_parser = subparsers.add_parser("inspect", help="inspect local LiTo safetensors metadata")
    inspect_parser.add_argument("root_path", nargs="?")
    inspect_parser.add_argument("--root", dest="command_root")
    inspect_parser.add_argument("--prefix", action="append", dest="prefixes")
    inspect_parser.add_argument("--limit", type=int, default=20)

    download_parser = subparsers.add_parser("download-command", help="print Apple CDN download commands")
    download_parser.add_argument("root_path", nargs="?")
    download_parser.add_argument("--root", dest="command_root")

    generate_parser = subparsers.add_parser("generate", help="run LiTo image-to-3DGS generation")
    generate_parser.add_argument("image_path")
    generate_parser.add_argument("--root", "--weights-root", dest="command_root", default=None)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--format", choices=("ply", "splat", "safetensors"), default="ply")
    generate_parser.add_argument(
        "--memory-profile",
        choices=LITO_MEMORY_PROFILES,
        default=LITO_DEFAULT_MEMORY_PROFILE,
    )
    generate_parser.add_argument("--num-steps", type=int, default=LITO_RECOMMENDED_NUM_STEPS)
    generate_parser.add_argument("--cfg-scale", type=float, default=LITO_RECOMMENDED_CFG_SCALE)
    generate_parser.add_argument("--seed", type=int, default=None)
    generate_parser.add_argument("--resolution", type=int, default=LITO_RECOMMENDED_RESOLUTION[0])
    generate_parser.add_argument("--render-size", type=int)
    generate_parser.add_argument(
        "--max-init-coords-per-batch",
        default="profile",
        metavar="{profile|none|N}",
        type=normalize_lito_init_coord_cap,
        help="checkpoint-backed init-coordinate cap: profile default, none for upstream coverage, or integer N",
    )
    generate_parser.add_argument("--print-metrics", action="store_true")
    generate_parser.add_argument(
        "--source-contract-smoke",
        action="store_true",
        help="run the synthetic source-contract smoke path instead of checkpoint-backed LiTo generation",
    )

    args = parser.parse_args(argv)

    if args.command == "validate":
        root = _root_arg(args, fallback=args.global_root)
        validation = validate(root)
        print(f"ready={validation.ready}")
        print(f"root={validation.root}")
        print(f"present={len(validation.present)}")
        print(f"missing={len(validation.missing)}")
        for missing in validation.missing:
            print(f"missing {missing}")
        return 0 if validation.ready else 1

    if args.command == "download-command":
        print(download_command(_root_arg(args, fallback=LITO_RAW_DEFAULT_ROOT)))
        return 0

    if args.command == "inspect":
        root = _root_arg(args, fallback=args.global_root)
        try:
            infos = inspect(root, prefixes=args.prefixes, limit=args.limit)
        except (FileNotFoundError, ValueError) as error:
            print(f"error={error}")
            return 1
        for info in infos:
            print(f"tensor {info.name} shape={info.shape} dtype={info.dtype} source={info.source}")
        return 0

    if args.command == "generate":
        root = _root_arg(args, fallback=args.global_root)
        try:
            result = LitoInferencePipeline(
                root,
                memory_profile=args.memory_profile,
                max_init_coords_per_batch=args.max_init_coords_per_batch,
                source_contract_smoke=args.source_contract_smoke,
            ).generate(
                args.image_path,
                output_path=args.output,
                output_format=args.format,
                num_steps=args.num_steps,
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                resolution=args.resolution,
                render_size=args.render_size,
            )
        except (FileNotFoundError, ValueError, LitoBackendUnavailable, LitoRealGenerationNotImplemented) as error:
            print(f"error={error}")
            return 1
        print(f"output={Path(args.output)}")
        print(f"format={args.format}")
        print(f"gaussians={int(result.gaussians['xyz_w'].shape[0])}")
        if args.source_contract_smoke:
            print("mode=source-contract-smoke")
        if args.print_metrics:
            print(metrics_to_json(result.metrics))
        return 0

    parser.error(f"unsupported command: {args.command}")


def _root_arg(args: argparse.Namespace, *, fallback: str) -> str:
    return getattr(args, "command_root", None) or getattr(args, "root_path", None) or fallback


__all__ = [
    "LitoGenerationResult",
    "LitoInferencePipeline",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
