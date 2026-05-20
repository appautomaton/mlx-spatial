"""HY-World-2.0 WorldMirror local asset and inference tooling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from .hyworld2_assets import (
    HYWORLD2_DEFAULT_ROOT,
    HYWORLD2_REPO_ID,
    inspect_hyworld2_checkpoint,
    hyworld2_download_command,
    validate_hyworld2_assets,
)
from .hyworld2_inference import (
    HYWORLD2_DEFAULT_HEADS,
    HYWORLD2_MEMORY_PROFILES,
    HyWorld2InferencePipeline,
    normalize_hyworld2_heads,
)
from .hyworld2_parity import (
    compare_hyworld2_parity_tensors,
    load_hyworld2_parity_bundle,
    parity_report_to_dict,
)
from .hyworld2_preprocess import HYWORLD2_DEFAULT_MEMORY_PROFILE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect local HY-World-2.0 WorldMirror assets")
    parser.add_argument("--root", default=HYWORLD2_DEFAULT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate the local WorldMirror asset layout")
    validate_parser.add_argument("root_path", nargs="?")
    validate_parser.add_argument("--root", dest="command_root")

    inspect_parser = subparsers.add_parser("inspect", help="inspect local WorldMirror checkpoint tensors")
    inspect_parser.add_argument("root_path", nargs="?")
    inspect_parser.add_argument("--root", dest="command_root")
    inspect_parser.add_argument("--prefix", action="append", dest="prefixes")
    inspect_parser.add_argument("--limit", type=int, default=20)

    download_parser = subparsers.add_parser("download-command", help="print the manual HF CLI command")
    download_parser.add_argument("root_path", nargs="?")
    download_parser.add_argument("--root", dest="command_root")

    reconstruct_parser = subparsers.add_parser("reconstruct", help="run staged HY-World WorldMirror MLX reconstruction")
    reconstruct_parser.add_argument("reconstruct_root")
    reconstruct_parser.add_argument("input")
    reconstruct_parser.add_argument("--output", required=True)
    reconstruct_parser.add_argument("--heads", default=",".join(HYWORLD2_DEFAULT_HEADS))
    reconstruct_parser.add_argument(
        "--memory-profile",
        choices=HYWORLD2_MEMORY_PROFILES,
        default=HYWORLD2_DEFAULT_MEMORY_PROFILE,
    )
    reconstruct_parser.add_argument("--trace-output")
    reconstruct_parser.add_argument("--fixture-tensors", action="store_true")
    reconstruct_parser.add_argument(
        "--mlx-device",
        choices=("default", "cpu", "gpu"),
        default="default",
        help="MLX device for the reconstruction process; use cpu for Torch CPU parity checks",
    )
    reconstruct_parser.add_argument(
        "--parity-output",
        help="optional MLX tensor bundle for dev-only PyTorch parity comparison",
    )

    parity_parser = subparsers.add_parser(
        "parity-compare",
        help="compare an MLX tensor bundle against a dev-only PyTorch reference bundle",
    )
    parity_parser.add_argument("reference")
    parity_parser.add_argument("actual")
    parity_parser.add_argument("--tensor", action="append", dest="tensors")
    parity_parser.add_argument("--atol", type=float, default=1e-4)
    parity_parser.add_argument("--rtol", type=float, default=1e-4)
    parity_parser.add_argument("--json-output")

    args = parser.parse_args(argv)
    root = getattr(args, "command_root", None) or getattr(args, "root_path", None) or args.root

    if args.command == "validate":
        validation = validate_hyworld2_assets(root)
        _print_validation(validation)
        return 0

    if args.command == "download-command":
        print(" ".join(hyworld2_download_command(root)))
        return 0

    if args.command == "inspect":
        validation = validate_hyworld2_assets(root)
        _print_validation(validation)
        if not validation.checkpoint_path.is_file():
            return 2
        infos = inspect_hyworld2_checkpoint(root, prefixes=args.prefixes)
        limit = max(args.limit, 0)
        for info in infos[:limit] if limit else infos:
            print(f"tensor {info.name} shape={info.shape} dtype={info.dtype}")
        if limit and len(infos) > limit:
            print(f"truncated={len(infos) - limit}")
        return 0

    if args.command == "reconstruct":
        try:
            heads = normalize_hyworld2_heads(args.heads)
        except ValueError as error:
            parser.error(str(error))
        previous_device = _set_mlx_device(args.mlx_device)
        try:
            result = HyWorld2InferencePipeline(args.reconstruct_root).reconstruct(
                args.input,
                output_path=args.output,
                heads=heads,
                memory_profile=args.memory_profile,
                fixture_tensors=args.fixture_tensors,
                parity_output_path=args.parity_output,
            )
        finally:
            if previous_device is not None:
                mx.set_default_device(previous_device)
        trace = result.trace
        trace.metadata["mlx_device"] = args.mlx_device
        payload = _jsonable(trace)
        if args.trace_output:
            trace_output = Path(args.trace_output)
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            trace_output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"completed={trace.completed_stages}")
        print(f"outputs={tuple(output.name for output in trace.outputs)}")
        print(f"requested_heads={trace.requested_heads}")
        print(f"memory_profile={trace.memory_profile}")
        if trace.blocker is not None:
            print(f"blocker_stage={trace.blocker.stage}")
            print(f"operation={trace.blocker.operation}")
            print(f"reason={trace.blocker.reason}")
            return 2
        return 0

    if args.command == "parity-compare":
        reference = load_hyworld2_parity_bundle(args.reference)
        actual = load_hyworld2_parity_bundle(args.actual)
        report = compare_hyworld2_parity_tensors(
            actual.tensors,
            reference,
            names=args.tensors,
            atol=args.atol,
            rtol=args.rtol,
        )
        payload = parity_report_to_dict(report)
        if args.json_output:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"passed={report.passed}")
        print(f"checked={len(report.comparisons)}")
        print(f"failed={report.failed_names}")
        for comparison in report.comparisons:
            if comparison.passed:
                continue
            print(
                f"mismatch {comparison.name} status={comparison.status} "
                f"max_abs_error={comparison.max_abs_error} reason={comparison.reason}"
            )
        return 0 if report.passed else 1

    parser.error(f"unsupported command: {args.command}")


def _set_mlx_device(device: str):
    if device == "default":
        return None
    previous = mx.default_device()
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    return previous


def _print_validation(validation) -> None:
    print(f"ready={validation.ready}")
    print(f"repo={HYWORLD2_REPO_ID}")
    print(f"root={validation.root}")
    print(f"model_dir={validation.model_dir}")
    print(f"checkpoint={validation.checkpoint_path}")
    print(f"config={validation.config_path if validation.config_path is not None else '<missing>'}")
    print(f"config_kind={validation.config_kind or '<missing>'}")
    print(f"present={len(validation.present)}")
    print(f"missing={len(validation.missing)}")
    for path in validation.missing:
        print(f"missing {path}")


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
