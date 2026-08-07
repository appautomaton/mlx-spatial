#!/usr/bin/env python3
"""Export cached decoded O-Voxel artifacts under an RSS/swap watchdog."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decoded_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--target-faces", type=int)
    parser.add_argument("--quality-preset", choices=("preview", "reference-target"), default="preview")
    parser.add_argument("--grid-size", type=int)
    parser.add_argument(
        "--uv-backend",
        choices=(
            "face-atlas",
            "native-chart",
            "xatlas-equivalent-native",
            "xatlas-global",
            "xatlas-clustered",
            "xatlas-parallel-spatial",
        ),
        default="face-atlas",
    )
    parser.add_argument(
        "--xatlas-parallel-chunks",
        type=int,
        help="spatial xatlas chunk count; required only by xatlas-parallel-spatial",
    )
    parser.add_argument("--remesh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--remesh-resolution", type=int)
    parser.add_argument("--remesh-repair-nonmanifold", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--simplify-backend",
        choices=("qem", "mlx-qem", "single-layer-qem", "single-layer-mlx-qem"),
    )
    parser.add_argument("--texture-postprocess", choices=("legacy-dilation", "telea"), default="legacy-dilation")
    parser.add_argument("--poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-rss-gib", type=float, default=24.0)
    parser.add_argument("--max-swap-growth-gib", type=float, default=0.0)
    parser.add_argument(
        "--require-utilization-probes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort when process CPU or Apple GPU utilization cannot be sampled.",
    )
    parser.add_argument("--terminate-grace-sec", type=float, default=10.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.worker:
        return _run_worker(args)
    return _run_watchdog(args)


def _validate_args(args: argparse.Namespace) -> None:
    if args.poll_interval_sec <= 0:
        raise ValueError("poll_interval_sec must be positive")
    if args.max_rss_gib <= 0:
        raise ValueError("max_rss_gib must be positive")
    if args.max_swap_growth_gib < 0:
        raise ValueError("max_swap_growth_gib must be non-negative")
    if args.terminate_grace_sec <= 0:
        raise ValueError("terminate_grace_sec must be positive")
    if args.uv_backend == "xatlas-parallel-spatial":
        if args.xatlas_parallel_chunks is None or args.xatlas_parallel_chunks <= 1:
            raise ValueError(
                "xatlas-parallel-spatial requires --xatlas-parallel-chunks greater than one"
            )
    elif args.xatlas_parallel_chunks is not None:
        raise ValueError(
            "--xatlas-parallel-chunks only applies to --uv-backend xatlas-parallel-spatial"
        )


def _run_worker(args: argparse.Namespace) -> int:
    from mlx_spatial.spatialkit import export_decoded_ovoxel_glb

    result = export_decoded_ovoxel_glb(
        args.decoded_dir,
        args.output,
        texture_size=args.texture_size,
        target_faces=args.target_faces,
        quality_preset=args.quality_preset,
        grid_size=args.grid_size,
        uv_backend=args.uv_backend,
        xatlas_parallel_chunks=args.xatlas_parallel_chunks,
        remesh=args.remesh,
        remesh_resolution=args.remesh_resolution,
        remesh_repair_nonmanifold=args.remesh_repair_nonmanifold,
        simplify_backend=args.simplify_backend,
        texture_postprocess=args.texture_postprocess,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "glb": str(result.glb.path),
                "diagnostics": str(result.diagnostics_path),
                "ready": result.diagnostics.get("result", {}).get("ready"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_watchdog(args: argparse.Namespace) -> int:
    monitor_dir = args.output.parent if args.output.suffix.lower() == ".glb" else args.output
    monitor_dir.mkdir(parents=True, exist_ok=True)
    log_path = monitor_dir / "conversion.log"
    samples_path = monitor_dir / "resource-samples.jsonl"
    summary_path = monitor_dir / "watchdog-summary.json"
    stage_events_path = monitor_dir / "stage-events.jsonl"
    stage_events_path.write_text("", encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    worker_environment = os.environ.copy()
    worker_environment["MLX_SPATIAL_STAGE_EVENTS_PATH"] = str(stage_events_path)
    max_rss_bytes = int(args.max_rss_gib * 1024**3)
    max_swap_growth_bytes = int(args.max_swap_growth_gib * 1024**3)
    baseline_swap = _swap_usage_bytes()
    baseline_swap_used = baseline_swap.get("swap_used_bytes")
    started = time.monotonic()
    sample_count = 0
    peak_rss = 0
    peak_swap_growth = 0
    abort_reason: str | None = None
    logical_cpu_count = os.cpu_count() or 1
    utilization_values: dict[str, list[float]] = {
        "process_cpu_percent": [],
        "process_cpu_capacity_percent": [],
        "system_cpu_capacity_percent": [],
        "gpu_device_utilization_percent": [],
        "gpu_renderer_utilization_percent": [],
        "gpu_tiler_utilization_percent": [],
    }
    stage_utilization_values: dict[str, dict[str, list[float]]] = {}
    previous_cpu_time_sec: float | None = None
    previous_cpu_sample_elapsed_sec: float | None = None

    with log_path.open("w", encoding="utf-8") as log, samples_path.open("w", encoding="utf-8") as samples:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=worker_environment,
        )
        while process.poll() is None:
            sample_elapsed_sec = time.monotonic() - started
            process_usage = _cpu_and_process_usage(process.pid, logical_cpu_count)
            process_cpu_time_sec = process_usage.get("process_cpu_time_sec")
            process_cpu_percent: float | None = None
            if (
                process_cpu_time_sec is not None
                and previous_cpu_time_sec is not None
                and previous_cpu_sample_elapsed_sec is not None
            ):
                elapsed_delta = sample_elapsed_sec - previous_cpu_sample_elapsed_sec
                if elapsed_delta > 0:
                    process_cpu_percent = max(
                        0.0,
                        (float(process_cpu_time_sec) - previous_cpu_time_sec) / elapsed_delta * 100.0,
                    )
            if process_cpu_time_sec is not None:
                previous_cpu_time_sec = float(process_cpu_time_sec)
                previous_cpu_sample_elapsed_sec = sample_elapsed_sec
            process_usage["process_cpu_percent"] = process_cpu_percent
            process_usage["process_cpu_capacity_percent"] = (
                None if process_cpu_percent is None else process_cpu_percent / logical_cpu_count
            )
            gpu_usage = _apple_gpu_usage()
            sample = {
                "elapsed_sec": sample_elapsed_sec,
                "pid": process.pid,
                "logical_cpu_count": logical_cpu_count,
                **process_usage,
                **gpu_usage,
                **_swap_usage_bytes(),
                "conversion_stage": _read_active_stage(stage_events_path),
            }
            rss = sample.get("rss_bytes")
            swap_used = sample.get("swap_used_bytes")
            swap_growth = (
                max(0, int(swap_used) - int(baseline_swap_used))
                if swap_used is not None and baseline_swap_used is not None
                else None
            )
            sample["swap_growth_bytes"] = swap_growth
            samples.write(json.dumps(sample, sort_keys=True) + "\n")
            samples.flush()
            sample_count += 1
            for key, values in utilization_values.items():
                value = sample.get(key)
                if value is not None:
                    values.append(float(value))
            stage_name = str(sample["conversion_stage"])
            stage_values = stage_utilization_values.setdefault(
                stage_name,
                {key: [] for key in utilization_values},
            )
            for key, values in stage_values.items():
                value = sample.get(key)
                if value is not None:
                    values.append(float(value))
            if args.require_utilization_probes:
                missing_probes = [
                    key
                    for key in ("process_cpu_time_sec", "gpu_device_utilization_percent")
                    if sample.get(key) is None
                ]
                if missing_probes:
                    abort_reason = f"required utilization probes unavailable: {', '.join(missing_probes)}"
            if rss is not None:
                peak_rss = max(peak_rss, int(rss))
                if int(rss) > max_rss_bytes:
                    abort_reason = f"process RSS {int(rss)} exceeded limit {max_rss_bytes}"
            if swap_growth is not None:
                peak_swap_growth = max(peak_swap_growth, int(swap_growth))
                if int(swap_growth) > max_swap_growth_bytes:
                    abort_reason = (
                        f"system swap growth {int(swap_growth)} exceeded limit {max_swap_growth_bytes}"
                    )
            if abort_reason is not None:
                _terminate(process, args.terminate_grace_sec)
                break
            time.sleep(args.poll_interval_sec)
        returncode = process.wait()

    summary = {
        "status": "aborted" if abort_reason is not None else ("completed" if returncode == 0 else "failed"),
        "returncode": returncode,
        "abort_reason": abort_reason,
        "elapsed_sec": time.monotonic() - started,
        "sample_count": sample_count,
        "peak_rss_bytes": peak_rss,
        "baseline_swap_used_bytes": baseline_swap_used,
        "peak_swap_growth_bytes": peak_swap_growth,
        "utilization": {
            "logical_cpu_count": logical_cpu_count,
            "process_cpu_percent_semantics": (
                "instantaneous child CPU time delta divided by wall-time delta; "
                "100 percent equals one fully occupied CPU core"
            ),
            "process_cpu_capacity_percent_semantics": "process CPU normalized across all logical CPU cores",
            "gpu_utilization_semantics": "system-wide Apple GPU utilization reported by AGXAccelerator",
            **{key: _summarize_values(values) for key, values in utilization_values.items()},
            "by_stage": {
                stage: {key: _summarize_values(values) for key, values in metrics.items()}
                for stage, metrics in sorted(stage_utilization_values.items())
            },
        },
        "limits": {
            "max_rss_bytes": max_rss_bytes,
            "max_swap_growth_bytes": max_swap_growth_bytes,
        },
        "samples_path": str(samples_path),
        "stage_events_path": str(stage_events_path),
        "log_path": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    if abort_reason is not None:
        return 2
    return returncode


def _cpu_and_process_usage(pid: int, logical_cpu_count: int) -> dict[str, float | int | None]:
    try:
        output = subprocess.check_output(
            ["ps", "-A", "-o", "pid=,time=,rss=,%cpu="],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "rss_bytes": None,
            "process_cpu_time_sec": None,
            "process_cpu_lifetime_percent": None,
            "system_cpu_capacity_percent": None,
        }
    return _parse_ps_usage(output, pid, logical_cpu_count)


def _parse_ps_usage(output: str, pid: int, logical_cpu_count: int) -> dict[str, float | int | None]:
    process_cpu_time: float | None = None
    process_cpu_lifetime: float | None = None
    process_rss: int | None = None
    system_cpu = 0.0
    valid_cpu_rows = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        try:
            row_pid = int(fields[0])
            row_cpu_time = _parse_ps_cpu_time(fields[1])
            row_rss = int(fields[2]) * 1024
            row_cpu = float(fields[3])
        except ValueError:
            continue
        system_cpu += row_cpu
        valid_cpu_rows += 1
        if row_pid == pid:
            process_cpu_time = row_cpu_time
            process_cpu_lifetime = row_cpu
            process_rss = row_rss
    cpu_count = max(1, int(logical_cpu_count))
    return {
        "rss_bytes": process_rss,
        "process_cpu_time_sec": process_cpu_time,
        "process_cpu_lifetime_percent": process_cpu_lifetime,
        "system_cpu_capacity_percent": None if valid_cpu_rows == 0 else min(100.0, system_cpu / cpu_count),
    }


def _parse_ps_cpu_time(value: str) -> float:
    days = 0
    time_value = value
    if "-" in value:
        day_value, time_value = value.split("-", 1)
        days = int(day_value)
    parts = [float(part) for part in time_value.split(":")]
    if len(parts) == 2:
        hours = 0.0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"unsupported ps CPU time value: {value!r}")
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def _apple_gpu_usage() -> dict[str, int | str | None]:
    empty: dict[str, int | str | None] = {
        "gpu_device_utilization_percent": None,
        "gpu_renderer_utilization_percent": None,
        "gpu_tiler_utilization_percent": None,
        "gpu_probe_source": None,
    }
    if sys.platform != "darwin":
        return empty
    try:
        output = subprocess.check_output(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return empty
    values = _parse_apple_gpu_usage(output)
    if values["gpu_device_utilization_percent"] is not None:
        values["gpu_probe_source"] = "IOKit AGXAccelerator PerformanceStatistics"
    return values


def _parse_apple_gpu_usage(output: str) -> dict[str, int | str | None]:
    def read(name: str) -> int | None:
        match = re.search(rf'"{re.escape(name)}"\s*=\s*([0-9]+)', output)
        return None if match is None else int(match.group(1))

    return {
        "gpu_device_utilization_percent": read("Device Utilization %"),
        "gpu_renderer_utilization_percent": read("Renderer Utilization %"),
        "gpu_tiler_utilization_percent": read("Tiler Utilization %"),
        "gpu_probe_source": None,
    }


def _summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"sample_count": 0, "mean": None, "peak": None}
    return {
        "sample_count": len(values),
        "mean": sum(values) / len(values),
        "peak": max(values),
    }


def _read_active_stage(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unavailable"
    stage_stack: list[str] = []
    saw_event = False
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        stage = event.get("stage")
        event_type = event.get("event")
        if not isinstance(stage, str):
            continue
        if event_type == "start":
            saw_event = True
            stage_stack.append(stage)
        elif event_type == "end":
            saw_event = True
            if stage in stage_stack:
                reverse_index = stage_stack[::-1].index(stage)
                del stage_stack[len(stage_stack) - 1 - reverse_index]
    if stage_stack:
        return stage_stack[-1]
    return "between_stages" if saw_event else "startup"


def _swap_usage_bytes() -> dict[str, int | None]:
    if sys.platform != "darwin":
        return {"swap_total_bytes": None, "swap_used_bytes": None, "swap_free_bytes": None}
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"swap_total_bytes": None, "swap_used_bytes": None, "swap_free_bytes": None}
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    values = {
        name: int(float(value) * scale[unit])
        for name, value, unit in re.findall(r"(total|used|free)\s*=\s*([0-9.]+)([KMGT])", output)
    }
    return {
        "swap_total_bytes": values.get("total"),
        "swap_used_bytes": values.get("used"),
        "swap_free_bytes": values.get("free"),
    }


def _terminate(process: subprocess.Popen[str], grace_sec: float) -> None:
    process.terminate()
    try:
        process.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
