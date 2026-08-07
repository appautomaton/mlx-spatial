"""Runtime stage timing and memory diagnostics for SpatialKit exports."""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")

PIXAL3D_MEMORY_POLL_INTERVAL_SEC = 0.25


class _MemoryStageScope:
    def __init__(self, monitor: _ProcessMemoryMonitor, stage: str):
        self._monitor = monitor
        self._stage = stage
        self._prior_stage = "idle"

    def __enter__(self) -> None:
        self._prior_stage = self._monitor._set_active_stage(self._stage)
        self._monitor._set_stage_boundary(self._stage, "start", self._monitor.sample(f"{self._stage}:start"))

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._monitor._set_stage_boundary(self._stage, "end", self._monitor.sample(f"{self._stage}:end"))
        self._monitor._set_active_stage(self._prior_stage)


class _ProcessMemoryMonitor:
    def __init__(
        self,
        *,
        poll_interval_sec: float = PIXAL3D_MEMORY_POLL_INTERVAL_SEC,
        sample_fn: Callable[[], dict[str, Any]] | None = None,
    ):
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        self._poll_interval_sec = float(poll_interval_sec)
        self._sample_fn = sample_fn or _memory_sample
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopped = False
        self._active_stage = "idle"
        self._sample_count = 0
        self._peak_current_rss_bytes: int | None = None
        self._peak_current_rss_label: str | None = None
        self._peak_current_rss_stage: str | None = None
        self._peak_max_rss_bytes: int | None = None
        self._peak_max_rss_label: str | None = None
        self._peak_max_rss_stage: str | None = None
        self._baseline_swap_used_bytes: int | None = None
        self._peak_swap_used_bytes: int | None = None
        self._peak_swap_growth_bytes: int | None = None
        self._peak_mlx_active_bytes: int | None = None
        self._peak_mlx_allocator_bytes: int | None = None
        self._peak_mlx_cache_bytes: int | None = None
        self._last_sample: dict[str, Any] | None = None
        self._stage_peaks: dict[str, dict[str, Any]] = {}

    @property
    def poll_interval_sec(self) -> float:
        return self._poll_interval_sec

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self.sample("monitor_start")
        thread = threading.Thread(
            target=self._poll_loop,
            name="mlx-spatial-memory-monitor",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            thread = self._thread
        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self._poll_interval_sec * 4.0))
        self.sample("monitor_stop")

    def sample(self, label: str) -> dict[str, Any]:
        sample = self._sample_fn()
        self._record(label, sample)
        return sample

    def track_stage(self, stage: str) -> _MemoryStageScope:
        return _MemoryStageScope(self, stage)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            stage_peaks = {stage: dict(values) for stage, values in sorted(self._stage_peaks.items())}
            return {
                "source": "process RSS/high-water RSS, MLX allocator counters, and macOS vm.swapusage",
                "poll_interval_sec": self._poll_interval_sec,
                "sample_count": self._sample_count,
                "peak_current_rss_bytes": self._peak_current_rss_bytes,
                "peak_current_rss_label": self._peak_current_rss_label,
                "peak_current_rss_stage": self._peak_current_rss_stage,
                "peak_max_rss_bytes": self._peak_max_rss_bytes,
                "peak_max_rss_label": self._peak_max_rss_label,
                "peak_max_rss_stage": self._peak_max_rss_stage,
                "baseline_swap_used_bytes": self._baseline_swap_used_bytes,
                "peak_swap_used_bytes": self._peak_swap_used_bytes,
                "peak_swap_growth_bytes": self._peak_swap_growth_bytes,
                "peak_mlx_active_bytes": self._peak_mlx_active_bytes,
                "peak_mlx_allocator_bytes": self._peak_mlx_allocator_bytes,
                "peak_mlx_cache_bytes": self._peak_mlx_cache_bytes,
                "last_sample": dict(self._last_sample) if self._last_sample is not None else None,
                "stage_peaks": stage_peaks,
            }

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_sec):
            self.sample("poll")

    def _set_active_stage(self, stage: str) -> str:
        with self._lock:
            prior = self._active_stage
            self._active_stage = stage
            if stage != "idle":
                self._stage_peaks.setdefault(stage, self._empty_stage_record())
            return prior

    def _set_stage_boundary(self, stage: str, boundary: str, sample: dict[str, Any]) -> None:
        current_rss = _sample_int(sample, "current_rss_bytes")
        max_rss = _sample_int(sample, "max_rss_bytes")
        swap_used = _sample_int(sample, "swap_used_bytes")
        mlx_active = _sample_int(sample, "mlx_active_bytes")
        mlx_peak = _sample_int(sample, "mlx_peak_bytes")
        mlx_cache = _sample_int(sample, "mlx_cache_bytes")
        with self._lock:
            record = self._stage_peaks.setdefault(stage, self._empty_stage_record())
            record[f"{boundary}_current_rss_bytes"] = current_rss
            record[f"{boundary}_max_rss_bytes"] = max_rss
            record[f"{boundary}_swap_used_bytes"] = swap_used
            record[f"{boundary}_mlx_active_bytes"] = mlx_active
            record[f"{boundary}_mlx_allocator_bytes"] = mlx_peak
            record[f"{boundary}_mlx_cache_bytes"] = mlx_cache

    def _record(self, label: str, sample: dict[str, Any]) -> None:
        current_rss = _sample_int(sample, "current_rss_bytes")
        max_rss = _sample_int(sample, "max_rss_bytes")
        swap_used = _sample_int(sample, "swap_used_bytes")
        mlx_active = _sample_int(sample, "mlx_active_bytes")
        mlx_peak = _sample_int(sample, "mlx_peak_bytes")
        mlx_cache = _sample_int(sample, "mlx_cache_bytes")
        with self._lock:
            self._sample_count += 1
            self._last_sample = dict(sample)
            stage = self._active_stage
            if current_rss is not None and (
                self._peak_current_rss_bytes is None or current_rss > self._peak_current_rss_bytes
            ):
                self._peak_current_rss_bytes = current_rss
                self._peak_current_rss_label = label
                self._peak_current_rss_stage = stage
            if max_rss is not None and (self._peak_max_rss_bytes is None or max_rss > self._peak_max_rss_bytes):
                self._peak_max_rss_bytes = max_rss
                self._peak_max_rss_label = label
                self._peak_max_rss_stage = stage
            if swap_used is not None:
                if self._baseline_swap_used_bytes is None:
                    self._baseline_swap_used_bytes = swap_used
                self._peak_swap_used_bytes = max(self._peak_swap_used_bytes or 0, swap_used)
                self._peak_swap_growth_bytes = max(
                    self._peak_swap_growth_bytes or 0,
                    swap_used - self._baseline_swap_used_bytes,
                )
            if mlx_active is not None:
                self._peak_mlx_active_bytes = max(self._peak_mlx_active_bytes or 0, mlx_active)
            if mlx_peak is not None:
                self._peak_mlx_allocator_bytes = max(self._peak_mlx_allocator_bytes or 0, mlx_peak)
            if mlx_cache is not None:
                self._peak_mlx_cache_bytes = max(self._peak_mlx_cache_bytes or 0, mlx_cache)
            if stage == "idle":
                return
            record = self._stage_peaks.setdefault(stage, self._empty_stage_record())
            record["sample_count"] += 1
            if current_rss is not None and (
                record["peak_current_rss_bytes"] is None or current_rss > record["peak_current_rss_bytes"]
            ):
                record["peak_current_rss_bytes"] = current_rss
                record["peak_current_rss_label"] = label
            if max_rss is not None and (
                record["peak_max_rss_bytes"] is None or max_rss > record["peak_max_rss_bytes"]
            ):
                record["peak_max_rss_bytes"] = max_rss
                record["peak_max_rss_label"] = label
            if swap_used is not None:
                record["peak_swap_used_bytes"] = max(record["peak_swap_used_bytes"] or 0, swap_used)
            if mlx_active is not None:
                record["peak_mlx_active_bytes"] = max(record["peak_mlx_active_bytes"] or 0, mlx_active)
            if mlx_peak is not None:
                record["peak_mlx_allocator_bytes"] = max(record["peak_mlx_allocator_bytes"] or 0, mlx_peak)
            if mlx_cache is not None:
                record["peak_mlx_cache_bytes"] = max(record["peak_mlx_cache_bytes"] or 0, mlx_cache)

    @staticmethod
    def _empty_stage_record() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "start_current_rss_bytes": None,
            "end_current_rss_bytes": None,
            "peak_current_rss_bytes": None,
            "peak_current_rss_label": None,
            "start_max_rss_bytes": None,
            "end_max_rss_bytes": None,
            "peak_max_rss_bytes": None,
            "peak_max_rss_label": None,
            "start_swap_used_bytes": None,
            "end_swap_used_bytes": None,
            "peak_swap_used_bytes": None,
            "start_mlx_active_bytes": None,
            "end_mlx_active_bytes": None,
            "peak_mlx_active_bytes": None,
            "start_mlx_allocator_bytes": None,
            "end_mlx_allocator_bytes": None,
            "peak_mlx_allocator_bytes": None,
            "start_mlx_cache_bytes": None,
            "end_mlx_cache_bytes": None,
            "peak_mlx_cache_bytes": None,
        }


def _timed_stage(
    diagnostics: dict[str, Any],
    name: str,
    fn: Callable[[], _T],
    *,
    memory_monitor: _ProcessMemoryMonitor | None = None,
) -> _T:
    start = time.perf_counter()
    _write_stage_event(name, "start")
    try:
        if memory_monitor is None:
            return fn()
        with memory_monitor.track_stage(name):
            return fn()
    except BaseException:
        if memory_monitor is not None:
            memory_monitor.stop()
        raise
    finally:
        elapsed = time.perf_counter() - start
        diagnostics["timings_sec"][name] = elapsed
        diagnostics["stages"].setdefault(name, {})["seconds"] = elapsed
        _write_stage_event(name, "end", seconds=elapsed)


def _observed_substage(
    name: str,
    fn: Callable[[], _T],
    timings: dict[str, float],
) -> _T:
    """Time a nested conversion stage and expose it to the external watchdog."""

    start = time.perf_counter()
    _write_stage_event(name, "start")
    try:
        return fn()
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = elapsed
        _write_stage_event(name, "end", seconds=elapsed)


def _write_stage_event(stage: str, event: str, *, seconds: float | None = None) -> None:
    path_value = os.environ.get("MLX_SPATIAL_STAGE_EVENTS_PATH")
    if not path_value:
        return
    record: dict[str, Any] = {
        "stage": stage,
        "event": event,
        "monotonic_sec": time.monotonic(),
    }
    if seconds is not None:
        record["seconds"] = float(seconds)
    with Path(path_value).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _sample_int(sample: dict[str, Any], key: str) -> int | None:
    value = sample.get(key)
    return None if value is None else int(value)


def _memory_sample() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = int(usage.ru_maxrss)
    max_rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
    swap = _macos_swap_usage()
    mlx_memory = _mlx_memory_counters()
    return {
        "pid": os.getpid(),
        "current_rss_bytes": _current_rss_bytes(),
        "max_rss_bytes": max_rss_bytes,
        **swap,
        **mlx_memory,
        "source": "ps rss plus resource.getrusage, MLX allocator counters, and vm.swapusage",
    }


def _current_rss_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(output.strip()) * 1024
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _macos_swap_usage() -> dict[str, int | None]:
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
    values: dict[str, int] = {}
    scales = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    for name, value, unit in re.findall(r"(total|used|free)\s*=\s*([0-9.]+)([KMGT])", output):
        values[name] = int(float(value) * scales[unit])
    return {
        "swap_total_bytes": values.get("total"),
        "swap_used_bytes": values.get("used"),
        "swap_free_bytes": values.get("free"),
    }


def _mlx_memory_counters() -> dict[str, int | None]:
    try:
        import mlx.core as mx
    except (ImportError, RuntimeError):
        return {"mlx_active_bytes": None, "mlx_peak_bytes": None, "mlx_cache_bytes": None}

    def read(name: str) -> int | None:
        counter = getattr(mx, name, None)
        if counter is None:
            return None
        try:
            return int(counter())
        except RuntimeError:
            return None

    return {
        "mlx_active_bytes": read("get_active_memory"),
        "mlx_peak_bytes": read("get_peak_memory"),
        "mlx_cache_bytes": read("get_cache_memory"),
    }


__all__ = [
    "PIXAL3D_MEMORY_POLL_INTERVAL_SEC",
    "_ProcessMemoryMonitor",
    "_observed_substage",
    "_timed_stage",
]
