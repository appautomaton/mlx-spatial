from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "spatialkit" / "export_cached_ovoxel.py"
SPEC = importlib.util.spec_from_file_location("export_cached_ovoxel", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
WATCHDOG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCHDOG)


def test_parse_ps_usage_reports_process_and_normalized_system_capacity() -> None:
    result = WATCHDOG._parse_ps_usage(
        """
          10   0:01.25  100  50.0
          20   2:03.50 2048 250.0
          30 1:02:03.75 300  20.0
        """,
        pid=20,
        logical_cpu_count=8,
    )

    assert result["rss_bytes"] == 2048 * 1024
    assert result["process_cpu_time_sec"] == 123.5
    assert result["process_cpu_lifetime_percent"] == 250.0
    assert result["system_cpu_capacity_percent"] == 40.0


def test_parse_ps_cpu_time_supports_minutes_hours_and_days() -> None:
    assert WATCHDOG._parse_ps_cpu_time("2:03.50") == 123.5
    assert WATCHDOG._parse_ps_cpu_time("1:02:03.75") == 3723.75
    assert WATCHDOG._parse_ps_cpu_time("2-01:02:03.25") == 176523.25


def test_parse_apple_gpu_usage_reads_agx_performance_statistics() -> None:
    result = WATCHDOG._parse_apple_gpu_usage(
        '"PerformanceStatistics" = {"Tiler Utilization %"=81,'
        '"Renderer Utilization %"=93,"Device Utilization %"=95}'
    )

    assert result["gpu_device_utilization_percent"] == 95
    assert result["gpu_renderer_utilization_percent"] == 93
    assert result["gpu_tiler_utilization_percent"] == 81


def test_summarize_values_preserves_count_mean_and_peak() -> None:
    assert WATCHDOG._summarize_values([]) == {"sample_count": 0, "mean": None, "peak": None}
    assert WATCHDOG._summarize_values([10.0, 20.0, 60.0]) == {
        "sample_count": 3,
        "mean": 30.0,
        "peak": 60.0,
    }


def test_read_active_stage_tracks_start_and_end_events(tmp_path: Path) -> None:
    events = tmp_path / "stage-events.jsonl"
    events.write_text(
        '{"event":"start","stage":"extract_mesh"}\n'
        '{"event":"end","stage":"extract_mesh"}\n'
        '{"event":"start","stage":"source_metrics"}\n',
        encoding="utf-8",
    )

    assert WATCHDOG._read_active_stage(events) == "source_metrics"


def test_read_active_stage_restores_parent_after_nested_stage(tmp_path: Path) -> None:
    events = tmp_path / "stage-events.jsonl"
    events.write_text(
        '{"event":"start","stage":"uv"}\n'
        '{"event":"start","stage":"uv.compute_charts"}\n'
        '{"event":"end","stage":"uv.compute_charts"}\n',
        encoding="utf-8",
    )

    assert WATCHDOG._read_active_stage(events) == "uv"


def test_read_active_stage_reports_between_stages_after_outer_end(tmp_path: Path) -> None:
    events = tmp_path / "stage-events.jsonl"
    events.write_text(
        '{"event":"start","stage":"uv"}\n'
        '{"event":"start","stage":"uv.pack"}\n'
        '{"event":"end","stage":"uv.pack"}\n'
        '{"event":"end","stage":"uv"}\n',
        encoding="utf-8",
    )

    assert WATCHDOG._read_active_stage(events) == "between_stages"
