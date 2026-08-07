from __future__ import annotations

from mlx_spatial.spatialkit.export import _ProcessMemoryMonitor, _timed_stage


def test_process_memory_monitor_records_stage_peaks_without_sample_log() -> None:
    samples = iter(
        [
            {
                "current_rss_bytes": 100,
                "max_rss_bytes": 120,
                "swap_used_bytes": 1_000,
                "mlx_active_bytes": 10,
                "mlx_peak_bytes": 20,
                "mlx_cache_bytes": 5,
                "source": "fake",
            },
            {
                "current_rss_bytes": 180,
                "max_rss_bytes": 200,
                "swap_used_bytes": 1_400,
                "mlx_active_bytes": 30,
                "mlx_peak_bytes": 40,
                "mlx_cache_bytes": 15,
                "source": "fake",
            },
            {
                "current_rss_bytes": 140,
                "max_rss_bytes": 210,
                "swap_used_bytes": 1_200,
                "mlx_active_bytes": 20,
                "mlx_peak_bytes": 50,
                "mlx_cache_bytes": 10,
                "source": "fake",
            },
        ]
    )
    monitor = _ProcessMemoryMonitor(sample_fn=lambda: next(samples))

    with monitor.track_stage("texture_bake"):
        monitor.sample("inside_texture_bake")

    summary = monitor.summary()
    stage = summary["stage_peaks"]["texture_bake"]
    assert summary["sample_count"] == 3
    assert summary["peak_current_rss_bytes"] == 180
    assert summary["peak_current_rss_label"] == "inside_texture_bake"
    assert summary["peak_current_rss_stage"] == "texture_bake"
    assert summary["peak_max_rss_bytes"] == 210
    assert summary["peak_max_rss_label"] == "texture_bake:end"
    assert summary["baseline_swap_used_bytes"] == 1_000
    assert summary["peak_swap_used_bytes"] == 1_400
    assert summary["peak_swap_growth_bytes"] == 400
    assert summary["peak_mlx_active_bytes"] == 30
    assert summary["peak_mlx_allocator_bytes"] == 50
    assert summary["peak_mlx_cache_bytes"] == 15
    assert stage["sample_count"] == 3
    assert stage["start_current_rss_bytes"] == 100
    assert stage["end_current_rss_bytes"] == 140
    assert stage["peak_current_rss_bytes"] == 180
    assert stage["start_swap_used_bytes"] == 1_000
    assert stage["end_swap_used_bytes"] == 1_200
    assert stage["peak_swap_used_bytes"] == 1_400
    assert stage["peak_mlx_active_bytes"] == 30
    assert stage["peak_mlx_allocator_bytes"] == 50
    assert stage["peak_mlx_cache_bytes"] == 15
    assert "samples" not in summary
    assert "samples" not in stage


def test_timed_stage_records_memory_stage_with_fake_samples() -> None:
    values = iter(
        [
            {"current_rss_bytes": 10, "max_rss_bytes": 10, "source": "fake"},
            {"current_rss_bytes": 12, "max_rss_bytes": 12, "source": "fake"},
        ]
    )
    monitor = _ProcessMemoryMonitor(sample_fn=lambda: next(values))
    diagnostics = {"timings_sec": {}, "stages": {}}

    result = _timed_stage(diagnostics, "write_glb", lambda: "ok", memory_monitor=monitor)

    assert result == "ok"
    assert diagnostics["timings_sec"]["write_glb"] >= 0.0
    assert diagnostics["stages"]["write_glb"]["seconds"] >= 0.0
    stage = monitor.summary()["stage_peaks"]["write_glb"]
    assert stage["start_current_rss_bytes"] == 10
    assert stage["end_current_rss_bytes"] == 12
    assert stage["peak_current_rss_bytes"] == 12
