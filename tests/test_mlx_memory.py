from mlx_spatial.mlx_memory import clear_mlx_cache, mlx_memory_snapshot, reset_mlx_peak_memory


def test_mlx_memory_helpers_are_safe_to_call():
    reset_mlx_peak_memory()
    clear_mlx_cache()

    snapshot = mlx_memory_snapshot()

    assert snapshot.active_bytes is None or snapshot.active_bytes >= 0
    assert snapshot.peak_bytes is None or snapshot.peak_bytes >= 0
    assert snapshot.as_dict() == {
        "active_bytes": snapshot.active_bytes,
        "peak_bytes": snapshot.peak_bytes,
    }
