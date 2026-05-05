"""Small MLX memory helpers used by staged local inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class MlxMemorySnapshot:
    """Point-in-time MLX allocator counters."""

    active_bytes: int | None
    peak_bytes: int | None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "active_bytes": self.active_bytes,
            "peak_bytes": self.peak_bytes,
        }


def mlx_memory_snapshot() -> MlxMemorySnapshot:
    """Return MLX active/peak memory counters when the runtime exposes them."""

    return MlxMemorySnapshot(
        active_bytes=_optional_memory_counter("get_active_memory"),
        peak_bytes=_optional_memory_counter("get_peak_memory"),
    )


def reset_mlx_peak_memory() -> None:
    """Reset MLX peak-memory accounting when supported."""

    reset = getattr(mx, "reset_peak_memory", None)
    if reset is not None:
        reset()


def clear_mlx_cache() -> None:
    """Synchronize outstanding work and release unused MLX allocator cache."""

    synchronize = getattr(mx, "synchronize", None)
    if synchronize is not None:
        synchronize()
    clear = getattr(mx, "clear_cache", None)
    if clear is not None:
        clear()


def _optional_memory_counter(name: str) -> int | None:
    counter = getattr(mx, name, None)
    if counter is None:
        return None
    return int(counter())
