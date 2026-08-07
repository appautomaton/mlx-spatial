"""Independent stage timing for inference pipelines."""

from __future__ import annotations

import time
from typing import Callable


class StageTimer:
    """Record independent stage durations and total pipeline wall time."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._pipeline_started = clock()
        self.values: dict[str, float] = {}

    def begin(self) -> float:
        return self._clock()

    def end(self, name: str, stage_started: float) -> None:
        self.values[name] = max(0.0, self._clock() - stage_started)

    def snapshot(self) -> dict[str, float]:
        return {
            **self.values,
            "total": max(0.0, self._clock() - self._pipeline_started),
        }


__all__ = ["StageTimer"]
