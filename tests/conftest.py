from __future__ import annotations

import mlx.core as mx


def pytest_sessionstart(session):
    mx.set_default_device(mx.cpu)
