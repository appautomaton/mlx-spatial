# Testing Strategy

Default test runs are CPU-bound and fast. `tests/conftest.py` sets MLX to the CPU device at session start, and pytest uses `-m "not heavy"` by default.

Use tiny tensors in default tests. Prefer shapes like `shape=(1, 4, 32)` for behavioral checks, not full model-scale shapes like `shape=(1, 1024, 1024)`, unless the test is explicitly marked heavy.

Mark tests with `@pytest.mark.heavy` when they load real files from `weights/`, require Metal-specific execution, or allocate model-scale tensors. Run them manually with:

```bash
uv run pytest -m heavy
```

The GitHub Actions fast-test workflow runs `uv run pytest` on every branch push and gives the pytest step a 5-minute timeout so leaked heavy tests fail quickly.
