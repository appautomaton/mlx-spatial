# Testing Strategy

Default test runs are CPU-bound and fast. `tests/conftest.py` sets MLX to the CPU device at session start, and pytest uses `-m "not heavy"` by default.

Use tiny tensors in default tests. Prefer shapes like `shape=(1, 4, 32)` for behavioral checks, not full model-scale shapes like `shape=(1, 1024, 1024)`, unless the test is explicitly marked heavy.

Mark tests with `@pytest.mark.heavy` when they load real files from `weights/`, require Metal-specific execution, or allocate model-scale tensors. Run them manually with:

```bash
export MLX_SPATIAL_TEST_SCRATCH="$(mktemp -d /tmp/mlx-spatial-test.XXXXXX)"
mkdir -p "$MLX_SPATIAL_TEST_SCRATCH"/{inputs,outputs,artifacts,logs,cache}
export PYTHONPYCACHEPREFIX="$MLX_SPATIAL_TEST_SCRATCH/cache/pycache"
uv run pytest -m heavy \
  --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest-heavy"
```

All generated test inputs, outputs, parity bundles, caches, logs, and browser
artifacts must stay below that task root. Optional local reference checkouts use
explicit environment variables such as `MLX_SPATIAL_TORCH_ROOT`; committed
tests and anchor metadata must not contain developer-machine absolute paths.

The GitHub Actions workflow runs the unified root suite, including
`tests/spatialkit`, on every branch push. The job creates the same isolated
scratch layout and has a 10-minute timeout so leaked heavy tests fail quickly.
