# Testing Strategy

The routine suite is bounded, self-contained, and representative. It includes
unit tests and miniature integration fixtures, while excluding tests marked
`heavy`, `real_assets`, or `torch_parity`.

```bash
uv run pytest
```

`tests/conftest.py` starts MLX on the CPU. A test may deliberately exercise
Metal when that boundary is important; such a test carries the `metal` marker.
Metal capability and test cost are separate concerns.

## Markers

| Marker | Meaning | Routine suite |
| --- | --- | --- |
| `integration` | Crosses multiple production component boundaries | Yes, unless paired with an exclusion marker |
| `metal` | Requires an Apple Metal device | Yes, when bounded |
| `real_assets` | Reads local weights, inputs, or uncommitted fixtures | No |
| `heavy` | Has substantial runtime or memory cost | No |
| `benchmark` | Measures resource or performance behavior; also marked `heavy` | No |
| `torch_parity` | Requires the opt-in PyTorch reference environment | No |

Useful focused commands are:

```bash
uv run pytest -m 'integration and not (heavy or real_assets or torch_parity)'
uv run pytest -m 'real_assets and not heavy'
uv run pytest -m heavy
uv run pytest -m benchmark
uv run pytest -m torch_parity
```

## Test Design

- Prefer the smallest tensor shapes that preserve the production branch under
  test.
- Keep one representative cross-component fixture per pipeline. Add narrower
  tests only when they protect a distinct contract or failure mode.
- Keep each supported pipeline's primary guard in
  `ROUTINE_PIPELINE_GUARDS`. A primary guard must remain an `integration` test
  and must not carry a routine-exclusion marker.
- Assert stable stage, schema, shape, and numerical-summary contracts. Do not
  accept a broad set of unrelated blockers as success.
- Record what a fixture covers and does not cover. Synthetic checkpoints must
  not be presented as real-weight numerical parity.
- Write all generated files below `tmp_path` or the task scratch root. The
  repository `outputs/` directory belongs to user-requested inference runs.

The bounded pipeline fixtures cover TRELLIS.2 selective INT8 inference,
Pixal3D orchestration and derived decoder replay, LiTo source-contract
generation, SAM3D CLI reconstruction, HY-World 2 reconstruction, and the full
MapAnything scene path. They remain runnable after local model weights are
removed.

## Isolated Runs

Use a task-specific scratch root for any focused or expensive run:

```bash
export MLX_SPATIAL_TEST_SCRATCH="$(mktemp -d /tmp/mlx-spatial-test.XXXXXX)"
mkdir -p "$MLX_SPATIAL_TEST_SCRATCH"/{inputs,outputs,artifacts,logs,cache}
export PYTHONPYCACHEPREFIX="$MLX_SPATIAL_TEST_SCRATCH/cache/pycache"
uv run pytest -m heavy \
  --basetemp "$MLX_SPATIAL_TEST_SCRATCH/artifacts/pytest-heavy"
```

Optional reference checkouts use explicit environment variables such as
`MLX_SPATIAL_TORCH_ROOT`. Committed fixtures and metadata must not contain
developer-machine absolute paths.

GitHub Actions runs the routine root suite, including `tests/spatialkit`, on
pushes and pull requests. The job uses isolated scratch storage and a 10-minute
timeout.
