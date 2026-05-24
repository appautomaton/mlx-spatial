# MLX Testing Strategy Plan

**Goal:** Establish a fast, CPU-bound MLX testing strategy with tiny shapes by default, segregated `@pytest.mark.heavy` tests, and a strict-timeout GitHub Actions CI workflow.

## Execution Routing and Topology
- **Topology:** Slices run serially (Slice 1 → 2 → 3 → 4).
- **Execution:** Direct execution for all slices.
- **Checkpoints:** None.

## Slice Sequence

### Slice 1: Pytest Configuration & CPU Default
**Objective:** Add a global pytest configuration to register the `heavy` marker, exclude it by default, and enforce MLX CPU execution.
**Acceptance criteria:**
- `pyproject.toml` (or `pytest.ini`) registers the `heavy` marker.
- Running `pytest` automatically excludes `heavy` tests via `addopts = "-m 'not heavy'"`.
- A global fixture (e.g., in `tests/conftest.py` `pytest_sessionstart` or similar auto-use fixture) calls `mlx.core.set_default_device(mlx.core.cpu)`.
**Verification:** `uv run pytest --collect-only` does not error, and inspecting `mx.default_device()` inside a test confirms CPU.
**Status:** complete
**Evidence:** changed `pyproject.toml`, `tests/conftest.py`, and `tests/test_pytest_config.py`; `uv run pytest --collect-only -q` collected 744 default tests with 26 deselected, `uv run pytest tests/test_pytest_config.py -q` passed, and the MLX CPU probe reported `Device(cpu, 0)`.
**Risks / next:** CPU default exposed a sparse convolution kernel-layout issue, fixed by making transposed sparse-conv kernels contiguous in `src/mlx_spatial/sam3d_slat.py` and `src/mlx_spatial/trellis2_decode.py`.

### Slice 2: Tag Existing Heavy Tests
**Objective:** Audit existing tests (specifically `test_lito_real_backend.py` and any tests hitting `weights/` or using large shapes) and mark them with `@pytest.mark.heavy`.
**Acceptance criteria:**
- Tests that load files from `weights/` or execute heavy Metal-intended computations are decorated with `@pytest.mark.heavy`.
- Running a default `uv run pytest` skips these tests and executes fast.
- Running `uv run pytest -m heavy` collects and runs them.
**Verification:** `uv run pytest tests/test_lito_real_backend.py` reports the majority of tests as "deselected", and completes almost instantly.
**Status:** complete
**Evidence:** marked real-weight LiTo/TRELLIS tests, explicit Metal tests, and model-scale fake DINO tests as `heavy`; `uv run pytest tests/test_lito_real_backend.py -q` passed with 31 selected fast tests and 17 deselected real-weight tests, and `uv run pytest -m heavy -q` passed with 24 passed, 2 skipped, and 744 deselected.
**Risks / next:** The original "majority deselected" wording was narrowed by audit to avoid hiding fake-weight LiTo backend coverage from the default suite.

### Slice 3: GitHub Actions CI Workflow
**Objective:** Create a GitHub Actions workflow to run the fast test suite on all pushes to any branch.
**Acceptance criteria:**
- `.github/workflows/test.yaml` is created.
- Triggers on `push` to *all* branches, so that builds run even if a PR is not explicitly opened yet.
- Uses `ubuntu-latest` and `astral-sh/setup-uv`.
- Runs `uv run pytest` (which relies on Slice 1's default skip).
- The test step has `timeout-minutes: 5` explicitly set to fail fast on memory/computation hangs.
**Verification:** Run `yamllint .github/workflows/test.yaml` (if available) or visually inspect the file for correct syntax and timeout.
**Status:** complete
**Evidence:** added `.github/workflows/test.yaml` with push-all-branches trigger, `ubuntu-latest`, `astral-sh/setup-uv@v5`, `uv run pytest`, and a 5-minute pytest-step timeout; PyYAML parsed the workflow successfully.
**Risks / next:** none.

### Slice 4: Testing Documentation
**Objective:** Document the testing boundary rules so human and agentic contributors know how to write fast MLX tests.
**Acceptance criteria:**
- `tests/README.md` or a section in the main `README.md` explains the strategy.
- Explains the requirement to use `shape=(1, 4, 32)` rather than `shape=(1, 1024, 1024)` in default tests.
- Explains what `@pytest.mark.heavy` is for and how CI is configured to enforce the 5-minute timeout.
**Verification:** `cat tests/README.md` shows the testing strategy section.
**Status:** complete
**Evidence:** added `tests/README.md`; `sed -n '1,80p' tests/README.md` shows the CPU-default, tiny-shape, `@pytest.mark.heavy`, manual heavy-run, and CI-timeout guidance.
**Risks / next:** none.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The plan is tightly scoped to pytest configuration, test markers, CI, and docs, so it does not alter runtime package APIs or model execution paths.
- Concern: The main regression risk is hiding useful coverage if `heavy` is applied too broadly or if the CPU fixture leaves Metal/backend-specific checks neither run by default nor reliably runnable with `pytest -m heavy`.
- Action: Execute Slice 2 with an explicit marker audit for weight-backed, large-shape, and Metal-specific tests, then verify both default `uv run pytest` and opt-in `uv run pytest -m heavy` collection behavior.
- Verified: STATUS.md, SPEC.md, PLAN.md, pyproject pytest config, existing workflow layout, selected LiTo/TRELLIS/Metal tests, `uv run pytest --collect-only -q`, and MLX CPU device probe.
