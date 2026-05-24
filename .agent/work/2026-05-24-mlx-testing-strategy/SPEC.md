# MLX Testing Strategy Spec

## Bounded Goal
Establish a reliable MLX testing strategy that forces CPU execution and uses tiny tensor shapes by default for fast, resource-efficient test runs, while relegating tests with real weights or large shapes to an opt-in `@pytest.mark.heavy` boundary, and wire this fast test suite into GitHub Actions CI for all branch pushes.

## Broader Intent
Keep the `mlx-spatial` test suite fast and lightweight without resorting to pure-Python shape-only mocks of `mlx.core`, ensuring we test actual tensor math compilation without the suite becoming a burden or crashing standard CPU CI runners. Tests serve as documentation and guardrails for agentic AI contributors.

## Work Scale and Shape
- Scale: capability
- Shape: coverage

## Selected Lenses
- **engineering**: Pytest configuration, fixture design, test segregation, memory/execution time limits, and CI automation.

## Target User
`mlx-spatial` contributors (both human and agentic AI) running tests locally or in standard CPU continuous integration environments.

## Scope Coverage Decisions
- **Included**:
  - A global pytest fixture/config to set `mx.set_default_device(mx.cpu)`.
  - Registration and configuration of a `@pytest.mark.heavy` marker that is skipped by default.
  - Converting or tagging existing tests to comply with the fast-by-default, tiny-shapes paradigm.
  - Adding a GitHub Actions workflow (`.github/workflows/test.yaml`) to run the default fast test suite on pushes to all branches.
  - Adding a strict timeout to the CI test run to ensure builds do not run forever if a heavy test accidentally leaks into the default run.
  - Adding documentation on how to write tests under this new strategy so agents and humans understand the boundaries.
- **Deferred to ROADMAP.md**: None.
- **Anti-goals**:
  - Mocking out `mlx.core` math operations (Approach B).
  - Enforcing that heavy tests strictly require Metal (they just require time/memory).
  - Expecting Apple Silicon runners in the default CI.
  - Running heavy tests in the default CI workflow.

## Constraints and Risks
- **Constraint**: Default tests must run instantly or near-instantly on a CPU.
- **Constraint**: Tests must compile and execute real MLX operations (no shape-only `MagicMock` over `mlx.core.array`).
- **Constraint**: CI is assumed to run on standard CPUs without MLX Metal acceleration (e.g., `ubuntu-latest`).
- **Risk**: Developers (or AI agents) might forget the standard and commit slow tests or tests with massive shapes, causing CI to hang. *Mitigation*: The test strategy documentation will explicitly cover the "tiny shape" requirement, and the GitHub Action will have a strict `timeout-minutes` (e.g., 5 or 10 minutes) at the job or step level to fail fast if a test hangs.

## Required Outcome
The test suite structure is modified to support fast CPU runs by default. A new GitHub Actions workflow is created that runs `pytest` (which skips `heavy` tests) on every push to any branch. The CI workflow is configured with a strict timeout to prevent infinite hangs. The `heavy` tests remain in the repo for manual execution but do not burden the automated CI.

## Acceptance Criteria
- **AC-1 (CPU Default)**: A global pytest configuration (e.g., in `conftest.py` or `pytest.ini`) ensures that `mx.set_default_device(mx.cpu)` is applied by default, preventing unexpected Metal allocations during standard test runs.
- **AC-2 (Heavy Marker)**: `@pytest.mark.heavy` is registered in the pytest configuration.
- **AC-3 (Default Skip)**: The pytest configuration is set up so that tests marked with `heavy` are excluded from the default `pytest` run (e.g., using `addopts = -m "not heavy"`).
- **AC-4 (Test Segregation)**: Existing long-running tests (like the ones currently hitting `weights/` in `test_lito_real_backend.py`) are tagged with `@pytest.mark.heavy`.
- **AC-5 (Tiny Shapes Examples)**: The core standard of "tiny shapes for fast tests" is applied to standard testing patterns (e.g., replacing large `mx.zeros` with `shape=(1, 4, 32)` in default tests).
- **AC-6 (Documentation)**: A testing strategy section is added to a relevant README or testing guide, outlining the difference between default tests (tiny shapes, CPU) and heavy tests.
- **AC-7 (GitHub Actions CI)**: A new workflow file `.github/workflows/test.yaml` is added. It triggers on `push` to *all branches*. It uses `uv` to install dependencies and runs `uv run pytest`.
- **AC-8 (CI Timeout Guard)**: The GitHub Actions `test` job (or the specific pytest step) has a `timeout-minutes` value set to prevent the build from running forever if a test hangs or allocates too much memory.

## Anti-Goals
- Implementing a shape-only mock for `mlx.core.array`.
- Enforcing that `@pytest.mark.heavy` tests strictly require a Metal GPU.
- Running heavy tests by default in standard CI environments.
