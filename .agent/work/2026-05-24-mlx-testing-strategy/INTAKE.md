# Intake: MLX Testing Strategy

## Scale and Shape
- Scale: capability
- Shape: coverage

## Objective
Establish a reliable testing strategy for MLX code that executes on a CPU backend with tiny random shapes for fast default tests, while relegating real-shape/real-weight tests to an opt-in `@pytest.mark.heavy` boundary to accommodate standard CPU CI runners.

## Broader Intent
Keep the test suite fast and lightweight without mocking MLX core out completely, preventing false confidence in tensor operations while preserving a tight developer loop.

## Target User
`mlx-spatial` contributors running tests locally or in standard CPU continuous integration environments.

## Desired Outcome
Running `pytest` executes quickly by default, compiling and running real MLX math on tiny shapes. Heavy tests with real weights or large shapes are preserved in the suite but skipped by default to ensure CI and local runs remain fast and do not crash on memory limits.

## Scope Boundary and Anti-Goals
- We will not implement a full "shape-only" mock for `mlx.core.array`.
- We will not enforce that `@pytest.mark.heavy` tests strictly require a Metal GPU; they test large dimensions/weights and can technically run on a CPU, albeit very slowly.
- We will not run heavy tests by default in standard CI environments.

## Rejected Framings
- **Shape-Only Mocking (Approach B):** Ruled out because mocking MLX arrays entirely leads to false confidence, hiding unsupported broadcasts or compilation errors.

## Scope Preservation
Preserves the user's full stated intent to properly test the MLX project without the tests becoming a burden.

## Scope Coverage
- **Included:**
  - Pytest configuration to enforce CPU execution by default (`mx.set_default_device(mx.cpu)`).
  - Introduction of the `@pytest.mark.heavy` marker, configured to skip by default.
  - Migrating existing test patterns to use tiny tensor shapes for baseline execution.
- **Anti-goals:**
  - Mocking out `mlx.core` math operations.
  - Assuming Apple Silicon runners in CI.

## Selected Approach
**Hybrid "Tagging" Strategy (Approach C) with Large Shapes/Weights.**
*Rationale:* Gives fast, tiny CPU executions for the daily dev loop and CI runners, while preserving heavy verification for manual or specialized execution, avoiding the false confidence of pure Python mocking.

## Key Assumptions and Risks
- **Assumption:** Executing MLX math on the CPU with tiny shapes is fast enough to keep the unit test suite lightweight and avoid resource hunger.
- **Risk:** Developers may forget to use tiny shapes when writing new default tests, causing CI to slow down or hit memory limits. Mitigation: Document the testing pattern and rely on code review or a timeout wrapper.

## Deferred Scope
None.
