# PLAN: mlx-spatialkit Metal GIL Release

## Goal

Implement `.agent/work/2026-05-27-mlx-spatialkit-metal-gil-release/SPEC.md` by hardening the Metal bake wait boundary and proving concurrent public API behavior.

## Ordered Slice Sequence

### Slice 1: Native GIL Release Boundary

**Objective:** Release the Python GIL only while the Metal command buffer is committed and waited on.
**Acceptance criteria:**
- `texture_bake.mm` releases the GIL only around non-Python Metal wait code.
- Existing deterministic texture bake tests still pass.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_texture_bake.py -q`
**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`
**Status:** complete
**Evidence:** `texture_bake.mm` now releases the GIL only around Metal command-buffer commit/wait/error inspection. Focused verification passed: `10 passed`.
**Risks / next:** CPU postprocessing and nanobind result construction remain GIL-held by design.

### Slice 2: Concurrent Bake Coverage And Docs

**Objective:** Add public API concurrency coverage and document the runtime boundary.
**Acceptance criteria:**
- A test runs multiple `bake_pbr_texture` calls through a thread pool and compares deterministic outputs.
- Runtime docs mention the GIL release boundary and memory-monitor benefit without overstating full export parallelism.
- Roadmap/Automaton state reflect this verified hardening cycle.
**Verification:** `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
**Touches:** `packages/mlx-spatialkit/tests/test_texture_bake.py`, `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`, `.agent/steering/ROADMAP.md`
**Status:** complete
**Evidence:** Added concurrent public API bake coverage and docs for the GIL release boundary. Package verification passed: `72 passed, 7 deselected`; `/tmp/mlx-spatialkit-dist-metal-gil` wheel/sdist built and archive checks passed; `git diff --check` passed.
**Risks / next:** this proves texture-bake API concurrency, not full export parallelism.

## Execution Routing And Topology

Default route: direct.

Parallel-safe groups: none. The runtime boundary and tests should land together so the contract stays precise.

## Aggregate Verification Commands

- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest tests/test_texture_bake.py -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run pytest -q`
- `cd packages/mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build --out-dir /tmp/mlx-spatialkit-dist-metal-gil`
