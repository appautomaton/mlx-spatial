# mlx-spatialkit Peak Memory Telemetry Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-peak-memory-telemetry-gate/SPEC.md`: add stage-level peak RSS telemetry to Pixal3D `export_pixal3d_glb` diagnostics.

## Architecture Approach

Use `.agent/work/2026-05-27-mlx-spatialkit-peak-memory-telemetry-gate/DESIGN.md`. Keep telemetry private to `export.py`, dependency-free beyond the standard library, and aggregate-only so memory monitoring does not become a new memory risk.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after the full verify gate; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Process Memory Monitor

**Objective:** Add a private thread-safe process memory monitor with deterministic unit coverage.

**Acceptance criteria:**
- Monitor records observed peak current RSS and max RSS high-water values.
- Monitor records per-stage start/end/peak aggregates without storing an unbounded sample log.
- Unit tests use a fake sample provider and do not depend on host memory behavior.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_memory_monitor.py -q`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_memory_monitor.py`

**Status:** complete
**Evidence:** Added `_ProcessMemoryMonitor` and `_MemoryStageScope` in `export.py` with lock-protected aggregate peaks, per-stage boundaries, and bounded summaries. Added deterministic fake-sample coverage in `tests/test_memory_monitor.py`. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_memory_monitor.py -q` passed with `2 passed`.
**Risks / next:** Slice 2 must prove the monitor is wired into the real Pixal3D export diagnostics.

### Slice 2: Export Diagnostics Integration

**Objective:** Wire the monitor into `export_pixal3d_glb` stage timing and diagnostics.

**Acceptance criteria:**
- `diagnostics["memory"]` includes sample source, poll interval, sample count, observed peaks, and `stage_peaks`.
- `memory_samples` labels such as `after_write_glb` remain present.
- Heavy real fixture diagnostics include stage peaks for `texture_bake` and `write_glb`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** Wired `_ProcessMemoryMonitor` into every `export_pixal3d_glb` `_timed_stage`, preserved existing `memory_samples`, and added heavy fixture assertions for `diagnostics["memory"]` and stage peaks. `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py -q -m heavy` passed with `2 passed, 3 deselected`. Latest `/tmp/mlx-spatialkit-reference-target-export-38128/diagnostics.json` reports `memory.sample_count=40`, `peak_current_rss_bytes=3591766016`, and stage peaks for `texture_bake`, `write_glb`, and `visual_compare`.
**Risks / next:** Slice 3 must document that the numbers are observed process RSS/high-water telemetry, not full system pressure or Metal allocator accounting.

### Slice 3: Docs And Full Verification

**Objective:** Document telemetry semantics and verify package/root/build hygiene.

**Acceptance criteria:**
- Docs explain that memory telemetry is observed host process RSS/high-water data, not full system or Metal allocator accounting.
- Full package tests pass.
- Root Pixal3D tests pass.
- Wheel/sdist build into `/tmp` passes and artifact inspection excludes generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, pycache, and pytest cache files.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 2

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** Updated `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, and `scripts/README.md` to describe observed process RSS/high-water telemetry and explicitly exclude full system pressure, Activity Monitor equivalence, MLX allocator state, and Metal heap residency. Full verification passed: `git diff --check`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q` passed with `48 passed, 2 deselected`; `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q` passed with `35 passed`; `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear` built the wheel and sdist under `/tmp`; artifact inspection found no generated outputs, inputs, diagnostics, GLBs, visual parity sidecars, pycache, or pytest cache entries.
**Risks / next:** Ready for independent Automaton verify. Deeper memory optimization and Metal allocator accounting remain deferred.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| MTG-01 | Slice 1 |
| MTG-02 | Slice 2 |
| MTG-03 | Slice 2 |
| MTG-04 | Slice 3 |
| MTG-05 | Slice 3 |

## Execution Notes

- Do not claim exact Activity Monitor equivalence.
- Do not add new runtime dependencies.
- Heavy/generated artifacts stay under `/tmp`.
- Broad thread goal remains open after this cycle; rendered GLB proof and deeper memory optimization remain separate work.
