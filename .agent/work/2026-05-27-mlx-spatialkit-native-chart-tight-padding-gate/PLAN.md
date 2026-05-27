# mlx-spatialkit Native Chart Tight Padding Gate Plan

## Goal

Execute `.agent/work/2026-05-27-mlx-spatialkit-native-chart-tight-padding-gate/SPEC.md`: tighten native-chart default padding to `0.01` and verify real fixture occupancy progress.

## Architecture Approach

Change only the Pixal3D backend default for `uv_backend="native-chart"`; keep face-atlas padding, explicit caller padding, UV generation, texture bake, and readiness semantics unchanged. Use the existing diagnostics to prove the new default and measured occupancy change.

## Execution Routing And Topology

- Default execution: direct, serial, continue after verification.
- Parallel-safe groups: none.
- Checkpoints: none.
- Commit rhythm: commit after full verify; no push, tag, publish, or release metadata changes.

## Ordered Slice Sequence

### Slice 1: Native-Chart Padding Default

**Objective:** Set native-chart default padding to `0.01` and prove reference-target native-chart occupancy advances.

**Acceptance criteria:**
- Resolver/unit and real-fixture tests assert `tile_padding=0.01` with `backend_default:native-chart`.
- Reference-target native-chart heavy gate reports UV-surface occupancy above `0.5396347045898438` or xatlas utilization ratio above `0.6494046474199308`.
- Parity diagnostics remain measured and non-ready.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_rejects_invalid_public_guards -q && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_reference_target_native_chart_backend_reports_readiness -q -m heavy`

**Touches:** `packages/mlx-spatialkit/src/mlx_spatialkit/export.py`, `packages/mlx-spatialkit/tests/test_real_pixal3d_export.py`

**Status:** complete
**Evidence:** resolver guard passed; reference-target native-chart heavy gate passed. Latest `/tmp` diagnostics report `tile_padding=0.01`, `tile_padding_source=backend_default:native-chart`, UV-surface occupancy `0.5597553253173828` vs Phase 28 `0.5396347045898438`, xatlas utilization ratio `0.6736181097830843` vs Phase 28 `0.6494046474199308`, final visible coverage `0.5597553253173828`, and `parity_ready=false`.
**Risks / next:** explicit 1M/4096 gate and docs remain.

### Slice 2: Upstream Gate, Docs, And Package Hygiene

**Objective:** Keep explicit 1M/4096 native-chart readiness intact, document the tighter native-chart default, and verify package hygiene.

**Acceptance criteria:**
- Explicit 1M/4096 native-chart heavy gate still passes upstream-setting and native-chart readiness.
- Docs mention `0.01` native-chart default padding without claiming xatlas equivalence.
- Full package tests, root Pixal3D tests, build, and artifact inspection pass with generated outputs under `/tmp`.

**Verification:** `git diff --check && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests/test_real_pixal3d_export.py::test_export_pixal3d_glb_native_chart_upstream_settings_passes_readiness_gate -q -m heavy && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv run --directory packages/mlx-spatialkit --reinstall-package mlx-spatialkit pytest tests -q && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv run pytest tests/test_pixal3d_export.py tests/test_pixal3d_pipeline.py -q && rm -rf /tmp/mlx-spatialkit-dist && UV_CACHE_DIR=/tmp/mlx-spatialkit-uv-cache uv build packages/mlx-spatialkit --out-dir /tmp/mlx-spatialkit-dist --clear`

**Depends on:** Slice 1

**Touches:** `packages/mlx-spatialkit/README.md`, `docs/pixal3d.md`, `scripts/README.md`

**Status:** complete
**Evidence:** `git diff --check` passed; explicit 1M/4096 native-chart heavy gate passed in `45.46s` with `tile_padding=0.01`, upstream readiness true, and native-chart quality-ready; package tests passed `64 passed, 7 deselected`; root Pixal3D tests passed `35 passed`; `/tmp` build produced wheel and sdist; artifact inspection reported wheel `bad 0` and sdist `bad 0`; docs grep found `0.01` native-chart padding language.
**Risks / next:** final verify passed; broader xatlas chart parity remains open.

## Requirement Traceability

| Spec AC | Covered by |
|---|---|
| NCTP-01 | Slice 1 |
| NCTP-02 | Slice 1 |
| NCTP-03 | Slice 1 |
| NCTP-04 | Slice 2 |
| NCTP-05 | Slice 2 |
| NCTP-06 | Slice 2 |

## Execution Notes

- Keep generated and heavy artifacts under `/tmp`.
- Do not change face-atlas defaults or explicit caller padding behavior.
- Do not relax readiness thresholds or parity readiness.
