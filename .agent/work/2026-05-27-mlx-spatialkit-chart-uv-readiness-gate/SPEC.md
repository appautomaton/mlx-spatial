# mlx-spatialkit Chart UV Readiness Gate Spec

## Bounded Goal

Make native chart UV export diagnostics distinguish artifact readiness from quality readiness, with explicit coverage and UV-utilization blockers when the chart backend writes a GLB but is not visually production-ready.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: diagnostics hardening
- Shape: quality gate

## Selected Lenses

- **engineering:** Turn chart UV candidate status into a structured readiness gate rather than a generic success label.
- **runtime:** Use existing texture bake diagnostics to explain whether chart exports are limited by backend wiring, UV-bin guards, sparse sampling, global coverage, or atlas occupancy.
- **product:** A real chart export that writes a GLB must not look production-ready when coverage/utilization evidence says it is not.

## Current Evidence

- Phase 14 proved `uv_backend="native-chart"` writes a GLB and bakes through `metal-uv-binned-nearest`.
- The real fixture chart export observed `final_visible_coverage_ratio=0.14284706115722656`, `uv_surface_texel_count=243931`, and `texture_pixel_count=1048576`, so the chart atlas occupies only about 23% of texels before dilation.
- Current `native_chart_uv_candidate` diagnostics say `status=candidate` and do not report coverage readiness, atlas occupancy, or chart-specific quality blockers.
- The default face-atlas path remains stable and must stay default.

## Required Outcome

1. `quality.native_chart_uv_candidate` reports `artifact_ready`, `quality_ready`, `status`, `checks`, and `quality_blockers`.
2. Chart exports report global coverage, UV-surface occupancy, UV-surface visible coverage, UV-bin diagnostics, chart count, duplicate ratio, and xatlas parity status in one structured object.
3. Native chart exports that write a GLB but miss coverage/utilization floors are labeled `status=quality_blocked`, not simply `candidate`.
4. `result.quality_warnings` includes a chart-specific warning when the native chart candidate is artifact-ready but quality-blocked.
5. Docs explain how to read chart readiness diagnostics and why this remains separate from xatlas parity.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| CUVR-01 | Readiness helper separates not-requested, artifact-blocked, quality-blocked, and quality-ready states. | Focused unit tests call `_native_chart_uv_candidate_status` with synthetic stats. |
| CUVR-02 | Real chart fixture is not hidden behind success. | Heavy chart export reports `artifact_ready=true`, `quality_ready=false`, `status=quality_blocked`, and chart-specific warning. |
| CUVR-03 | Readiness diagnostics identify the actual low-coverage issue. | Heavy chart export reports failed global coverage and UV-surface occupancy checks, with observed ratios. |
| CUVR-04 | Existing default/reference gates remain stable. | Full package tests, root Pixal3D tests, build, and artifact inspection pass. |
| CUVR-05 | Docs match the contract. | Docs describe chart readiness as an opt-in candidate gate, not xatlas parity or a default replacement. |

## Scope Coverage Decisions

- **Included:** readiness summary helper, result warning propagation, focused tests, real fixture assertions, docs, full regression.
- **Deferred:** improving chart packing, switching defaults, removing `not_xatlas_chart_parity`, exact perceptual scoring.
- **Anti-goals:** relaxing quality thresholds, claiming production chart parity, changing decoded model outputs, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep `uv_backend="face-atlas"` as the default.
- Do not change Metal bake behavior in this phase.
- Keep chart readiness thresholds explicit and diagnostics-first.
- Keep generated and heavy artifacts under `/tmp`.

## Risks

- **Terminology risk:** `quality_blocked` could be misread as export failure. Mitigation: expose both `artifact_ready` and `quality_ready`.
- **Threshold risk:** Coverage/utilization floors are first-pass production-readiness floors, not xatlas parity. Mitigation: docs and `xatlas_chart_parity=false`.

## Blocking Questions Or Assumptions

Assumption: the next correct step is to make chart export readiness truthful before optimizing chart packing quality in a later phase.
