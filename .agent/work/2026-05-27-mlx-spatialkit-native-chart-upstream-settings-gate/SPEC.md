# mlx-spatialkit Native Chart Upstream Settings Gate Spec

## Bounded Goal

Promote the passing explicit `target_faces=1000000`, `texture_size=4096`, `uv_backend="native-chart"` Pixal3D export path into a verified real-fixture gate with honest readiness diagnostics and docs.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native export readiness gate
- Shape: coverage/parity evidence hardening

## Selected Lenses

- **engineering:** Convert a passing 1M/4096 native-chart probe into a durable regression gate without hiding visual/reference mismatch semantics.
- **runtime:** Keep the path on existing native C++/Metal hot paths and generated artifacts under `/tmp`.
- **product:** Make upstream-setting readiness readable while preserving the remaining xatlas chart parity boundary.

## Current Evidence

- Manual probe output: `/tmp/mlx-spatialkit-native-chart-upstream-settings-probe-32257/diagnostics.json`.
- Probe settings: `quality_preset="reference-target"`, `target_faces=1000000`, `texture_size=4096`, `uv_backend="native-chart"`, `chart_angle_degrees=45.0`.
- Probe result: `artifact_ready=true`, `quality.upstream_export_settings.all_passed=true`, `native_chart_uv_candidate.status=quality_ready`.
- Probe geometry/texture: final faces `911927`, target reached true, chart count `118575`, global coverage `0.5028553009033203`, UV-surface visible coverage `1.0`.
- Probe boundaries: visual comparison `all_passed=false` because the checked-in reference GLB is the 1024 reference-target artifact, while deferred parity boundaries shrink to `["not_xatlas_chart_parity"]`.
- Probe memory: peak current RSS about `4.133 GiB` at `visual_compare`.

## Required Outcome

1. A heavy real-fixture test covers explicit 1M/4096 native-chart Pixal3D export.
2. The test asserts artifact readiness, upstream-setting readiness, native chart quality readiness, UV-bin guard pass, memory diagnostics, and GLB viewer compatibility.
3. The test asserts the honest visual/reference boundary: 1024-reference visual comparison is not all-passed because face-count and texture-resolution checks mismatch, while only xatlas chart parity remains deferred.
4. Docs explain that native-chart 1M/4096 readiness closes the upstream-setting boundary for this backend but does not claim xatlas chart parity or reference-target visual comparability against the 1024 GLB.
5. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCUSG-01 | Explicit 1M/4096 native-chart path is a real fixture gate. | Heavy test writes GLB/diagnostics under `/tmp` and asserts artifact readiness. |
| NCUSG-02 | Upstream-setting readiness is explicit. | Heavy test asserts `quality.upstream_export_settings.all_passed=true` and all checks pass. |
| NCUSG-03 | Native chart readiness is explicit. | Heavy test asserts `native_chart_uv_candidate.status=quality_ready`, coverage floors pass, and `xatlas_chart_parity=false`. |
| NCUSG-04 | Visual comparison remains honest. | Heavy test asserts reference visual comparison is not all-passed for 1M/4096 versus 1024, and only xatlas remains deferred. |
| NCUSG-05 | Docs match the readiness boundary. | Docs mention native-chart 1M/4096 readiness without claiming xatlas or 1024-reference visual parity. |
| NCUSG-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Heavy explicit 1M/4096 native-chart test, diagnostics assertions, docs, regression/build verification.
- **Deferred:** xatlas chart parity, CUDA/cuMesh remesh parity, using the native-chart backend as the default, perceptual visual equivalence beyond deterministic GLB/texture metrics.
- **Anti-goals:** changing decoded model outputs, relaxing thresholds, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Do not change runtime behavior unless the new gate exposes a real bug or a misleading diagnostic.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Runtime risk:** The 1M/4096 native-chart gate is heavier than the 1024 reference gate. Mitigation: keep it marked `@pytest.mark.heavy` and assert memory diagnostics.
- **Evidence risk:** The checked-in reference GLB is 1024, so visual `all_passed=false` is expected for explicit 1M/4096. Mitigation: assert the exact failed visual checks and the remaining deferred parity boundary.

## Blocking Questions Or Assumptions

Assumption: a 45-second local 1M/4096 native-chart gate is acceptable as a heavy test because it covers a production-critical export boundary.
