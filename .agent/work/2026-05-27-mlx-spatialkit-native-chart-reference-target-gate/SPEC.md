# mlx-spatialkit Native Chart Reference-Target Gate Spec

## Bounded Goal

Promote the passing `reference-target` native-chart Pixal3D export path into a verified real-fixture gate with clear diagnostics and docs.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native export readiness gate
- Shape: coverage/parity evidence hardening

## Selected Lenses

- **engineering:** Convert a passing manual probe into a durable regression gate without broadening runtime behavior.
- **runtime:** Keep the path under existing native C++/Metal hot paths and `/tmp` artifacts.
- **product:** Strengthen visual-comparability evidence for the intended Pixal3D reference-target path while preserving explicit parity boundaries.

## Current Evidence

- Manual probe output: `/tmp/mlx-spatialkit-native-chart-reference-target-probe-27288/diagnostics.json`.
- Probe settings: `quality_preset="reference-target"`, `uv_backend="native-chart"`, `texture_size=1024`, target faces resolved to `212542`.
- Probe result: `artifact_ready=true`, `production_quality_ready=true`, `quality_warnings=[]`.
- Probe native chart candidate: `status=quality_ready`, `global_coverage_ratio=0.5047416687011719`, `uv_surface_final_visible_coverage_ratio=1.0`, `xatlas_chart_parity=false`.
- Probe visual comparison: `all_passed=true`, face-count ratio `0.9344882423238701`, texture resolution match true, deferred boundaries `["not_xatlas_chart_parity", "not_1m_face_export_setting_parity"]`.

## Required Outcome

1. A heavy real-fixture test covers `quality_preset="reference-target"` with `uv_backend="native-chart"`.
2. The test asserts production readiness, native chart quality readiness, visual comparison pass, UV-bin guard pass, raw/exact/final coverage separation, and explicit deferred parity boundaries.
3. Docs explain that reference-target native-chart now passes scalar production/visual gates while xatlas and 1M/4096 parity remain deferred.
4. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCRTG-01 | Reference-target native-chart path is a real fixture gate. | Heavy test writes GLB/diagnostics under `/tmp` and asserts `production_quality_ready=true`. |
| NCRTG-02 | Native chart readiness is explicit. | Heavy test asserts `native_chart_uv_candidate.status=quality_ready`, `quality_blockers=[]`, and `xatlas_chart_parity=false`. |
| NCRTG-03 | Visual comparison remains honest. | Heavy test asserts visual summary passes and deferred boundaries retain xatlas and 1M setting parity. |
| NCRTG-04 | Diagnostics remain high-signal. | Heavy test asserts raw/exact/final coverage and surface-fill counts remain separate. |
| NCRTG-05 | Docs match the readiness boundary. | Docs mention reference-target native-chart readiness without claiming xatlas or upstream-setting parity. |
| NCRTG-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Heavy reference-target native-chart test, diagnostics assertions, docs, regression/build verification.
- **Deferred:** 1M/4096 native-chart gate, browser-render native-chart proof, xatlas chart parity, default UV backend switch.
- **Anti-goals:** changing decoded model outputs, relaxing thresholds, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Do not change runtime behavior unless the new gate exposes a real bug.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Runtime risk:** The heavy gate adds another real-fixture path. Mitigation: keep it explicit and `@pytest.mark.heavy`.
- **Evidence risk:** Passing scalar/PNG comparison is not xatlas chart parity. Mitigation: assert deferred parity boundaries in the test and docs.

## Blocking Questions Or Assumptions

Assumption: reference-target native-chart readiness is stable enough to promote from `/tmp` probe to heavy regression gate.
