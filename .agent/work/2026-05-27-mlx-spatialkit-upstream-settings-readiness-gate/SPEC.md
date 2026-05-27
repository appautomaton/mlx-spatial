# mlx-spatialkit Upstream Settings Readiness Gate Spec

## Bounded Goal

Make explicit Pixal3D upstream export settings (`target_faces=1000000`, `texture_size=4096`) artifact-ready and coverage-threshold-ready in spatialkit diagnostics without claiming xatlas chart parity.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: capability hardening
- Shape: parity and runtime diagnostics

## Selected Lenses

- **engineering:** Support the upstream Pixal3D 1M/4096 export-setting path through native C++/Metal behavior and explicit diagnostics.
- **runtime:** Keep the high-density atlas path memory-observable and bounded.
- **product:** A developer should know whether the native backend can run upstream-like export settings, separately from xatlas chart parity.

## Current Evidence

Vendored Pixal3D uses:

- `decimation_target=1000000`
- `texture_size=4096`
- `remesh=True`
- `remesh_band=1`
- `remesh_project=0`

A `/tmp` spatialkit probe with `quality_preset="reference-target"`, `target_faces=1000000`, and `texture_size=4096` produced:

- `artifact_ready=true`
- `production_quality_ready=false`
- `target_faces=1000000`
- `target_faces_source=explicit`
- `simplified_faces=911927`
- `target_reached=true`
- `texture_size=4096`
- `final_visible_coverage_ratio=0.4261552095413208`
- `fallback_radius=13`
- `dilation_max_passes=13`
- observed peak RSS about `3.93 GB`

The native path can execute the upstream-sized setting, but the high-density face-atlas texture fill budget underfills coverage. Existing production thresholds also compare against the checked-in 1024 reference GLB, so they are not the right proof surface for upstream-setting readiness.

## Required Outcome

1. High-density 4096 face-atlas exports use a larger bounded fallback/dilation floor so explicit 1M/4096 export reaches final visible coverage threshold.
2. Pixal3D diagnostics include an explicit upstream export-setting readiness section separate from checked-in 1024 reference visual parity.
3. Upstream export-setting readiness can pass only when target faces, texture size, backend tier, target reach, face retention, artifact readiness, and coverage checks pass.
4. Visual comparison deferrals remove `not_1m_face_export_setting_parity` only when upstream export-setting readiness passes; xatlas chart parity remains deferred.
5. Docs explain that this closes the 1M/4096 setting boundary but does not implement xatlas charting, upstream CUDA remesh, or exact perceptual parity.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| USET-01 | Native high-density atlas fill has bounded larger budgets for 4096 dense atlases. | Focused/real diagnostics show `fallback_radius >= 24` and `dilation_max_passes >= 26` for 1M/4096. |
| USET-02 | Upstream setting readiness diagnostics are explicit. | Unit/focused test asserts target, texture, target-reach, face-retention, backend, artifact, and coverage checks. |
| USET-03 | Real 1M/4096 Pixal3D fixture passes upstream-setting readiness. | Heavy test under `/tmp` reports `upstream_export_settings.all_passed=true` and coverage >= 0.50. |
| USET-04 | Deferred visual boundary is truthful. | Default reference-target keeps `not_1m_face_export_setting_parity`; 1M/4096 readiness removes it while keeping `not_xatlas_chart_parity`. |
| USET-05 | Docs and hygiene hold. | Docs updated; package/root/build verification and artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** adaptive high-density fill floor, upstream-setting readiness diagnostics, 1M/4096 heavy real fixture gate, visual-boundary filtering, docs, package/root/build verification.
- **Deferred:** xatlas chart parity, CUDA/cuMesh remesh parity, exact perceptual scoring, replacing the checked-in 1024 reference GLB, changing model inference or decoded NPZ artifacts.
- **Anti-goals:** claiming xatlas parity, relaxing coverage thresholds, adding dependencies, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep heavy/generated artifacts under `/tmp`.
- Do not change decoded model outputs.
- Do not relax the existing final coverage threshold.
- Do not remove `not_xatlas_chart_parity`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Runtime cost risk:** Larger fallback/dilation budgets may make 1M/4096 slower. Mitigation: keep budgets bounded and assert memory diagnostics.
- **False parity risk:** Passing 1M/4096 settings could be mistaken for xatlas or CUDA remesh parity. Mitigation: keep xatlas charting explicitly deferred.
- **Reference mismatch risk:** The checked-in GLB reference is 1024 and 212k faces. Mitigation: keep upstream-setting readiness separate from reference visual comparison.

## Blocking Questions Or Assumptions

Assumption: closing `not_1m_face_export_setting_parity` means the native export can run explicit `target_faces=1000000` and `texture_size=4096` with artifact readiness, target reach, production-tier native backend, and coverage readiness; it does not mean xatlas chart parity or upstream CUDA remesh parity.
