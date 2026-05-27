# mlx-spatialkit Chart UV Export Gate Spec

## Bounded Goal

Add an opt-in native chart-UV backend to `export_pixal3d_glb` and prove it on the real Pixal3D decoded fixture without changing the default face-atlas export path or claiming xatlas parity.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: export-path hardening
- Shape: parity candidate gate

## Selected Lenses

- **engineering:** Move native chart UVs from an isolated primitive into the real Pixal3D export API through an explicit backend switch.
- **runtime:** Keep the existing face-atlas default stable while chart UV exports bake through the binned Metal path and keep memory diagnostics.
- **product:** Developers should see whether the chart candidate is artifact-ready on real decoded Pixal3D data, with xatlas parity still reported honestly as deferred.

## Current Evidence

- Phase 13 added `make_native_chart_uvs`, but `export_pixal3d_glb` still always calls `make_face_atlas_uvs`.
- Real Pixal3D heavy tests assert the default UV backend remains `face-atlas` and texture bake backend remains `metal-face-atlas-nearest`.
- The remaining visual parity boundary still includes `not_xatlas_chart_parity`.
- The binned Metal UV path is already available for non-atlas UV meshes and reports bin diagnostics.

## Required Outcome

1. `export_pixal3d_glb` accepts `uv_backend="face-atlas"` or `uv_backend="native-chart"` plus a bounded `chart_angle_degrees` setting.
2. The default remains `face-atlas`, preserving existing Pixal3D behavior and diagnostics.
3. `uv_backend="native-chart"` calls `make_native_chart_uvs`, records requested/resolved backend settings, preserves chart diagnostics, and writes GLB metadata that identifies the UV backend.
4. Real Pixal3D decoded fixture export with the native chart backend writes a GLB under `/tmp`, bakes through `metal-uv-binned-nearest`, and reports nonzero UV-bin diagnostics.
5. Docs explain the chart backend as opt-in candidate export evidence, while xatlas chart parity remains deferred.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| CUVX-01 | Public export API validates and records the UV backend contract. | Focused tests cover default resolution, native-chart resolution, invalid backend, and invalid chart angle. |
| CUVX-02 | Default Pixal3D export behavior is unchanged. | Existing package/root tests still assert face-atlas and `metal-face-atlas-nearest` defaults. |
| CUVX-03 | Native chart backend is wired into the real export path. | Heavy fixture with `uv_backend="native-chart"` writes GLB and diagnostics under `/tmp`. |
| CUVX-04 | Native chart export uses the binned Metal path. | Heavy fixture diagnostics show `uv.stats.backend=native-chart-atlas`, `texture_bake.stats.backend=metal-uv-binned-nearest`, and nonzero UV-bin diagnostics. |
| CUVX-05 | Chart backend is not presented as xatlas parity. | Docs and visual diagnostics keep `not_xatlas_chart_parity` when comparing against the reference GLB. |

## Scope Coverage Decisions

- **Included:** export API switch, backend validation, chart angle validation, diagnostics/metadata, real-fixture chart backend proof, docs, full regression.
- **Deferred:** making chart UV the default, xatlas-quality chart cutting/packing, exact perceptual parity, replacing upstream xatlas/cuMesh semantics.
- **Anti-goals:** relaxing quality thresholds, changing decoded model outputs, hiding chart backend limitations, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep `face-atlas` as the default backend.
- Keep native chart implementation in C++ and call it through the existing Python wrapper.
- Keep binned Metal texture bake diagnostics visible for chart exports.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Simple chart packing may underperform the paired-triangle face atlas on real data. Mitigation: opt-in backend and truthful diagnostics.
- **Runtime risk:** Real chart exports may produce many charts and stress UV-bin reference guards. Mitigation: keep existing bin guard and prove behavior with a heavy fixture.
- **Regression risk:** Export API changes may affect existing callers. Mitigation: default-preserving API and full package/root regression.

## Blocking Questions Or Assumptions

Assumption: this phase should prove opt-in real-fixture chart export readiness before any later phase considers switching Pixal3D defaults or removing `not_xatlas_chart_parity`.
