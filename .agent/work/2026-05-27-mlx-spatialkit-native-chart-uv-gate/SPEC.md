# mlx-spatialkit Native Chart UV Gate Spec

## Bounded Goal

Add an opt-in native chart-UV generator that groups connected smooth faces into charts, packs chart UVs, and bakes through the binned Metal UV path without claiming full xatlas parity.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: capability hardening
- Shape: native UV unwrap foundation

## Selected Lenses

- **engineering:** Replace the face-atlas-only UV story with a real native charting primitive that can share vertices within charts and split only at chart boundaries.
- **runtime:** Reuse the binned non-atlas Metal bake path from Phase 12 so chart UVs avoid all-face scans.
- **product:** Developers should be able to exercise a native chart path and see truthful diagnostics before it becomes the default Pixal3D export path.

## Current Evidence

- Current spatialkit Pixal3D exports use `make_face_atlas_uvs`, which duplicates vertices per face and encodes two triangles per atlas tile.
- Phase 12 added `metal-uv-binned-nearest`, making arbitrary UV meshes bakeable without scanning every face per texel.
- Vendored Pixal3D uses CUDA `cumesh.uv_unwrap` and nvdiffrast UV rasterization; native spatialkit still explicitly defers `not_xatlas_chart_parity`.
- A first native chart generator can be useful only if it is labeled as a native chart candidate and not sold as xatlas equivalence.

## Required Outcome

1. `mlx-spatialkit` exposes a native `make_native_chart_uvs` API returning `NativeUvMesh`.
2. The native implementation groups adjacent faces into charts using a configurable normal-angle threshold and duplicates vertices per chart rather than per face.
3. Chart UVs are packed into a bounded atlas with diagnostics for chart count, output vertices/faces, duplicated vertex ratio, chart packing grid, and threshold.
4. Chart UV outputs bake through `metal-uv-binned-nearest`, proving the Phase 12 path is usable by native chart meshes.
5. Docs explain this as native chart-candidate generation, while `not_xatlas_chart_parity` remains deferred until real Pixal3D heavy exports are switched and visually proven.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| CUV-01 | Public API and native binding exist. | Focused tests import `make_native_chart_uvs` from `mlx_spatialkit` and inspect stats. |
| CUV-02 | Connected coplanar faces share one chart and reuse vertices within that chart. | Square fixture returns one chart and fewer output vertices than face-atlas duplication. |
| CUV-03 | Creases or disconnected components split charts. | Focused fixture with a hard normal break reports multiple charts. |
| CUV-04 | Chart UVs bake through binned Metal path. | Texture bake test with chart UV mesh reports `metal-uv-binned-nearest` and sampled texels. |
| CUV-05 | Existing face-atlas and real Pixal3D gates remain intact. | Package tests, heavy Pixal3D test, docs, build, and artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** native connected smooth-face chart grouping, chart-local vertex reuse, simple chart packing, Python API, tests, docs, heavy regression.
- **Deferred:** replacing Pixal3D default UV backend, xatlas-quality chart cutting/packing, CUDA/cuMesh parity, exact perceptual proof.
- **Anti-goals:** claiming xatlas parity, changing decoded model outputs, relaxing thresholds, adding runtime dependencies, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native C++.
- Keep `make_face_atlas_uvs` unchanged and default export behavior unchanged.
- Use the existing `NativeUvMesh` contract.
- Keep chart packing deterministic and bounded.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Simple smooth-connected charting is not xatlas and may distort UVs on complex curved surfaces. Mitigation: label as native chart candidate and keep xatlas parity deferred.
- **Regression risk:** API changes may affect imports. Mitigation: add focused public API tests and run package/root suites.
- **Runtime risk:** Very large connected charts may be slower to bake than face atlas. Mitigation: do not switch real Pixal3D default in this phase.

## Blocking Questions Or Assumptions

Assumption: implementing an opt-in native chart generator is the right next step because Phase 12 already made arbitrary chart UV baking scalable enough for focused proof, while switching heavy Pixal3D exports needs a later visual-quality gate.
