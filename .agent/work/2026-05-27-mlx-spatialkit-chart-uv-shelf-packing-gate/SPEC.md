# mlx-spatialkit Chart UV Shelf Packing Gate Spec

## Bounded Goal

Replace the native chart UV equal-grid packer with deterministic aspect-aware shelf packing and prove it improves real Pixal3D chart UV occupancy without changing the default face-atlas export path.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: packing algorithm hardening

## Selected Lenses

- **engineering:** Improve the native C++ chart packer itself instead of adjusting thresholds or hiding poor coverage.
- **runtime:** Keep the binned Metal bake path and default face-atlas path stable while changing only opt-in native chart UV geometry.
- **product:** Chart exports should show measurable progress toward visually useful output, and diagnostics should say whether quality is still blocked.

## Current Evidence

- Phase 15 reports native chart exports as artifact-ready but quality-blocked.
- A `/tmp` chart-angle probe showed chart angle does not solve the blocker: coverage stayed around `0.138..0.144` and UV-surface occupancy stayed around `0.222..0.233`.
- Current chart packing uses one equal square grid cell per chart, which wastes atlas area for skinny or rectangular chart projections.
- The default `export_pixal3d_glb` path remains `uv_backend="face-atlas"` and must stay default.

## Required Outcome

1. `make_native_chart_uvs` uses deterministic aspect-aware shelf packing for chart rectangles instead of equal square grid cells.
2. Chart UV stats report shelf packing diagnostics: packing backend, packing efficiency, row count, packed bounds, chart count, and duplicate ratio.
3. Focused tests prove coplanar charts still reuse vertices and hard creases still split charts.
4. The real Pixal3D chart export improves UV-surface occupancy over the previous verified baseline of `0.23263072967529297`.
5. Docs describe shelf packing as an improved native chart candidate, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| CUVP-01 | Native chart packing is deterministic shelf packing. | Focused chart tests assert `packing=aspect-shelf-charts` and shelf diagnostics. |
| CUVP-02 | Existing chart grouping behavior remains intact. | Coplanar and hard-crease tests still pass. |
| CUVP-03 | Real chart export occupancy improves. | Heavy chart fixture reports `uv_surface_occupancy_ratio > 0.23263072967529297`. |
| CUVP-04 | Default export behavior remains unchanged. | Full package/root tests still pass, including default face-atlas assertions. |
| CUVP-05 | Docs match the quality boundary. | Docs say shelf packing improves the candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** C++ shelf packing, stats, focused tests, real fixture proof, docs, regression/build verification.
- **Deferred:** full xatlas-quality chart cuts, chart rotation search, bin-packing optimality, default backend switch, removing readiness blockers if thresholds still fail.
- **Anti-goals:** changing decoded model outputs, changing Metal bake behavior, relaxing readiness thresholds, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native C++.
- Keep deterministic output for the same input mesh and parameters.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Shelf packing may improve occupancy but not reach quality-ready thresholds. Mitigation: readiness diagnostics remain truthful.
- **Regression risk:** UV layout changes may affect focused tests and GLB writes. Mitigation: run package/root tests and real fixture chart export.
- **Runtime risk:** Packing many charts must remain CPU-light. Mitigation: use simple sort plus shelf pass, no iterative optimal packing.

## Blocking Questions Or Assumptions

Assumption: a deterministic shelf packer is the right next increment because angle changes did not materially improve the chart export coverage blocker.
