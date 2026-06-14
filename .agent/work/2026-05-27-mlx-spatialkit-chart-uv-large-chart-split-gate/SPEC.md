# mlx-spatialkit Chart UV Large Chart Split Gate Spec

## Bounded Goal

Add deterministic oversized-chart splitting to native chart UV generation and prove the real Pixal3D chart candidate improves or exposes a stronger bounded-quality diagnostic without changing the default export path.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: native chart quality improvement
- Shape: C++ chart-cut hardening

## Selected Lenses

- **engineering:** Add real chart seams for oversized smooth charts instead of relying only on normal-threshold grouping.
- **runtime:** Keep splitting deterministic, bounded, and CPU-light; no global unwrap solve or external dependency.
- **product:** Move the native chart candidate closer to useful output while diagnostics remain honest if readiness is still blocked.

## Current Evidence

- Phase 17 local-frame/PCA projection improved real fixture occupancy to `0.38708019256591797` and global coverage to `0.24212932586669922`.
- The chart candidate still reports `quality_blocked` below the `0.50` floors.
- A `/tmp` angle sweep after local projection showed only marginal occupancy movement (`~0.3805..0.3897`) and did not solve the blocker.
- Diagnostics still show `max_chart_faces=6220`, so one oversized smooth chart can dominate projection distortion and UV-bin candidate load.

## Required Outcome

1. `make_native_chart_uvs` splits oversized source charts into deterministic spatial chunks before projection/packing.
2. Chart UV stats report split diagnostics: pre-split chart count, split chart count, split max-face target, split count, and post-split max chart faces.
3. Focused tests prove small charts remain unchanged and oversized coplanar charts split deterministically while preserving face count and in-range UVs.
4. The real Pixal3D native-chart export either improves `uv_surface_occupancy_ratio` over `0.38708019256591797` or records a stronger bounded-quality diagnostic that proves oversized charts are no longer the active bottleneck.
5. Docs describe large-chart splitting as a native chart candidate improvement, not xatlas parity or a default backend switch.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| LCSG-01 | Oversized charts split deterministically. | Focused tests assert split stats and stable output for a synthetic oversized chart. |
| LCSG-02 | Small chart behavior remains intact. | Existing coplanar and hard-crease tests still pass. |
| LCSG-03 | Real chart export advances the diagnostic boundary. | Heavy fixture reports occupancy above `0.38708019256591797` or `max_chart_faces` below the split target with truthful quality blockers. |
| LCSG-04 | Default export behavior remains unchanged. | Full package/root Pixal3D tests still pass, including default face-atlas assertions. |
| LCSG-05 | Docs match the parity boundary. | Docs say splitting improves the native candidate but does not claim xatlas parity or switch defaults. |

## Scope Coverage Decisions

- **Included:** deterministic oversized-chart splitting, stats, focused tests, real fixture proof, docs, regression/build verification.
- **Deferred:** full xatlas-quality chart cuts, LSCM/ABF unwrap, overlap removal, default backend switch, readiness-threshold relaxation.
- **Anti-goals:** changing decoded model outputs, changing Metal bake behavior, changing chart quality floors, writing heavy artifacts outside `/tmp`, tagging/pushing/releasing.

## Constraints

- Keep implementation native C++ and deterministic for identical inputs.
- Keep runtime bounded: no recursive unbounded search, global solve, or external dependency.
- Preserve face count and valid in-range UVs.
- Keep generated and heavy artifacts under `/tmp`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Quality risk:** Splitting can add seams and duplicate vertices without increasing texture occupancy. Mitigation: accept only real fixture improvement or an explicit diagnostic boundary shift showing oversized charts are no longer active.
- **Runtime risk:** More charts increase packing and binning work. Mitigation: fixed split target and no iterative global optimization.
- **Regression risk:** UV layout changes can affect GLB texture bake results. Mitigation: focused chart tests, real chart heavy fixture, package/root tests, and build checks.

## Blocking Questions Or Assumptions

Assumption: oversized smooth charts are the next measurable bottleneck because angle changes barely moved occupancy and diagnostics still show `max_chart_faces=6220`.
