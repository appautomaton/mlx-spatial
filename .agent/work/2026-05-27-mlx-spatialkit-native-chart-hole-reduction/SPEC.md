# mlx-spatialkit Native Chart Hole Reduction Spec

## Bounded Goal

Identify and reduce the native-chart UV holes that keep `mlx-spatialkit` below the Pixal3D xatlas reference, while preserving honest diagnostics and production-readiness contracts.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: multi-slice quality improvement
- Shape: real-fixture geometry/UV export improvement

## Selected Lenses

- **engineering:** Improve native chart generation where evidence shows the gap lives, not by loosening thresholds or adding xatlas as a dependency.
- **runtime:** Keep the hot path native C++/Metal, deterministic, bounded, memory-safe, and compatible with the current Python API.
- **product:** Make the generated native GLB visually closer to the intended Pixal3D/xatlas export while keeping non-equivalence explicit until proven.

## Current Evidence

- The fresh native GLB at `/tmp/mlx-spatialkit-native-chart-reference-target-open/model.glb` has SHA `c945ba348c7978eea91ea3c63805a4774489131084da063516059535538107d8`; the xatlas reference GLB has SHA `d519633db296f8a047df0cc2f948a5875d78181ccec448d8ea66869e28d83fea`.
- Current reference-target native-chart metrics:
  - `chart_rect_fill_ratio=0.5764121465152018`
  - `uv_surface_occupancy_ratio=0.5693244934082031`
  - `xatlas_utilization_ratio=0.685133792850289`
  - `shelf_packing_efficiency=0.9952422592195517`
- A chart-angle probe under `/tmp/mlx-spatialkit-chart-angle-probe` found `45.0` degrees still best among `30,45,60,75,90`, so changing the public chart angle is not the next quality lever.
- Shelf packing is already near full; the remaining hole is primarily chart-local projection/splitting fill, not global shelf packing.

## Required Outcome

1. The spec and plan treat native-chart hole reduction as one coherent program, not a roadmap entry per tweak.
2. Baseline diagnostics identify whether a candidate change improves real fixture UV-surface occupancy or only reshuffles local metrics.
3. At least one bounded native chart-generation policy or C++ change improves reference-target native-chart UV-surface occupancy or xatlas-utilization ratio beyond the verified `0.5693244934082031` / `0.685133792850289` baseline.
4. Focused tests cover the changed chart-generation policy without real fixtures.
5. Heavy real-fixture tests assert the quality improvement and preserve `production_quality_ready=true`, `parity_ready=false`, and `xatlas_utilization_equivalence=false` until the gap is actually closed.
6. Docs explain the improved policy and the still-open xatlas parity boundary.
7. Package/root/build verification remains clean, with heavy generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| NCHR-01 | Roadmap granularity is corrected. | ROADMAP has one active native-chart hole-reduction phase, not per-tweak micro phases. |
| NCHR-02 | Baseline and lever selection are evidence-backed. | PLAN records the angle probe and identifies chart-local fill as the active blocker. |
| NCHR-03 | Native chart quality improves. | Heavy reference-target gate asserts `uv_surface_occupancy_ratio > 0.5693244934082031` or `xatlas_utilization_ratio > 0.685133792850289`. |
| NCHR-04 | No false parity claim. | Tests assert `parity_ready=false`, `xatlas_chart_parity=false`, and `xatlas_utilization_equivalence=false` unless a future slice genuinely closes them. |
| NCHR-05 | Focused coverage exists. | Unit tests cover the changed chart-generation policy deterministically. |
| NCHR-06 | Docs match behavior. | README, Pixal3D docs, and script docs describe the policy and remaining boundary. |
| NCHR-07 | Repo/package hygiene holds. | Package tests, root Pixal3D tests, and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Hole taxonomy, bounded native chart-generation policy or C++ changes, focused tests, real-fixture quality gates, docs, package/root/build verification.
- **Deferred:** Full xatlas-equivalent charting, replacing the default backend, adding xatlas as a package dependency, external remeshing libraries.
- **Anti-goals:** relaxing readiness thresholds, claiming xatlas parity from partial improvement, pushing, tagging, publishing, or release metadata changes.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Keep hot-path implementation native C++/Metal; Python may orchestrate tests and diagnostics only.
- Preserve memory-safety and bounded work: no unbounded chart search, no hidden O(texture_pixels * faces) fallback, no generated artifacts in package builds.

## Risks

- **Local metric trap:** Improving chart rect fill alone may not improve actual UV-surface occupancy. Mitigation: heavy tests gate on UV-surface occupancy or xatlas-utilization ratio.
- **Performance risk:** More split/projection candidates can slow native chart generation. Mitigation: keep candidate counts bounded and expose diagnostics.
- **False completion risk:** A modest improvement is not xatlas parity. Mitigation: keep deficit diagnostics and parity flags false.

## Blocking Questions Or Assumptions

Assumption: The next meaningful lever is chart-local projection/splitting because shelf packing is already near full and chart-angle probing did not improve the real fixture.
