# mlx-spatialkit Xatlas Parity Diagnostics Gate Spec

## Bounded Goal

Add structured xatlas chart-parity diagnostics for native-chart Pixal3D exports so the remaining parity boundary is measured instead of represented only by a boolean false/deferred label.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: diagnostics/contract hardening
- Shape: parity evidence

## Selected Lenses

- **engineering:** Replace the current low-signal xatlas boundary with quantitative diagnostics grounded in the checked-in Pixal3D reference trace and native chart stats.
- **runtime:** Reuse existing native chart and texture-bake diagnostics; do not add xatlas to `mlx-spatialkit` runtime dependencies.
- **product:** Make it clear what remains non-equivalent before calling native-chart exports production-equivalent to the intended Pixal3D export path.

## Current Evidence

- Root dev environment has `xatlas`; `packages/mlx-spatialkit` intentionally does not.
- Checked-in reference trace reports `unwrap_backend="xatlas-parallel-spatial"`, `unwrap_chart_count=51953`, `unwrap_utilization=0.8309683442115784`, and `texture_size=1024`.
- Native-chart reference-target diagnostics currently pass scalar readiness while keeping `xatlas_chart_parity=false`.
- Native-chart 1M/4096 diagnostics pass upstream-setting readiness and leave only `not_xatlas_chart_parity` deferred.

## Required Outcome

1. Reference trace loading preserves xatlas chart count and utilization fields.
2. Native-chart exports write `quality.xatlas_chart_parity` with reference/backend fields, native chart fields, ratios, checks, and a non-ready status.
3. Heavy real-fixture tests assert the structured parity diagnostics on both reference-target and explicit 1M/4096 native-chart paths.
4. Focused tests cover the helper contract without loading real fixtures.
5. Docs explain that xatlas parity is now measured, still not claimed, and not a package dependency.
6. Package/root/build verification remains clean with generated artifacts under `/tmp`.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| XPDG-01 | Reference xatlas fields are loaded. | Focused test asserts `unwrap_chart_count` and `unwrap_utilization` flow into parity diagnostics. |
| XPDG-02 | Native-chart parity diagnostics are structured. | Diagnostics include status, readiness bool, reference/native chart fields, chart-count ratio, utilization ratio, and checks. |
| XPDG-03 | No false parity claim. | Diagnostics keep `parity_ready=false`, `xatlas_chart_parity=false`, and `deferred_boundary="not_xatlas_chart_parity"`. |
| XPDG-04 | Real fixture paths assert the contract. | Heavy native-chart reference-target and 1M/4096 tests assert structured xatlas diagnostics. |
| XPDG-05 | Docs match the boundary. | Docs state that parity is measured but not solved and xatlas is not a spatialkit runtime dependency. |
| XPDG-06 | Repo/package hygiene holds. | Full package/root tests and `/tmp` wheel/sdist artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Structured diagnostics, helper tests, real-fixture assertions, docs, package/root/build verification.
- **Deferred:** Implementing xatlas-equivalent native charting, adding xatlas as a package dependency, switching defaults.
- **Anti-goals:** relaxing readiness thresholds, claiming xatlas parity, tagging/pushing/releasing.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Do not add runtime dependency on `xatlas` in `packages/mlx-spatialkit`.
- Do not tag, release, publish, push, or change release metadata.

## Risks

- **Evidence risk:** Ratios can be misread as equivalence. Mitigation: diagnostics include explicit `parity_ready=false` and docs keep xatlas parity deferred.
- **Contract drift risk:** Reference trace fields may be absent in future fixtures. Mitigation: helper reports missing reference fields as blocked/measured-unavailable instead of passing.

## Blocking Questions Or Assumptions

Assumption: measuring the xatlas boundary before attempting a native replacement is the correct next step because it prevents hidden stub behavior and gives future implementation targets.
