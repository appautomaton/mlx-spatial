# mlx-spatialkit Xatlas Deficit Diagnostics Gate Spec

## Bounded Goal

Make the remaining native-chart versus xatlas utilization deficit explicit in `mlx-spatialkit` diagnostics so passing native-chart readiness cannot be confused with xatlas-equivalent chart quality.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant Metal/C++ hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no half-stubbed behavior hidden behind successful tests.

## Work Scale And Shape

- Scale: diagnostics/contract hardening
- Shape: parity boundary evidence

## Selected Lenses

- **engineering:** Add explicit deficit fields and failed equivalence checks on top of the existing xatlas ratio diagnostics.
- **runtime:** Reuse existing native chart and texture bake stats; do not add xatlas as a package dependency or run xatlas at export time.
- **product:** Make the remaining quality gap readable even when `native_chart_uv_candidate.quality_ready=true`.

## Current Evidence

- Checked-in xatlas reference trace reports `unwrap_backend="xatlas-parallel-spatial"`, `unwrap_chart_count=51953`, and `unwrap_utilization=0.8309683442115784`.
- Latest verified native-chart reference-target diagnostics report UV-surface occupancy `0.5693244934082031` and xatlas-utilization ratio `0.685133792850289`.
- Native-chart reference-target and 1M/4096 paths can pass scalar readiness while `quality.xatlas_chart_parity.parity_ready=false`.

## Required Outcome

1. `quality.xatlas_chart_parity` includes explicit deficit fields for native occupancy versus reference xatlas utilization.
2. Diagnostics include a named utilization-equivalence check with a documented target, and that check fails on current native-chart real fixtures.
3. Existing parity fields remain intact: `parity_ready=false`, `xatlas_chart_parity=false`, and `deferred_boundary="not_xatlas_chart_parity"`.
4. Focused helper tests cover passing measurement, missing-reference, and non-native-chart states without loading real fixtures.
5. Heavy native-chart fixture tests assert that scalar native-chart readiness can pass while xatlas utilization equivalence fails.
6. Docs explain the deficit fields and the non-equivalence boundary without claiming xatlas parity or adding xatlas as a runtime dependency.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| XDDG-01 | Deficit fields are explicit. | Focused test asserts utilization deficit and ratio gap fields. |
| XDDG-02 | Equivalence check cannot hide. | Focused and heavy tests assert `xatlas_utilization_equivalence.passed=false` for current native-chart output. |
| XDDG-03 | Existing no-parity contract holds. | Tests assert `parity_ready=false`, `xatlas_chart_parity=false`, and `not_xatlas_chart_parity`. |
| XDDG-04 | Real fixture proves the user-facing boundary. | Heavy reference-target and 1M/4096 native-chart tests assert native readiness can pass while xatlas equivalence fails. |
| XDDG-05 | Docs match diagnostics. | README, Pixal3D docs, and script docs describe deficit fields and xatlas non-equivalence. |
| XDDG-06 | Repo/package hygiene holds. | Targeted package tests, root Pixal3D tests, and `/tmp` build/artifact inspection pass. |

## Scope Coverage Decisions

- **Included:** Deficit diagnostics, equivalence check, helper tests, real-fixture assertions, docs, package/root/build verification.
- **Deferred:** Implementing xatlas-equivalent chart cutting/packing, changing the default UV backend, adding xatlas to package runtime dependencies.
- **Anti-goals:** claiming xatlas parity, relaxing readiness thresholds, tagging, pushing, publishing, or release metadata changes.

## Constraints

- Keep generated and heavy artifacts under `/tmp`.
- Do not add runtime dependency on `xatlas` in `packages/mlx-spatialkit`.
- Preserve current native-chart readiness semantics; this change adds clarity, not a new passing parity claim.

## Risks

- **Threshold interpretation risk:** A utilization target could be mistaken for full xatlas parity. Mitigation: name it utilization equivalence only and keep backend equivalence/parity false.
- **Diagnostics drift risk:** Future reference traces may omit xatlas fields. Mitigation: missing-reference paths remain measurement-incomplete instead of passing.

## Blocking Questions Or Assumptions

Assumption: A `0.95` utilization-ratio target is a useful diagnostics threshold for "close enough to compare seriously" while full xatlas parity still requires backend/topology equivalence.
