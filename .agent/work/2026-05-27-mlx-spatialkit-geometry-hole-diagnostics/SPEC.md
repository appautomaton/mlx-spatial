# mlx-spatialkit Geometry Hole Diagnostics Spec

## Bounded Goal

Make native geometry hole risk measurable in `mlx-spatialkit` exports so visible holes are not mistaken for UV-padding or triangle-count guesses.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, and visually comparable Pixal3D exports before production readiness is claimed.

## Selected Lenses

- **engineering:** Add native topology evidence before implementing repair heuristics.
- **runtime:** Keep diagnostics in C++ over existing mesh buffers; no Python mesh walks on heavy exports.
- **product:** Explain whether visible holes are likely geometry boundaries, simplification loss, or UV/chart coverage.

## Current Evidence

- Latest live reference-target native export: `/tmp/mlx-spatialkit-native-chart-reference-target-export-80378/diagnostics.json`.
- Geometry is close to the reference by count: `198618` final faces versus xatlas reference `212542` (`0.9344882423238701` ratio).
- Export mesh still has `23822` boundary edges and `2` connected components, while `nonmanifold_edges=0`.
- UV parity is still false: `uv_surface_occupancy_ratio=0.5768346786499023`, `xatlas_utilization_ratio=0.6941716645020964`.

## Required Outcome

1. Native mesh diagnostics report boundary-loop shape, not only raw boundary-edge count.
2. Export diagnostics preserve boundary-loop metrics for source and final export meshes.
3. Tests distinguish open-but-valid boundaries from nonmanifold blockers.
4. Real Pixal3D heavy coverage asserts the metrics exist and keeps existing production/non-parity contracts honest.
5. Docs explain that the current small-hole investigation is geometry-boundary evidence first; full hole repair is deferred until the loop topology is quantified.
6. Heavy/generated artifacts stay under `/tmp`; no push, tag, publish, or release metadata work.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| GHD-01 | Boundary-loop diagnostics exist in native metrics. | `mesh_metrics` returns loop count, small-loop count, max loop size, and open-chain count. |
| GHD-02 | Focused tests cover loop diagnostics. | Unit tests assert a closed square boundary loop and a nonmanifold blocker case. |
| GHD-03 | Export diagnostics expose final geometry hole risk. | Heavy reference-target diagnostics include boundary loop metrics under `export_metrics.metrics`. |
| GHD-04 | No false geometry or xatlas readiness claim. | Existing production quality and xatlas non-equivalence assertions remain intact. |
| GHD-05 | Docs/roadmap are current. | ROADMAP current state and spatialkit/Pixal3D docs mention geometry-boundary diagnostics and latest UV metrics. |
| GHD-06 | Repo/package hygiene holds. | Focused tests, heavy reference-target test, package tests, and root Pixal3D tests pass. |

## Scope Coverage Decisions

- **Included:** Native boundary-loop metrics, Python diagnostics propagation through existing metric calls, focused tests, heavy export assertion, docs/roadmap update.
- **Deferred:** Hole filling, remeshing, xatlas-equivalent unwrap, changing production readiness thresholds, changing the default backend.
- **Anti-goals:** Claiming the visible hole is solved, relaxing parity checks, adding xatlas, or adding slow Python topology passes.

## Constraints

- Keep topology analysis native C++ and bounded by existing mesh edge counts.
- Preserve current export output unless diagnostics reveal a future repair-safe policy.
- Keep all heavy generated artifacts under `/tmp`.

## Risks

- **Boundary ambiguity:** Some boundaries can be intentional object openings. Mitigation: record loop shape without treating every boundary as a blocker.
- **False progress:** Diagnostics alone do not repair holes. Mitigation: this cycle explicitly gates only observability; repair remains a future quality slice.
