# mlx-spatialkit Repair Policy Contract Spec

## Bounded Goal

Make the verified small-boundary-loop repair policy explicit and configurable across the native API, Python wrapper, Pixal3D export settings, diagnostics, tests, and docs.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and visually comparable Pixal3D exports before production readiness is claimed.

## Selected Lenses

- **engineering:** Turn a hard-coded repair policy into a visible contract instead of hiding behavior behind passing tests.
- **runtime:** Keep the default at the verified native cap-3 repair, with `0` as an explicit disable path.
- **product:** Let callers understand and reproduce the geometry/UV tradeoff from diagnostics.

## Current Evidence

- Verified default policy fills triangular closed boundary loops only: `small_boundary_loop_fill_max_edges=3`.
- Heavy reference-target evidence: `boundary_loop_count=1872 < 2594`, xatlas-utilization ratio `0.6828063257125282`, visual comparison passes, xatlas parity remains false.
- The policy is currently a C++ constant and is not visible as a caller setting in `simplify_mesh` or `export_pixal3d_glb`.

## Required Outcome

1. Native `simplify_mesh` accepts `small_boundary_loop_fill_max_edges` with default `3` and supports `0` to disable repair.
2. Python `mlx_spatialkit.simplify_mesh` exposes the same parameter.
3. `export_pixal3d_glb` exposes and records the same setting in diagnostics.
4. Focused tests cover default repair, disabled repair, invalid negative values, and the 4-edge preservation policy.
5. Heavy Pixal3D reference-target export still passes with default `3`.
6. Docs and roadmap describe the repair policy as configurable, with the current default and remaining xatlas boundary.
7. Heavy/generated artifacts stay under `/tmp`; no push, tag, publish, or release metadata work.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| RPC-01 | Native repair cap is explicit. | C++ binding has `small_boundary_loop_fill_max_edges=3`; negative values reject. |
| RPC-02 | Python API exposes the cap. | `mlx_spatialkit.simplify_mesh(..., small_boundary_loop_fill_max_edges=0)` disables fill. |
| RPC-03 | Pixal3D export diagnostics expose the cap. | `diagnostics.settings.small_boundary_loop_fill_max_edges == 3` by default. |
| RPC-04 | Focused tests cover policy semantics. | Unit tests assert default fill, disabled no-fill, and 4-edge preservation. |
| RPC-05 | Heavy default behavior remains verified. | Reference-target native-chart heavy test passes with the same topology/visual/xatlas honesty gates. |
| RPC-06 | Docs/roadmap are current. | Spatialkit/Pixal3D docs and ROADMAP mention configurable cap/default/disable. |
| RPC-07 | Repo/package hygiene holds. | Focused tests, heavy test, package/root tests, and `/tmp` build inspection pass. |

## Scope Coverage Decisions

- **Included:** Native/Python/export parameter, validation, diagnostics, tests, docs.
- **Deferred:** Adaptive repair selection, larger/open boundary repair, xatlas parity.
- **Anti-goals:** Changing the verified default, claiming xatlas parity, adding external remesh dependencies.

## Constraints

- Keep default behavior stable at cap `3`.
- Keep repair disabled only through explicit value `0`.
- Preserve native guards against degenerate, duplicate, and nonmanifold patch faces.
