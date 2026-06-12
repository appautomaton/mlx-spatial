# Design: Reference-Parity UV Unwrap

Decisions only; requirements live in SPEC.md (UVU-01..11).

## File layout (additive)

- `cpp/uv_metrics.cpp` (+ hpp): UV-validity/quality metrics — overlap count, flipped-triangle count, per-chart stretch (Sander L2/Linf texel stretch), atlas utilization. Pure functions over `(vertices, faces, uvs, chart_ids)`; report-only, consumed by stats and tests. Bound in `bindings.cpp`.
- `cpp/uv_unwrap.cpp` (+ hpp): the two-stage backend — `ConeClusterer` (stage A) + `ChartBuilder`/`ChartParameterizer`/`ChartPacker` (stage B). New `make_reference_uvs(...)` entry mirroring the `make_native_chart_uvs` return contract (NativeUvMesh: vertices/faces/uvs/vmap + stats).
- `glb_writer.cpp` untouched except sharing helpers via `mesh_common.hpp` where they already fit. Existing backends and stats keys untouched (UVU-11).
- Export wiring confined to `export.py`: validator widening (`:1075-1079`), backend dispatch beside `make_native_chart_uvs` (`:113-140`), stage contract + parity summary (`:1227-1340`, `:1682-1840`).

## Stage A — cone clustering (behavior of `atlas.cu:1071`)

Greedy chart agglomeration as cost-ordered merge of the chart-adjacency graph (the CUDA kernels collapse chart-adjacency "edges"; we do the same serially): each chart keeps a normal cone (axis + half-angle) and area/perimeter; merge cost = cone-widening cost + `area_penalty_weight * area` + `perimeter_area_ratio_weight * (perimeter/area)`; a merge is admissible only if the merged cone half-angle ≤ threshold. Heap-driven like `QemSimplifier`: named comparator `(cost ASC, edge_id ASC, version DESC)`, lazy invalidation, compaction — reuse that pattern verbatim (it is the proven deterministic/O(E log E) shape). `refine_iterations=0`, `global_iterations=1`, `smooth_strength=1` at production knobs: implement refine/smooth loops per `atlas.cu` semantics but they are no-ops/single-pass at the pinned values — verify them at non-default values only via unit tests, not fixture parity.

## Stage B — xatlas-equivalent per cluster

1. **Chart growth** (`ChartOptions` semantics): within each stage-A cluster, seed-and-grow charts with cost = `normal_deviation*2.0 + roundness*0.01 + straightness*6.0 + normal_seam*4.0 + texture_seam*0.5`, stop at `max_cost 2.0`; `max_iterations=1` (single grow+reseed pass).
2. **Parameterization:** PCA/orthographic projection first; accept if flip-free and stretch under threshold (xatlas accepts planar charts the same way). Otherwise LSCM: sparse conformal system, two pinned extremal boundary vertices, solved by native conjugate gradient on the normal equations (dependency-light, no Eigen). Bounded iterations; deterministic (fixed traversal order, no parallel reduction in the solve).
3. **Overlap repair:** exact UV triangle-overlap test per chart (uv_metrics primitive); a chart with flips/self-overlap after LSCM is split (deepest-cost face seam, bounded depth) and re-parameterized. Hard invariant: emit zero overlaps or fail loudly in stats — never ship overlapping UVs silently.
4. **Packing:** rotate charts to convex-hull axis, then shelf/skyline pack with `padding`/`bilinear` gutter semantics and `resolution`/`texels_per_unit` scaling. Reuse the aspect-shelf scaffolding shape from `glb_writer.cpp:1240+` only where it matches; padding semantics are new.

## Oracle topology (the parity anchor)

pip `xatlas` (version-pinned, `/tmp/uvoracle-venv`) is the same CPU C++ library CuMesh vendors. Anchor generation script (`tests/tools/gen_uv_oracle_anchors.py`, runnable only when xatlas importable): export both fixtures' QEM 50k meshes from cached NPZ, run **our** stage-A clusters through pip xatlas per-cluster (exactly the reference composition, `cumesh.py:408`), plus one whole-mesh xatlas run as secondary sanity. Record `{xatlas_version, chart_count, utilization, stretch L2/Linf distribution, seam ratio}` per fixture into committed `tests/data/uv_oracle_anchors.json`. Parity tests read the JSON constants and must pass **without** xatlas installed (UVU-10).

## Honest gate

`parity_ready` (hardcoded `False` at `export.py:1730/1823`, consumed `:2335-2348`) becomes computed: True only when backend is the new reference one AND measured overlap==0 AND stretch/utilization/chart-count within anchor tolerances. `unwrap_reference` naming: backend id `"xatlas-equivalent-native"` satisfies `startswith("xatlas")` (`:1249`) — allowed only because the behavior genuinely implements the two-stage reference; SPEC anti-goal forbids renaming a heuristic.

## Determinism & perf discipline (QEM lessons, applied from slice 1)

Ordered containers only; no pointer-keyed maps; cross-process PYTHONHASHSEED test at engine level. No O(F²): heap agglomeration (A), per-cluster local growth (B1), CG with capped iterations (B2), shelf pack O(C log C). Memory budgets asserted as **delta** RSS. `rm -rf /tmp/mlx-spatialkit-build` for clean rebuilds after constant changes.
