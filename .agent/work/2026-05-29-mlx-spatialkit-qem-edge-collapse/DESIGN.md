# DESIGN — Native QEM Edge-Collapse Simplifier (rev. post eng-review)

Scope: the **mechanism** for SPEC `2026-05-29-mlx-spatialkit-qem-edge-collapse`. Behavior-reference only (CuMesh `simplify_step(lambda_edge_length, lambda_skinny, thresh)`); no CUDA line-port. CPU-first, dependency-light. This revision folds in `reviews/engineering-findings.md` (B1–B4, R1–R8).

## Why not extend clustering
`cluster_mesh` (`simplify.cpp:264`) bins vertices into voxels and rebuilds faces — non-adjacent vertices merge, shared edges lose a triangle, watertight input re-tears (turtle 0→41, violin 0→52). Edge-collapse removes one **interior edge** at a time under validity guards, so a closed manifold stays closed **by construction**.

## Data structures (additive new region in simplify.cpp)
- Reuse quadric infra: `Quadric` (`:135`), `add_plane_to_quadric` (`:193`), `evaluate_quadric` (`:207`), `vertex_plane_quadrics` (`:225`); and `mesh_common::MeshData`.
- **Adjacency (B3 — determinism-critical):** per-vertex incident-face lists and per-edge→incident-triangle maps used by the collapse/re-push loop **MUST be sorted `std::vector` (keyed by `edge_id`/face index) or `std::map` — never `std::unordered_map`.** Unordered iteration order varies by ASLR hash seed across processes/builds and diverges the collapse order on equal-cost edges (the codebase already worked around this at `simplify.cpp:1376-1380`).
- **Cost heap:** binary min-heap of `(cost, edge_id, version)` with per-edge `version` for **lazy invalidation** (stale pops skipped). **Comparator (named struct, unit-testable):** min on **`(cost ASC, edge_id ASC, version DESC)`** — for equal `(cost, edge_id)` the newest version sorts first so the stale-skip path is deterministic.
- **Heap compaction (R3):** rebuild the heap from live edges when its size exceeds **3× live-edge count** (O(E), keeps memory linear; bounds the ~10× lazy-entry blowup at remesh-output scale).

## Per-collapse math
- Edge `(a,b)`: combined quadric `Q = Q_a + Q_b`.
- **Optimal vertex:** solve the 3×3 system minimizing `vᵀQv`; if singular/ill-conditioned, fall back to best of `{midpoint, a, b}` by quadric error, with an **explicit tiebreak (B3, mirror `cluster_mesh:335-342`): error → distance-to-midpoint → fixed vertex-index order.** Cost = `evaluate_quadric(Q, v*)` + `lambda_edge_length·|a−b|²` + `lambda_skinny`·(sliver penalty). Penalties only **reorder**; they never substitute for a guard.

## Validity guards — what preserve watertightness (QEM-02). A collapse is REJECTED if any holds:
1. **Boundary lock:** either endpoint is a boundary vertex (interior-only collapse in v1). On a watertight input this is a no-op; on residual loops it freezes those edges (see "Stall detection").
2. **Link condition (manifold-preserving):** the set of common one-ring neighbors of `a` and `b` must be **exactly** the two opposite vertices of the incident triangles. Necessary but **not sufficient** alone — pair with guard 3.
3. **Vertex-fan / pinch test (B1 — the dominant failure mode):** after retargeting `b→a`, the surviving faces incident to `a` must form a **single edge-connected fan** (one umbrella). A second disconnected fan is a non-manifold **vertex** (a pinch) — invisible to `nonmanifold_edges`, so it must be prevented here and asserted via the new `nonmanifold_vertices` metric.
4. **Low-valence / fold-over (R7):** reject if the collapse would (a) make any surviving face a **canonical-face duplicate** of another in the merged 1-ring (valence-3 / tetrahedron fold), or (b) drop the mesh below the **4-face tetrahedron floor**.
5. **Normal-flip (R6):** reject when `dot(normal_pre, normal_post) <= cos(theta_min)` (a **positive threshold**, not `< 0`) over any triangle in the merged 1-ring — closes the near-zero-area sliver blind spot.
6. **Degeneracy (R6):** hard reject on post-collapse triangle **area/aspect below a fixed threshold** (not merely a cost penalty).

After a valid collapse: remove the 2 incident triangles, retarget `b→a`, `Q_a += Q_b`, bump `version` on all touched edges, re-push their costs (iterating the **sorted** adjacency).

## Loop / target / stall detection (R4)
Pop min-cost valid edge; collapse; repeat until `face_count ≤ target_faces` **or no valid collapse remains**. The existing `target_reached` stat (`simplify.cpp:1850`) already distinguishes "hit target" from "exhausted" — assert it so a **boundary-lock stall reads as under-decimation, not success**.

## Input-prep composition (watertight input — SPEC key decision)
Reference/production path: `remesh(repair_nonmanifold=True)` → `nonmanifold_edges==0` but opens small loops (turtle 291 / violin 1111) → bounded `small_boundary_loop_fill` (existing, `cumesh-perimeter-centroid-fan`, cap 64 edges) closes what it can → **then** QEM. Record residual boundary loops (bounded); locked boundaries are why guard 1 + stall detection matter.

## Diagnostics / stat-field contract (B4 + M2/M4)
The shared `simplify_mesh` stat block (`simplify.cpp:1820-1862`) reads a populated clustering `ClusterResult best` **unconditionally** (`best.cluster_count`, `best.quadric_representative_*`, …). The QEM path has no `best`, so it **MUST FORK its own stat emission** — both on the main return and the **backend-agnostic early-return at `:1714`** (input already ≤ target) — rather than route through that block. **Governing rule (M4):** the forked qem stat dict's **keyset must EQUAL** a clustering call's keyset on the same input (set-equality), not an enumerated subset — so a missing field is caught regardless of any hand-listed enumeration. Clustering-specific fields with no QEM analogue (`cluster_count`, `grid_resolution`, `degenerate_faces_removed`, `duplicate_faces_removed`, `nonmanifold_faces_removed`, `representative_vertices_selected`, `representative_selection_strategy`, `quadric_representative_*`, plus `add_small_loop_fill_stats`/`add_pre_simplify_loop_fill_stats` families) → emit **0 / "n/a" sentinels** (grep-confirmed no downstream reader does arithmetic on them). New QEM fields: `qem_collapses_applied`, `qem_collapses_rejected_by_guard`, `qem_geometric_error_mean/max`, `qem_input_faces` (= `source_faces`). Set `qem_simplification_backend="native-qem-edge-collapse"`, `qem_equivalence_status="edge-collapse"`. **Blocker logic (`production_blocker_values` :1543):** for `qem`, push only `missing_narrow_band_dc_remesh` (cleared by `export.py:472-478` post-remesh), **never** `missing_qem_edge_collapse_simplification`; `quality_tier=="production"` when blockers empty.

## Vertex-manifold metric — report-only, not gated (M1)
The new `nonmanifold_vertices` (pinch) counter from `mesh_metrics.cpp` is the **correctness oracle** (asserted on synthetics in S2 and fixtures in S5). It is added as a **reported count/class only**; the export status gate (`export.py:1445`, which branches on `nonmanifold_edges>0`) is left **unchanged**, because clustering output legitimately pinches and adding the counter to that gate would flip existing clustering fixtures (`test_real_pixal3d_export.py:292/:539`) from `rendered_visual_blocked` to `artifact_blocked`. Gate-level pinch honesty is deferred; the QEM correctness proof rests on the metric + guards + assertions, not the gate.

## Opt-in wiring (B2 — do NOT touch the preset; M3 + F1)
`_simplifier_backend_for_quality_preset` and its tests (`:580/:615-617/:988/:1079/:1210`) stay **unchanged**. Add an explicit `export_pixal3d_glb(simplify_backend: str | None = None)` param that, when `"qem"`, overrides `requested_simplifier_backend` (`export.py:302/463`); default `None` preserves preset behavior exactly. **Validation (F1):** a Python-side resolver accepts only `None`|`"qem"` and raises a clear `ValueError` otherwise (mirroring `_resolve_pixal3d_uv_backend`), so unknown strings fail fast rather than as a late C++ `nb::value_error`. **Watertight-input guard (M3 — zero silent failures):** QEM boundary-locks, so on a non-watertight input it stalls into a vacuously-"watertight" under-decimated mesh; therefore `simplify_backend="qem"` **requires `remesh=True` + `remesh_repair_nonmanifold=True`** — reject with `ValueError` if not set, OR auto-enable the watertight input-prep when `qem` is selected (planning picks one; S4 asserts the chosen behavior, not just the happy path). Default `remesh` is `False` (`export.py:237`), so this guard is mandatory.

## Determinism assumption (B3 footnote)
Quadric accumulation is IEEE-754 `double` over a fixed face order; determinism assumes ARM64 / strict-IEEE (no FP80, no `-ffast-math`) — true today (Apple-only, `CMakeLists.txt`). A future Linux/x86 port must audit this.

## Out of scope (deferred per SPEC)
xatlas-equivalent unwrap; Telea inpaint / GPU trilinear; full 1M-vertex/4096 reference-scale (prove the mechanism at a tractable target; record reference-scale deferred if CPU perf impractical); Metal acceleration of the collapse.
