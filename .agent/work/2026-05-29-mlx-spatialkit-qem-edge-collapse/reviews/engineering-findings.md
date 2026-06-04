# Engineering Review Findings — QEM Edge-Collapse (verdict: needs_correction)

Source: 4-reviewer adversarial pass (Opus on guard-correctness; Sonnet on perf/determinism/regression), grounded against `simplify.cpp`, `mesh_metrics.cpp`, `export.py`, and the existing test suite. Apply these in `auto-plan` (amend PLAN.md + DESIGN.md), then re-run `auto-eng-review`.

## Blocking (must fix before execution)

### B1 — Pinch (non-manifold *vertex*) is the dominant failure mode and is INVISIBLE to QEM-02's metrics
- `mesh_metrics.cpp:146-155` counts `nonmanifold_edges` (edges with face-count>2) only; there is **no non-manifold-vertex metric** anywhere in `cpp/` or `src/`. A collapse that pinches the surface at a vertex passes every QEM-02 check and is reported `topology_clear`.
- **Fix (DESIGN):** the collapse engine must enforce the **link condition** AND a **vertex-fan test** (after retarget, vertex `a`'s surviving incident faces must form a single edge-connected umbrella). The skinny *penalty* only reorders cost — guards must *reject*.
- **Fix (verification):** add a `nonmanifold_vertices` counter to `mesh_metrics.cpp` (or a per-vertex fan-count assertion) and assert it stays 0 in S1/S4. Without this, QEM-02 cannot prove its own contract.

### B2 — Slice 3 as written regresses existing reference-target tests + violates the default-preservation anti-goal
- PLAN S3 says route the `reference-target` preset through QEM via `_simplifier_backend_for_quality_preset` (`export.py:1091-1095`). That breaks `test_real_pixal3d_export.py:1210` (`== 'topology-aware'`), `:580` (`requested_simplifier_backend == 'topology-aware'`), and `:614-624` (backend/algorithm asserts).
- **Fix:** do **NOT** change `_simplifier_backend_for_quality_preset`. Gate QEM on an **explicit opt-in** (e.g. `backend="qem"` / a new `simplify_backend` param to `export_pixal3d_glb`). Keep the preset→backend mapping and its tests green. Update S3 acceptance accordingly.

### B3 — Determinism: adjacency/re-push containers must be ordered, not `std::unordered_map`
- DESIGN mirrors `DirectedEdgeKeyHash` (`simplify.cpp:474`, unordered). If the per-vertex incident-face list / per-edge adjacency iterated by the cost re-push loop is an `unordered_map`, iteration order varies by ASLR hash seed across processes/builds → divergent collapse order on equal-cost edges. The codebase already hit this and works around it (`simplify.cpp:1376-1380` sorts seeds extracted from an unordered_map).
- **Fix (DESIGN):** mandate sorted `std::vector` (by `edge_id`) or `std::map` for collapse-loop containers; define the heap comparator explicitly as **(cost ASC, edge_id ASC, version DESC)** as a named struct; specify the singular-fallback equal-error tiebreak (mirror `cluster_mesh:335-342`: error → distance → index).
- **Fix (verification):** S1's "equal arrays on a second call" is same-process only (same hash seed) — add a **cross-process** determinism check (subprocess re-run + byte compare, or two `PYTHONHASHSEED` values) in S1 or S5.

### B4 — `add_backend_stats` emits ~15 clustering-specific fields that tests assert field-by-field; QEM must emit them or KeyError
- `simplify.cpp:1571-1596` + the `simplify_mesh` body emit `cluster_count`, `grid_resolution`, `degenerate_faces_removed`, `representative_vertices_selected`, `quadric_representative_*`, etc.; asserted at `test_mesh_processing.py:200-247` and `test_real_pixal3d_export.py:740-775`.
- **Fix (DESIGN contract):** enumerate the exact stat-field set for `backend="qem"`; emit 0/sentinel values for all legacy clustering fields so no test KeyErrors, plus the new QEM fields. Pin `production_blocker_values` for qem: push only `missing_narrow_band_dc_remesh` (so `export.py:472-478` clears it post-remesh), never `missing_qem_edge_collapse_simplification`; assert `quality_tier=="production"` when blockers empty.

## Risks (handle in execution, assert explicitly)

- **R1 — Input size ≠ target size.** The ~50k is the *output* target; QEM's *input* is the remesh output at res1024 (~500k–2M faces, undocumented). Measure & assert `simplify_stats['source_faces']` (`simplify.cpp:1838`) early in S3; word the QEM-07 budget as "input N faces → target M faces < T s".
- **R2 — QEM-07 is pass/fail only.** Add numeric assertions reusing existing hooks: `diagnostics['timings_sec']['simplify_mesh'] < BUDGET_S` and `memory['stage_peaks']['simplify_mesh']['peak_current_rss_bytes'] < BUDGET_B`; add `'simplify_mesh'` to `required_stages`.
- **R3 — Heap peak ~10× live edges** (lazy invalidation, no compaction) → OOM risk at remesh-output scale. Rebuild/compact heap when size > 3× live-edge count; assert a memory bound.
- **R4 — Boundary lock can silently STALL.** If input-prep leaves residual loops (`small_boundary_loop_fill` caps at 64 edges; turtle 291 / violin 1111 loops from `repair_nonmanifold`), locked boundary edges can exhaust the heap before target → silent under-decimation that still "passes" watertight. Assert `target_reached` (reuse `target_not_reached` semantics) distinct from watertightness; bound + record residual loops.
- **R5 — QEM-05 contrast not machine-asserted.** In S4, run `backend="spatial-cluster"` on the same input and assert `clustering boundary_loop_count > qem boundary_loop_count` — don't leave it to the human checkpoint.
- **R6 — Normal-flip blind spot.** Reject when `dot(pre,post) <= cos(theta_min)` (positive threshold), and make degeneracy a hard area/aspect reject, not just a cost penalty.
- **R7 — Link-condition low-valence cases.** Add guards for valence-3 / tetrahedron fold-over (canonical-face duplicate collision in merged 1-ring; tetrahedron floor of 4 faces); cover with a non-icosphere synthetic (icosphere is uniformly high-valence and won't exercise it).
- **R8 — resolve_backend throw.** Add the `qem` arm + update the message string (`simplify.cpp:1540`); confirm `test_mesh_processing.py:849-853` still matches.

## Verification-scope corrections
- **S1** must include adversarial synthetics beyond icosphere: thin double-sided sheet (pinch/fold), high-valence hub, valence-3 case — each paired with the B1 vertex-manifold assertion.
- **S2** verification must also run the non-heavy export unit tests that hard-assert blocker strings (`test_export_quality_summary_*`, `test_topology_blocker_map_*`) — they need no Metal/fixtures and catch wiring regressions fast.
- **S5** explicitly name `test_real_pixal3d_export.py` (heavy + non-heavy) as a regression target; add the cross-process determinism check.

## Matrix (verdict-driving low scores)
Edge case coverage 3 (B1) · Test strategy 3–4 (R2/R5) · Data flow clarity 4 (B3) · Dependency risk 4 (R1). Architecture fit 6–7 and Rollback safety 7 are healthy — the change is additive and cleanly revertible once the contract gaps above are closed.
