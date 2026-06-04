# Engineering Review Findings — Round 2 (verdict: needs_correction)

Round-2 closure gate on the revised plan. **10 of 12 prior findings (B1–B3, R1, R2, R4–R8) are genuinely closed** — mechanism specified AND a verification that fails if the fix is absent, grounded in source. The corrections, however, introduced **two execution-breaking gaps** plus a contract loophole. Apply these in `auto-plan` (round 3), then re-run `auto-eng-review`. All items below are verified against code.

## Must-fix before execution

### M1 (was NR3) — S1's `nonmanifold_vertices` must NOT enter the status gate
- `export.py:1445`: `if artifact_blockers or nonmanifold_edges > 0: status = "artifact_blocked"` is the **top of status precedence**. Heavy clustering tests assert `topology["status"] == "rendered_visual_blocked"` at `test_real_pixal3d_export.py:292` and `:539` (both `spatial-cluster`). Clustering merges non-adjacent vertices (the exact pinch mode the new metric detects), so adding `or nonmanifold_vertices > 0` to the gate flips those fixtures → `artifact_blocked`, breaking the tests under S6's heavy re-run.
- **Fix:** S1 adds `nonmanifold_vertices` as a **reported count only** (additive key in metrics + the topology `classes` block), and does **NOT** alter the status-determining gate at `:1445`. Rewrite the S1 line "so the production gate is honest" → "reported as a count/class; the status gate is unchanged." Gate-honesty for pinches is deferred (the B1 correctness proof rests on the metric + synthetics + S2/S5 assertions, not the gate).

### M2 (was NR6) — S2 must fork its own qem stat-return path (self-containment + chain integrity)
- The shared stat block (`simplify.cpp:1844-1860`) reads `best.cluster_count`, `best.grid_resolution`, `best.representative_vertices_selected`, `best.quadric_representative_*` **unconditionally** from a `ClusterResult best`. PLAN orders S2 (engine) **before** S3 (sentinel rewrite) and S2 `Depends on: S1` only. As written, `simplify_mesh(backend="qem")` in S2 routes through this block with no populated `best` → garbage/crash, so S2's own verification (`-k qem`, which inspects the returned mesh/stats) fails before S3 lands.
- **Fix:** S2 introduces its **own qem return path / stat dict** (self-contained, emitting at least the fields S2's tests read), so S2 is verifiable independently; S3 then refines it to the full sentinel keyset. Also covers the backend-agnostic **early-return** at `simplify.cpp:1714` (NR7) — S2/S3 must emit the qem contract there too.

### M3 (was NR1) — `simplify_backend="qem"` needs a watertight-input guard (zero silent failures)
- `export.py:434` `simplify_source = cleaned`; remesh runs only `if remesh:` and `remesh` defaults `False` (`:237`). So `simplify_backend="qem"` with the default `remesh=False` feeds QEM the raw, non-watertight cleaned mesh; boundary-lock (DESIGN guard 1) then freezes every boundary edge → the loop stalls (`target_not_reached`) and yields an under-decimated mesh that still passes a *vacuous* watertight check.
- **Fix (S4):** when `simplify_backend="qem"`, either **require** `remesh=True` (+ `remesh_repair_nonmanifold=True`) with a clear `ValueError`, or **auto-enable** the watertight input-prep. S4 acceptance asserts the rejection/auto-enable, not just the happy path.

### M4 (was B4, partial) — pin the qem stat contract by keyset-equality, not an enumerated subset
- DESIGN's sentinel enumeration (~15 fields) is **incomplete** vs the real emitter: the shared path also emits `unreferenced_vertices_removed`, `min_component_faces`, `candidate_faces_considered`, `accepted_faces`, `simplified`, `pre_simplify_faces/vertices`, plus the entire `add_small_loop_fill_stats` / `add_pre_simplify_loop_fill_stats` families (~30 fields asserted at `test_real_pixal3d_export.py:740-775`). Those heavy field tests run **topology-aware**, not qem (opt-in), so they won't catch a qem-path omission.
- **Fix (S3):** S3 acceptance asserts the qem `simplify_stats` **keyset equals a `spatial-cluster`/`topology-aware` call's keyset** (set-equality on the same input), so any missing sentinel KeyErrors in CI regardless of the enumeration. Update DESIGN to state the governing rule (keyset-equality) over the partial list.

## Assert during execution (followups, non-blocking)
- **F1 (NR2):** add a Python-side validator for `simplify_backend` (accept `None`|`"qem"`, reject others with `ValueError` mirroring `_resolve_pixal3d_uv_backend`), with a non-heavy unit test — avoids a late C++ `nb::value_error` deep in the pipeline.
- **F2 (NR5):** S5 must **pin and record** the `(quality_preset, target_faces, target_faces_source)` the qem proof exercises; a preview-target proof is a legitimate tractable-scale proof (SPEC allows) but will **not** clear the reference-target production gate — state this explicitly so the proof's scope is unambiguous.
- **F3 (R3, partial):** heap-compaction (rebuild at >3× live edges) is verified only by aggregate `peak_current_rss_bytes < BUDGET_B` at fixture scale, which proves the symptom (no OOM), not that compaction fires; acceptable given reference-scale is deferred, but optionally add a direct "compaction triggered" assertion.

## OK / confirmed safe (no action)
- **NR4:** grep of `src/` found **no** arithmetic/ratio/type-specific reader of the clustering sentinel fields — a `0`/`"n/a"` sentinel cannot raise `TypeError` or skew a ratio (only equality asserts in topology-aware-only tests). B4's sentinel *values* are safe; M4 is about the *keyset*, not the values.
- **NR8/R8:** the `resolve_backend` throw-message change is safe (substring match at `test_mesh_processing.py:852`); `_topology_blocker_map` unit test defaults the new field to 0.
- **NR9:** the S1→S6 dependency order is otherwise sound; only M1 (S1 gate) and M2 (S2/S3 stat coupling) needed fixing.
