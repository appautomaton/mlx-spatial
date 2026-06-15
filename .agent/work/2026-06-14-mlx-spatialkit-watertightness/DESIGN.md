# DESIGN — watertightness via seam classification + reconciliation

Compact design for the non-obvious parts. Full context in `SPEC.md`.

## Core insight

The nonmanifold repair (`cpp/remesh.cpp:45-50`) resolves non-manifold vertices by **duplicating them at the same position** — creating boundary edges between coincident vertices. These are **coincident duplicate-position seams**: topologically open, geometrically zero-gap, render watertight. The reference has ~25k of them (`trace.json:135`) and looks fine. QEM is boundary-locked so it preserves them.

So `boundary_loop_count` conflates two very different things:
- **Coincident seam** — boundary loop where each vertex has a duplicate at the same position (within `eps`) also on the boundary; welding the duplicates closes it with zero geometry change. Harmless to render; fixing it removes texture/normal seams.
- **Genuine gap** — a real open hole with no coincident partner; needs a filled patch.

## Metric (Slice 1)

Classify each residual boundary loop into `coincident_seam_loop_count` vs `genuine_open_boundary_loop_count` (+ edge counts), surfaced in simplify stats / `mesh_metrics`. Classification: a boundary vertex is *seam-paired* if another boundary vertex shares its position within `eps` (deterministic spatial hash). A loop is a coincident seam iff all its vertices are seam-paired and welding preserves manifoldness; else genuine.

## Reconciliation (Slice 2) — weld, don't fill

Reconcile coincident seams by **welding duplicate-position boundary vertices**, not by filling. Weld safety condition: a weld is applied only if merging the duplicate pair leaves every incident edge with ≤ 2 faces (no new non-manifold edge) and flips no face normal. Deterministic order: sort candidate welds by (position, min-vertex-id). This is geometrically safe (merges coincident points) and reverses the split that opened the seam.

## Genuine-gap closure (Slice 3) — validated, conditional

Only if Slice 1 finds genuine gaps. Replace the **unvalidated** centroid-fan append in `fill_reference_clean_boundary_loops` (`cpp/simplify.cpp:1055`) with robust triangulation (ear-clip / constrained) gated by full patch validation: degenerate, duplicate, non-manifold edge, normal-flip, and self-intersection rejection. Centroid-fan is retained only for small convex loops where it is provably safe.

## Honest gate + anti-gaming (Slice 4)

Geometry-watertightness verdict in the reference stage contract keyed on `genuine_open_boundary_loop_count == 0` AND coincident seams reconciled — not raw `boundary_loop_count`. Anti-gaming invariants enforced as acceptance: face-count lower bound (no mass deletion), connected-component count preserved, surface-area/bounds/volume deltas within tolerance, no selective boundary-face deletion. Determinism: identical output cross-process.

## Non-goals

No QEM cost/guard changes (thin-feature is a separate change). No UV/texture rework. Weld/fill only on the reference-target QEM path.
