# SPEC — mlx-spatialkit native GLB watertightness

## Bounded goal

Eliminate **genuine open-boundary defects** (visible holes / seams) in the native reference-target GLB export on both fixtures, while reconciling the harmless coincident duplicate-position seams that inflate the boundary metric — i.e. make the output render watertight without adding spurious geometry or regressing manifoldness/parity.

## Broader intent

Close the first of two recorded geometry-quality defects (`.agent/wiki/LEARNINGS.md`): the holes/seams the user sees in the generated asset. Thin-feature loss (violin string) is the *other* defect and is out of scope. **One change, not a roadmap.**

## Work scale & shape

- **Scale:** focused single-outcome change in the remesh→QEM geometry path (+ its honest gate + tests), front-loaded with a short characterization step.
- **Shape:** quality-coverage with a **deviation-from-reference** target (the reference is *not* watertight; our improvement must be stated honestly, like the UV-unwrap zero-overlap guarantee).

## Mechanism (root cause — evidence-verified, corrected after Codex review)

1. Remesh **without** nonmanifold repair is boundary-free (`tests/test_real_pixal3d_export.py:334`).
2. The QEM path requires `remesh_repair_nonmanifold=True` (`export.py:401-404`). That repair (`cpp/remesh.cpp:45-50`, applied `:431`) resolves non-manifold vertices by **duplicating them at the same position** ("each other fan gets a duplicate vertex at the same position … without removing any face") — i.e. vertex-splitting that creates **coincident duplicate-position boundary seams**.
3. QEM edge-collapse is **boundary-locked** (`cpp/simplify.cpp:2024-2027`) so it preserves those seams; the residual is recorded as `qem_pre_fill_residual_boundary_loops` (`cpp/simplify.cpp:2661,2720`).
4. The bounded pre-QEM filler closes only simple/small loops; the reference-clean filler **appends centroid-fan faces with NO patch validation** (`cpp/simplify.cpp:1055`) — the `PatchRejectReason` guards belong to a *different* filler (`fill_small_boundary_loops`, `:1100`).
5. Tests pin the residual as topology-preserved (`:2031,:2198,:2132`: `boundary_loop_count == qem_pre_fill_residual_boundary_loops`, residual ≤ 64).

**Baseline reality:** the comparison-baseline GLB (a prior MLX export, not ground truth) is **not watertight** — `boundary_edges: 25105` (`inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json:135`) — and its remesh branch skips the final cleanup/fill (`vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:164-187`). It renders watertight *because the boundary edges are coincident seams, not visible gaps.*

## Required outcome

On the reference-target QEM path (`quality_preset="reference-target"`, `remesh=True`, `remesh_repair_nonmanifold=True`, `simplify_backend="qem"`), both fixtures (main `pixal3d-1024-cascade` + `violin-bow`):
- **Genuine open boundaries → 0**: every residual boundary loop that is NOT a coincident duplicate-position seam is closed by a *validated* patch.
- **Coincident seams reconciled**: duplicate-position boundary vertices are welded (or proven harmless), so the mesh renders watertight and texture/normal seams at those boundaries are removed.
- **No regression**: `nonmanifold_edges == 0`, `nonmanifold_vertices == 0`, no new degenerate/duplicate faces, face-count not collapsed, connected-component count preserved.

Approach is **outcome-bounded**. Candidate approaches for the plan (ranked by Codex + verified): (a) **duplicate-position seam reconciliation / vertex-weld before QEM** (geometrically safe — merges coincident verts; addresses root cause) — *recommended primary*; (b) topology-preserving remesh repair / QEM guards around non-manifold vertices instead of splitting them; (c) validated closure of any genuine non-coincident gaps via robust triangulation (NOT raw centroid-fan); (d) higher `remesh_resolution` — **mitigation/diagnostic only, not a correctness fix.**

## Acceptance criteria

1. **Characterization (first slice):** on both fixtures via the exact QEM path, classify every residual boundary loop as coincident-seam vs genuine-gap, and confirm whether visible holes/seams actually exist. The target metric is set from this (genuine-gap count, not raw `boundary_loop_count`).
2. Genuine-gap count `== 0` on both fixtures; coincident seams welded so render-watertight.
3. `nonmanifold_edges == 0` **and** `nonmanifold_vertices == 0` preserved.
4. No new `degenerate_faces`/`duplicate_faces`; `export_blocking_reasons == []`.
5. **Anti-gaming:** face-count has a lower bound (no mass deletion), connected-component count preserved, surface-area/bounds/volume deltas within tolerance, and no selective boundary-face deletion to drop `boundary_edges`. Any fill patch passes degenerate/duplicate/non-manifold/normal-flip/self-intersection validation.
6. **Determinism:** identical output across two in-process runs and cross-process.
7. **Honest gate:** add a real geometry-watertightness verdict to the reference stage contract keyed on the genuine-gap metric (current status is mostly algorithm-name based, `export.py:1386`); an anti-gaming test flips it back when a genuine gap is present. Update the residual-pinning tests (`:2031,:2198,:2132`) to the new target.
8. A new heavy test exercises the exact config (`reference-target` + `remesh=True` + `remesh_repair_nonmanifold=True` + `simplify_backend="qem"`) on both fixtures; existing heavy suite stays green within budgets (<600 s / <12 GiB per fixture).

## Anti-goals

- Not thin-feature/sharp-feature preservation (violin string) — separate change.
- Not weakening QEM's closed-manifold-by-construction guards; not emitting non-manifold output.
- Not "match the reference's boundary metric" — the reference is non-watertight; do not regress to 25k boundary edges, and do not over-fill harmless coincident seams with spurious geometry.
- Not UV/texture/sampling re-work, GPU-trilinear, or reference-scale (4096/1M).
- Not a roadmap / multi-phase decomposition.

## Scope coverage

- **Included:** characterize residual loops; reconcile coincident seams; close genuine gaps with validated patches; honest watertightness gate; real test at the exact config.
- **Deferred / not in scope:** thin-feature preservation; the broader non-manifold-tolerant remesh rewrite beyond what seam reconciliation needs; GPU-trilinear; reference-scale.

## Key risks

- **Centroid-fan is unsafe** for non-convex/non-planar/branched loops and the reference-clean filler currently appends them unvalidated (`simplify.cpp:1055`); existing validation detects neither self-intersection nor normal-flip. Any genuine-gap closure must use robust triangulation + full patch validation.
- **Metric ≠ visible defect:** `boundary_loop_count > 0` may be mostly harmless coincident seams (the reference has 25k). Targeting raw zero would add spurious geometry and diverge from reference. The characterization slice de-risks this; if no genuine visible holes exist on the correct path, the change narrows to seam reconciliation + honest metric.
- **Welding coincident verts** can itself reintroduce the non-manifold condition the repair split apart — must weld only where it preserves manifoldness (or pair with QEM non-manifold-vertex guards).

## Assumptions

- The reference-target QEM path is the one that matters; default `preview` path unaffected.
- The user's visible "holes" are on this path (earlier gross holes were a separate bad-param checkpoint, already explained).
- Source NPZ + reference `model.glb` + reference `trace.json` are present under `inputs/mlx-spatialkit/` (verified).

## Evidence anchors

- `cpp/remesh.cpp:45-50,431` (duplicate-position vertex-split repair); `cpp/simplify.cpp:2024-2027` (QEM boundary lock), `:2661,2720` (residual metric), `:1055` (unvalidated centroid-fan), `:1100` (guarded filler), `:151` (PatchRejectReason).
- `cpp/mesh_metrics.cpp:18-34,236` (BoundaryTopology metrics).
- `src/mlx_spatialkit/export.py:333` (signature/flags), `:401-404` (QEM manifold-input requirement), `:1386,1498-1560` (stage-contract verdicts).
- `tests/test_real_pixal3d_export.py:334` (remesh no-repair watertight), `:2031,:2198,:2132` (residual pinned tests), `:700` (the native-chart non-QEM test — NOT the target path).
- `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json:135` (reference boundary_edges=25105); `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:164-187` (reference remesh branch skips cleanup/fill).

## Review: Engineering (Codex gpt-5.5, xhigh, read-only) — 2026-06-15

Verdict: **needs_correction** → corrections applied to this SPEC. Verified findings folded in: (1) root cause is remesh duplicate-position vertex-splitting, not "simplify input-prep"; (2) the reference is NOT watertight (25k boundary edges) so the naive 0-loops target was wrong — reframed to genuine-gap elimination + seam reconciliation; (3) `:827-833` was the wrong (non-QEM) anchor — corrected to the `qem_pre_fill_residual_boundary_loops` tests; (4) `fill_reference_clean_boundary_loops` appends unvalidated centroid-fans — added patch-validation + robust-triangulation requirement; (5) added anti-gaming acceptance (face-count/component/volume preservation, no boundary-face deletion, nonmanifold_vertices/degenerate/duplicate checks) and a real geometry-watertightness gate. Open item the plan resolves first: characterize coincident-seam vs genuine-gap to fix the true target metric.
