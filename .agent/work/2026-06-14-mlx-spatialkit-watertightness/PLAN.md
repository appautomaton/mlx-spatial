# PLAN — mlx-spatialkit native GLB watertightness

**Goal:** Eliminate genuine open-boundary defects and reconcile harmless coincident duplicate-position seams on the reference-target QEM path so both fixtures **render watertight** (no visible holes), no manifold/parity regression.

**Done = visual watertightness** (user decision, 2026-06-15): the **browser render-proof** (no visible holes in the rendered views, user-confirmed) is the acceptance gate. Topology metrics (`genuine_open_boundary_loop_count == 0`, seams welded, manifold preserved) are *supporting evidence*, not the hard bar. See `SPEC.md` + `DESIGN.md`.

## Architecture approach

Discovery-first, render-validated. The reference is *not* watertight (coincident seams); so we (1) render the current output and measure the genuine-vs-seam split, (2) weld coincident seams (root-cause-safe), (3) close any genuine gaps with validated triangulation, (4) gate on a **browser render-proof** backed by the genuine metric + anti-gaming. Every mutation slice carries a render check, since the user judges by eye. No QEM cost/guard changes. Detail in `DESIGN.md`.

## Execution routing and topology

- **Order:** serial; each slice depends on the prior.
- **Continuation:** continue through all slices after each verification passes; execution windows are context batches, not stopping points.
- **Checkpoints:** after Slice 1 (`decision`) — user confirms the visible defect from baseline renders + the seam-vs-genuine split decides the path before any geometry mutation; after Slice 4 (`human-verify`) — user signs off the final render-proof, which is the definition of done.
- **Subagent route:** Slices 2–4 `subagent recommended` (correctness-critical native C++/Metal; one implementer per slice — serial, so orthogonal in time), consistent with the user's agentic-team request for watertightness.
- **Parallel-safe groups:** none.

## Ordered slice sequence

### Slice 1: Visual baseline + seam-vs-gap classification metric + characterization

Required:
**Objective:** Establish the visual baseline (render the *current* reference-target output for both fixtures) AND add a metric that classifies each residual boundary loop as coincident duplicate-position seam vs genuine gap, so the true target and visible defect are both grounded before any geometry mutation.
**Acceptance criteria:**
- Both fixtures' current reference-target GLBs rendered via `scripts/spatialkit/render_glb_visual_parity.cjs` (iso/front/top views); screenshots saved to `/tmp` for user inspection.
- New fields `coincident_seam_loop_count` / `genuine_open_boundary_loop_count` (+ edge counts) in `mesh_metrics`/simplify stats, surfaced through `export_pixal3d_glb` diagnostics.
- Unit test: a synthetic coincident-seam mesh classifies as seam; a synthetic real-hole mesh classifies as genuine.
- Heavy run on both fixtures (`reference-target`, `remesh=True`, `remesh_repair_nonmanifold=True`, `simplify_backend="qem"`) emits per-fixture seam vs genuine counts; written to a characterization summary alongside the renders.
**Verification:** `cd packages/mlx-spatialkit && pytest tests/test_mesh_processing.py -k seam_classification` (unit) and `pytest -m heavy tests/test_real_pixal3d_export.py -k watertight_characterization` (both fixtures, counts emitted); renders present in `/tmp`.

Defaults / overrides:
**Checkpoint after:** decision
**Checkpoint reason:** Present the rendered screenshots + per-fixture `genuine_open_boundary_loop_count`/`coincident_seam_loop_count`. The user (a) confirms which holes/views are the actual visible defect, and (b) the metric decides the path: seam-reconciliation only (genuine == 0) vs seam-reconciliation + genuine-gap closure (genuine > 0). Both confirmations gate geometry mutation. If the current render shows no visible holes, surface that — the fix may reduce to seam reconciliation + honest metric.
**Touches:** `cpp/mesh_metrics.cpp`, `cpp/simplify.cpp` (stats), `src/mlx_spatialkit/export.py` (surface), `tests/`, `scripts/spatialkit/render_glb_visual_parity.cjs` (invoke only)
**Produces:** baseline renders + classification metric + characterization numbers (no geometry mutation)

### Slice 2: Coincident duplicate-position seam reconciliation (weld)

Required:
**Objective:** Weld coincident duplicate-position boundary vertices on the reference-target QEM path, where the weld preserves manifoldness, so coincident seams are removed and the mesh renders watertight without added geometry.
**Acceptance criteria:**
- `coincident_seam_loop_count == 0` on both fixtures after reconciliation.
- `nonmanifold_edges == 0` AND `nonmanifold_vertices == 0` preserved; no new degenerate/duplicate faces.
- Welds applied in deterministic order; no normal-flipped face; weld-safety (≤2 faces/edge) holds.
- Face-count/component-count/surface-area within tolerance (no geometry added/lost beyond the weld).
- **Render check:** post-weld browser render of both fixtures shows no visible holes/seams at the welded boundaries; the violin string is not visibly worsened or deleted (thin-feature improvement is out of scope, but no regression).
**Verification:** `cd packages/mlx-spatialkit && pytest -m heavy tests/test_real_pixal3d_export.py -k watertight_seam_weld` (both fixtures: seam count → 0, manifold preserved, determinism) + post-weld render via `render_glb_visual_parity.cjs` saved to `/tmp` for inspection.
**Execution:** subagent recommended
**Depends on:** Slice 1
**Touches:** `cpp/simplify.cpp` / a new finalization pass, `cpp/mesh_metrics.cpp`

### Slice 3: Validated genuine-gap closure (conditional on Slice 1)

Required:
**Objective:** Close genuine (non-coincident) gaps with robust triangulation gated by full patch validation, replacing the unvalidated centroid-fan append (`cpp/simplify.cpp:1055`).
**Conditional / non-blocking:** Real closure work only if Slice 1 found `genuine > 0`. If `genuine == 0` AND Slice 2's render passes (no visible holes), the visible defect is already fixed — the centroid-fan validation-hardening becomes an **optional, non-blocking** cleanup and does not gate change completion.
**Acceptance criteria:** (apply when genuine > 0; otherwise this slice is non-blocking hardening)
- `genuine_open_boundary_loop_count == 0` on both fixtures, and post-fill render shows the holes closed.
- Every fill patch passes degenerate/duplicate/non-manifold/normal-flip/self-intersection validation (unit test rejects each bad-patch class).
- No self-intersecting or normal-flipped patch reaches output; centroid-fan retained only for provably-safe small convex loops.
**Verification:** `cd packages/mlx-spatialkit && pytest tests/test_mesh_processing.py -k patch_validation` (unit, bad-patch rejection) and `pytest -m heavy tests/test_real_pixal3d_export.py -k watertight_genuine_gap` (genuine count → 0) + render confirms closure.
**Execution:** subagent recommended
**Depends on:** Slice 2

### Slice 4: Visual render-proof gate + honest metric gate + anti-gaming + test re-pin

Required:
**Objective:** Make the **browser render-proof the acceptance gate** (no visible holes, user-confirmed) backed by a geometry-watertightness verdict in the reference stage contract, enforce anti-gaming invariants, re-pin the residual tests, and add an end-to-end test at the exact config on both fixtures.
**Acceptance criteria:**
- **Primary gate (visual):** final browser render of both fixtures via `render_glb_visual_parity.cjs` shows no visible holes in iso/front/top views; `rendered_visual_ready`/`browser_rendered_visual_proof` true AND the screenshots are human-confirmed hole-free. (The boolean only proves non-blank render — human inspection is required, per README:237-239.)
- **Supporting metric gate:** stage-contract verdict flips `reference_matched` only when `genuine_open_boundary_loop_count == 0` and coincident seams reconciled; an anti-gaming test flips it back when a genuine gap or a deletion-gamed mesh is present.
- Anti-gaming invariants verified: face-count lower bound, connected-component preservation, surface-area/bounds/volume deltas within tolerance, no selective boundary-face deletion.
- Residual-pinning tests (`tests/test_real_pixal3d_export.py:2031,2198,2132`) updated to the new target; new heavy test exercises `reference-target`+`remesh=True`+`remesh_repair_nonmanifold=True`+`simplify_backend="qem"` on both fixtures.
- Cross-process determinism holds; full heavy suite green within budgets (<600 s / <12 GiB per fixture); no new runtime deps.
**Verification:** `cd packages/mlx-spatialkit && pytest tests/test_real_pixal3d_export.py -k "watertight_gate or anti_gaming"` (non-heavy gate logic) and `pytest -m heavy tests/test_real_pixal3d_export.py` (full heavy suite green, budgets) + final render artifacts in `/tmp` for human sign-off.
**Execution:** subagent recommended
**Depends on:** Slice 2 (+ Slice 3 when genuine gaps exist)
**Checkpoint after:** human-verify
**Checkpoint reason:** User signs off on the final rendered screenshots (no visible holes) — the visual gate is the definition of done and only a human can confirm it.

## Requirement traceability

| SPEC acceptance | Slice |
|---|---|
| AC1 characterization / true metric | 1 |
| AC2 genuine→0 + seams welded | 2 (seams), 3 (genuine) |
| AC3 nonmanifold edges+vertices==0 | 2, 3 |
| AC4 no degenerate/duplicate; blockers empty | 2, 3, 4 |
| AC5 anti-gaming (count/component/volume/no-deletion) | 4 |
| AC6 determinism | 2, 4 |
| AC7 honest gate + test re-pin | 4 |
| AC8 real e2e test at exact config + budgets | 4 |

## Aggregate verification

| Slice | Command |
|---|---|
| 1 | `pytest tests/test_mesh_processing.py -k seam_classification` + `pytest -m heavy ... -k watertight_characterization` |
| 2 | `pytest -m heavy tests/test_real_pixal3d_export.py -k watertight_seam_weld` |
| 3 | `pytest tests/test_mesh_processing.py -k patch_validation` + `pytest -m heavy ... -k watertight_genuine_gap` |
| 4 | `pytest ... -k "watertight_gate or anti_gaming"` + `pytest -m heavy tests/test_real_pixal3d_export.py` |

Build note: `rm -rf /tmp/mlx-spatialkit-build` before a clean rebuild; heavy artifacts to `/tmp`; never sweep `/tmp` mid heavy-suite.

## Review: Engineering (Codex gpt-5.5, xhigh, read-only) — 2026-06-15

Verdict: **aligned_with_gaps** → edits applied. Codex confirmed the plan keeps thin-feature out of scope and the decomposition is sound, but flagged that the plan gated on *metrics* while the user judges by *eye*. Folded in: (1) Slice 1 renders the current output as a visual baseline (verified machinery: `scripts/spatialkit/render_glb_visual_parity.cjs`, export `visual_compare` stage); (2) the Slice 1 checkpoint now requires user visual confirmation of the defect + the seam-vs-genuine decision; (3) Slice 4's primary acceptance is the **browser render-proof** (human-confirmed no visible holes), metrics demoted to supporting; (4) Slice 3 filler-hardening made explicitly non-blocking when `genuine == 0` and Slice 2's render passes; (5) violin-string non-regression guard added to Slices 2/3.

User decisions (2026-06-15): **done = visual watertightness** (render-proof is the gate, not raw metrics); **visual baseline = render current output, user confirms** the real defect before geometry mutation.
