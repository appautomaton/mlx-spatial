# mlx-spatialkit QEM Edge-Collapse Simplification Spec

## Bounded Goal

Add a native, dependency-light **QEM (quadric-error-metric) edge-collapse** simplifier to `mlx-spatialkit` that reduces a watertight narrow-band remesh to a target face count **while preserving its closed manifold topology** — so the decimated GLB geometry stays watertight on both fixtures, instead of the current clustering simplifier that re-tears it.

## Broader Intent

Production-quality native geometry/export parity with the Pixal3D/TRELLIS reference. This is **stage 2 of 4** on the production-geometry path: narrow-band remesh (DONE — verified change `2026-05-28-mlx-spatialkit-narrow-band-remesh`) → **QEM edge-collapse (this change)** → xatlas-equivalent UV unwrap → texture postprocess. The reference does `remesh → simplify(QEM)`; our remesh closes all holes but the existing **clustering** simplifier re-opens them (turtle 0→41 boundary loops, violin 0→52 at ~50k faces, measured), so a topology-preserving QEM simplifier is the precondition for a clean, decimated, textured GLB.

## Work Scale And Shape

- **Scale:** capability-sized — a new simplifier backend plus the input-prep needed to keep its output watertight.
- **Shape:** reference parity + structural addition (a new `simplify_mesh` backend), verified by topology + fidelity metrics on two fixtures (engineering/runtime).

## Selected Lenses

- **engineering:** the core is the QEM edge-collapse algorithm, its correctness (manifold/boundary preservation), and honest wiring into the existing quality/blocker logic.
- **runtime:** edge-collapse over millions of edges is the cost/memory risk; CPU-first with a measured budget, Metal only if required.

## Required Outcome (parity target)

Reference (behavior-only, no CUDA/vendor line-port): CuMesh `CuMesh.simplify` (`/tmp/CuMesh/cumesh/cumesh.py:320`) — iterative QEM edge-collapse via `simplify_step(lambda_edge_length, lambda_skinny, thresh)` looping to a target with `thresh` escalation; kernel `/tmp/CuMesh/src/simplify.cu:531`; called right after remesh in `o_voxel.postprocess.to_glb`'s remesh branch. Native parity requires:

1. A native **QEM edge-collapse backend** in `packages/mlx-spatialkit/cpp/simplify.cpp` — per-vertex 4×4 quadrics accumulated from incident face planes, collapse the lowest-error *valid* edge (with edge-length + skinny-triangle penalties), preserving manifoldness and boundaries; iterate to the target face count. Exposed as a `simplify_mesh(backend=…)` option **alongside** (not replacing) the existing `spatial-cluster` / `topology-aware` clustering backends.
2. **Topology preservation** — the differentiator from clustering: on a watertight manifold input, simplification introduces **no** new boundary loops / open chains / non-manifold edges (clustering turns the remesh's 0 → 41/52).
3. **Honest diagnostics** — `simplify_stats["qem_simplification_backend"]` leaves `"not_implemented"`, `qem_equivalence_status` reflects edge-collapse, and `missing_qem_edge_collapse_simplification` clears from `production_backend_blockers` when QEM is used (the `_topology_blocker_map` / `_pixal3d_reference_stage_contract` logic in `export.py` already reads these).
4. **End-to-end watertight decimated mesh** on both fixtures: `remesh → (manifold/closure prep) → QEM` at the production target yields a mesh that is both reduced and watertight (loops/branched/open/nonmanifold within a stated bound), materially better than clustering.

## Constraints

- Dependency-light native C++/Metal: do **not** add MLX, Torch, CUDA, or xatlas as required dependencies (carry-forward). CPU-first; any Metal kernel keeps a CPU fallback.
- Behavior-reference only; do not line-port `simplify.cu` or vendor code.
- Do **not** remove or rename the existing clustering simplifier backends, public knobs, diagnostics, or tests; QEM is additive.
- Reuse existing native primitives where they fit (`cpp/triangle_bvh.hpp` for any fidelity-projection; the existing `small_boundary_loop_fill` for bounded closure prep).
- Heavy artifacts (GLBs, renders, scratch) stay under `/tmp`; no release/tag/push.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| QEM-01 | QEM edge-collapse backend exists and is callable. | `simplify_mesh(backend="qem"…)` reduces faces toward a target via quadric-error edge collapse; unit test on a synthetic closed mesh shows real face reduction and a still-closed manifold. |
| QEM-02 | Topology-preserving simplification (the differentiator). | On a watertight manifold input, post-QEM `boundary_loop_count` / `boundary_open_chain_count` / `boundary_branched_open_chain_count` / `nonmanifold_edges` do **not** increase versus the input (contrast clustering: 0 → 41/52). |
| QEM-03 | Quadric-error fidelity, not random decimation. | The simplified surface stays within a recorded geometric-error bound of the pre-simplify surface (e.g., bounded mean/max closest-point distance); collapse order is quadric-error-driven. |
| QEM-04 | QEM production blocker clears honestly. | `qem_simplification_backend` ≠ `"not_implemented"`; `qem_equivalence_status` reflects edge-collapse; `missing_qem_edge_collapse_simplification` is absent from `production_backend_blockers` when QEM is used. |
| QEM-05 | Two-fixture watertight-decimated proof. | turtle/city AND violin/bow: `remesh → QEM` at the target face count → `mesh_metrics` watertight within the stated bound (materially better than clustering), with honest readiness (production parity still gated on unwrap/texture). |
| QEM-06 | Determinism. | Same input + target → identical output mesh. |
| QEM-07 | Runtime/memory recorded and bounded. | QEM at the proven target completes within a recorded budget; no OOM. |
| QEM-08 | Dependency-light preserved. | No new required MLX/Torch/CUDA/xatlas deps; native build + import succeed; any Metal path falls back to CPU. |
| QEM-09 | Regression contracts preserved. | Existing clustering backends, `test_mesh_processing` / simplify / export / remesh suites, and all public knobs/diagnostics stay green and intact. |
| QEM-10 | Pipeline integration + diagnostics. | `export_pixal3d_glb` can select QEM for the reference/production preset (preset-driven or opt-in, default behavior preserved until proven); diagnostics record backend, before/after face counts, and the quadric-error summary. |

## Scope Coverage Decisions

- **Included:** the QEM edge-collapse backend; topology preservation; quadric-error fidelity; blocker/diagnostics wiring; pipeline integration; two-fixture watertight-decimated proof; determinism; perf; regression; **the bounded manifold/closure input-prep needed to give QEM a watertight input** (reuse `repair_nonmanifold` + the existing `small_boundary_loop_fill` to close the small repair-induced loops — exact composition is a planning decision; this change owns the watertight-decimated *outcome*).
- **Deferred (own follow-on changes):** xatlas-equivalent UV unwrap; texture postprocess / Telea-equivalent inpaint; full reference-scale 1M-vertex / 4096-texture parity (unless accepted during planning); Metal acceleration unless CPU perf proves insufficient.
- **Anti-goals:** removing or regressing the existing clustering simplifiers; line-porting CUDA; spawning another micro-spec chain; claiming production-quality GLB from QEM alone (unwrap + texture still pending); declaring success on face-count reduction alone without the watertight-preservation check.

## Risks

- **Input watertightness (the loops↔nonmanifold tradeoff).** QEM edge-collapse needs a manifold input; the remesh's `repair_nonmanifold` gives `nonmanifold_edges == 0` but opens small boundary loops (turtle 291, violin 1111), while pure remesh is closed but non-manifold. To deliver a *fully* watertight decimated mesh, this change must close those small repair-induced loops (bounded `small_boundary_loop_fill`) before/within QEM, or bound and record any residual. **This is the key planning decision; the SPEC commits to the watertight-decimated outcome + the topology-preserving QEM mechanism, not to a specific input-prep recipe.**
- **CPU perf at high face counts** (priority-queue edge collapse over millions of edges). Mitigation: measure (QEM-07); prove the mechanism at a tractable target; Metal/reference-scale deferred unless needed.
- **Correctness of collapse guards** (no collapses across boundaries, no normal flips / non-manifold creation). Mitigation: standard QEM validity checks + unit tests on synthetic meshes.
- **Overfitting** to one asset. Mitigation: two-fixture evidence (QEM-05).

## Assumptions

- `/tmp/CuMesh` and `vendors/TRELLIS.2/o-voxel` remain available locally as behavior references.
- QEM operates downstream of `remesh(repair_nonmanifold=True)` (strictly-manifold input); the small repair-induced boundary loops are closed by bounded existing hole-fill so QEM preserves a watertight surface.
- The mechanism is proven at a tractable target face count (e.g., the 50k preview target); reference-scale (1M-vertex / 212,542-face) parity is recorded as deferred if CPU perf or time makes it impractical in this change.
- Existing readiness vocabulary and the `production_quality_ready` gate remain valid — QEM clears the QEM blocker only; full production parity stays gated on the unwrap and texture follow-ons.
