# mlx-spatialkit Narrow-Band Remesh Spec

## Bounded Goal

Implement a native narrow-band dual-contour remesh stage in `mlx-spatialkit`'s export path so Pixal3D GLB geometry is rebuilt into a watertight manifold (zero open/branched boundary chains and zero unfilled boundary loops) on both the Pixal3D 1024 cascade and an independent violin/bow fixture.

## Broader Intent

Reach production-quality geometry/export parity with the Pixal3D/TRELLIS reference (`o_voxel.postprocess.to_glb(..., remesh=True)`). Remesh is the keystone topology stage: it structurally eliminates the open-boundary/branched-chain defects that the 2026-05-27 hole-repair chain could not close (the just-verified rebaseline measured 331/147 branched open chains and 532/330 unfilled boundary loops on the two fixtures, with `simple_open_chains == 0` — i.e., the residual is exactly the topology remesh regenerates). QEM edge-collapse simplification and xatlas-equivalent unwrap are explicit follow-on changes, not part of this one.

## Work Scale And Shape

- **Scale:** capability-sized — one new native pipeline stage plus one supporting refactor.
- **Shape:** reference parity + structural addition (engineering/runtime), verified by topology metrics on two fixtures.

## Selected Lenses

- **engineering:** the change is a native backend stage; correctness and reuse of existing primitives dominate.
- **runtime:** narrow-band sampling at resolution 1024 is the main cost/memory risk; CPU-first with a measured budget, Metal only if required.

## Required Outcome (parity target)

The reference meshing process is `o_voxel.postprocess.to_glb(..., remesh=True, remesh_band=1, remesh_project=0)` → `cumesh.remeshing.remesh_narrow_band_dc` (`/tmp/CuMesh/cumesh/remeshing.py:24`; dual-vertex placement = mean of edge intersections, `/tmp/CuMesh/src/remesh/simple_dual_contour.cu:145`). Native parity is **behavior-only** (no CUDA line-port) and requires:

1. A native `remesh_narrow_band` backend taking `(vertices, faces, resolution, band, project)` → triangle mesh, reusing the existing CPU `TriangleBvh` (point→triangle distance / closest-point / barycentric, `packages/mlx-spatialkit/metal/texture_bake.mm:341`) and the existing dual-contour connectivity skeleton + tables (`packages/mlx-spatialkit/cpp/flexi_dual_grid.cpp`).
2. `TriangleBvh` (+ `closest_point_on_triangle`, `distance2_aabb`, `ClosestPointResult`) hoisted to a shared header and reused by both the texture bake and the remesh, with the bake path behaviorally unchanged.
3. Remesh wired between `clean_mesh` and `simplify_mesh` in `export.py`, **opt-in** (default preserves current behavior until two-fixture proof), with diagnostics: backend name, resolution, band, active-voxel count, timing, and before/after vertex/face/boundary counts; the `remesh_backend` stat leaves `"not_implemented"` and the `missing_narrow_band_remesh` topology blocker clears when remesh is on.

The full engineering mapping (primitive-by-primitive status, parameters, build path, risks, slice breakdown) is normative in [spec/remesh-port-design.md](spec/remesh-port-design.md).

## Constraints

- Dependency-light native only: do not add MLX, Torch, CUDA, or xatlas as required dependencies (carry-forward of the rebaseline contract). CPU-first; any Metal kernel keeps a CPU fallback.
- Behavior reference only; do not line-port CUDA or vendor code.
- Pixal3D parameters in the native [-0.5, 0.5]³ frame: `center=0`, `resolution=grid_size` (1024), `scale=(res+3·band)/res`, `band=1`, `project=0`, `eps=band·scale/res`.
- Remesh inserts between `clean_mesh` and `simplify_mesh`; default off until proven, so the just-verified rebaseline behavior does not regress.
- Heavy artifacts (GLBs, renders, scratch) stay under `/tmp`.
- Do not expand the legacy hole-repair heuristics (earclip / centroid-fan / branched-cycle); remesh supersedes the need for them.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| RMS-01 | Native remesh backend exists and is callable. | A `remesh_narrow_band` binding + Python wrapper consume `(vertices, faces, resolution, band, project)` and return a valid triangle mesh; unit test on a known closed shape (sphere/cube). |
| RMS-02 | TriangleBvh reuse without bake regression. | `TriangleBvh` hoisted to a shared header; existing `test_texture_bake.py` / `test_glb_writer.py` / `test_glb_compare.py` pass unchanged. |
| RMS-03 | Hole closure on both fixtures (the remesh deliverable). | Post-remesh `mesh_metrics` reports `boundary_loop_count == 0`, `boundary_open_chain_count == 0`, `boundary_branched_open_chain_count == 0` for the Pixal3D 1024 cascade AND the violin/bow fixture. **Amended 2026-05-28 (user-approved, post-verify):** `nonmanifold_edges == 0` is moved out of this criterion to the QEM/cleanup follow-on — the reference (`o_voxel.postprocess.to_glb` remesh branch) does not repair non-manifold edges after remesh (it feeds QEM directly), and the double-walled narrow-band DC trades non-manifold edges for boundary loops (measured: vertex-split repair drives `nonmanifold_edges`→0 but opens 291/1111 boundary loops), so simultaneous all-zero is a downstream property, not a remesh-stage one. An opt-in `repair_nonmanifold` capability ships for use as manifold input-prep for QEM. |
| RMS-04 | Surface fidelity preserved. | Max closest-point distance from remeshed vertices to the original surface ≤ a recorded bound (~2 voxel widths); shape silhouette unchanged on both fixtures. |
| RMS-05 | Pipeline integration + honest diagnostics. | Remesh runs between clean and simplify (opt-in); diagnostics record backend / resolution / band / active-voxels / timing / before-after counts; `remesh_backend` ≠ `"not_implemented"` and `missing_narrow_band_remesh` blocker clears when on. |
| RMS-06 | Dependency-light preserved. | No new required MLX/Torch/CUDA/xatlas deps; build + import succeed with current deps; any Metal path falls back to CPU. |
| RMS-07 | Determinism. | Same input + params → identical output mesh across runs. |
| RMS-08 | Runtime/memory recorded and bounded. | Heavy two-fixture remesh completes within a stated budget recorded in diagnostics; no OOM at resolution 1024. |
| RMS-09 | Two-fixture evidence. | RMS-03 / 04 / 08 hold on both lineages, not one. |
| RMS-10 | Regression contracts preserved. | Texture coordinate order, PBR packing, extraction/metrics, GLB writer/viewer, and existing non-heavy + heavy suites stay green. |

## Scope Coverage Decisions

- **Included:** TriangleBvh hoist/refactor; narrow-band voxel build; `simple_dual_contour` UDF dual-vertex placement; normal-alignment quad split; opt-in pipeline integration + diagnostics; two-fixture watertight + fidelity + perf proof; regression protection.
- **Deferred (own follow-on changes):** QEM edge-collapse simplification; xatlas-equivalent UV unwrap; making remesh the default preset behavior; Metal acceleration (only if CPU perf insufficient); `project_back`/surface snap (Pixal3D uses `project=0`); **`nonmanifold_edges == 0` (moved here from RMS-03 by the 2026-05-28 amendment — the QEM/cleanup change owns strictly-manifold output; the opt-in `repair_nonmanifold` native capability is available now as manifold input-prep, at the cost of small boundary loops).**
- **Anti-goals:** another micro-spec chain for one problem; new heavy required dependencies; expanding the legacy hole-repair heuristics; line-porting CUDA; claiming production parity from remesh alone (QEM/unwrap still pending).

## Risks

- **Perf/memory at resolution 1024 narrow band on CPU.** Mitigation: measure early (RMS-08); Metal-ize the UDF/DC hot loop only if needed, keeping a CPU fallback. (Precedent: the bake already runs ~1M `closest_point` queries within the 433s heavy suite.)
- **Crossing-flag corner convention + neighbor-offset matching (DC correctness).** Mitigation: cross-check `simple_dual_contour.cu` against `flexi_dual_grid.cpp`; unit-test on analytic shapes.
- **eps shell ≈ 1-voxel inflation can thin or bridge fine features (violin strings/bow).** Mitigation: RMS-04 fidelity + the violin fixture; band/resolution are tunable parameters.
- **Hoisting `TriangleBvh` could alter bake behavior.** Mitigation: RMS-02 regression.

## Assumptions

- `/tmp/CuMesh` and `vendors/TRELLIS.2/o-voxel` remain available locally as behavior references.
- The verified rebaseline's diagnostics (`mesh_metrics`, `topology_blocker_map`) are the measurement basis for RMS-03 / 05.
- Resolution 1024 (decode resolution) is the production target; lower preview resolutions are acceptable interim test sizes.
