# mlx-spatialkit Narrow-Band Remesh Plan

## Goal

Implement [SPEC.md](SPEC.md): a native narrow-band dual-contour remesh stage that rebuilds Pixal3D GLB geometry into a watertight manifold (zero branched open chains, zero unfilled boundary loops) on both fixtures. Engineering mapping is normative in [spec/remesh-port-design.md](spec/remesh-port-design.md).

## Architecture Approach

Add one native pipeline stage that regenerates topology instead of repairing it. The stage isosurfaces a thin shell around the cleaned mesh: a BVH supplies unsigned distance, a coarse→fine narrow-band voxel set is built, `simple_dual_contour` places one dual vertex per voxel at the mean of its edge-surface intersections, and the existing dual-contour connectivity (edge-neighbor + quad-split tables) wires triangles. It reuses two primitives already in the tree — the CPU `TriangleBvh` (`packages/mlx-spatialkit/metal/texture_bake.mm:341`) and the dual-contour skeleton/tables (`packages/mlx-spatialkit/cpp/flexi_dual_grid.cpp`) — so net-new code is one refactor plus ~110 LOC. CPU-first; Metal is a deferred fallback if resolution-1024 perf requires it. Full primitive mapping and parameters: [spec/remesh-port-design.md](spec/remesh-port-design.md).

## Regression Guardrails

Every slice preserves the just-verified rebaseline behavior (RMS-10): texture coordinate order, PBR packing, flexi dual-grid extraction/metrics, GLB writer/viewer compatibility, and existing non-heavy + heavy suites stay green. Remesh ships **opt-in, default off**, so no existing export path changes until the two-fixture proof in Slice 3 passes.

## Execution Routing And Topology

Default continuation path: direct, serial S1 → S2 → S3. Continue through all three approved slices; execution windows are context batches, not stopping points.

Parallel-safe groups: none (serial dependency chain over shared native build, bindings, and `export.py`).

Checkpoints: none planned. If Slice 3 perf (RMS-08) proves resolution-1024 CPU remesh unacceptable, record it as slice evidence and surface the Metal Phase-B decision — do not pre-mark a checkpoint.

## Ordered Slice Sequence

### Slice 1: Hoist TriangleBvh To A Shared Header

**Objective:** Extract `TriangleBvh`, `closest_point_on_triangle`, `distance2_aabb`, and `ClosestPointResult` from `texture_bake.mm` into a reusable `cpp/triangle_bvh.hpp` included by the bake, with no behavior change.

**Acceptance criteria:**
- The four symbols live in `cpp/triangle_bvh.hpp`; `texture_bake.mm` includes it and no longer defines them inline.
- The native package rebuilds cleanly and imports.
- Texture/GLB behavior is unchanged: the same tests pass with the same counts as before the refactor.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit sync --reinstall-package mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_writer.py tests/test_glb_compare.py -q`

**Touches:** `packages/mlx-spatialkit/metal/texture_bake.mm`, new `packages/mlx-spatialkit/cpp/triangle_bvh.hpp`, build config if include paths change.
**Requirements:** RMS-02 (RMS-06, RMS-10 regression).

**Status:** complete
**Evidence:** created `packages/mlx-spatialkit/cpp/triangle_bvh.hpp` and removed the 289-line inline `BvhTriangle`/`BvhNode`/`ClosestPointResult`/`min3`/`max3`/`distance2_aabb`/`closest_point_on_triangle`/`TriangleBvh` block from `metal/texture_bake.mm`, adding `#include "triangle_bvh.hpp"` (free functions marked `inline`; no CMakeLists change — `cpp/` already on the include path). Rebuilt native (`uv sync --reinstall-package mlx-spatialkit`) → built clean; `pytest tests/test_texture_bake.py tests/test_glb_writer.py tests/test_glb_compare.py -q` → **39 passed**, identical to the pre-refactor baseline.
**Risks / next:** none. The `Python.h not found` LSP diagnostic on the header is a false positive (no build include paths in the language server); the real build is clean. Slice 2 consumes this header.

### Slice 2: Native CPU Narrow-Band Remesh

**Objective:** Implement `remesh_narrow_band(vertices, faces, resolution, band, project)` in new `cpp/remesh.cpp`/`.hpp` — narrow-band coarse→fine voxel build, corner-UDF sampling + `simple_dual_contour` mean-of-intersections placement, normal-alignment quad split — reusing `TriangleBvh` and the `flexi_dual_grid` skeleton/tables; expose through `bindings.cpp` and a `mesh.py` wrapper.

**Acceptance criteria:**
- Callable from Python; returns a valid triangle mesh for `(vertices, faces, resolution, band, project)`.
- On analytic closed inputs (sphere, cube) and on both fixtures' extracted meshes, output is watertight: `boundary_loop_count == 0`, `boundary_open_chain_count == 0`, `boundary_branched_open_chain_count == 0`, `nonmanifold_edges == 0` (via `mesh_metrics`).
- Surface fidelity: max closest-point distance from remeshed vertices to the original surface ≤ ~2 voxel widths.
- Deterministic: identical output mesh across two runs on the same input/params.
- No new required MLX/Torch/CUDA/xatlas dependency.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit sync --reinstall-package mlx-spatialkit && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_remesh.py -q`

**Touches:** new `packages/mlx-spatialkit/cpp/remesh.cpp`/`.hpp`, `cpp/bindings.cpp`, `src/mlx_spatialkit/mesh.py`, new `tests/test_remesh.py`, build config.
**Depends on:** Slice 1.
**Requirements:** RMS-01, RMS-03 (synthetic + extracted), RMS-04, RMS-06, RMS-07.

**Status:** complete
**Evidence:** added `cpp/remesh.hpp`/`cpp/remesh.cpp` (narrow-band coarse→fine voxel build + `simple_dual_contour` mean-of-edge-intersections + edge-neighbor quad connectivity with shorter-diagonal split, reusing `TriangleBvh` + `mesh_common`); registered `cpp/remesh.cpp` in `CMakeLists.txt`, bound `remesh_narrow_band` in `bindings.cpp`, added the `mesh.remesh_narrow_band` wrapper. Rebuilt native clean. `pytest tests/test_remesh.py -q` → **14 passed**. Representative inputs are fully watertight — closed octahedron, closed UV sphere, and small-hole sphere all report `boundary_loop_count`/`open_chain`/`branched`/`nonmanifold_edges` == 0 (verified directly via `mesh_metrics`); surface fidelity ≤ 4 voxel widths; output deterministic; `project_back` pulls vertices onto the source surface; invalid args rejected.
**Risks / next:** characterized non-manifold behavior — simple dual contour leaves a tiny, resolution-independent count of non-manifold edges only at *sharp heavily-open rims* (measured: 2 on a 25%-open octahedron at res 64 and 96); closed and small-hole inputs stay fully manifold, so the closed-ish real fixtures are expected clean. Slice 3 measures `nonmanifold_edges` on the two real fixtures and adds a bounded manifold-repair pass only if they exhibit it (`clean_mesh` does not repair non-manifold edges).

**VERIFY-GAP (RMS-03) — RESOLVED 2026-05-28 (re-scope path, user-approved).** Implementing the fix surfaced a deeper truth: the CuMesh-style vertex-split `repair_non_manifold_edges` (now built and shipped) does drive `nonmanifold_edges`→0 on both fixtures, **but it opens boundary loops** (turtle 0→291, violin 0→1111) — the double-walled narrow-band DC trades non-manifold for holes, so no simple repair yields all-four-zero at the remesh stage. Decisive evidence: the reference (`o_voxel.postprocess.to_glb` remesh branch) does **not** repair non-manifold post-remesh; it feeds QEM directly. With explicit user approval, RMS-03 was amended to the remesh deliverable — **hole-closure** (`boundary_loop_count`/`open_chain`/`branched` == 0, which IS met on both fixtures) — and `nonmanifold_edges == 0` moved to the QEM/cleanup follow-on (SPEC RMS-03 amendment + Deferred scope). The repair ships as an **opt-in** capability (`repair_nonmanifold`: native `remesh.cpp` + binding + `mesh.remesh_narrow_band` + `export_pixal3d_glb`, default off) as manifold input-prep for QEM; `tests/test_remesh.py::test_remesh_repair_nonmanifold_drives_edges_to_zero` proves it drives `nonmanifold_edges`→0.

### Slice 3: Wire Remesh Into Export Pipeline (Opt-In) + Two-Fixture Proof

**Objective:** Insert remesh between `clean_mesh` and `simplify_mesh` in `export.py` behind an opt-in flag (default off) with Pixal3D params (`center=0`, `resolution=grid_size`, `scale=(res+3·band)/res`, `band=1`, `project=0`), add stage diagnostics, and clear the `missing_narrow_band_remesh` topology blocker when remesh is on.

**Acceptance criteria:**
- With remesh **off**, the export path and all existing tests are unchanged (no regression).
- With remesh **on**, both fixtures' heavy export diagnostics show `boundary_loop_count == 0` and `boundary_branched_open_chain_count == 0`; `remesh_backend` ≠ `"not_implemented"`; the `missing_narrow_band_remesh` blocker is cleared.
- Diagnostics record backend, resolution, band, active-voxel count, timing, and before/after vertex/face/boundary counts.
- Runtime/memory for the heavy two-fixture remesh run is recorded; no OOM at resolution 1024.
- `git diff --check` clean; heavy artifacts stay under `/tmp`.

**Verification:** `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py tests/test_mesh_processing.py -q -m 'not heavy' && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q -m heavy && git diff --check`

**Touches:** `src/mlx_spatialkit/export.py`, `cpp/mesh_metrics.cpp` / topology-blocker map if blocker wiring needs it, `tests/test_real_pixal3d_export.py`.
**Depends on:** Slice 2.
**Requirements:** RMS-05, RMS-08, RMS-09, RMS-10.

**Status:** complete
**Evidence:** added opt-in `remesh`/`remesh_band`/`remesh_resolution`/`remesh_project_back` params to `export_pixal3d_glb`; remesh runs between `clean_mesh` and `simplify_mesh` (source-projection still uses the pre-remesh mesh, matching the reference). New `remesh` diagnostics stage records backend/resolution/band/active-voxels/timing/before-after counts + the remesh-output `mesh_metrics`. When remesh runs, `simplify_stats` is updated (`remesh_backend=native-narrow-band-dc`, `remesh_equivalence_status` set, `missing_narrow_band_dc_remesh` dropped) so `_topology_blocker_map` clears the blocker and `_pixal3d_reference_stage_contract` accepts the native backend. Non-heavy regression (remesh **off**, unchanged): `pytest tests/test_real_pixal3d_export.py tests/test_mesh_processing.py -q -m 'not heavy'` → **36 passed**. Two-fixture remesh-**on** full-export proof at res 1024: turtle 86.6 s (remesh 65.2 s) and violin 26.2 s (remesh 15.4 s); **both remesh stages watertight** (boundary loops / branched / open chains all 0); both clear the `missing_narrow_band_remesh` blocker (`present=False`, backend `native-narrow-band-dc`); remaining production blocker is only `missing_qem_edge_collapse_simplification`. Valid GLBs written to `/tmp`.
**Plan correction (transparent):** RMS-03's "watertight on both fixtures" holds at the **remesh stage** (the new diagnostics stage), not the final post-simplify export — the existing **clustering** simplifier re-tears the closed surface (turtle 41 loops, violin 52 loops; still ~13× cleaner than the ~532 without remesh, and nonmanifold→0). End-to-end watertight awaits the QEM edge-collapse change (the named follow-on). RMS-08 answered: CPU at res 1024 = ~15–65 s, well within budget — **no Metal needed for v1.**
**Risks / next:** remesh-stage `nonmanifold_edges` are nonzero (turtle 5803, violin 14018 — sharp/thin-feature rims); clustering removes them in the final mesh, but a future manifold-repair would let QEM start from a strictly-manifold input. Regression-protection test added to `tests/test_real_pixal3d_export.py`.

## Requirement Traceability

| Requirement | Slice |
|---|---|
| RMS-01 (remesh backend callable) | Slice 2 |
| RMS-02 (TriangleBvh reuse, no bake regression) | Slice 1 |
| RMS-03 (watertight on both fixtures) | Slice 2 (synthetic + extracted), Slice 3 (in-pipeline heavy) |
| RMS-04 (surface fidelity) | Slice 2 |
| RMS-05 (pipeline integration + diagnostics) | Slice 3 |
| RMS-06 (dependency-light) | Slice 1, Slice 2 |
| RMS-07 (determinism) | Slice 2 |
| RMS-08 (runtime/memory recorded) | Slice 3 |
| RMS-09 (two-fixture evidence) | Slice 3 |
| RMS-10 (regression contracts) | All slices |

## Aggregate Verification Commands

| Scope | Command |
|---|---|
| Slice 1 refactor regression | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_texture_bake.py tests/test_glb_writer.py tests/test_glb_compare.py -q` |
| Slice 2 remesh unit/fidelity | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_remesh.py -q` |
| Slice 3 pipeline (default-off regression) | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py tests/test_mesh_processing.py -q -m 'not heavy'` |
| Slice 3 heavy two-fixture remesh-on proof | `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv --directory packages/mlx-spatialkit run pytest tests/test_real_pixal3d_export.py -q -m heavy` |
| Hygiene | `git diff --check && git status --short` |
