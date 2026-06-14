# Remesh Port Design — `remesh_narrow_band_dc` → mlx-spatialkit (native, no CUDA/Torch/MLX)

Normative engineering reference for the narrow-band remesh change. Source-grounded; every claim carries a `file:line` anchor.

## Verdict: LOW risk. ~80% is reuse of primitives the package already ships.

The one primitive expected to be the hard new build — point→triangle-mesh unsigned distance + closest point + barycentric (the `cuBVH` role) — **already exists** as a complete CPU `TriangleBvh` in `texture_bake.mm`. The dual-contour connectivity (edge-neighbor tables, quad→triangle split, coord→index hashmap) **already exists** in `flexi_dual_grid.cpp`. Net new code is **one refactor + ~3 small functions (~110 LOC)**. Feasibility is not in question; the only real risk is *perf at resolution 1024*, which is measurable and Metal-able.

## 1. What remesh actually is (6 steps)

`remesh_narrow_band_dc` (`/tmp/CuMesh/cumesh/remeshing.py:24`) rebuilds topology from scratch by isosurfacing a thin shell around the input surface:

1. Build a BVH over (vertices, faces) for unsigned-distance queries.
2. **Narrow band:** coarse→fine voxel subdivision, keep voxels where `|UDF − eps| < 0.87·cell` (`eps = band·scale/res`). → active voxel coords.
3. Dedup active-voxel corners → grid vertices; sample `UDF − eps` at each (signed: <0 inside shell).
4. **`simple_dual_contour`:** per voxel, one dual vertex = **mean of its 12 edges' surface intersections**; record 3 axis sign-change flags. (`/tmp/CuMesh/src/remesh/simple_dual_contour.cu:145`)
5. **Connectivity:** each crossed edge → quad of its 4 neighbor voxels' dual vertices → split to 2 triangles by best normal-alignment.
6. (Optional) project vertices back to the original surface. **Pixal3D uses `remesh_project=0` → skipped.**

Key realization: **step 4+5 = `flexi_dual_grid.cpp`, but the dual vertex and crossing flags come from a sampled UDF instead of the decoded learned fields.** Same skeleton, different source of the scalar.

## 2. Primitive-by-primitive mapping (kernel-level data flow)

| Reference step | Reference source | mlx-spatialkit status | Action |
|---|---|---|---|
| BVH build | `cuBVH(v,f)` (`bvh.py:11`) | ✅ **EXISTS** `TriangleBvh` `texture_bake.mm:341` | Hoist to shared header |
| `unsigned_distance(pts)` | `bvh.py:54` | ✅ **EXISTS** `bvh.closest_point(p).distance2` → `sqrt` (`texture_bake.mm:370`, `closest_point_on_triangle` :235) | Reuse |
| closest-point + barycentric `uvw` (project_back) | `bvh.py:70` | ✅ **EXISTS** `ClosestPointResult.barycentric` (`texture_bake.mm:89`) | Not needed (project=0); available if ever |
| narrow-band coarse→fine subdivision | `remeshing.py:104-141` (Torch) | 🟡 **NEW** ~40 LOC | Plain nested loop driving BVH UDF; keep / subdivide-×8 |
| `hashmap_insert/lookup_3d` | `src/hash/hash.cu` (GPU) | ✅ **EQUIVALENT** `std::unordered_map<Coord3,…>` (`flexi_dual_grid.cpp:135`) | Reuse pattern |
| `get_sparse_voxel_grid_active_vertices` (corner dedup) | `src/remesh/svox2vert.cu` | 🟡 **NEW** ~15 LOC | Insert 8 corners/voxel into a map |
| `simple_dual_contour` (mean-of-intersections + flags) | `src/remesh/simple_dual_contour.cu` | 🟡 **NEW** ~50 LOC | Confirmed mean (not QEF) — trivial |
| quad gen from edge-neighbors + validity | `remeshing.py:191-210` | ✅ **EXISTS** `flexi_dual_grid.cpp:172-199` w/ identical `kEdgeNeighborVoxelOffset` | Reuse |
| quad→triangle split | `remeshing.py:214-233` | ✅ **EXISTS** `flexi_dual_grid.cpp:218-227` w/ identical `kQuadSplit1/2` | Swap criterion |
| split criterion | normal-alignment | 🟡 **TWEAK** ~20 LOC | flexi uses learned `split_weight`; remesh uses max normal dot |
| project_back | `remeshing.py:238-250` | ⚪ **SKIP** (`remesh_project=0`) | — |

## 3. The two decisive reuses

**`TriangleBvh` = the entire `cuBVH` role, already here:**
- `texture_bake.mm:341` `class TriangleBvh` — built from `mesh_common::MeshData`; `closest_point()` at :370 with AABB pruning (`distance2_aabb` :215) and ordered child traversal; `closest_point_on_triangle` :235 returns exact point + barycentric (`ClosestPointResult` :89).
- Already exercised on the source mesh at `texture_bake.mm:1074-1088` (texel→original-mesh projection — the reference's `postprocess.py:254` BVH step). The `"knn"` mode (`sparse_knn_sample_attributes` :901) is only a miss *fallback*, not the primary path. An exact mesh-distance structure is proven in production here.

**`flexi_dual_grid.cpp` = the dual-contour connectivity, already here:**
- `kEdgeNeighborVoxelOffset` (`:104`) is byte-identical to `remesh_narrow_band_dc.edge_neighbor_voxel_offset` (`remeshing.py:72`).
- `kQuadSplit1/kQuadSplit2` (`:109`) identical to the reference quad-split tables (`remeshing.py:80`).
- The voxel loop + 4-neighbor gather + validity + quad→tri (`:172-227`) is line-for-line the reference's step-5 connectivity.

## 4. New-code inventory (precise)

- **R1 (refactor, no behavior change):** hoist `TriangleBvh` + `closest_point_on_triangle` + `distance2_aabb` + `ClosestPointResult` from `texture_bake.mm` into `cpp/triangle_bvh.hpp`. De-dups, makes it linkable from remesh. *Lowest-risk first slice; pure move.*
- **N1 — narrow band (~40 LOC):** coarse→fine voxel set via `bvh` UDF; output active `Coord3` list. Reuses `Coord3`/`Coord3Hash` from `flexi_dual_grid.cpp:24`.
- **N2 — corner UDF + `simple_dual_contour` (~50 LOC):** dedup corners → sample `sqrt(distance2) − eps` → per-voxel mean-of-intersections + 3 flags. Formula confirmed `simple_dual_contour.cu:62-68,146-149`.
- **N3 — connectivity + normal-split (~20 LOC):** copy `flexi_dual_grid.cpp:172-227`; replace the `split_weight` test with the two-triangulation normal-dot test (`remeshing.py:220-233`).
- **Glue:** `bindings.cpp` entry `remesh_narrow_band(vertices, faces, resolution, band, project)`; `mesh.py` wrapper; insert in `export.py` between `clean_mesh` and `simplify_mesh`.

## 5. Exact parameters for Pixal3D

From `vendors/Pixal3D/inference.py:268` / `app.py:522` (`remesh=True, remesh_band=1, remesh_project=0`) and `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:166-180`, in the native [-0.5, 0.5]³ frame that `flexi_dual_grid` already uses:

```
center     = 0                         (aabb.mean; extraction is symmetric)
resolution = grid_size  (=1024)        (decode resolution)
scale      = (res + 3·band)/res · 1.0  (≈1.003, the padded extent)
band       = 1
eps        = band·scale/res  (≈1 voxel) # isosurface = UDF − eps  → ~1-voxel shell
project    = 0                          # geometry stays on shell; color sampled from
                                        #   original via the bake's existing BVH projection
```

## 6. Pipeline integration

Reference order is `fill_holes → [remesh] → simplify`. Native mapping in `export.py`:

```
extract_flexi_dual_grid → clean_mesh → [REMESH (new, opt-in)] → simplify_mesh → uv → bake → glb
```

After remesh the mesh is a clean watertight manifold — which is *also the precondition that makes QEM edge-collapse safe* (the follow-on change). Downstream UV/bake interfaces are unchanged.

## 7. Build path & perf

- **Phase A — CPU reference port (ship first).** Reuse `TriangleBvh` + flexi skeleton; no MLX/Torch/CUDA, honoring the dependency-light constraint. **Perf precedent:** the heavy bake already runs ~1M `closest_point` queries and the full heavy suite finishes in ~433s. A 1024 narrow band is the same order of magnitude of queries, so CPU is plausible — *but must be measured (RMS-08)*.
- **Phase B — Metal (only if Phase A is too slow).** The UDF-over-voxels and dual-contour passes are embarrassingly parallel; mirror the existing `metal/texture_bake.mm` + `metal/kernels/texture_bake.metal` pattern. Optional, with CPU fallback.

## 8. Correctness validation (without a CUDA box)

- **Watertightness** — the goal: reuse `mesh_metrics` → assert `boundary_loop_count == 0` and `boundary_branched_open_chain_count == 0` post-remesh on **both** fixtures. Direct refutation of today's 331/147 branched chains.
- **Surface fidelity** — Hausdorff/chamfer of remeshed vs original surface bounded by ~1 voxel (the eps shell). Sample via the same `TriangleBvh`.
- **Parity (if a CUDA box is reachable offline)** — diff against a real `cumesh.remeshing` output on one fixture; otherwise the watertight + fidelity gates stand alone.

## 9. Risks / verify during impl (not before)

1. **Active-voxel count at res 1024** — memory/time unknown; *measure in Phase A* before deciding on Metal. The only real risk.
2. **`intersected`-flag corner convention** — set on the `(u==1,v==1)` edge (`simple_dual_contour.cu:71-81`); must line up with `kEdgeNeighborVoxelOffset`'s 4-voxel gather and winding (`intersected_dir` flips the split, `remeshing.py:215-233`). Fully specified in those two files; needs care.
3. **eps shell ≈ 1-voxel inflation** — at 1024 this is ≈0.001 in normalized space and color is resampled from the original via the bake BVH, so visually negligible; confirm on the violin (thin structures).
4. **`closest_point_on_triangle` extraction** — confirm it doesn't depend on `texture_bake.mm`-local state when hoisted (it shouldn't; pure geometry).

## 10. Proposed slices

- **Slice 1** — Hoist `TriangleBvh` → `cpp/triangle_bvh.hpp` (no behavior change) + regression test that the bake path is unchanged. (RMS-02)
- **Slice 2** — CPU `remesh_narrow_band` (N1+N2+N3) + binding + wrapper; gate: watertight + fidelity on analytic shapes and both fixtures. (RMS-01, RMS-03, RMS-04, RMS-07)
- **Slice 3** — Wire remesh into `export.py` (opt-in flag, default off); diagnostics + `missing_narrow_band_remesh` blocker clears; prove branched-chains/boundary-loops → 0 in two-fixture diagnostics; record runtime/memory. (RMS-05, RMS-08, RMS-09, RMS-10)
- *(Follow-on, separate changes: QEM edge-collapse on the now-clean manifold; then xatlas-equivalent unwrap.)*
