# Pixal3D Export Reference Port Design

## Architecture Decision

Keep `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` as the Python orchestration boundary, but move the quality-critical Pixal3D export stages behind native C++/Objective-C++/Metal APIs that match the Pixal3D `o_voxel.postprocess.to_glb` stage contract.

Python remains responsible for:

- loading decoded NPZ files
- resolving presets and output paths
- calling native APIs
- writing diagnostics and GLB artifacts
- releasing stage-local arrays

Native code owns:

- mesh cleanup and hole fill
- remesh/simplification
- xatlas-behavior-compatible native unwrap
- UV raster/interpolate
- original-mesh BVH projection
- sparse PBR voxel sampling
- texture seam fill/inpaint-equivalent behavior

## Reference Stage Map

| Reference stage | Source | spatialkit owner |
|---|---|---|
| Decoded NPZ validation | `packages/mlx-spatialkit/cpp/pixal3d_contracts.cpp` | keep, extend diagnostics only |
| FlexiDualGrid extraction | `vendors/TRELLIS.2/o-voxel/src/convert` | `cpp/flexi_dual_grid.cpp` |
| Initial hole fill | `/tmp/CuMesh/src/clean_up.cu:450` | `cpp/mesh_cleanup.cpp` / `cpp/simplify.cpp` |
| Source BVH | `/tmp/CuMesh/cumesh/bvh.py:54` | new native BVH module |
| Narrow-band DC remesh | `/tmp/CuMesh/cumesh/remeshing.py:24` | new native remesh module |
| QEM simplify/repair | `/tmp/CuMesh/src/simplify.cu:531` | replace/extend `cpp/simplify.cpp` |
| xatlas unwrap | `/tmp/CuMesh/cumesh/cumesh.py:408` | native unwrap/packing with xatlas/CuMesh behavioral metrics; no mandatory xatlas dependency |
| UV raster/interpolate | `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:229` | `metal/texture_bake.mm` + kernels |
| Original-mesh projection | `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:252` | native BVH projection before sampling |
| Trilinear voxel sampling | `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:258` | Metal sparse-grid sampler |
| Texture inpaint/fill | `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:287` | native seam fill with truthful sample/fill stats |
| GLB write | existing spatialkit GLB writer | keep, update metadata only |

## Heuristic Policy

Recent local fixes such as projected ear clipping, alternate triangulation, branch-cycle fill, native chart packing, nearest-voxel fallback, and broad texture dilation are not accepted as the Pixal3D quality path by default. Each must become one of:

- reference-matched behavior
- explicitly experimental/non-equivalent behavior
- removed or disabled from the reference preset

The quality preset should pass because reference-critical stages are implemented, not because diagnostics thresholds were relaxed.

## Dependency Policy

- `xatlas` is a behavior reference for chart/packing metrics in this cycle, not a required external or vendored dependency.
- Adding a required or vendored `xatlas` implementation needs a new explicit decision.
- CuMesh and o-voxel MIT code may be used as implementation reference or copied with notices when appropriate.
- `nvdiffrast` is behavior reference only; do not copy or line-port its CUDA implementation.

## Verification Shape

Unit tests prove individual native contracts. Heavy tests prove the decoded Pixal3D fixture under `/tmp`. Visual acceptance uses generated GLB diagnostics, embedded texture inspection, existing reference GLB comparison, and manual Preview/browser inspection when appropriate.
