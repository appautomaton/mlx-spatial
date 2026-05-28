# Roadmap

## Phase 1: Pixal3D GLB Quality Reference Port

- status: done
- change: `2026-05-27-mlx-spatialkit-pixal3d-export-reference-port`
- objective: Make `mlx-spatialkit` generate high-quality Pixal3D GLB artifacts by following the Pixal3D/o-voxel/CuMesh export contract and xatlas behavior metrics without adding an unapproved xatlas dependency.
- why now: Recent native GLBs became structurally better but showed color/granularity smear around repaired areas; the next work needs a source-grounded export reference port before more implementation changes.
- likely outputs: One source-grounded plan, C++/Metal mesh/UV/bake changes, focused unit tests, real Pixal3D heavy fixture verification under `/tmp`, visual comparison artifacts, and docs aligned to actual readiness.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-pixal3d-export-reference-port/SPEC.md` | user-stated | `vendors/Pixal3D/inference.py:263` | `vendors/TRELLIS.2/o-voxel/o_voxel/postprocess.py:14` | `/tmp/CuMesh/src/clean_up.cu:450`
- exit signal: The real decoded Pixal3D fixture writes inspectable GLBs under `/tmp`, tests and diagnostics prove the implemented reference-critical stages and explicit blockers, remaining parity gaps are explicit, and no generated heavy artifacts pollute the repo.

## Deferred or Not Now

- Release, tag, publish, or push work is not part of this roadmap cycle.
- Copying or line-porting `nvdiffrast` CUDA code is out of scope; it is behavior reference only.
- Broad model-inference changes outside decoded Pixal3D NPZ to GLB export are out of scope.
