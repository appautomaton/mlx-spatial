---
active_change: sam3d-mlx-mesh-decoder-algorithm-parity
stage: complete
---

# Status

## Current Change

- active change: `sam3d-mlx-mesh-decoder-algorithm-parity`
- current stage: `complete`

## What Is True Now

- Slices 1-7 are complete.
- Live SAM3D human-object reconstruction wrote `outputs/sam3d/human-object/gaussians.ply` and `outputs/sam3d/human-object/mesh.glb` with trace reaching `glb-export` and no blocker.
- Blender headless import, focused SAM3D tests, full pytest, and diff check passed.

## Next Step

Prepare commit cleanup/review for the completed SAM3D mesh decoder algorithm parity change.

## Open Risks

- The basic GLB is geometry plus vertex color only; official-quality layout optimization and textured/material postprocess remain future work.
- Large mesh extraction requires the `large` memory profile on this 128 GB machine.
