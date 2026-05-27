# Pixal3D GPU-Safe Projection Spec

## Bounded Goal

Fix the real-weight Pixal3D projection failure caused by `mx.linalg.inv` on the
GPU by replacing the camera-transform inverse with GPU-supported rigid
transform math.

## Source Evidence

The real Pixal3D smoke run with local Pixal3D, DINOv3, NAF, and MoGe assets
reached projection conditioning and failed with:

```text
ValueError: [linalg::inv] This op is not yet supported on the GPU.
```

The failure occurs in `project_pixal3d_points_to_image` when inverting the
front-view camera transform.

## Requirements

| ID | Requirement |
| --- | --- |
| PXGPU-01 | Remove the runtime dependency on `mx.linalg.inv` from Pixal3D projection. |
| PXGPU-02 | Preserve upstream-equivalent projection results for rigid camera transforms. |
| PXGPU-03 | Add regression coverage for the rigid transform inverse/projection path. |
| PXGPU-04 | Rerun the real Pixal3D smoke and record the next proven stage or blocker. |

## Constraints

- Runtime code remains Torch-free and CUDA-free.
- Keep the fix inside Pixal3D projection math unless evidence shows a wider
  shared primitive issue.
- Do not change public package versioning in this cycle.

## Acceptance Criteria

- `rg -n "mx\.linalg\.inv|linalg\.inv" src/mlx_spatial/pixal3d_projection.py`
  finds no unsupported inverse in the Pixal3D projection path.
- Targeted projection tests pass.
- Focused Pixal3D tests pass.
- The real downloaded-weight smoke no longer fails with the GPU
  `mx.linalg.inv` error.
