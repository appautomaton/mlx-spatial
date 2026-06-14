# Pixal3D DINOv3 Conditioning Spec

## Bounded Goal

Wire Pixal3D inference to the existing MLX DINOv3 hidden-state forward path so `mlx-spatial` can build sparse-stage Pixal3D projection conditioning directly from an input image when local DINOv3 assets are available.

## Broader Intent

The previous Pixal3D change established assets, CLI, projection math, and structured blockers, but runtime image conditioning still required caller-supplied `projection_hidden_states`. This change removes that artificial manual boundary for the sparse stage while preserving honest blockers for later Pixal3D flow/decoder/export work.

## Selected Lenses

- **engineering:** Reuses the existing TRELLIS.2 DINOv3 MLX implementation rather than forking a second image encoder.
- **runtime:** Keeps DINOv3 loading explicit, local, and memory-conscious; no Torch, Transformers, vendor, or CUDA runtime imports.

## Required Outcome

1. `Pixal3DInferencePipeline.generate(...)` accepts a local DINOv3 root and, when `projection_hidden_states` is not supplied, runs image preprocessing plus MLX DINOv3 hidden-state extraction.
2. The Pixal3D CLI and `scripts/pixal3d/generate.py` expose `--dino-root` with the existing local convention `weights/dinov3-vitl16-pretrain-lvd1689m`.
3. Missing or invalid DINOv3 assets return an actionable `image-conditioning` blocker that names the DINOv3 root and download path, without opening or executing large model tensors unnecessarily.
4. With fake or valid DINOv3 assets, the runtime reaches the existing sparse projection artifact boundary from an image without caller-supplied hidden states.
5. Docs and trace metadata describe the DINOv3 root, image-conditioning status, and remaining blocker accurately.

## Constraints

- Base runtime must remain free of Torch, TorchVision, Transformers, CUDA-only packages, `vendors/Pixal3D`, and vendor imports.
- Do not implement Pixal3D sparse FlowEuler sampling, NAF high-resolution features, decoders, MoGe auto-camera, or GLB export in this cycle.
- Existing TRELLIS.2 DINOv3 behavior and tests must remain intact.
- Avoid eager Pixal3D checkpoint loads; this cycle should only add DINOv3 image-conditioning execution and sparse projection artifact production.

## Risks

- Real DINOv3 ViT-L/16 execution can be memory-heavy. The runtime must use the existing staged DINOv3 helper and record memory snapshots rather than introducing an unbounded custom path.
- Pixal3D uses DINOv3 hidden states for multiple resolutions. This cycle is limited to the sparse-stage 512px path because that is the current implemented projection/export boundary.
- Tests should use fake DINOv3 roots to prove wiring without requiring multi-GB assets.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| PXDINO-01 | Runtime exposes explicit DINOv3 root configuration | `Pixal3DInferencePipeline.generate(..., dino_root=...)`, `mlx-spatial-pixal3d generate --dino-root`, and `scripts/pixal3d/generate.py --dino-root` exist and help output is clear. |
| PXDINO-02 | Missing DINOv3 assets block precisely | Pixal3D manual-FOV generation with no local DINO root returns an `image-conditioning` blocker with root/download metadata instead of the old "not wired" blocker. |
| PXDINO-03 | Image-to-projection path works with fake DINOv3 assets | A test image plus fake DINOv3 root reaches `projection-conditioning:ss`, writes `sparse_projection.npz`, and then blocks at `sparse-structure-flow`. |
| PXDINO-04 | Runtime dependency boundary remains clean | AST forbidden import scan over `src/mlx_spatial` passes. |
| PXDINO-05 | Pixal3D and shared DINOv3 regressions pass | Pixal3D targeted tests and TRELLIS.2 DINOv3/forward tests pass. |

## Scope Coverage Decisions

- **Included:** Pixal3D sparse-stage DINOv3 image hidden-state extraction, CLI/script parameter plumbing, trace/docs updates, fake-weight tests, and dependency hygiene.
- **Deferred:** shape/texture DINOv3+NAF conditioning, Pixal3D sparse sampling, sparse decoder handoff, SLat cascade execution, MoGe auto-camera, and GLB/PBR export.
- **Anti-goals:** Treating this as full Pixal3D GLB support, loading PyTorch/Transformers at runtime, or adding new bundled weights.
