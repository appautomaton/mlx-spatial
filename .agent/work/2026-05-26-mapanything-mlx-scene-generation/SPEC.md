# MapAnything MLX Scene Generation Spec

## Bounded Goal

Implement MLX-native MapAnything image-only scene generation in `mlx-spatial`, using the local `facebook/map-anything` best-performance weights and the two official Desk images to produce inspectable scene predictions without Torch, UniCeption, OpenCV, or vendored runtime imports.

## Broader Intent

The user goal is not an encoder-prefix smoke path. The intended outcome is a real MapAnything inference path in `mlx-spatial`: load two images for one scene, run the model core in MLX, and emit scene-level outputs close enough to the vendored PyTorch pipeline to support parity checks and later product integration.

## Work Scale and Shape

- Scale: capability
- Shape: parity-driven feature implementation with runtime and packaging constraints

## Selected Lenses

- **product:** Adds a scene-level MapAnything reconstruction path that Apple Silicon users can run from local assets.
- **engineering:** Ports the inference-time model core and postprocessing in verifiable slices while keeping Torch and vendor code out of runtime.
- **runtime:** Keeps execution memory-bounded on AppleGPU/MLX and avoids accidental CUDA/Torch assumptions.

## Target User or Stakeholder

Developers using `mlx-spatial` on Apple Silicon who want local MapAnything scene generation from multiple images, with dev-time parity evidence against the official implementation.

## Reference Sources

- Vendored reference implementation: `vendors/map-anything`
- Existing MLX foundation change: `.agent/work/2026-05-26-mapanything-mlx-parity-backbone`
- Local model assets: `weights/map-anything/{config.json,model.safetensors,README.md}`
- Official Desk image input: `inputs/map-anything/desk`
- Dev-only parity environment: `uv run --group torch-ref ...`

## Required Outcome

The repo gains an MLX MapAnything scene pipeline that:

1. Uses the existing MapAnything asset/config/preprocess/prefix modules as the starting point.
2. Loads all inference-time weights needed by the image-only path: DINOv2 encoder, info-sharing transformer, dense head, pose head, scale head, and required normalization/token parameters.
3. Runs the image-only `model.infer(...)` equivalent for a folder of two Desk images without importing Torch, UniCeption, OpenCV, or `vendors/map-anything` from runtime package code.
4. Produces scene predictions with at least these fields: images, depth, confidence, mask, camera intrinsics, camera poses or extrinsics, and world points.
5. Writes an inspectable generated-scene artifact. A `.npz` prediction bundle is required; a mesh/point-cloud export such as `.glb` or `.ply` is included when it can be implemented without breaking the runtime dependency boundary.
6. Provides dev-only Torch reference capture for the same two-image scene so MLX outputs can be checked at meaningful intermediate and final boundaries.
7. Preserves packaging hygiene: local weights, inputs, vendors, and scratch reference dumps are not included in wheel or sdist artifacts.

## Constraints

- Runtime package code must not depend on or import Torch, TorchVision, UniCeption, OpenCV, CUDA-only packages, or `vendors/map-anything`.
- Torch is allowed only through the `torch-ref` group and only for dev/reference capture under `tools/` or opt-in tests.
- Local CC-BY-NC 4.0 `facebook/map-anything` weights are research/runtime assets in the checkout, not redistributable package data.
- `weights/`, `inputs/`, `vendors/`, and `/tmp` debug/reference files must not be bundled into release artifacts.
- Training, fine-tuning, DA3 implementation, Apache model variant support, Gradio UI reproduction, and non-image geometric input modes are out of scope.
- The previous encoder-prefix API/tests must keep working unless the new full pipeline intentionally extends them in a backward-compatible way.
- If `.glb` export requires a new heavy or incompatible dependency, the accepted generated artifact for this change is `.npz` plus a lightweight point-cloud export path that respects runtime constraints. This does not remove the requirement to generate actual scene predictions.

## Risks

- **Full encoder scale:** Porting all 24 DINOv2 giant blocks can exceed comfortable AppleGPU memory if tensors are retained. Mitigation: run layers sequentially, avoid unnecessary residual captures, and add smaller tests plus real-weight opt-in smoke.
- **Info-sharing transformer complexity:** MapAnything packs per-view features, per-view class/register tokens, and a global scale token through alternating attention. Mitigation: preserve token-layout parity in a dedicated slice before heads.
- **Head/postprocess drift:** Dense ray/depth, pose, scale, intrinsics, depth-z, masks, and world points are easy to make shape-correct but semantically wrong. Mitigation: capture vendored reference tensors and compare both intermediate and final fields.
- **Dependency creep:** Vendored visualization/export helpers may pull in libraries unsuitable for package runtime. Mitigation: implement a minimal local exporter or keep export to `.npz`/`.ply` when needed.
- **Memory/device instability:** Real two-image Desk inference may stress MLX Metal memory. Mitigation: keep an explicit real-weight smoke gate and document any memory profile limits in test evidence.

## Acceptance Criteria

| ID | Requirement | Check |
|---|---|---|
| MA-SCENE-01 | Scene pipeline API exists | `mlx_spatial` exposes a MapAnything scene pipeline that accepts `weights/map-anything` and an image folder, returning structured scene predictions. |
| MA-SCENE-02 | Runtime deps stay clean | Runtime dependencies and package code do not add Torch, TorchVision, UniCeption, OpenCV, CUDA-only packages, or vendor imports. |
| MA-SCENE-03 | Full DINOv2 image encoder runs in MLX | The MLX encoder loads all required encoder block weights and returns per-view feature maps/tokens used by MapAnything, with parity against dev-captured reference boundaries. |
| MA-SCENE-04 | Info-sharing transformer runs in MLX | The MLX info-sharing module matches the reference token layout, alternating attention behavior, global scale token handling, and intermediate feature outputs needed by downstream heads. |
| MA-SCENE-05 | Prediction heads run in MLX | Dense, pose, and scale heads load required checkpoint groups and produce ray/depth/pose/confidence/mask/scale tensors with checked shapes and parity gates. |
| MA-SCENE-06 | Postprocessing produces scene tensors | The inference path returns depth, intrinsics, camera poses/extrinsics, confidence, masks, images, and world points in the same semantic layout as the vendored image-only demo. |
| MA-SCENE-07 | Desk two-image scene generation works | A local smoke command/test runs `inputs/map-anything/desk` through MLX with `weights/map-anything` and writes a prediction artifact under `/tmp` or an explicit output path. |
| MA-SCENE-08 | Dev reference capture is opt-in | Torch reference generation is guarded by `MAPANYTHING_TORCH_REF=1` and uses local weights with Torch Hub disabled. |
| MA-SCENE-09 | AppleGPU memory stays bounded | Real-weight Desk smoke either completes on MLX/Metal within practical memory limits or records a concrete implementation blocker tied to the failing model boundary. |
| MA-SCENE-10 | Existing behavior and packaging remain healthy | Existing MapAnything prefix tests still pass, targeted new tests pass, `uv lock --check` passes, and package artifact checks exclude local assets/vendors. |

## Scope Coverage Decisions

- **Included:** image-only inference pipeline, local best-performance CC-BY-NC weights, two-image Desk input, full model-core inference port, scene tensor postprocessing, generated prediction artifact, dev-only PyTorch reference capture, and package/runtime hygiene.
- **Deferred:** training, DA3, Apache weights, non-image input modalities, Gradio UI, quality benchmarking across datasets, commercial/default weight policy, and full UI/CLI product polish beyond a minimal runnable surface.
- **Anti-goals:** claiming completion from prefix smoke, generating only tensor summaries, importing vendored code at runtime, bundling weights, or replacing MLX inference with a PyTorch wrapper.

## Blocking Questions or Assumptions

No blocking question remains. The operating assumption is that execution can be staged through parity boundaries, but the change is not complete until actual scene predictions are generated from the Desk pair through MLX or a concrete model-boundary blocker is proven by implementation evidence.

## Anti-Goals

- Full MapAnything training or fine-tuning.
- DA3 or external monocular-depth pipeline implementation.
- Apache model variant implementation.
- Gradio app parity.
- Runtime Torch/TorchVision/UniCeption/OpenCV/vendor-source dependency.
- Weight redistribution or package bundling of `weights/map-anything`.
- Treating encoder-prefix parity as end-to-end scene generation.
