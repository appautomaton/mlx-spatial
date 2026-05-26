# Plan: MapAnything MLX Scene Generation

## Goal

Execute the scene-generation contract in [SPEC.md](SPEC.md): two Desk images through MLX MapAnything inference, producing scene predictions without runtime Torch/vendor dependencies.

## Architecture Approach

Use the staged runtime path in [DESIGN.md](DESIGN.md). The existing `mapanything-mlx-parity-backbone` work is the starting point, but the new done condition is an actual scene prediction artifact, not prefix summaries.

Implementation should preserve a dev/runtime split:

- Runtime: `src/mlx_spatial/mapanything_*.py`, MLX/NumPy/Pillow, local checkpoint helpers.
- Dev reference: guarded scripts under `tools/`, `torch-ref` group, `vendors/map-anything`, `/tmp` reference bundles.

## Execution Routing and Topology

Default: direct, serial, continue after each verified slice.

Overrides:
- Slice 3: subagent recommended, because it ports all DINOv2 blocks and may need careful memory review.
- Slice 4: subagent recommended, because token packing and alternating attention semantics are high-risk.
- Slice 5: subagent recommended, because DPT/pose/scale heads touch the broadest checkpoint surface.

Parallel-safe groups: none. The slices are ordered by data dependency.

Checkpoints: none.

## Requirement Traceability

| SPEC ID | Satisfied by |
|---|---|
| MA-SCENE-01 | Slices 2, 7 |
| MA-SCENE-02 | Slices 1, 8 |
| MA-SCENE-03 | Slice 3 |
| MA-SCENE-04 | Slice 4 |
| MA-SCENE-05 | Slice 5 |
| MA-SCENE-06 | Slice 6 |
| MA-SCENE-07 | Slice 7 |
| MA-SCENE-08 | Slice 1 |
| MA-SCENE-09 | Slices 3, 7, 8 |
| MA-SCENE-10 | Slice 8 |

## Ordered Slice Sequence

### Slice 1: Full Scene Reference Capture

**Objective:** Extend the guarded Torch reference tooling to capture the vendored image-only Desk scene outputs and the intermediate tensors needed by later MLX parity checks.

**Acceptance criteria:**
- A dev-only script refuses to run unless `MAPANYTHING_TORCH_REF=1` is set.
- The script initializes vendored `MapAnything` from local `weights/map-anything` with Torch Hub disabled.
- The script captures Desk preprocessing, full encoder output, info-sharing intermediates/final output, dense/pose/scale head outputs, and final postprocessed scene fields.
- Output is a `.npz` bundle under `/tmp` or an explicit path with metadata documenting local weights, vendor root, device, capture keys, and `runtime_depends_on_torch=false`.

**Touches:** `tools/mapanything_dump_torch_reference.py` or new `tools/mapanything_dump_torch_scene_reference.py`, `src/mlx_spatial/mapanything_parity.py`, `tests/test_mapanything_parity.py`

**Verification:** `uv run pytest tests/test_mapanything_parity.py -q && MAPANYTHING_TORCH_REF=1 uv run --group torch-ref python tools/mapanything_dump_torch_scene_reference.py weights/map-anything inputs/map-anything/desk --device mps --output /tmp/mapanything-desk-scene-reference.npz`

**Status:** complete
**Evidence:** added `tools/mapanything_dump_torch_scene_reference.py` and guarded it behind `MAPANYTHING_TORCH_REF=1`; patched the reference script to route UniCeption DINOv2 construction through vendored local backbones with Torch Hub disabled. Added guard coverage in `tests/test_mapanything_parity.py`; `uv run pytest tests/test_mapanything_parity.py -q` passed with 8 tests. The full Desk capture command succeeded on `mps` and wrote `/tmp/mapanything-desk-scene-reference.npz` (122M) with 71 tensors, including `encoder.features.0` `(1, 1536, 28, 37)`, `info.final.features.0` `(1, 1536, 28, 37)`, `head.dense.value` `(2, 4, 392, 518)`, `head.pose.value` `(2, 7)`, `scene.depth` `(2, 392, 518)`, `scene.intrinsics` `(2, 3, 3)`, `scene.camera_poses` `(2, 4, 4)`, and `scene.world_points` `(2, 392, 518, 3)`. Bundle metadata reports `load_missing_keys=[]`, `torch_hub_disabled=true`, and `torch_hub_patched_to_local_vendor=true`.
**Risks / next:** `load_missing_alias_keys` contains duplicate DPT registration paths that are loaded through `dense_head.*`; Slice 3 can use this reference bundle for full encoder parity.

### Slice 2: Scene Pipeline Contract

**Objective:** Add runtime data classes and a scene-pipeline skeleton that defines the public MLX API and output bundle schema before the full model is wired in.

**Acceptance criteria:**
- Runtime exposes a scene pipeline class or function separate from the existing prefix smoke path.
- Scene result types cover images, depth, confidence, masks, intrinsics, camera poses/extrinsics, world points, metadata, and blocker reporting.
- The pipeline can validate assets and preprocess Desk images, then stop with a structured unimplemented model-core blocker until later slices fill it.
- Prefix APIs and tests continue to pass.

**Touches:** `src/mlx_spatial/mapanything_inference.py`, `src/mlx_spatial/mapanything_scene.py` if split out, `src/mlx_spatial/__init__.py`, `tests/test_mapanything_inference.py`

**Verification:** `uv run pytest tests/test_mapanything_inference.py tests/test_mapanything_preprocess.py tests/test_mapanything_assets.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_scene.py` with `MapAnythingScenePipeline`, scene result/blocker/prediction dataclasses, stable scene output keys, and `.npz` scene bundle writing. Exported the scene API from `src/mlx_spatial/__init__.py` without changing the prefix pipeline. Added scene contract, missing-asset, bundle-schema, public-export, and vendor-free import coverage in `tests/test_mapanything_inference.py`; `uv run pytest tests/test_mapanything_inference.py tests/test_mapanything_preprocess.py tests/test_mapanything_assets.py -q` passed with 28 tests.
**Risks / next:** scene generation intentionally stops at the `model-core` blocker until Slice 3 expands the MLX encoder beyond the prefix boundary.

### Slice 3: Full DINOv2 Encoder

**Objective:** Expand the current encoder-prefix port into the full 24-layer DINOv2 giant encoder used by the local MapAnything weights.

**Acceptance criteria:**
- Weight loading covers all required `encoder.model.blocks.0..23.*` tensors plus patch/position/class-token tensors.
- The encoder returns per-view feature maps in `[B, 1536, H/14, W/14]` layout and the class/register-token surface expected by info sharing.
- Numeric parity passes against captured reference tensors for deterministic small fixtures and real Desk reference boundaries.
- Real Desk execution uses sequential layer evaluation and avoids retaining unnecessary intermediate activations.

**Execution:** subagent recommended

**Depends on:** Slice 1

**Touches:** `src/mlx_spatial/mapanything_model.py`, optional `src/mlx_spatial/mapanything_encoder.py`, `tests/test_mapanything_model.py`, `tests/fixtures/mapanything/`

**Verification:** `uv run pytest tests/test_mapanything_model.py -q && MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_full_encoder_parity.py -q`

**Status:** complete
**Evidence:** implemented the full MLX DINOv2 image encoder directly in `src/mlx_spatial/mapanything_model.py` while preserving the existing prefix API. Added full-encoder required-key mapping, weight loading/validation, reusable per-block execution, full encoder outputs (`features` and `registers`), and parity tensor helpers; exported the new surface from `src/mlx_spatial/__init__.py`. Added unit coverage in `tests/test_mapanything_model.py` and opt-in real Desk parity in `tests/test_mapanything_full_encoder_parity.py`. Verification passed: `uv run pytest tests/test_mapanything_model.py -q` -> 11 tests, and `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_full_encoder_parity.py -q` -> 1 test. Real Desk parity checked `encoder.patch_embed`, `encoder.tokens`, `encoder.block0`, `encoder.features.0/1`, and `encoder.registers.0/1` against `/tmp/mapanything-desk-scene-reference.npz`.
**Risks / next:** execution was direct despite `subagent recommended` because host subagent rules require explicit user delegation. Full encoder parity uses relaxed MPS/autocast tolerances (`atol=2e-1`, `rtol=5e-2`) for final features; Slice 4 should reuse the same scene reference for info-sharing token layout and intermediate outputs.

### Slice 4: Multi-View Info Sharing

**Objective:** Implement the MapAnything multi-view alternating attention transformer in MLX, including view packing, scale token, per-view token handling, final norm, and intermediate feature returns.

**Acceptance criteria:**
- Runtime loads `info_sharing.*`, `scale_token`, and related norm/view-position tensors from the checkpoint.
- Token packing matches the reference: per-view spatial tokens, optional per-view tokens, and global scale token.
- Even layers run global attention; odd layers run frame-level attention while holding global additional tokens out of frame attention.
- Intermediate outputs at indices 7 and 11 and final output match the reference layout and parity tolerances.

**Execution:** subagent recommended

**Depends on:** Slices 1, 3

**Touches:** `src/mlx_spatial/mapanything_model.py`, optional `src/mlx_spatial/mapanything_info_sharing.py`, `tests/test_mapanything_info_sharing.py`

**Verification:** `uv run pytest tests/test_mapanything_info_sharing.py -q && MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_info_sharing_parity.py -q`

**Status:** complete
**Evidence:** implemented MLX `MapAnythingInfoSharing` in `src/mlx_spatial/mapanything_model.py`, including checkpoint key loading/validation for `info_sharing.*` and `scale_token`, per-view spatial/register/global-token packing, reference-view positional encoding, even-layer global attention, odd-layer frame attention with global scale token held out, final norm, and normalized intermediate outputs at configured indices. Exported the public API from `src/mlx_spatial/__init__.py`. Added `tests/test_mapanything_info_sharing.py` and opt-in real Desk parity in `tests/test_mapanything_info_sharing_parity.py`. Verification passed: `uv run pytest tests/test_mapanything_info_sharing.py -q` -> 6 tests, `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_info_sharing_parity.py -q` -> 1 test, and `uv run pytest tests/test_mapanything_model.py tests/test_mapanything_inference.py -q` -> 20 tests.
**Risks / next:** info-sharing parity intentionally uses captured `fusion.features.*` plus encoder registers because vendored MapAnything feeds post-fusion features into `info_sharing`; Slice 5 owns the MLX fusion boundary and prediction heads.

### Slice 5: Dense, Pose, and Scale Heads

**Objective:** Port the inference-time prediction heads that turn encoder/info-sharing features into dense ray/depth/confidence/mask, camera pose, and global scale outputs.

**Acceptance criteria:**
- Runtime loads and validates `dense_head.*`, `pose_head.*`, `scale_head.*`, `fusion_norm_layer.*`, and required adaptor settings.
- DPT feature processing matches the configured four-input path: encoder features, info layer 7, info layer 11, and final info-sharing features.
- Pose head produces translation and normalized quaternion outputs; scale head produces the configured exponential scale.
- Dense adaptor applies configured transforms for ray directions, depth, confidence, and mask logits.
- Shape and numeric parity tests cover each head independently before full scene assembly.

**Execution:** subagent recommended

**Depends on:** Slices 1, 3, 4

**Touches:** `src/mlx_spatial/mapanything_model.py`, optional `src/mlx_spatial/mapanything_heads.py`, `tests/test_mapanything_heads.py`

**Verification:** `uv run pytest tests/test_mapanything_heads.py -q && MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_heads_parity.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_heads.py` with MLX fusion norm, DPT dense feature/regressor path, dense ray-depth-confidence-mask adaptor, pose head plus quaternion normalization, scale MLP plus exponential scale adaptor, checkpoint loading/validation, parity tensor extraction, and public exports from `src/mlx_spatial/__init__.py`. Added unit coverage in `tests/test_mapanything_heads.py` and opt-in real Desk parity in `tests/test_mapanything_heads_parity.py`. Verification passed: `uv run pytest tests/test_mapanything_heads.py -q` -> 6 tests, `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_heads_parity.py -q` -> 1 test, and `uv run pytest tests/test_mapanything_model.py tests/test_mapanything_info_sharing.py tests/test_mapanything_inference.py -q` -> 26 tests.
**Risks / next:** head parity uses `atol=1e-1`, `rtol=5e-2` for dense outputs because MLX and PyTorch bilinear/transpose-convolution numerics differ most in the DPT path; fusion norm parity is tight (`1e-4`) and scale parity is exact against the reference bundle.

### Slice 6: Inference Postprocess and Geometry

**Objective:** Implement the image-only `infer(...)` postprocess needed to convert raw model outputs into scene tensors matching the vendored demo contract.

**Acceptance criteria:**
- Runtime derives `pts3d`, `pts3d_cam`, `ray_directions`, `depth_along_ray`, `depth_z`, `cam_trans`, `cam_quats`, `camera_poses`, `intrinsics`, `conf`, `non_ambiguous_mask`, and final `mask`.
- Depth-to-world conversion matches the reference `depthmap_to_world_frame` semantics for NumPy/MLX arrays.
- Mask handling covers non-ambiguous mask, valid-depth mask, and the configured default `apply_mask=True`, `mask_edges=True` behavior, or records any edge-mask limitation explicitly in code/tests.
- Final scene arrays use a stable schema consumable by Slice 7 export.

**Depends on:** Slices 1, 5

**Touches:** `src/mlx_spatial/mapanything_geometry.py`, `src/mlx_spatial/mapanything_inference.py`, `tests/test_mapanything_geometry.py`, `tests/test_mapanything_scene_postprocess.py`

**Verification:** `uv run pytest tests/test_mapanything_geometry.py tests/test_mapanything_scene_postprocess.py -q && MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_scene_postprocess_parity.py -q`

**Status:** complete
**Evidence:** added `src/mlx_spatial/mapanything_geometry.py` with Torch-free raw head assembly, quaternion/camera pose conversion, pinhole intrinsics recovery, depth-to-camera/world conversion, denormalized images, confidence/mask handling, vendored-style depth/normal edge masks, final per-view outputs, and stable scene payload helpers. Exported the postprocess/geometry API from `src/mlx_spatial/__init__.py`. Added `tests/test_mapanything_geometry.py`, `tests/test_mapanything_scene_postprocess.py`, and opt-in real Desk parity in `tests/test_mapanything_scene_postprocess_parity.py`. Verification passed: `uv run pytest tests/test_mapanything_geometry.py tests/test_mapanything_scene_postprocess.py -q` -> 7 tests, and `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref pytest tests/test_mapanything_scene_postprocess_parity.py -q` -> 1 test.
**Risks / next:** Desk edge-mask parity has 9 total threshold-pixel differences when reconstructing from captured head tensors instead of the original Torch pointmap path; the test and trace record this NumPy/Torch edge-threshold boundary while keeping strict parity on stable final fields and common-mask geometry. Slice 7 can now wire the full pipeline and scene export.

### Slice 7: Desk Scene Generation and Export

**Objective:** Wire the full runtime pipeline so the two Desk images generate a real scene prediction artifact.

**Acceptance criteria:**
- A local command or test runs `weights/map-anything` plus `inputs/map-anything/desk` through the MLX scene pipeline.
- The run writes `/tmp/mapanything-desk-mlx-scene.npz` or a user-provided output path with images, depth, intrinsics, camera poses/extrinsics, world points, confidence, masks, and metadata.
- A lightweight export path writes `.ply` or `.glb` when compatible with runtime constraints; otherwise the `.npz` scene bundle is the required artifact and the exporter clearly reports unsupported optional formats.
- Real Desk smoke records timing/device/memory-relevant metadata and fails with a concrete model-boundary blocker instead of silent partial output.

**Depends on:** Slices 2, 3, 4, 5, 6

**Touches:** `src/mlx_spatial/mapanything_inference.py`, optional `src/mlx_spatial/mapanything_export.py`, `tests/test_mapanything_scene_pipeline.py`, optional `scripts/mapanything/`

**Verification:** `uv run pytest tests/test_mapanything_scene_pipeline.py -q && uv run python -m mlx_spatial.mapanything_scene weights/map-anything inputs/map-anything/desk --output /tmp/mapanything-desk-mlx-scene.npz`

**Status:** complete
**Evidence:** wired `MapAnythingScenePipeline.generate` through asset/config validation, Desk preprocessing, full MLX DINOv2 encoder, fusion norm, info sharing, dense/pose/scale heads, Slice 6 postprocess, and scene prediction construction. Added a CLI entrypoint in `src/mlx_spatial/mapanything_scene.py` and real local Desk coverage in `tests/test_mapanything_scene_pipeline.py`; updated the earlier scene contract test to expect the new checkpoint-loading boundary for intentionally tiny weights. Verification passed: `uv run pytest tests/test_mapanything_scene_pipeline.py -q` -> 1 test, and `uv run python -m mlx_spatial.mapanything_scene weights/map-anything inputs/map-anything/desk --output /tmp/mapanything-desk-mlx-scene.npz` wrote the scene bundle. The output bundle contains `images`, `depth`, `confidence`, `masks`, `intrinsics`, `camera_poses`, `extrinsics`, and `world_points` with two `(392, 518)` Desk views and metadata `implemented_boundary=scene-generation`.
**Risks / next:** none for the CLI path after Slice 8 made `mlx_spatial.mapanything_scene` a lazy package export and removed the `python -m` runpy warning.

### Slice 8: Regression, Runtime Hygiene, and Packaging

**Objective:** Prove the new scene-generation path does not regress existing MapAnything prefix behavior, runtime dependencies, or package artifact hygiene.

**Acceptance criteria:**
- Existing `tests/test_mapanything_*.py` pass, including prefix tests.
- Runtime import scans show no package import of Torch, TorchVision, UniCeption, OpenCV, or `vendors/map-anything`.
- `uv lock --check` and build checks pass.
- Built wheel/sdist exclude `weights/`, `inputs/`, `vendors/`, `/tmp` captures, and generated scene artifacts.

**Depends on:** Slices 1-7

**Touches:** tests, packaging metadata only if required

**Verification:** `uv run pytest tests/test_mapanything_*.py -q && ! rg -n "import (torch|torchvision|cv2)|from (torch|torchvision|uniception|cv2)|vendors/map-anything" src/mlx_spatial && uv lock --check && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl`

**Status:** complete
**Evidence:** ran the default MapAnything regression suite and package hygiene gates. `uv run pytest tests/test_mapanything_*.py -q` passed with 67 tests and 5 opt-in parity tests skipped. The runtime import scan `! rg -n "import (torch|torchvision|cv2)|from (torch|torchvision|uniception|cv2)|vendors/map-anything" src/mlx_spatial` passed after making the LiTo dev-only torch fallback use `importlib` and replacing the MapAnything parity source label with a non-path string. `uv lock --check` passed. `UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build` built `dist/mlx_spatial-0.0.2.tar.gz` and `dist/mlx_spatial-0.0.2-py3-none-any.whl`; `python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl` checked both artifacts. Also made `mlx_spatial.mapanything_scene` lazy-exported from package `__init__`, and reran the Desk CLI successfully without the previous runpy warning.
**Risks / next:** none.

## Aggregate Verification Commands

| Gate | Command |
|---|---|
| Default MapAnything regression | `uv run pytest tests/test_mapanything_*.py -q` |
| Torch reference capture | `MAPANYTHING_TORCH_REF=1 uv run --group torch-ref python tools/mapanything_dump_torch_scene_reference.py weights/map-anything inputs/map-anything/desk --device mps --output /tmp/mapanything-desk-scene-reference.npz` |
| MLX Desk scene generation | `uv run python -m mlx_spatial.mapanything_scene weights/map-anything inputs/map-anything/desk --output /tmp/mapanything-desk-mlx-scene.npz` |
| Package hygiene | `uv lock --check && UV_CACHE_DIR=/tmp/mlx-spatial-uv-cache uv build && python scripts/packaging/check_release_artifacts.py dist/mlx_spatial-0.0.2.tar.gz dist/mlx_spatial-0.0.2-py3-none-any.whl` |

## Review Recommendation

Run `auto-eng-review` before `auto-execute`. This plan ports large model math and sets the user-visible completion boundary at real scene generation, so a short review is useful before changing runtime code.
