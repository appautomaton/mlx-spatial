# HY-World-2.0 WorldMirror MLX Inference

## Bounded Goal

Build a staged MLX inference infrastructure for HY-World-2.0 WorldMirror 2.0 that can reconstruct scene/world geometry from image or video-frame inputs, starting with depth, normals, camera metadata, and point-cloud PLY outputs, while preserving official pipeline semantics and strict memory guards.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- The parity target is the official HY-World-2.0 WorldMirror 2.0 PyTorch pipeline, not the older HunyuanWorld-Mirror repo and not Voyager.
- Learned inference must run through MLX in the shipped path; PyTorch, CUDA, FSDP, NCCL, `gsplat`, and training runtime dependencies are not allowed.
- The implementation must be staged by head/output so large runs can enable depth, normals, points, cameras, and Gaussian attributes independently.
- Memory safety is a first-class behavior: frame count, image size, attention/activation size, and head execution must have explicit guards or chunking rather than risking OS-level crashes.
- MLX usage should follow repo best practice: keep model tensors in MLX, control lazy evaluation with deliberate `mx.eval` boundaries, avoid unnecessary NumPy round-trips in learned compute, and expose structured blockers when resources exceed configured limits.
- Output paths must stay under `outputs/` and weights/inputs must stay outside committed artifacts.
- The command surface should fit the existing local 3D stack: `trellis2` remains object-centric generation, while `hyworld2` becomes scene/world reconstruction.
- If official weights or config files are missing, the command must report a precise setup blocker instead of attempting partial or fake inference.

## Required Behavior

- Add a `hyworld2` command surface for validating assets and running staged reconstruction from an image directory, video frames, or a supported single input path.
- Validate the expected HY-World-2.0 WorldMirror 2.0 checkpoint layout, including `model.safetensors` and config metadata, before inference begins.
- Load and route WorldMirror components in a modular MLX pipeline that mirrors official preprocessing, backbone, and selectable prediction heads.
- Support staged head selection, with at least depth, normals, camera metadata, and point-cloud output represented in the public contract; Gaussian attributes are included as a later staged head, not a required first live output.
- Write concrete reconstruction artifacts under `outputs/hyworld2/`, including machine-readable trace metadata, depth/normal outputs, camera metadata when available, and point-cloud PLY when the points head succeeds.
- Return structured blockers with deepest completed stage, requested heads, resolved frame count, estimated memory pressure, checkpoint status, and missing implementation surface when a stage cannot run exactly.
- Provide deterministic fixture coverage so the pipeline can be verified without downloading the full checkpoint.

## Acceptance Criteria

- `uv run mlx-spatial-hyworld2 validate weights/hy-world-2` reports either a valid WorldMirror 2.0 asset layout or a structured missing-asset blocker naming exact expected files.
- `uv run mlx-spatial-hyworld2 reconstruct weights/hy-world-2 inputs/hyworld2/demo --output outputs/hyworld2/demo --heads depth,normal,points --memory-profile balanced` either writes real staged outputs or stops with a structured blocker that identifies the deepest completed stage.
- The implementation includes MLX-native modules or adapters for the official preprocessing and model-stage contracts needed by depth, normals, cameras, and points; no shipped inference path imports PyTorch or CUDA-only packages.
- Memory guards are covered by tests and include frame-count/image-size limits plus at least one model-stage activation guard or chunked execution path.
- Fixture tests prove that staged head selection works, trace metadata is emitted, output path validation rejects paths outside `outputs/`, and missing weights/configs fail cleanly.
- If real HY-World-2.0 WorldMirror 2.0 weights are present locally, a live 518-resolution reconstruction command can produce a non-empty point-cloud PLY and trace metadata for a small input set.
- Existing TRELLIS.2 tests continue passing after adding the HY-World command surface.

## Blocking Questions Or Assumptions

- Assumption: the local checkpoint will be placed under `weights/hy-world-2/HY-WorldMirror-2.0/` or a CLI-supplied equivalent path.
- Assumption: first live reconstruction can use a small frame count and the official 518 input size before broader video-scale testing.
- Assumption: Gaussian attribute export is in scope for the staged architecture, but CUDA-style Gaussian rasterization or preview is deferred.
- Assumption: NumPy is acceptable for file assembly, PLY writing, and non-learned postprocess, but learned model compute remains MLX.

## Anti-Goals

- Do not port HunyuanWorld Voyager in this change.
- Do not add PyTorch, CUDA, FSDP, NCCL, `gsplat`, `flash-attn`, `xfuser`, or training dependencies to the shipped MLX path.
- Do not fake point clouds, cameras, normals, depth maps, or Gaussian outputs when exact model stages are missing.
- Do not attempt real-time 3DGS rendering, camera-path video synthesis, GLB scene packaging, or GUI/workbench preview in this spec.
- Do not change the TRELLIS.2 command behavior except where shared CLI packaging or tests require non-behavioral integration.

## Recommended Next Skill

`auto-plan`
