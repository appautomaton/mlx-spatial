# SAM3D MLX Mesh Decoder Algorithm Parity Spec

## Bounded Goal

Port the SAM3D Objects mesh decoder path closely enough that the existing MLX `reconstruct` command can produce a real Blender-readable `mesh.glb` from the generated SLat coordinates/features without using PyTorch, CUDA, or fake geometry.

## Selected Lenses

- Product
- Engineering
- Runtime

## Context

The current SAM3D MLX path runs the official sample through MLX MoGe, official preprocessing, SS condition fusion, SS ShortCut flow, SS decoder, SLat FlowMatching, gaussian decoder, and binary official-field `gaussians.ply` export. When `--glb-output` is requested it blocks at `mesh-decoder`. This change closes that mesh gap using algorithm parity as the target, not a quick Gaussian-to-mesh approximation.

## Constraints

- Runtime must remain PyTorch/CUDA-free: no shipped imports of `torch`, `spconv`, `gsplat`, `nvdiffrast`, `kaolin`, or vendored runtime modules.
- Learned inference stays in MLX. NumPy is acceptable for sparse coordinate indexing, dense-grid assembly, FlexiCubes-style extraction support code, and GLB writing.
- The mesh path must use `slat_decoder_mesh` outputs and official-style mesh feature interpretation: `sdf`, `deform`, `weights`, and optional `color`.
- If mesh extraction cannot run within memory or unsupported math boundaries, the command must return a structured blocker at the deepest completed stage and must not write a fake `.glb`.
- The existing `gaussians.ply` path must keep working and must not regress.
- First GLB acceptance is Blender-readable non-empty geometry, not final mesh quality or texture baking.
- Live target remains the official single-object notebook samples already used in this repo, starting with `human_object` and `kidsroom`.

## Required Behavior

- `mlx-spatial-sam3d reconstruct ... --glb-output outputs/.../mesh.glb` runs past the current `mesh-decoder` blocker for at least one official sample.
- The command writes both `gaussians.ply` and `mesh.glb` when both output paths are requested.
- Trace metadata records mesh decoder input shape, mesh feature layout, sparse subdivision counts, dense extraction resolution, vertex count, face count, GLB bytes, and any cleanup or blocker reason.
- The mesh decoder follows the active official architecture enough to preserve algorithm semantics:
  - shared SLat decoder transformer on `slat_decoder_mesh` weights,
  - two `SparseSubdivideBlock3d` stages,
  - final mesh feature layout,
  - `SparseFeatures2Mesh`-style conversion from sparse cube features to dense vertex/cube fields,
  - FlexiCubes-style surface extraction or a clearly documented exact subset needed for inference.
- If `--glb-output` is omitted, existing Gaussian-only completion still exits `0`.

## Acceptance Criteria

- Unit tests cover:
  - mesh decoder config parsing and tensor loading;
  - sparse subdivide coordinate/feature expansion;
  - `SparseSubdivideBlock3d` fixture behavior;
  - mesh feature channel layout for `sdf`, `deform`, `weights`, and color;
  - sparse cube-to-vertex aggregation;
  - dense SDF/deform/weight assembly for a deterministic tiny fixture;
  - FlexiCubes-style extraction on a tiny deterministic field producing non-empty vertices/faces;
  - GLB writer validation for mesh decoder output.
- Integration tests cover:
  - fake SLat mesh fixture writes a non-empty GLB and trace reaches `glb-export`;
  - Gaussian-only SAM3D path still exits cleanly;
  - requested GLB never falls back to fake point, cube, or Gaussian-derived mesh geometry.
- Live verification covers:
  - `uv run mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx vendors/sam-3d-objects/notebook/images/human_object/image.png --mask vendors/sam-3d-objects/notebook/images/human_object/0.png --moge-root weights/moge-vitl-mlx --output outputs/sam3d/human-object/gaussians.ply --glb-output outputs/sam3d/human-object/mesh.glb --seed 42 --memory-profile large --trace-output outputs/sam3d/human-object/trace.json`
  - output `gaussians.ply` exists and has official Gaussian fields;
  - output `mesh.glb` exists, is non-empty, and imports in Blender headless if Blender is available;
  - trace reaches `glb-export` with no blocker and nonzero vertex/face counts.
- Regression checks pass:
  - focused SAM3D tests,
  - full `uv run pytest -q`,
  - `git diff --check`.

## Blocking Questions Or Assumptions

- Assumption: a basic Blender-readable GLB is sufficient for this slice even if topology is rough.
- Assumption: exact numerical PyTorch parity is deferred; algorithm parity and artifact validity are the acceptance bar.
- Assumption: dense extraction at SAM3D mesh resolution is acceptable on this 128 GB machine, but memory guards still must be present.
- Assumption: if a direct FlexiCubes port proves too large for one implementation slice, the plan may split extraction into a bounded first subset while preserving strict blockers and no fake GLB.

## Anti-Goals

- Do not generate mesh geometry from Gaussian splats as a shortcut.
- Do not add PyTorch, CUDA, `spconv`, `gsplat`, `nvdiffrast`, `kaolin`, or vendored runtime imports to the shipped command.
- Do not implement texture baking, UV unwrap, remeshing, mesh cleanup quality passes, SAM3D Body, auto-segmentation, or multi-object scene assembly in this change.
- Do not claim official-quality mesh output from the first GLB milestone.
- Do not broaden the SAM3D CLI contract beyond the existing `reconstruct` command unless required for trace or guard metadata.
