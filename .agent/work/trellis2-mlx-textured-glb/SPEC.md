# SPEC: TRELLIS.2 MLX Textured GLB

## Bounded Goal

Extend the current MLX TRELLIS.2 shape pipeline so an RGB or RGBA image can produce a real textured `.glb` asset through MLX inference and Mac-native export surfaces, with structured blockers instead of fake outputs when an exact stage is not yet implemented.

## Selected Lenses

- product
- engineering
- runtime

## Constraints

- Runtime inference must not depend on PyTorch or CUDA; vendored TRELLIS.2 / trellis-mac code may be used only as reference material.
- Architecture parity is the target: match upstream TRELLIS.2 routing, module semantics, checkpoint tensor usage, and shape/texture coupling; strict numerical parity is not required for the first implementation.
- The existing shape path remains the geometry base: MLX RMBG, DINOv3 conditioning, sparse structure, shape SLat, shape decoder, FlexiDualGrid mesh.
- Texture implementation must use the local TRELLIS.2 texture checkpoints and route through texture SLat and texture decoder rather than painting a placeholder color.
- Exported assets must stay under ignored `outputs/` and must be readable by Blender.
- Memory guards must remain first-class: if token counts, decoder expansion, baking size, or GLB assembly exceed configured limits, the command must stop with a precise blocker.
- The first command should support `--pipeline-type 512`; cascade texture support may be implemented only where needed by upstream routing and should be guarded if incomplete.
- Existing OBJ shape generation behavior must continue passing.

## Required Behavior

- Add a user-facing textured generation command, tentatively:

```bash
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png \
  --output outputs/trellis2/image-textured.glb \
  --pipeline-type 512 \
  --seed 42 \
  --rmbg-root weights/rmbg2
```

- The command must run the existing image preprocessing and shape geometry path before texture generation.
- The command must run MLX texture SLat using the same sparse structure, conditioning, and upstream texture route semantics.
- The command must run the MLX texture decoder using the shape outputs and subdivision/voxel guides required by upstream TRELLIS.2.
- The command must construct the mesh/voxel representation needed for texture baking or equivalent Mac-native texture projection.
- The command must produce a textured GLB that Blender can import with geometry and material/texture data.
- If any stage is incomplete, the command must report:
  - blocker stage,
  - blocker operation,
  - deepest completed stage,
  - relevant token/resolution/shape metadata,
  - next implementation slice.

## Acceptance Criteria

- `generate-textured` rejects non-`.glb` outputs with a precise format blocker and rejects paths outside `outputs/`.
- `generate-textured` rejects missing texture checkpoints or incomplete texture route metadata before doing expensive compute.
- Texture SLat route selection matches upstream TRELLIS.2 for `512`, `1024`, `1024_cascade`, and `1536_cascade`, even if non-512 textured export is initially guarded.
- Texture SLat consumes the actual shape SLat/shape coordinates expected by upstream semantics, not synthetic texture input.
- Texture decoder reaches a concrete texture output representation or returns a structured blocker identifying the exact missing decoder/export operation.
- GLB export writes a real `.glb` with mesh geometry and material/texture data; no placeholder color, fake UVs, or empty material counts as success.
- Blender headless import succeeds on the generated GLB and reports at least one mesh object and at least one material or image texture.
- The current shape OBJ path continues to work:

```bash
uv run mlx-spatial-trellis2 generate-shape weights/trellis2 inputs/trellis2/image.png \
  --output outputs/trellis2/image-512-rmbg-shape.obj \
  --pipeline-type 512 \
  --seed 42 \
  --rmbg-root weights/rmbg2 \
  --decoder-token-limit 1200000
```

- Default test suite passes with local fixtures and without requiring gated model downloads:

```bash
uv run pytest -q
```

- Live verification, when local weights are present, includes:

```bash
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png \
  --output outputs/trellis2/image-textured.glb \
  --pipeline-type 512 \
  --seed 42 \
  --rmbg-root weights/rmbg2
```

## Blocking Questions Or Assumptions

- Assumption: local `weights/trellis2` includes the texture SLat and texture decoder checkpoints required by the upstream image-to-3D texture path.
- Assumption: a Python/NumPy/MLX GLB writer or a small repo-local writer is acceptable if it avoids PyTorch/CUDA and produces Blender-readable files.
- Assumption: UV unwrap and texture baking may use Mac-native non-MLX libraries if they are export utilities, not learned model runtime.
- Assumption: architecture parity allows small numeric differences from PyTorch/trellis-mac as long as the module graph, conditioning, tensor semantics, and checkpoint usage match.
- Blocking question for planning: whether to use an existing lightweight GLB/UV dependency or implement a minimal internal GLB writer and texture atlas path.

## Anti-Goals

- Do not implement training or differentiable rendering.
- Do not require CUDA, PyTorch, torchvision, or nvdiffrast at runtime.
- Do not fake a textured GLB by assigning a constant material or embedding the input image as an unrelated texture.
- Do not make `generate-shape` write `.glb`; textured GLB belongs to the new textured generation command.
- Do not target strict PyTorch tensor parity in this change.
- Do not make 1024/1536 cascade textured GLB the first required live output unless 512 parity is already working.
