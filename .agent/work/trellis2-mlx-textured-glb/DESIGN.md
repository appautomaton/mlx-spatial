# DESIGN: TRELLIS.2 MLX Textured GLB

## Architecture Summary

The textured pipeline extends the working shape pipeline instead of creating a parallel path:

```text
image
  -> MLX RMBG / alpha preprocessing
  -> MLX DINOv3 conditioning
  -> sparse structure sampling and coordinates
  -> shape SLat
  -> shape decoder / FlexiDualGrid fields / mesh
  -> texture SLat
  -> texture decoder
  -> mesh/voxel texture baking
  -> GLB export
```

The implementation should keep the shape path as the source of geometry truth. Texture generation must consume the same sparse structure, shape SLat, shape decoder outputs, and subdivision/voxel guides that upstream TRELLIS.2 uses.

## Runtime Boundary

Allowed runtime:

- MLX for learned tensor compute.
- NumPy/Pillow for index construction, image buffers, and texture atlas assembly.
- A small pure-Python GLB writer or a lightweight non-PyTorch export dependency if planning confirms it is worth adding.
- Blender only for verification, not as a runtime export dependency.

Disallowed runtime:

- PyTorch, torchvision, CUDA, nvdiffrast, or vendored Python modules as required inference/runtime dependencies.
- Fake material output that does not consume texture SLat and texture decoder results.

## Core Components

### Command Surface

Add `generate-textured` to the TRELLIS.2 CLI. It should share the same asset roots, pipeline type, seed, and guard style as `generate-shape`, plus texture/export guards.

The command returns either:

- a textured `.glb` under `outputs/`, or
- a structured blocker with deepest completed stage and relevant shapes/token counts.

### Texture SLat

Texture SLat should reuse existing SLat route/config infrastructure where possible. It must validate that it is consuming real shape SLat/layout data, not fake placeholders.

Key outputs:

- texture SLat coordinates/features,
- route metadata,
- conditioning resolution,
- token counts,
- blockers for unsupported cascade or guard violations.

### Texture Decoder

The texture decoder should follow upstream architecture semantics:

- load texture decoder config/checkpoint,
- use texture SLat `from_latent` projection,
- consume shape decoder subdivision/guide metadata where upstream requires it,
- produce a concrete texture representation suitable for baking/export.

The first implementation may checkpoint at a raw texture representation if GLB is blocked, but it must not claim textured export success until the GLB contains material/texture data.

### Mesh / Voxel / Baking

The shape mesh alone is not enough for upstream-style texturing. The implementation needs an intermediate representation equivalent to the upstream mesh/voxel coupling:

- mesh vertices/faces from FlexiDualGrid,
- voxel or field metadata needed to map decoder texture outputs onto surface samples,
- UV coordinates or an atlas,
- baked image data.

Use small deterministic fixtures before live-weight export.

### GLB Export

GLB export should be minimal but real:

- binary GLB 2.0,
- one or more meshes,
- valid indices/accessors/bufferViews,
- material referencing a base-color texture or equivalent image payload,
- Blender import verification.

## Data And Blocker Trace

The textured result should expose stage trace metadata comparable to `generate-shape`:

- completed stages,
- output names,
- texture route,
- shape and texture token counts,
- decoder field shapes,
- atlas/image dimensions,
- exported byte size,
- blocker details when incomplete.

## Verification Strategy

Use three layers:

- Unit fixtures for format/path policy, route selection, texture decoder shape contracts, baking/GLB writer structure.
- Integration fixture that uses fake shape/texture fields and writes a Blender-readable textured GLB.
- Live local verification using real weights and `inputs/trellis2/image.png`.

Full live verification may be expensive. Every expensive command must keep token/memory guards and produce structured failure if the guard is exceeded.
