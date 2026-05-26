# MapAnything

MapAnything in `mlx-spatial` is the image-only multi-view scene geometry path
for Meta's `facebook/map-anything` checkpoint. It takes related scene images and
writes an inspectable scene tensor bundle.

This runtime does not use Torch, TorchVision, UniCeption, OpenCV, CUDA, or
vendor Python code. Those are only used in explicit dev parity workflows.

## Assets

Download the public HF checkpoint into the ignored local weights folder:

```bash
uv run hf download facebook/map-anything \
  --local-dir weights/map-anything
```

The expected local layout is:

```text
weights/map-anything/
  config.json
  model.safetensors
```

Validate it:

```bash
uv run mlx-spatial-mapanything validate weights/map-anything
```

The `facebook/map-anything` weights are not bundled in the package. Respect the
upstream model license and access terms.

## Run

Use the repository script for the recommended path:

```bash
python scripts/mapanything/generate_scene.py inputs/map-anything/desk \
  --output-dir outputs/mapanything/desk-script
```

Or call the installed package CLI directly:

```bash
uv run mlx-spatial-mapanything generate inputs/map-anything/desk \
  --root weights/map-anything \
  --output outputs/mapanything/desk/scene.npz
```

Defaults:

| Setting | Value |
| --- | --- |
| weights root | `weights/map-anything` |
| resize mode | `fixed_mapping` |
| frame stride | `1` |
| patch size | read from `config.json`, normally `14` for `facebook/map-anything` |
| normalization | DINOv2 mean/std |
| postprocess | apply final mask and mask image edges |
| output artifact | compressed `.npz` scene bundle |

Do not pass patch size manually for normal inference. The pipeline reads it from
the checkpoint config and uses it to choose patch-aligned target sizes.

## Output Schema

`scene.npz` uses view-first arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `images` | `[V,H,W,3]` | preprocessed RGB views |
| `depth` | `[V,H,W]` | dense depth |
| `confidence` | `[V,H,W]` | dense confidence |
| `masks` | `[V,H,W]` | final valid masks |
| `intrinsics` | `[V,3,3]` | recovered pinhole intrinsics |
| `camera_poses` | `[V,4,4]` | camera-to-world poses |
| `extrinsics` | `[V,4,4]` | world-to-camera transforms |
| `world_points` | `[V,H,W,3]` | dense world-space point maps |

The `.npz` also stores `__metadata_json__` with trace-oriented metadata.

## Torch Reference Parity

The original Torch pipeline records final scene tensors with `scene.*` prefixes:

```text
scene.images
scene.depth
scene.conf
scene.final_masks
scene.intrinsics
scene.camera_poses
scene.world_points
```

The MLX bundle uses clean top-level keys for the same semantic outputs:

```text
images
depth
confidence
masks
intrinsics
camera_poses
world_points
```

It also writes `extrinsics`, derived from `camera_poses`, because many viewers
and downstream geometry tools need world-to-camera matrices.

Dev-only Torch reference capture is guarded by:

```bash
MAPANYTHING_TORCH_REF=1 uv run --group torch-ref \
  python tools/mapanything_dump_torch_scene_reference.py \
  weights/map-anything inputs/map-anything/desk \
  --output /tmp/mapanything-desk-scene-reference.npz
```

Runtime generation does not require that command.

## What It Is Not

The supported MapAnything artifact is not a mesh and not a Gaussian Splat PLY.
A colored point cloud can be derived from `world_points`, `images`, and `masks`
for inspection, but that is a viewer/export step outside the runtime contract.

Temporary HTML or PLY viewers under `/tmp` are useful for visual checks. They
should not be treated as the formal package output.
