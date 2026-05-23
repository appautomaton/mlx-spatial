# HY-WorldMirror 2.0

HY-WorldMirror is the scene/world reconstruction path in `mlx-spatial`. It
does not take object masks and is not the right tool for isolated object
reconstruction. Use SAM3D or TRELLIS.2 for object-centric inputs.

## Assets

No MLX conversion step is required. The runtime reads Tencent's downloaded
safetensors and config layout directly:

```text
weights/hy-world-2/
weights/hy-world-2/HY-WorldMirror-2.0/config.json
weights/hy-world-2/HY-WorldMirror-2.0/model.safetensors
```

Print the download command:

```bash
uv run mlx-spatial-hyworld2 download-command weights/hy-world-2
```

After downloading, validate the layout:

```bash
uv run mlx-spatial-hyworld2 validate weights/hy-world-2
```

## Inputs

Supported inputs are:

- one RGB/RGBA scene image
- a directory of `.jpg`, `.jpeg`, `.png`, or `.webp` scene frames

The frame directory path is sorted lexically, then capped by the selected memory
profile. Use consistent framing and orientation across frames. This path ignores
object masks.

Better inputs are scene-level views with useful camera motion and enough visual
overlap. A single image can run, but it gives the model less geometric evidence.

## Run

Recommended script path:

```bash
python scripts/hyworld2/generate_scene.py inputs/sam3d/kidsroom/image.png \
  --output-dir outputs/hyworld2/kidsroom-scene-script
```

For a frame directory:

```bash
python scripts/hyworld2/generate_scene.py inputs/hyworld2/small-room-8 \
  --memory-profile balanced \
  --output-dir outputs/hyworld2/small-room-8-balanced
```

## Outputs

The release-ready heads are `camera,depth,normal,points`:

```text
outputs/hyworld2/<run>/
  camera_params.json
  depth/
  normal/
  points/points.ply
  trace.json
```

The optional `gs` head is not exposed by the script because Gaussian preview and
export are not release-ready.

## Memory Profiles

| Profile | Target size | Max frames | Use case |
| --- | ---: | ---: | --- |
| `safe` | 392 | 2 | lowest memory smoke run |
| `balanced` | 518 | 8 | practical multi-frame default |
| `large` | 952 | 32 | official-resolution path for one or a few frames |

`large` matches the official 952px target, but exact full attention grows
quickly with token count. If a multi-frame run blocks on the attention guard,
switch to `balanced` before lowering output heads.

## Trace

`trace.json` records selected images, memory profile, processed image size,
token count, requested heads, completed stages, outputs, and any blocker. Use it
when comparing runs or reporting failures.
