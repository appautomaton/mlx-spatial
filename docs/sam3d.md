# SAM3D

This package supports a local MLX path for SAM 3D Objects. The Python package does not include SAM3D weights.

Upstream references:

- Runtime model bundle: https://huggingface.co/appautomaton/sam-3d-objects-mlx
- Model repo: https://huggingface.co/facebook/sam-3d-objects
- Code repo: https://github.com/facebookresearch/sam-3d-objects
- License source: the upstream model and code pages identify the SAM License as the governing license.

The public `appautomaton/sam-3d-objects-mlx` bundle is the recommended runtime download for `mlx-spatial`. The upstream Meta model remains the source-of-truth for license and provenance; its Hugging Face repo is gated and is only needed for maintainer conversion/audit work.

## Local Asset Layout

Use these ignored paths:

```text
weights/sam-3d-objects/       # original upstream checkpoint bundle
weights/sam-3d-objects-mlx/   # converted MLX safetensors mirror
weights/sam-3d-objects-mlx/moge/model.safetensors
```

The converted SAM3D root is expected to contain `pipeline.yaml` plus checkpoint files under `checkpoints/`. The bundled MoGe root is expected at `weights/sam-3d-objects-mlx/moge/` and contains `model.safetensors`.

## Download Runtime Bundle

Download the ready MLX bundle from the public Hugging Face repo:

```bash
uv run hf download appautomaton/sam-3d-objects-mlx \
  --local-dir weights/sam-3d-objects-mlx
uv run mlx-spatial-sam3d validate weights/sam-3d-objects-mlx
uv run mlx-spatial-sam3d inspect weights/sam-3d-objects-mlx
```

This is the normal user setup path. No local SAM3D conversion is required when using this bundle. From an installed package, use the same commands without `uv run`.

## Maintainer Conversion Path

Print the manual Hugging Face command for the gated upstream source bundle:

```bash
uv run mlx-spatial-sam3d download-command --root weights/sam-3d-objects
```

Convert the official checkpoint bundle to safetensors:

```bash
uv run mlx-spatial-sam3d convert weights/sam-3d-objects \
  --output-root weights/sam-3d-objects-mlx \
  --moge-root weights/moge-vitl \
  --moge-output-root weights/sam-3d-objects-mlx/moge
```

The SAM3D conversion audit compares converted tensors against the original checkpoint sources and belongs with the model bundle, not in the Python package.

## Recommended Inference Command

Use the repository script for a stable user-facing command:

```bash
python scripts/sam3d/reconstruct.py inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output-dir outputs/sam3d/living-room-script
python scripts/sam3d/inspect_trace.py outputs/sam3d/living-room-script/trace.json
```

Equivalent package CLI:

```bash
uv run mlx-spatial-sam3d reconstruct weights/sam-3d-objects-mlx \
  inputs/sam3d/living-room/image.png \
  --mask inputs/sam3d/living-room/mask-3.png \
  --output outputs/sam3d/living-room-primary-default/gaussians.ply \
  --trace-output outputs/sam3d/living-room-primary-default/trace.json \
  --memory-profile balanced
```

`--moge-root weights/sam-3d-objects-mlx/moge` is the default when using the public `appautomaton/sam-3d-objects-mlx` bundle. Pass it only when using a different MoGe root.

Recommended defaults:

- seed: `42`
- memory profile: `balanced`
- stage 1 steps: package default
- stage 2 steps: package default
- strict quality gates: on
- quality-warning bypass: off
- mask selection: the runtime uses the exact `--mask` path supplied by the user

## Inputs

SAM3D reconstruction requires:

- RGB/RGBA image.
- Binary object mask.
- Converted SAM3D safetensors root.
- Bundled converted MoGe root, unless an external pointmap is supplied with `--pointmap`.

## Outputs

The main output is a Gaussian PLY:

```text
outputs/sam3d/<run>/gaussians.ply
outputs/sam3d/<run>/trace.json
```

The trace records completed stages, selected mask, sparse-structure occupancy and geometry metrics, Gaussian count, Gaussian xyz ranges, and opacity quality. Inspect it with:

```bash
python scripts/sam3d/inspect_trace.py outputs/sam3d/<run>/trace.json
```

## Quality Gates

Strict quality mode blocks before writing junk outputs when sparse occupancy is saturated, geometry collapses to a flat axis, or Gaussian opacity/geometry metrics are outside the configured nominal bands.

This matters because an inference can still produce a bad object for a bad input or mask. The runtime blocks those outputs.

## PLY Coordinates

SAM3D sparse structure uses grid axes in depth, height, width order. The Gaussian PLY writer preserves the pipeline coordinate order used by the decoder and renderer. A generic 3DGS viewer may interpret PLY columns as a different world convention, so a valid PLY can appear rotated or axis-swapped in a third-party viewer.

Do not "fix" this by swapping only at PLY write time unless the full camera/world convention is also handled consistently.

## Edge Blur

Some blur or softness around Gaussian edges is normal. The PLY stores Gaussian splats with opacity, scaling, rotation, and color fields; it is not a watertight mesh. Blurry boundaries become suspicious when the trace also reports saturated sparse occupancy, flat geometry, extreme opacity, non-finite fields, or an obviously wrong mask.
