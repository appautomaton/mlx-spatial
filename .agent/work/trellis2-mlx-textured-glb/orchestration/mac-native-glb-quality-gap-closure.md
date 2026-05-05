# Mac-Native GLB Quality Gap Closure

## Gap Identified

The first live textured GLB was Blender-readable but used the internal per-face atlas path. That made GLBs very large, duplicated mesh vertices heavily, and left the live export short of the `trellis-mac` fallback architecture: mesh cleanup, simplification, xatlas unwrap, and scipy KDTree texture baking.

## Changes

- Added Mac-native export dependencies: `xatlas`, `scipy`, and `fast-simplification`.
- Added an explicit dependency gate so live `generate-textured` blocks cleanly when export dependencies are unavailable.
- Added GLB mesh postprocess before texture baking:
  - remove degenerate faces,
  - remove duplicate faces,
  - compact unreferenced vertices,
  - remove tiny disconnected components,
  - fill bounded clean holes,
  - simplify before UV unwrap,
  - report boundary and non-manifold edge metrics.
- Added xatlas UV unwrap as the default live GLB UV backend.
- Added scipy `cKDTree` texture baking from decoder voxels with inverse-distance weights.
- Added UV gap fill through scipy ndimage.
- Extended `generate-textured` trace outputs with mesh postprocess, unwrap, bake coverage, and backend metadata.
- Added `--glb-target-faces` and `--xatlas-face-guard` so higher-detail unwraps can be attempted explicitly without making the live default an open-ended CPU job.

## Verification

- Focused tests:

```sh
uv run pytest -q tests/test_trellis2_export.py tests/test_trellis2_inference.py tests/test_trellis2_tools.py
```

Result: `73 passed`.

- Full suite:

```sh
uv run pytest -q
```

Result: `241 passed, 5 skipped`.

- Diff hygiene:

```sh
git diff --check
```

Result: passed.

- Live default command:

```sh
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-512-textured.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --slat-steps 1 --decoder-token-limit 1000000 --texture-size 1024
```

Result: completed through `mesh-export` in `1:36.26` and wrote `outputs/trellis2/image-512-textured.glb` with `3361692` bytes.

- Blender import:

```text
GLB_OK 1 1 2 main_vertices 74762 main_faces 45981
```

- Texture check:

```text
GLB glTF 2 3361692 meshes 1 materials 1 images 2 textures 2
IMAGE 0 image/png (1024, 1024) alpha_nonzero 1.000000 rgb_nonzero 1.000000
IMAGE 1 image/png (1024, 1024) alpha_nonzero 1.000000 rgb_nonzero 1.000000
PRIM 0 0 triangles 45981 mode 4 attrs ['POSITION', 'TEXCOORD_0']
```

## Runtime Finding

The plan's `200000` face target was tested with `--glb-target-faces 200000 --xatlas-face-guard 250000`. The process stayed CPU-bound inside `xatlas::ComputeCharts` for over 20 minutes while holding about 10 GB RSS. Memory was not the limiting factor, but the runtime is not acceptable as the default iteration path.

The shipped default is therefore the verified practical path: `50000` target faces and a `75000` xatlas guard. The higher-detail path remains available explicitly through the new flags.

## Remaining Gaps

- The Metal `trellis-mac` stack may still be needed for a faster high-detail unwrap/bake path.
- Full `cumesh`-equivalent complex topology repair is not implemented.
- Texture refinement is still a deterministic voxel bake, not a full parity port of every upstream texture polish stage.
- Live verification is still centered on the 512 route.
