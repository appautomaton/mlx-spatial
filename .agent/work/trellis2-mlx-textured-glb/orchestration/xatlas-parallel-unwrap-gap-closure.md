# Xatlas Parallel Unwrap Gap Closure

## Gap Identified

The high-detail `200000` face GLB export was not memory-bound. It stayed CPU-bound inside one global `xatlas::ComputeCharts` job for over 20 minutes while using only a few cores. On a many-core Mac, that looked like low total CPU utilization.

## Changes

- Replaced the live bake path's simple `xatlas.parametrize(...)` call with an `xatlas.Atlas.generate(...)` wrapper.
- Added unwrap metrics:
  - backend,
  - input/output vertex and face counts,
  - elapsed seconds,
  - chunk count,
  - chart count,
  - atlas utilization when available.
- Added spatial chunking for high-face meshes:
  - split faces by centroid along spatial axes,
  - compact each chunk,
  - run xatlas per chunk in a thread pool,
  - pack chunk UV islands into one atlas,
  - merge remapped vertices/faces/UVs for GLB export.
- Added `generate-textured --xatlas-parallel-chunks`.
  - `0` auto-selects chunking by face count.
  - positive values force a chunk count.

## Verification

- Focused tests:

```sh
uv run pytest -q tests/test_trellis2_export.py tests/test_trellis2_inference.py tests/test_trellis2_tools.py
```

Result: `75 passed`.

- Full suite:

```sh
uv run pytest -q
```

Result: `243 passed, 5 skipped`.

- Diff hygiene:

```sh
git diff --check
```

Result: passed.

- Synthetic xatlas smoke:

```text
SMOKE_XATLAS faces 28322 backend xatlas-parallel-spatial chunks 4 out_faces 28322 charts 5 elapsed 0.048 uv_min 0.009999999776482582 uv_max 0.9900000095367432
```

- Live default run after the wrapper change:

```sh
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-512-textured.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --slat-steps 1 --decoder-token-limit 1000000 --texture-size 1024
```

Result: completed through `mesh-export` in `1:37.58`, wrote `3361692` bytes.

- Live high-detail chunked run:

```sh
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-512-textured-200k-parallel.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --slat-steps 1 --decoder-token-limit 1000000 --texture-size 1024 --glb-target-faces 200000 --xatlas-face-guard 250000 --xatlas-parallel-chunks 4
```

Result: completed through `mesh-export` in `2:54.14`, wrote `8517700` bytes.

- High-detail GLB inspection:

```text
GLB outputs/trellis2/image-512-textured-200k-parallel.glb glTF 2 8517700 meshes 1 materials 1 images 2 textures 2
IMAGE 0 (1024, 1024) alpha_nonzero 1.000000 rgb_nonzero 1.000000
IMAGE 1 (1024, 1024) alpha_nonzero 1.000000 rgb_nonzero 1.000000
TRIANGLES 171114 attrs ['POSITION', 'TEXCOORD_0']
```

- Blender import:

```text
GLB_OK 1 1 2 main_vertices 252087 main_faces 171114
```

## Remaining Gap

Chunked xatlas unwrap improves practical CPU use and avoids the single long global charting job, but it introduces additional UV seams compared with one global xatlas atlas. The next quality/performance step is still a Metal unwrap/bake path or a more topology-aware chunking strategy.
