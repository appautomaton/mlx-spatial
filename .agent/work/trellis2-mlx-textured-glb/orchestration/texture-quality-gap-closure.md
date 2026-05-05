# Texture Quality Gap Closure

## Gap Identified

The first live GLB was structurally valid but not quality-valid enough:

- The base-color texture was only `256x256`.
- Roughly one third of texels had useful coverage, leaving black/transparent atlas holes.
- The bake used the deterministic internal face atlas rather than an upstream-style unwrap/refinement path.
- The GLB imported in Blender, but visual texture quality was limited by export/bake behavior rather than by file-format validity.

## Changes

- Filled uncovered atlas texels with a deterministic nearest-covered-texel pass before PNG export.
- Gamma-corrected base color for GLB texture output.
- Added `--texture-size` to `generate-textured`.
- Raised the default textured export size to `1024`.
- Added guards and tests for invalid texture sizes.

## Verification

- `uv run pytest -q tests/test_trellis2_export.py tests/test_trellis2_inference.py tests/test_trellis2_tools.py` -> `69 passed`.
- `uv run pytest -q` -> `235 passed, 5 skipped`.
- `git diff --check` -> passed.
- Live command:

```sh
uv run mlx-spatial-trellis2 generate-textured weights/trellis2 inputs/trellis2/image.png --output outputs/trellis2/image-textured.glb --pipeline-type 512 --seed 42 --rmbg-root weights/rmbg2 --slat-steps 1 --decoder-token-limit 1000000
```

- Artifact: `outputs/trellis2/image-textured.glb`
- Bytes: `169822388`
- Embedded base-color texture: `1024x1024` RGBA, 100% nonzero alpha, 100% nonzero RGB.
- Blender headless import: `GLB_OK 1 1 2`.

## Remaining Gaps

- The internal per-face atlas duplicates vertices and produces very large GLBs.
- Upstream-quality UV unwrap or an xatlas-equivalent packer is not implemented.
- Upstream mesh cleanup/fill-holes and optional simplification are not implemented.
- Texture sampling is still a deterministic Mac-native exporter path, not a full parity port of every upstream texture-baking utility.
- Live verification is still centered on the 512 path.
