# mlx-spatialkit 4096 Texture Coverage Gate Design

## Current Gap

The native Metal bake marks surface texels as:

```text
1 sampled exactly
2 missing
3 out of grid
4 fallback-filled
```

During the Metal pass, missing exact voxel samples use a bounded nearest-voxel
fallback radius. After the Metal pass, C++ dilation fills remaining surface
texels from nearby sampled/fallback texels. The current fixed budgets were
enough for the 1024 fixture but underfill larger 4096 atlas tiles.

## Adaptive Budgets

Resolve fill budgets from the effective atlas tile size:

```text
tile_pixels = texture_size / max(atlas_cols, atlas_rows)
fallback_radius = clamp(ceil(tile_pixels * 2), 12, 24)
dilation_passes = clamp(ceil(tile_pixels * 2), 8, 64)
```

Fallback for non-atlas UVs stays at the current lower budget:

```text
fallback_radius = 12
dilation_passes = 8
```

This keeps small textures unchanged while giving 4096 face-atlas exports enough
nearest fallback and dilation budget to fill larger in-tile gaps. The caps
prevent unexpected runaway GPU/CPU work.

## Diagnostics

Keep current fields and make them more useful:

- `dilation_max_passes`: resolved adaptive pass budget
- `dilation_pass_count`: actual passes that filled at least one texel
- `dilation_filled_texel_count`: texels filled by CPU dilation
- `fallback_radius`: resolved Metal nearest fallback radius

## Boundary

This is a texture coverage readiness fix for high-resolution native face-atlas
exports. It does not implement xatlas charting or 1M-face export parity.
