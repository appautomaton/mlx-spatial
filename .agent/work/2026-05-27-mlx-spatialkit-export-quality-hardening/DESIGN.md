# mlx-spatialkit Export Quality Hardening Design

## Quality Failure

The current native path is fast and structurally valid, but visually weak:

```text
Pixal3D decoded texture voxels
        |
        v
Metal bake: one rounded voxel lookup per UV texel
        |
        +-- exact hit -> colored texel
        +-- miss      -> zero RGBA texel
        |
        v
GLB writer embeds sparse baseColor PNG unchanged
```

The real turtle fixture produced only about `1.15%` visible base-color texels, while tests accepted `sampled_texel_count > 0`.

## Target Shape

Keep Python thin and make native quality explicit:

```text
UV raster texels
  -> exact sparse voxel sample
  -> bounded native fallback/fill for misses
  -> final coverage diagnostics
  -> embedded GLB baseColor verification
```

Texture diagnostics must separate raw exact hits from final visible coverage so speed and quality tradeoffs are visible.

## Texture Strategy

Implement the first production-useful fix in native code:

1. Preserve the current Metal exact-sample pass and coverage status.
2. Add native fallback/fill for missing surface texels.
   - Preferred first implementation: native tile-aware or status-aware dilation/fill over UV-rasterized surface texels.
   - Acceptable follow-up: sparse neighbor/trilinear model-space fallback where exact voxel lookup misses.
3. Track status categories:
   - `no_face`
   - `exact_hit`
   - `fallback_filled`
   - `missing_after_fallback`
   - `out_of_grid`
4. Report:
   - `uv_surface_texel_count`
   - `exact_sampled_texel_count`
   - `fallback_filled_texel_count`
   - `missing_texel_count`
   - `visible_base_color_texel_count`
   - `raw_coverage_ratio`
   - `final_visible_coverage_ratio`

This can make the GLB visually coherent before implementing a full xatlas/kdtree parity path.

## Mesh Simplification Strategy

The current simplifier is a deterministic face-stride reducer. That is acceptable only as a preview simplifier if diagnostics say so.

This change should not hide that limitation:

- Rename/report the backend as `face-stride-preview`.
- Add `quality_tier: preview`.
- Add warning/export metadata so `ready=True` means "artifact written", not "production-quality remesh parity".
- Add tests that prevent the face-stride reducer from being labeled production or quality-aware.

A QEM/native remesher is the likely next phase, but it is not required to fix the immediate sparse-color turtle output.

## Verification Principle

Do not verify only `glTF` magic bytes. For the real fixture, verify the embedded GLB baseColor PNG payload or equivalent diagnostics:

```text
model.glb
  -> JSON chunk
  -> baseColor image bufferView
  -> PNG decode / coverage count
  -> reject sparse-dot output
```

Generated GLBs and extracted PNGs stay under `/tmp`.
