# mlx-spatialkit Reference Visual Parity Gate Design

## Current Gap

The reference-target gate now proves:

```text
production_quality_ready = true
backend = topology-aware
face_count_ratio ~= 0.934
final_coverage_ratio ~= 0.602
```

That is necessary but not enough for the broader goal. There is no durable
artifact that compares the generated GLB against the checked-in reference GLB in
a way a developer can inspect later.

## Comparator Shape

Add a small package-level GLB comparison module:

```text
candidate.glb
reference.glb
  -> parse GLB JSON + BIN
  -> summarize mesh/material/image structure
  -> extract embedded baseColor / metallicRoughness PNGs
  -> decode PNG coverage with standard filters
  -> compare metrics
  -> write report JSON + texture PNGs + preview HTML
```

This is post-export diagnostics, not an inference or export hot path. It should
avoid browser/Pillow dependencies and operate on existing GLB bytes.

## Export Integration

When `export_pixal3d_glb(..., quality_preset="reference-target")` has a
reference trace, resolve the sibling reference `model.glb`. After writing the
candidate GLB:

```text
output/
  model.glb
  diagnostics.json
  visual_parity/
    visual_parity.json
    index.html
    candidate_base_color.png
    reference_base_color.png
```

`diagnostics.json` should include a compact `visual_comparison` section with the
report path and pass/fail summary. The detailed report lives in the sidecar JSON.

## Metrics

Minimum report fields:

- candidate/reference mesh counts: vertices, indices, faces
- candidate/reference material/image counts
- base-color PNG dimensions, alpha coverage, RGB coverage
- face-count ratio
- base-color alpha/RGB coverage ratios versus reference
- texture-resolution match
- pass/fail checks
- deferred parity boundaries: xatlas chart parity, 4096 texture parity, 1M-face
  export-setting parity

## Readiness Boundary

The visual-comparison report can support the current production gate, but it is
not a browser-rendered visual proof. If later we need rendered screenshots, that
should be another cycle with an explicit browser/renderer dependency and visual
acceptance threshold.
