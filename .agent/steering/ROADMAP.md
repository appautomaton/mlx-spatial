# Roadmap

## Phase 37: Texture Gutter Fill

- status: done
- change: `2026-05-27-mlx-spatialkit-texture-gutter-fill`
- objective: Add native post-bake texture gutter fill so linear-filtered GLB viewers do not bleed black no-face texels around UV islands.
- why now: Geometry holes and repair policy are now measured; the next visual-quality risk is texture seam robustness in the native export path.
- likely outputs: Native gutter fill, diagnostics, focused texture test, real Pixal3D heavy gate, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-texture-gutter-fill/SPEC.md`
- exit signal: Gutter fill improves no-face RGB/MR seam texels while preserving UV-surface and visible-alpha coverage semantics, and the real reference-target native-chart export stays green.

## Current State

- Last verified change: `2026-05-27-mlx-spatialkit-texture-gutter-fill`.
- Prior geometry-hole investigation/reduction is consolidated into the current native export line: boundary-loop diagnostics are present, triangular small-loop repair is configurable as `small_boundary_loop_fill_max_edges=3`, and `0` disables it for comparison runs.
- Current reference-target native-chart baseline: final closed boundary loops `1872`, nonmanifold edges `0`, UV-surface occupancy `0.5673904418945312`, xatlas-utilization ratio `0.6828063257125282`, and xatlas chart parity remains false.
- Texture bake now fills a bounded no-face RGB/MR gutter around UV islands for linear-filter seam robustness while preserving alpha, coverage status, UV-surface counts, and visible-coverage ratios. Visual comparison separates raw RGB footprint from visible RGB coverage. Latest real-fixture diagnostics showed `gutter_filled_texel_count=453197`.
- Explicit 1M/4096 native-chart exports are upstream-setting ready, with xatlas parity still the open quality-equivalence gap.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
- Zero-padding default switch is deferred until seam behavior is safer and separately justified.
