# 2026-05-27 mlx-spatialkit Legacy Rollup

This file consolidates the 55 `2026-05-27-mlx-spatialkit-*` work directories as historical evidence for the active Pixal3D GLB quality rebaseline. The old folders stay intact and are not edited; this rollup is the working index.

## Canonical Takeaway

The 2026-05-27 work was one debugging chain, not a durable backlog. It moved `mlx-spatialkit` from decoded Pixal3D NPZ fixtures toward native GLB output, then exposed the harder truth: the backend can emit valid and browser-renderable GLBs, but the outputs are still not production-quality GLBs.

The remaining blocker is production-quality native geometry/export parity, not more readiness plumbing. The current problem is concentrated in open-boundary topology, missing true QEM edge-collapse, missing narrow-band remesh, non-reference unwrap behavior, and secondary texture/postprocess/normal effects that become hard to interpret while topology remains rough.

## Consolidated Buckets

| Bucket | Historical work dirs | What they proved |
|---|---|---|
| Native GLB core and runtime | `native-glb-core`, `metal-gil-release`, `peak-memory-telemetry-gate` | The native package boundary, decoded NPZ contract, C++/Metal path, GLB writer, and heavy `/tmp` runtime policy are the right foundation. |
| Reference and production parity | `pixal3d-export-reference-port`, `production-remesh-parity`, `native-geometry-backend-tier`, `native-simplification-parity`, `native-atlas-coverage-parity`, `4096-texture-coverage-gate` | The reference path is larger than local tuning: production parity depends on remesh, QEM, unwrap, bake, and reference-scale settings. |
| UV and atlas behavior | `chart-uv-*`, `native-chart-*`, `low-fill-chart-split-gate`, `small-chart-splitting`, `split-position-search`, `xatlas-*` | Native charting improved enough for evidence, but it is not xatlas-equivalent and should not be treated as production unwrap parity. |
| Texture, bake, and materials | `export-quality-hardening`, `texture-gutter-fill`, `uv-raster-binning-gate`, `uv-surface-fill-gate`, `chart-normal-drift-gate`, `glb-viewer-compatibility-gate` | Texture coordinate order and PBR packing are mostly settled; fill/postprocess/render-padding still need attribution tests and must not hide geometry defects. |
| Geometry repair and holes | `geometry-hole-diagnostics`, `open-boundary-diagnostics`, `small-boundary-loop-fill`, `small-loop-fill-balance`, `repair-policy-contract`, `repair-cap-alignment`, `centroid-fan-hole-fallback`, `earclip-hole-repair`, `alternative-triangulation-repair`, `branched-cycle-repair`, `quad-loop-repair`, `native-chart-hole-reduction`, `parity-boundary-coherence-gate` | Small clean-loop filling helped, but residual simple/branched open chains and non-reference topology remain visual blockers. |
| Visual proof and readiness semantics | `reference-visual-parity-gate`, `browser-render-visual-proof-gate`, `rendered-visual-correctness`, `readiness-semantics`, `upstream-settings-readiness-gate`, `native-chart-reference-target-gate` | Browser/GLB comparison proof is useful evidence, but visible pixels or scalar comparison passes cannot mean production-quality readiness by themselves. |

## Decisions To Preserve

- Treat the old work dirs as forensic evidence, not as the active backlog.
- Do not fan the next implementation back into many micro-specs.
- Keep the working signals already established: Pixal3D `batch-x-y-z` texture coordinate order, PBR channel packing, flexible dual-grid extraction, manifest-bound A/B/C comparisons, and browser blank/framing proof.
- Keep readiness terms strict: `artifact_ready`, `rendered_visual_ready`, `browser_rendered_visual_proof`, `production_quality_ready`, and `production_equivalence_ready` are separate.
- Keep heavy GLBs, screenshots, browser renders, and scratch diagnostics under `/tmp`.

## Next Implementation Focus

After the rebaseline is verified, the next focused implementation should target production-quality geometry/export parity, with the following order:

1. Use current two-fixture diagnostics to choose the first topology fix that can improve real GLB quality.
2. Implement or explicitly block simple/branched open-chain repair, true QEM edge-collapse, and narrow-band remesh in that order of evidence.
3. Re-check UV unwrap, texture postprocess, material/normal, and viewer effects only after topology is no longer the dominant blocker.
4. Verify on both the base Pixal3D 1024 cascade fixture and the independent violin/bow lineage.

This is a focused production-quality path, not another readiness or screenshot-tuning loop.
