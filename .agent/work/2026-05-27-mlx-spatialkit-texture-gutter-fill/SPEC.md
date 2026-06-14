# mlx-spatialkit Texture Gutter Fill Spec

## Bounded Goal

Add a native post-bake texture gutter fill that colors no-face padding texels around UV islands without inflating UV-surface coverage metrics.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant native hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and visually comparable Pixal3D exports before production readiness is claimed.

## Work Scale And Shape

- **Scale:** medium native hardening slice.
- **Shape:** visual-quality robustness in the C++/Metal texture export path, with diagnostics and real-fixture verification.

## Selected Lenses

- **engineering:** Fix a concrete texture-export seam risk instead of hiding it behind coverage pass/fail.
- **runtime:** Keep the hot path native-owned, bounded, deterministic, and memory-aware.
- **product:** Improve macOS Preview/GLB viewer visual robustness without claiming xatlas unwrap parity.

## Current Evidence

- Native-chart reference-target exports pass production gates, but xatlas utilization parity remains false: latest default ratio is about `0.6828063257125282`.
- A `/tmp` probe showed `tile_padding=0.0` raises utilization ratio to about `0.6855710536883002`, but zero padding can increase linear-filter seam risk.
- The GLB writer emits linear texture samplers (`magFilter=9729`, `minFilter=9729`), so no-face texels adjacent to UV islands can bleed black/transparent RGB into rendered edges.
- Existing surface fill colors missing UV-surface texels; it does not color no-face gutter texels outside UV triangles.

## Required Outcome

1. Native texture bake fills a bounded no-face gutter around visible texels after UV-surface fill.
2. Gutter fill copies RGB and metallic/roughness data but does not make no-face texels count as UV surface or visible alpha coverage.
3. Visual comparison keeps raw RGB footprint separate from visible RGB coverage so transparent gutter RGB cannot inflate pass/fail coverage.
4. Diagnostics report whether gutter fill ran, pass count, filled texel count, and no-face accounting.
5. Focused tests prove gutter RGB/MR fill improves neighboring no-face texels while preserving coverage semantics.
6. Real Pixal3D reference-target native-chart export still passes quality, viewer, visual, memory, and xatlas-honesty gates.
7. Docs explain gutter fill as seam robustness, not xatlas parity or a UV-utilization fix.
8. Heavy/generated artifacts stay under `/tmp`; no push, tag, publish, or release metadata work.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| TGF-01 | Gutter fill is native, bounded, and deterministic. | Native code has a pass-limited no-face gutter fill after surface fill. |
| TGF-02 | Coverage metrics stay honest. | Tests assert gutter-filled no-face texels do not increase `uv_surface_texel_count` or alpha-visible coverage. |
| TGF-03 | Visual comparison does not treat transparent gutter RGB as visible coverage. | PNG/GLB comparison tests separate raw RGB footprint from visible RGB coverage. |
| TGF-04 | Diagnostics expose the policy. | Texture stats include `gutter_fill_enabled`, `gutter_filled_texel_count`, and `gutter_fill_pass_count`. |
| TGF-05 | Focused texture test covers seam behavior. | Unit test observes nonzero RGB/MR in no-face gutter texels with unchanged alpha/coverage counts. |
| TGF-06 | Real fixture remains green. | Reference-target native-chart heavy test passes with existing xatlas non-parity assertions. |
| TGF-07 | Docs/roadmap are current. | Spatialkit/Pixal3D docs describe texture gutter fill and the remaining xatlas boundary. |
| TGF-08 | Repo/package hygiene holds. | Focused tests, heavy test, package/root tests, and `/tmp` build inspection pass. |

## Scope Coverage Decisions

- **Included:** native post-bake gutter fill, stats, focused tests, heavy Pixal3D verification, docs, compact roadmap update.
- **Deferred:** zero-padding default switch, xatlas-equivalent unwrap, full perceptual scoring, GPU kernel rewrite for the gutter pass.
- **Anti-goals:** inflating coverage metrics, changing decoded model outputs, changing GLB sampler filtering, claiming xatlas parity.

## Constraints And Risks

- Keep no-face texels outside UV-surface accounting even when their RGB/MR values are filled.
- Keep alpha unchanged for gutter texels so existing visible-coverage metrics stay comparable.
- Bound the pass count to avoid unbounded texture expansion and keep memory behavior predictable.
- CPU-native postprocess is acceptable for this slice because the existing surface-fill postprocess is already native CPU; a Metal rewrite remains deferred unless this becomes a measured bottleneck.
