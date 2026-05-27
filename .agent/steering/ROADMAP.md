# Roadmap

## Phase 1: Spatialkit Export Quality Hardening

- status: done
- change: `2026-05-27-mlx-spatialkit-export-quality-hardening`
- objective: Make the real Pixal3D 1024 cascade turtle fixture export as a visually coherent colored GLB and make preview-quality limitations explicit.
- why now: The native GLB core is verified, but the current output has only about 1.15% visible base-color coverage and the simplifier is still preview-quality.
- likely outputs: Native texture coverage diagnostics, native texture fill/fallback, embedded GLB texture quality tests, real-fixture heavy gate, simplifier quality-tier diagnostics, docs.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-export-quality-hardening/SPEC.md`
- exit signal: Fast package tests, heavy real fixture test, package artifact cleanliness check, and docs verification pass with generated artifacts under `/tmp`.

## Phase 2: Reference-Target Readiness Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-production-remesh-parity`
- objective: Add a Pixal3D reference-target preset and threshold-gated production-readiness diagnostics so preview output cannot be mistaken for production parity.
- why now: The preview spatial-cluster export is artifact-ready, but it needs a real reference-target gate before deeper native remesh/UV/texture work can be judged honestly.
- likely outputs: Reference-target preset, threshold diagnostics, explicit native geometry blocker, real-fixture heavy gate, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-production-remesh-parity/SPEC.md`
- exit signal: Reference-target heavy fixture passes as artifact-ready but keeps production readiness false with explicit threshold failures and `/tmp` artifacts.

## Phase 3: Native Remesh And Texture Parity

- status: done
- change: `2026-05-27-mlx-spatialkit-native-atlas-coverage-parity`
- objective: Improve native atlas utilization so reference-target Pixal3D exports move closer to production texture coverage without external unwrap dependencies.
- why now: Current evidence shows face-count/topology pass, but global final coverage fails because one-triangle-per-tile face atlas wastes texture area.
- likely outputs: Native paired-triangle atlas packing, UV utilization diagnostics, real-fixture coverage gate, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-native-atlas-coverage-parity/SPEC.md`
- exit signal: Reference-target heavy fixture shows final visible coverage around `0.602` under `/tmp`; production readiness remains threshold-controlled and false because the backend tier is still preview.

## Phase 4: Native Remesh Backend Tier

- status: done
- change: `2026-05-27-mlx-spatialkit-native-geometry-backend-tier`
- objective: Replace or augment `spatial-cluster` with a non-preview native remesh/simplification backend so the reference-target export can pass backend-tier readiness.
- why now: Current reference-target evidence shows face count, topology, coverage, and raw reporting pass; backend tier is the only remaining measured production blocker.
- likely outputs: Native topology-aware geometry backend, explicit backend selection contract, production-tier self-diagnostics, heavy reference-target gate, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-native-geometry-backend-tier/SPEC.md`
- exit signal: Reference-target export reports `topology-aware`, `quality_tier=production`, all production thresholds pass, and `production_quality_ready=true` on the heavy decoded fixture under `/tmp`.

## Phase 5: Reference Visual Parity Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-reference-visual-parity-gate`
- objective: Add deterministic GLB visual-comparison diagnostics and reviewer-friendly sidecar artifacts for spatialkit reference-target exports against the checked-in Pixal3D reference GLB.
- why now: The scalar production gate passes, but the broader goal still needs visual comparability evidence rather than only threshold readiness.
- likely outputs: GLB/PNG inspection utility, visual parity JSON/HTML sidecar, reference-target diagnostics integration, heavy fixture gate, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-reference-visual-parity-gate/SPEC.md`
- exit signal: Reference-target heavy fixture writes visual comparison artifacts under `/tmp` and diagnostics show face, texture, and coverage comparability against the reference GLB.

## Phase 6: Peak Memory Telemetry Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-peak-memory-telemetry-gate`
- objective: Add stage-level peak host-memory telemetry to Pixal3D exports so diagnostics report observed RSS peaks during major native export stages.
- why now: The export quality gates are stronger, but sparse boundary memory samples can still under-explain real process memory seen by local monitors.
- likely outputs: Thread-safe process RSS monitor, per-stage memory peak diagnostics, heavy fixture assertions, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-peak-memory-telemetry-gate/SPEC.md`
- exit signal: Heavy reference-target export writes `diagnostics.memory.stage_peaks` under `/tmp`, with full package/root/build verification passing.

## Phase 7: Browser Render Visual Proof Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-browser-render-visual-proof-gate`
- objective: Add dev-only browser-rendered screenshot proof for Pixal3D reference-target GLBs so visual comparability is backed by real browser rendering, not only GLB/texture metrics.
- why now: The existing visual parity report passes deterministic checks but still explicitly defers browser-rendered visual proof.
- likely outputs: Playwright/Three render script, `/tmp` screenshot/report artifacts, real fixture browser proof, augmented visual parity report, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-browser-render-visual-proof-gate/SPEC.md`
- exit signal: Real reference-target export renders candidate/reference GLBs in Chrome, writes browser render artifacts under `/tmp`, and removes the browser-rendered proof deferral from the generated visual parity report.

## Phase 8: 4096 Texture Coverage Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-4096-texture-coverage-gate`
- objective: Make native Pixal3D reference-target exports pass production texture coverage at `texture_size=4096` without relaxing thresholds.
- why now: A 4096 probe writes a valid GLB but fails production coverage because fixed 8-pass dilation underfills larger atlas tiles.
- likely outputs: Adaptive native dilation budget, 4096 heavy fixture gate, memory diagnostics evidence, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-4096-texture-coverage-gate/SPEC.md`
- exit signal: 4096 reference-target export under `/tmp` reports `production_quality_ready=true` with final coverage above threshold while xatlas and 1M-face boundaries remain explicit.

## Phase 9: Parity Boundary Coherence Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-parity-boundary-coherence-gate`
- objective: Make visual-comparison deferred parity boundaries reflect only production gaps that remain open after the 4096 texture and browser-render proof gates.
- why now: `glb_compare.py` still emits stale `not_4096_texture_parity` and `not_browser_rendered_visual_proof` labels even though those gates are verified done.
- likely outputs: Updated boundary labels, focused and heavy diagnostics assertions, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-parity-boundary-coherence-gate/SPEC.md`
- exit signal: Visual-comparison diagnostics retain only xatlas chart parity and 1M-face export-setting parity as deferred boundaries, with focused/heavy tests and docs aligned.

## Phase 10: Upstream Settings Readiness Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-upstream-settings-readiness-gate`
- objective: Make explicit Pixal3D upstream export settings (`target_faces=1000000`, `texture_size=4096`) artifact-ready and coverage-ready in spatialkit diagnostics without claiming xatlas chart parity.
- why now: A `/tmp` 1M/4096 probe reaches `artifact_ready=true` and `target_reached=true`, but texture coverage is still below threshold and diagnostics do not separately prove upstream-setting readiness.
- likely outputs: High-density native texture fill floor, upstream-setting readiness diagnostics, real 1M/4096 heavy gate, visual boundary filtering, docs, package/root verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-upstream-settings-readiness-gate/SPEC.md`
- exit signal: Explicit 1M/4096 export under `/tmp` reports upstream-setting readiness true, removes only the 1M-face setting deferral, and keeps xatlas chart parity deferred.

## Phase 11: GLB Viewer Compatibility Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-glb-viewer-compatibility-gate`
- objective: Make native Pixal3D GLBs structurally friendlier to macOS Preview/Quick Look and strict viewers by adding normals, chunking large primitives into uint16-indexed local primitives, and reporting compatibility readiness.
- why now: Browser-rendered proof passes, but user-visible Preview behavior can still look point-like or uncolored because the native GLB currently has no normals and a single large uint32-indexed primitive.
- likely outputs: Native normal generation, uint16 primitive chunking, viewer-compatibility diagnostics, real-fixture gate, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-glb-viewer-compatibility-gate/SPEC.md`
- exit signal: Real reference-target export under `/tmp` reports GLB viewer compatibility ready with normals and uint16-only primitives while xatlas chart parity remains deferred.

## Phase 12: UV Raster Binning Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-uv-raster-binning-gate`
- objective: Make arbitrary chart UV texture baking production-viable by replacing the non-atlas all-faces-per-pixel scan with a bounded UV-space face-bin Metal path and diagnostics.
- why now: The remaining xatlas/chart parity boundary cannot be closed honestly while arbitrary UV meshes would fall back to an O(texture_pixels * faces) scan.
- likely outputs: CPU UV bin construction, Metal binned UV raster lookup, bin diagnostics and guards, focused/stress tests, real fixture regression, docs.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-uv-raster-binning-gate/SPEC.md`
- exit signal: Non-atlas UV bake reports `metal-uv-binned-nearest` with bounded candidate counts, while existing face-atlas and heavy Pixal3D gates still pass and xatlas chart parity remains explicit.

## Phase 13: Native Chart UV Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-native-chart-uv-gate`
- objective: Add an opt-in native chart-UV generator that groups connected smooth faces into packed charts and bakes through the binned Metal UV path without claiming xatlas parity.
- why now: Phase 12 made arbitrary UV raster baking scalable; the next production gap is native chart generation itself before real Pixal3D exports can move away from face-atlas UVs.
- likely outputs: Native chart UV binding/API, chart diagnostics, focused chart/crease tests, binned texture bake proof, docs, heavy regression.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-native-chart-uv-gate/SPEC.md`
- exit signal: `make_native_chart_uvs` produces charted UV meshes with truthful diagnostics, bakes through `metal-uv-binned-nearest`, and existing Pixal3D gates remain unchanged.

## Phase 14: Chart UV Export Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-chart-uv-export-gate`
- objective: Add an opt-in native chart UV backend to `export_pixal3d_glb` and prove it on the real Pixal3D decoded fixture without changing the default face-atlas backend.
- why now: The chart UV primitive is verified, but it is not yet wired into the real Pixal3D export path or proven with real fixture diagnostics.
- likely outputs: UV backend API contract, chart backend diagnostics/metadata, real fixture heavy gate, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-export-gate/SPEC.md`
- exit signal: Real fixture export with `uv_backend="native-chart"` writes a GLB under `/tmp`, bakes through `metal-uv-binned-nearest`, and keeps xatlas chart parity deferred.

## Phase 15: Chart UV Readiness Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-chart-uv-readiness-gate`
- objective: Make native chart UV export diagnostics distinguish artifact readiness from quality readiness, with explicit coverage and UV-utilization blockers.
- why now: The chart backend writes a GLB, but the real fixture shows low global texture coverage; successful tests must not hide that quality gap.
- likely outputs: Chart readiness summary, chart-specific warnings, real fixture quality-blocked assertions, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-readiness-gate/SPEC.md`
- exit signal: Real chart export reports artifact-ready but quality-blocked with failed coverage/utilization checks and xatlas chart parity still deferred.

## Phase 16: Chart UV Shelf Packing Gate

- status: done
- change: `2026-05-27-mlx-spatialkit-chart-uv-shelf-packing-gate`
- objective: Replace native chart equal-grid packing with deterministic aspect-aware shelf packing and prove real Pixal3D chart UV occupancy improves.
- why now: Angle probes did not improve chart coverage; the verified blocker is atlas utilization from the current chart packer.
- likely outputs: Native shelf packer, packing diagnostics, focused chart tests, real fixture occupancy gate, docs, package/root/build verification.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-chart-uv-shelf-packing-gate/SPEC.md`
- exit signal: Real chart export improves UV-surface occupancy above the Phase 15 baseline while default face-atlas exports remain unchanged.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
