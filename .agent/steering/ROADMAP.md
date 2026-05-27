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

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
