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

- status: pending
- change:
- objective: Improve the native remesh, UV, and texture path until spatialkit real-fixture output is visually comparable to the intended Pixal3D export path.
- why now: Deferred from the readiness-gate cycle because current evidence shows face-count/topology can pass while preview-tier backend and global texture coverage still fail production thresholds.
- likely outputs: Non-preview native remesh backend or equivalent, higher UV utilization, stronger Metal texture bake/fill path, production quality tier, runtime/memory benchmarks.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-production-remesh-parity/PLAN.md`
- exit signal: Reference-target heavy fixture passes production thresholds or records a narrower verified blocker for the next native backend cycle.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
