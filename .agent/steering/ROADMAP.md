# Roadmap

## Phase 1: Spatialkit Export Quality Hardening

- status: done
- change: `2026-05-27-mlx-spatialkit-export-quality-hardening`
- objective: Make the real Pixal3D 1024 cascade turtle fixture export as a visually coherent colored GLB and make preview-quality limitations explicit.
- why now: The native GLB core is verified, but the current output has only about 1.15% visible base-color coverage and the simplifier is still preview-quality.
- likely outputs: Native texture coverage diagnostics, native texture fill/fallback, embedded GLB texture quality tests, real-fixture heavy gate, simplifier quality-tier diagnostics, docs.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-export-quality-hardening/SPEC.md`
- exit signal: Fast package tests, heavy real fixture test, package artifact cleanliness check, and docs verification pass with generated artifacts under `/tmp`.

## Phase 2: Production Remesh And Texture Parity

- status: active
- change: `2026-05-27-mlx-spatialkit-production-remesh-parity`
- objective: Move spatialkit toward upstream-style production export quality with quality-aware native simplification/remeshing, higher face targets, and stronger texture parity.
- why now: Deferred from the immediate sparse-color fix because full 1M-face / 4096-texture parity is larger than this hardening cycle.
- likely outputs: Bounded native QEM/remesh implementation or equivalent, production quality tier, reference GLB/trace comparison, runtime/memory benchmarks.
- evidence: `.agent/work/2026-05-27-mlx-spatialkit-export-quality-hardening/SPEC.md`
- exit signal: Spatialkit real-fixture output is visually comparable to the intended Pixal3D export path at production-quality settings.

## Deferred or Not Now

- Release, tag, publish, or push work is explicitly not part of this roadmap cycle.
