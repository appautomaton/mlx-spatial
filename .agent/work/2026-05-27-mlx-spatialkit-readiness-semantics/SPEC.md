# SPEC: mlx-spatialkit Readiness Semantics

## Bounded Goal

Make Pixal3D export diagnostics distinguish scalar native export quality from full Pixal3D production-equivalence readiness, so open parity boundaries cannot be hidden behind `production_quality_ready=true`.

## Broader Intent

This preserves the larger `mlx-spatialkit` goal: a dependable native export backend for `mlx-spatial` that is fast, memory-aware, honest about quality gaps, and not treated as production-ready before it is visually comparable to the intended Pixal3D export path.

## Work Scale And Shape

- scale: small/medium
- shape: diagnostics contract hardening with tests and docs
- selected lenses: engineering, runtime

## Constraints And Risks

- Keep existing `production_quality_ready` behavior as the scalar reference-target threshold signal unless a test proves it is unsafe to preserve.
- Add stricter production-equivalence diagnostics instead of renaming existing fields across the codebase.
- Do not add xatlas, CUDA, remesh, browser-render, or new generated fixture dependencies in this change.
- Heavy outputs stay under `/tmp`; repo docs and tests remain lightweight.
- Risk: if field names are vague, callers may still misread scalar readiness as full Pixal3D parity.

## Required Outcome

Pixal3D export diagnostics must expose a stricter equivalence summary that:

- reports whether the export is safe to treat as Pixal3D production-equivalent;
- lists remaining parity boundaries, including xatlas chart parity and upstream-setting/visual-comparison gaps when present;
- records scalar readiness, artifact readiness, upstream setting readiness, xatlas parity readiness, and deterministic visual-comparison readiness as separate checks;
- surfaces the stricter flag in `diagnostics["result"]` and GLB metadata;
- keeps existing scalar quality, native-chart, xatlas parity, and upstream-setting diagnostics intact.

## Acceptance Criteria

- Current reference-target exports may still report `production_quality_ready=true`, but must report `production_equivalence_ready=false` while `not_xatlas_chart_parity` or upstream-setting parity remains open.
- Native-chart reference-target diagnostics must show scalar native-chart readiness separately from xatlas equivalence readiness.
- Explicit 1M/4096 diagnostics must not remove xatlas as a remaining production-equivalence boundary.
- Unit tests cover the new equivalence helper and result contract.
- Real Pixal3D heavy fixture coverage checks the stricter field on at least the native-chart reference-target path.
- Docs explain `production_quality_ready` versus `production_equivalence_ready` without claiming production parity.

## Anti-Goals

- Do not implement xatlas, remesh, or new mesh repair in this change.
- Do not change the model inference path or decoded NPZ contracts.
- Do not push, tag, publish, or create release artifacts.
