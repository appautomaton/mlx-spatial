# mlx-spatialkit Native Simplification Parity Spec

## Bounded Goal

Replace the `face-stride-preview` simplifier with a native geometry-aware simplification baseline and add reference-parity metrics so `mlx-spatialkit` can move toward production Pixal3D GLB export quality without hiding remaining gaps.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: native hot paths, quality-first geometry, explicit diagnostics, real fixture proof, and no successful tests that mask preview-only behavior.

This change is Phase 2 progress. It does not claim full upstream remeshing parity unless the evidence proves it; it removes the most obvious geometry stub and creates the metrics needed to evaluate the next remesh/unwrap/texture improvements.

## Work Scale And Shape

- Scale: capability hardening
- Shape: native algorithm + parity metrics + real-fixture verification

## Selected Lenses

- **engineering:** Native C++ owns simplification; Python stays orchestration and tests.
- **runtime:** Real Pixal3D meshes have millions of faces, so the first quality-aware algorithm must be near-linear or otherwise bounded.
- **product:** Users should no longer see an export called successful while its mesh was produced by arbitrary face sampling.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers who need a local native GLB export path that is honest about quality and steadily converges toward the intended Pixal3D output.

## Evidence From Current Repo

- `packages/mlx-spatialkit/cpp/simplify.cpp` currently selects evenly spaced source faces, compacts vertices, and reports `face-stride-preview`.
- Latest hardening run improved texture visibility, but diagnostics still report `production_quality_ready=false` because the simplifier is preview-tier.
- The reference Pixal3D trace at `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json` reports:
  - source mesh faces: `8,304,022`
  - final faces: `212,542`
  - final vertices: `95,281`
  - unwrap backend: `xatlas-parallel-spatial`
  - bake backend: `xatlas-kdtree`
  - raw coverage: about `0.413`
  - final coverage: `1.0`
- Current spatialkit real fixture export uses `50,000` faces and preview simplification.

## Required Outcome

1. A native geometry-aware simplification backend replaces `face-stride-preview` as the default.
2. Simplifier diagnostics expose backend, algorithm, quality tier, target/source/final face counts, component/topology summaries, and whether the target was reached.
3. Tests prove the new simplifier uses all source geometry coherently, not arbitrary face striding.
4. Real-fixture diagnostics compare spatialkit output against the checked-in Pixal3D reference trace on face count, coverage, simplifier backend, and readiness tier.
5. Production readiness remains false unless the evidence meets the documented production threshold.

## Constraints

- `mlx-spatialkit` remains independent from MLX, Torch, and root `mlx-spatial` runtime imports.
- No Python per-face or per-vertex simplification hot loops.
- The first replacement algorithm must be memory-aware for multi-million-face fixtures.
- Heavy GLBs, diagnostics, extracted textures, and benchmark output stay under `/tmp`.
- Do not tag, release, publish, or push.

## Risks

- **Algorithm quality risk:** A simple native clustering algorithm may be better than face stride but still below production remesh quality. Mitigation: label the tier honestly and add reference metrics.
- **Runtime risk:** Full QEM edge-collapse can be too slow for 8M faces in the first pass. Mitigation: use a bounded native spatial-clustering baseline first, then make QEM a follow-up if needed.
- **Topology risk:** Clustering can introduce degenerates or nonmanifold edges. Mitigation: reuse cleanup/metrics after simplification and gate tests on blockers.
- **Scope risk:** Upstream parity also depends on unwrap and bake. Mitigation: this change improves simplification and parity measurement only; remaining unwrap/bake gaps stay explicit.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| SKP-01 | Default simplifier is no longer `face-stride-preview`. | `tests/test_mesh_processing.py` asserts the new backend/algorithm and fails on face-stride labeling. |
| SKP-02 | Simplifier is geometry-aware and native-owned. | C++ implementation consumes all vertices/faces through native code; tests use structured meshes where face-stride would leave fragmented output. |
| SKP-03 | Simplifier diagnostics are actionable. | Stats include backend, algorithm, quality tier, source/target/final face counts, target reached flag, and topology summary fields. |
| SKP-04 | Real fixture exports under `/tmp` with the new simplifier. | Heavy test records the new backend and keeps GLB/diagnostics under `/tmp`. |
| SKP-05 | Reference trace metrics are loaded and compared. | Tests compare real fixture output with `inputs/mlx-spatialkit/pixal3d-1024-cascade-glb-reference/trace.json` on face count and coverage fields. |
| SKP-06 | Production readiness is honest. | Diagnostics keep `production_quality_ready=false` unless parity thresholds are actually met. |
| SKP-07 | Repo/package cleanliness holds. | Full package tests, heavy test, root smoke test, build artifact check, and `git status --short` confirm no generated artifacts are tracked. |

## Scope Coverage Decisions

- **Included:** native simplifier replacement, simplifier diagnostics, synthetic geometry tests, real-fixture reference metrics, docs if behavior wording changes, package/root verification.
- **Deferred:** exact QEM production remesher if spatial clustering is insufficient, 1M-face/4096-texture settings, replacing face-atlas UVs with production xatlas unwrap, final texture parity beyond current fallback/fill path.
- **Anti-goals:** Python simplification loops, pretending the first geometry-aware simplifier is production-grade without evidence, release/tag work, or checked-in generated GLBs.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: a native spatial vertex-clustering simplifier is the right first replacement because it is bounded and memory-aware for 8M-face fixtures; if verification shows it is insufficient, the next cycle should implement a stronger QEM/remesh backend.
