# mlx-spatialkit Native Geometry Backend Tier Spec

## Bounded Goal

Replace the reference-target Pixal3D export's preview-only `spatial-cluster` simplifier blocker with a distinct native topology-aware geometry backend whose production-tier claim is proven by synthetic topology tests and the real Pixal3D fixture gate.

## Broader Intent

`mlx-spatialkit` should become the dependable native export backend for `mlx-spatial`: quality-first geometry, performant C++/Metal hot paths, consistent diagnostics/contracts, memory-aware execution, real-fixture proof, and no hidden half-stubbed behavior behind passing tests.

This change is Phase 4. It targets the last currently measured production-readiness blocker after the native paired-atlas cycle: reference-target face count, topology, final coverage, and raw coverage reporting pass, but backend tier still fails because the simplifier reports `quality_tier=geometry_aware_preview`.

## Work Scale And Shape

- Scale: capability hardening
- Shape: native geometry algorithm + backend selection contract + real-fixture production gate

## Selected Lenses

- **engineering:** Native C++ owns production geometry work; Python only selects presets, passes backend intent, and records diagnostics.
- **runtime:** The reference fixture starts from about `8.3M` source faces and targets about `212k` final faces, so the backend must be bounded, memory-aware, and measured.
- **product:** A reference-target export should not remain blocked by a preview-tier label once measured geometry, topology, and texture thresholds pass.

## Target User Or Stakeholder

Apple Silicon `mlx-spatial` developers validating decoded Pixal3D NPZ artifacts into GLB output through the `mlx-spatialkit` companion backend.

## Current Evidence

- `packages/mlx-spatialkit/cpp/simplify.cpp` reports `backend=spatial-cluster`, `algorithm=native_spatial_vertex_clustering`, `quality_tier=geometry_aware_preview`, and `production_ready=false`.
- `packages/mlx-spatialkit/src/mlx_spatialkit/export.py` computes `production_quality_ready` from topology, face-count, coverage, raw-coverage reporting, reference availability, preset, and backend-tier checks. Backend tier passes only when simplifier `quality_tier == "production"`.
- Latest reference-target diagnostics under `/tmp` show:
  - target faces: `212542`
  - final faces: `198618`
  - face-count ratio: `0.9344882423238701`
  - final visible coverage: `0.6019515991210938`
  - topology blockers: none
  - backend-tier actual: `geometry_aware_preview`
  - production readiness: `false`
- The checked-in Pixal3D reference trace reports `final_faces=212542`, `final_vertices=95281`, `raw_coverage_ratio=0.41324901580810547`, `coverage_ratio=1.0`, `unwrap_backend=xatlas-parallel-spatial`, and `bake_backend=xatlas-kdtree`.

## Required Outcome

1. A native backend distinct from `spatial-cluster` is available for reference-target geometry simplification/remeshing.
2. The new backend reports an explicit algorithm, quality tier, production-readiness flag, source/target/final counts, topology cleanup counts, and bounded runtime/memory diagnostics.
3. Reference-target export selects the new backend only when it is available and keeps preview/default exports on the honest preview path.
4. A production-tier claim is not a label flip: tests must prove the backend is not aliased to `spatial-cluster`, consumes source geometry coherently, and passes topology/face-count/coverage gates on the real fixture before `production_quality_ready=true`.
5. If the new backend fails a measured gate, diagnostics must report the exact blocker and keep `production_quality_ready=false`; the old preview-tier blocker must not be hidden behind a renamed stub.
6. Heavy generated artifacts stay under `/tmp`; repo/package artifacts stay clean.

## Production Geometry Targets

| ID | Target | Required evidence |
|---|---|---|
| GBT-01 | Backend identity | Reference-target simplification reports a backend other than `spatial-cluster` and an algorithm other than `native_spatial_vertex_clustering`. |
| GBT-02 | Topology exportability | Export metrics report no degenerate, duplicate, or nonmanifold export blockers. |
| GBT-03 | Face-count parity | Final faces are within `80-125%` of the checked-in reference final faces. |
| GBT-04 | Texture compatibility | Final visible coverage remains at least `0.50` of reference final coverage and raw coverage remains reported. |
| GBT-05 | Production-tier honesty | `quality_tier=production` is allowed only when backend self-diagnostics and the reference-target threshold gate pass. |
| GBT-06 | Runtime hygiene | Heavy fixture diagnostics include timings/RSS samples and write generated GLB/diagnostics only under `/tmp`. |

## Constraints

- Keep `mlx-spatialkit` independent from MLX, Torch, xatlas, and root `mlx-spatial` runtime imports.
- Do not add Python per-face/per-vertex geometry hot loops.
- Do not relax `_production_thresholds()` to make a backend pass.
- Do not rename `spatial-cluster` to a production backend without a distinct native algorithm and tests that would fail on the current implementation.
- Do not tag, release, publish, push, or change release metadata.
- Keep generated heavy outputs under `/tmp`.

## Risks

- **Algorithm risk:** A topology-aware native backend may be slower or visually worse than spatial clustering on the real fixture. Mitigation: gate production readiness on measured fixture output and keep blockers explicit.
- **Runtime risk:** Full edge-collapse/QEM over millions of faces can be memory-expensive. Mitigation: use bounded native data structures, record RSS/timing, and keep heavy runs under `/tmp`.
- **Quality risk:** A production-tier backend claim can become meaningless if it only changes labels. Mitigation: require backend identity, algorithm identity, topology tests, and heavy threshold proof.
- **Integration risk:** Changing simplifier routing may regress preview/default exports. Mitigation: keep preview route behavior covered separately from reference-target route.

## Acceptance Criteria

| AC | Requirement | Check |
|---|---|---|
| GBT-01 | A non-preview native geometry backend exists and is not an alias of `spatial-cluster`. | Native/package tests assert backend and algorithm identity, stats fields, and topology output on synthetic meshes. |
| GBT-02 | Reference-target routing selects the new backend without changing preview routing. | Export tests prove `quality_preset="reference-target"` requests the new backend while preview/default remains preview-tier. |
| GBT-03 | Production readiness remains threshold-gated. | Tests prove `production_quality_ready` depends on backend tier plus topology, face-count, coverage, raw-reporting, and reference checks; thresholds are not relaxed. |
| GBT-04 | Real Pixal3D fixture proves or blocks the production-tier claim. | Heavy test writes under `/tmp` and asserts backend diagnostics, threshold details, memory samples, and either `production_quality_ready=true` or a specific measured blocker unrelated to a renamed preview backend. |
| GBT-05 | Docs and scripts describe the new boundary honestly. | Docs explain which backend is selected, what production-ready means, and what remains deferred if the gate does not pass. |
| GBT-06 | Repo/package hygiene holds. | `git diff --check`, package tests, root Pixal3D tests, heavy fixture test, build artifact inspection, and `git status --short` stay clean. |

## Scope Coverage Decisions

- **Included:** native backend contract, backend selection/routing for reference-target exports, topology-aware native geometry implementation, synthetic tests, real Pixal3D heavy gate, docs, package/root/build verification.
- **Deferred:** exact xatlas-equivalent unwrap, 4096 texture baking, 1M-face upstream export setting parity, root internal exporter replacement, release/tag/publish work.
- **Anti-goals:** label-only production promotion, Python hot-loop remeshing, checked-in heavy outputs, threshold relaxation, hiding a failed native backend behind `artifact_ready=true`.

## Blocking Questions Or Assumptions

No blocking questions remain.

Assumption: the next useful backend should be a native topology-aware simplifier/remesher selected for reference-target exports. If implementation proves that a full production claim is not justified, the verified outcome must be a measured backend-specific blocker, not a preview-tier relabel.
